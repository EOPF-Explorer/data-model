"""STAC item builder for S1 GRD RTC Zarr V3 stores."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
import pyproj
import pystac
import zarr

SAR_EXT = "https://stac-extensions.github.io/sar/v1.0.0/schema.json"
SAT_EXT = "https://stac-extensions.github.io/sat/v1.0.0/schema.json"
PROJ_EXT = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
RENDER_EXT = "https://stac-extensions.github.io/render/v1.0.0/schema.json"
DATACUBE_EXT = "https://stac-extensions.github.io/datacube/v2.2.0/schema.json"
TIMESTAMPS_EXT = "https://stac-extensions.github.io/timestamps/v1.1.0/schema.json"

ZARR_MEDIA_TYPE = "application/vnd.zarr; version=3"

_ORBIT_PREFERENCE = ("ascending", "descending")
# Short suffix for orbit-keyed asset names (gamma0-rtc-backscatter-asc / -desc).
_ORBIT_SHORT = {"ascending": "asc", "descending": "desc"}

# γ⁰ RTC backscatter is float32 with a NaN fill at every resolution level; the arrays carry no attrs
# in the store so these product invariants are hardcoded (see the store hierarchy in data_api/s1_rtc).
GAMMA0_DTYPE = "float32"
GAMMA0_NODATA = "nan"
GAMMA0_UNIT = "gamma0 (linear power)"
BORDER_MASK_DTYPE = "uint8"
GSD = 10


def _rgb_render(orbit: str) -> dict[str, object]:
    """Build the dual-pol RGB composite render config for the given orbit group.

    Produces a 3-band false-colour composite (R=VV, G=VH, B=VV/VH ratio) that
    titiler renders into previews/tiles. ``bidx=[1]`` selects the single time
    slice from each multi-band variable; the single ``rescale`` pair (linear
    gamma0 units) is applied to every band.
    """
    vv = f"/{orbit}:vv"
    vh = f"/{orbit}:vh"
    return {
        "title": "VV, VH, VV/VH composite",
        "expression": f"{vv};{vh};({vv})/({vh})",
        # Linear gamma0 stretch tuned for the S1 RTC product (the 0.0,0.1 default was too bright).
        "rescale": [[0.0, 0.2]],
        "bidx": [1],
        "tilesize": 256,
    }


def _gamma0_bands() -> list[dict[str, object]]:
    """STAC 1.1 band objects for the two polarisations carried by a γ⁰ RTC asset."""
    return [
        {
            "name": pol,
            "description": f"γ⁰ RTC backscatter, {pol.upper()} polarization",
            "data_type": GAMMA0_DTYPE,
            "nodata": GAMMA0_NODATA,
            "unit": GAMMA0_UNIT,
        }
        for pol in ("vv", "vh")
    ]


def _utm_to_wgs84(proj_code: str, utm_bbox: list[float]) -> tuple[float, float, float, float]:
    """Convert UTM [xmin, ymin, xmax, ymax] to WGS84 (west, south, east, north)."""
    xmin, ymin, xmax, ymax = utm_bbox
    transformer = pyproj.Transformer.from_crs(proj_code, "EPSG:4326", always_xy=True)
    xs = [xmin, xmax, xmin, xmax]
    ys = [ymin, ymin, ymax, ymax]
    lons, lats = transformer.transform(xs, ys)
    return min(lons), min(lats), max(lons), max(lats)


def _bbox_to_geometry(bbox: list[float]) -> dict[str, object]:
    """A closed rectangular Polygon for a WGS84 [west, south, east, north] bbox."""
    west, south, east, north = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [[west, south], [east, south], [east, north], [west, north], [west, south]]
        ],
    }


def _open_root(zarr_store: str) -> zarr.Group:
    """Open the cube root, preferring consolidated metadata.

    A cube grown by appending a time-slice to an *existing same-orbit* group can end up without root
    consolidated metadata (re-consolidating an append on the S3 store is unreliable). The builder must
    not require it — fall back to reading the hierarchy directly, exactly as titiler does. See the
    data-model issue on the S1 RTC consolidated-metadata regression.
    """
    try:
        return zarr.open_consolidated(zarr_store, zarr_format=3)
    except ValueError as exc:
        if "consolidated metadata" not in str(exc).lower():
            raise
        return zarr.open_group(zarr_store, mode="r", zarr_format=3)


def build_s1_rtc_stac_item(zarr_store: str, collection_id: str) -> pystac.Item:
    """Build a STAC item from a consolidated S1 GRD RTC Zarr V3 store.

    Parameters
    ----------
    zarr_store:
        Local path or ``s3://`` URI to the Zarr store.
    collection_id:
        STAC collection ID to attach to the item.

    Returns
    -------
    pystac.Item

    Raises
    ------
    ValueError
        If the store contains no acquisitions.
    """
    # TEMPORARY (#246): the store is written as s1-rtc-{tile}.zarr so its filename equals
    # the item id, which titiler-eopf reconstructs as the render path (it ignores the asset
    # href). Revert this prefix to "s1-grd-rtc-" when titiler-eopf#108 lands.
    tile_id = Path(zarr_store).name.removeprefix("s1-rtc-").removesuffix(".zarr")

    root = _open_root(zarr_store)

    all_times_ns: list[int] = []
    wgs84_bboxes: list[tuple[float, float, float, float]] = []
    # Per present orbit, in preference order: the metadata needed for assets, projection and datacube.
    present: list[dict[str, object]] = []

    for orbit_dir in _ORBIT_PREFERENCE:
        if orbit_dir not in root:
            continue
        og = cast("zarr.Group", root[orbit_dir])
        attrs = dict(og.attrs)
        proj_code = cast("str", attrs["proj:code"])
        utm_bbox = cast("list[float]", attrs["spatial:bbox"])

        r10m = cast("zarr.Group", og["r10m"])
        times = np.array(cast("zarr.Array", r10m["time"])).tolist()
        if not times:
            continue

        # proj:shape / proj:transform live on the r10m group attrs in real stores; read best-effort so
        # minimal/legacy stores without them still build (just without those projection refinements).
        r10m_attrs = dict(r10m.attrs)
        all_times_ns.extend(times)
        wgs84_bboxes.append(_utm_to_wgs84(proj_code, utm_bbox))
        present.append(
            {
                "orbit": orbit_dir,
                "proj_code": proj_code,
                "utm_bbox": utm_bbox,
                "shape": r10m_attrs.get("spatial:shape"),
                "transform": r10m_attrs.get("spatial:transform"),
            }
        )

    if not all_times_ns:
        raise ValueError(f"No acquisitions found in Zarr store: {zarr_store}")

    # Temporal range
    start_dt = dt.datetime.fromtimestamp(min(all_times_ns) / 1e9, tz=dt.UTC)
    end_dt = dt.datetime.fromtimestamp(max(all_times_ns) / 1e9, tz=dt.UTC)

    # WGS84 bbox union across all present orbit directions
    west = min(b[0] for b in wgs84_bboxes)
    south = min(b[1] for b in wgs84_bboxes)
    east = max(b[2] for b in wgs84_bboxes)
    north = max(b[3] for b in wgs84_bboxes)
    wgs84_bbox = [west, south, east, north]

    geometry = _bbox_to_geometry(wgs84_bbox)

    # The preferred orbit (ascending if present) drives the single-valued projection fields and the
    # default render/preview; every present orbit gets its own first-class asset below.
    preferred = present[0]
    preferred_orbit = cast("str", preferred["orbit"])
    preferred_proj_code = cast("str", preferred["proj_code"])
    preferred_bbox = cast("list[float]", preferred["utm_bbox"])

    properties: dict[str, object] = {
        "start_datetime": start_dt.isoformat(),
        "end_datetime": end_dt.isoformat(),
        "title": f"Sentinel-1 GRD RTC γ⁰ — tile {tile_id}",
        "description": (
            "Radiometric-terrain-corrected (RTC) γ⁰ backscatter datacube from Sentinel-1 GRD, "
            "reprojected onto the Sentinel-2 MGRS/UTM grid."
        ),
        # `updated` (timestamps extension) tracks this metadata build. `created` is intentionally
        # omitted: it means the item's creation instant, which the store does not record — using an
        # acquisition time would misuse the field, and a build-time value would churn on every append.
        "updated": dt.datetime.now(tz=dt.UTC).isoformat(),
        # Identity invariants (constant across the cube; platform is per-acquisition so omitted here —
        # a cube can mix S1A and S1C).
        "constellation": "sentinel-1",
        "instruments": ["c-sar"],
        "gsd": GSD,
        # SAR extension
        "sar:instrument_mode": "IW",
        "sar:frequency_band": "C",
        "sar:center_frequency": 5.405,
        "sar:polarizations": ["VV", "VH"],
        "sar:product_type": "GRD",
        # Projection extension
        "proj:code": preferred_proj_code,
        "proj:bbox": preferred_bbox,
        # Render extension: dual-pol RGB composite for previews/tiles (defaults to the preferred orbit)
        "renders": {"rgb": _rgb_render(preferred_orbit)},
    }
    if preferred["shape"] is not None:
        properties["proj:shape"] = preferred["shape"]
    if preferred["transform"] is not None:
        properties["proj:transform"] = preferred["transform"]

    stac_extensions = [SAR_EXT, PROJ_EXT, RENDER_EXT, DATACUBE_EXT, TIMESTAMPS_EXT]

    # sat:orbit_state is single-valued, so it's only meaningful when the cube holds a single orbit. A
    # dual-orbit cube would mislabel half its slices — omit it there (per-acquisition items, which are
    # single-orbit, carry the real value). Only declare the SAT extension when the field is set.
    if len(present) == 1:
        properties["sat:orbit_state"] = preferred_orbit
        stac_extensions.append(SAT_EXT)

    # Datacube extension. The time axis is irregularly sampled, so it lists its discrete `values` (the
    # acquisition instants across all orbit groups, sorted) — their count is the number of time steps,
    # and the list stays modest (bounded by the tile's acquisitions). The regular x/y axes instead carry
    # extent + step (their element count is derivable, and the exact pixel count is in proj:shape);
    # enumerating their ~10⁴ coordinates would not scale.
    epsg = pyproj.CRS.from_user_input(preferred_proj_code).to_epsg()
    xmin, ymin, xmax, ymax = preferred_bbox
    time_values = [
        dt.datetime.fromtimestamp(t / 1e9, tz=dt.UTC).isoformat() for t in sorted(set(all_times_ns))
    ]
    time_dim: dict[str, object] = {
        "type": "temporal",
        "extent": [start_dt.isoformat(), end_dt.isoformat()],
        "values": time_values,
    }
    if len(present) > 1:
        # The cube merges two per-orbit sub-cubes (disjoint time axes) onto a shared grid. Orbit is an
        # attribute of each acquisition, not an independent axis, so it is conveyed via the per-orbit
        # assets rather than a (sparse) orbit dimension — note that here to avoid misreading the axis.
        time_dim["description"] = (
            "Acquisition instants across both orbit directions (union); each step belongs to a single "
            "orbit — the ascending/descending groups are exposed as separate assets and as items in "
            "the per-acquisition collection."
        )
    x_dim: dict[str, object] = {
        "type": "spatial",
        "axis": "x",
        "extent": [xmin, xmax],
        "reference_system": epsg,
    }
    y_dim: dict[str, object] = {
        "type": "spatial",
        "axis": "y",
        "extent": [ymin, ymax],
        "reference_system": epsg,
    }
    if preferred["transform"] is not None:
        transform = cast("list[float]", preferred["transform"])
        x_dim["step"] = transform[0]
        y_dim["step"] = transform[4]
    properties["cube:dimensions"] = {"time": time_dim, "x": x_dim, "y": y_dim}
    # `variable_type` is the datacube field name (not `type`); the border mask is auxiliary, not data.
    properties["cube:variables"] = {
        "vv": {"dimensions": ["time", "y", "x"], "variable_type": "data", "unit": GAMMA0_UNIT},
        "vh": {"dimensions": ["time", "y", "x"], "variable_type": "data", "unit": GAMMA0_UNIT},
        "border_mask": {"dimensions": ["time", "y", "x"], "variable_type": "auxiliary"},
    }

    item = pystac.Item(
        id=f"s1-rtc-{tile_id}",
        geometry=geometry,
        bbox=wgs84_bbox,
        datetime=None,
        properties=properties,
        stac_extensions=stac_extensions,
        collection=collection_id,
    )

    store_str = str(zarr_store)
    item.add_asset(
        "zarr-store",
        pystac.Asset(
            href=store_str,
            media_type=ZARR_MEDIA_TYPE,
            roles=["data"],
            title="Sentinel-1 GRD RTC Zarr store",
        ),
    )

    # One γ⁰ asset per present orbit group (fixes the duplicate-href vv/vh ambiguity and the missing
    # descending asset): VV/VH are addressable as named `bands`, not indistinguishable duplicate assets.
    # A separate border-mask asset exposes the valid-data mask variable in the same group.
    for info in present:
        orbit = cast("str", info["orbit"])
        short = _ORBIT_SHORT[orbit]
        group_href = f"{store_str}/{orbit}"
        item.add_asset(
            f"gamma0-rtc-backscatter-{short}",
            pystac.Asset(
                href=group_href,
                media_type=ZARR_MEDIA_TYPE,
                roles=["data"],
                title=f"γ⁰ RTC backscatter ({orbit})",
                extra_fields={
                    "bands": _gamma0_bands(),
                    "data_type": GAMMA0_DTYPE,
                    "nodata": GAMMA0_NODATA,
                    "unit": GAMMA0_UNIT,
                    "gsd": GSD,
                },
            ),
        )
        item.add_asset(
            f"border-mask-{short}",
            pystac.Asset(
                href=group_href,
                media_type=ZARR_MEDIA_TYPE,
                roles=["data"],
                title=f"Valid-data mask ({orbit})",
                extra_fields={
                    "bands": [
                        {
                            "name": "border_mask",
                            "description": "Valid-data mask (0 = border/no-data, non-zero = valid)",
                            "data_type": BORDER_MASK_DTYPE,
                            "nodata": 0,
                        }
                    ],
                    "gsd": GSD,
                },
            ),
        )

    return item


# ============================================================================
# Per-acquisition item construction (one queryable item per cube `time` slice)
# ============================================================================

# Default the cube preview to the most recent acquisition covering most of the tile, so a browser shows
# fresh near-full data rather than the oldest slice.
COVERAGE_THRESHOLD = 0.80


class Slice(NamedTuple):
    """One cube time slice: its orbit group, acquisition instant, and tile coverage fraction (0..1)."""

    orbit: str
    dt: dt.datetime
    coverage: float


def pick_slice(slices: list[Slice]) -> Slice | None:
    """Choose the slice the cube preview should default to.

    The most recent acquisition with coverage strictly above ``COVERAGE_THRESHOLD``; if none clears it,
    the highest-coverage slice (ties broken by most recent). Spans both orbit groups. Returns ``None``
    for an empty cube.
    """
    if not slices:
        return None
    good = [s for s in slices if s.coverage > COVERAGE_THRESHOLD]
    if good:
        return max(good, key=lambda s: s.dt)
    return max(slices, key=lambda s: (s.coverage, s.dt))


def slice_coverages(zarr_store: str) -> list[Slice]:
    """Per-slice tile coverage from the cube, across both orbit groups.

    Reads ``border_mask`` at the cheap ``r720m`` overview only (~150x150). Coverage is the fraction of
    **valid** pixels; the S1Tiling border mask is stored with ``fill_value=0`` for the border, so valid
    = non-zero. ``time`` is raw int64 ns (as :func:`build_s1_rtc_stac_item` reads it) -> UTC datetime.
    """
    root = _open_root(zarr_store)
    out: list[Slice] = []
    for orbit in _ORBIT_PREFERENCE:
        if orbit not in root:
            continue
        level = cast("zarr.Group", cast("zarr.Group", root[orbit])["r720m"])
        mask = np.asarray(cast("zarr.Array", level["border_mask"]))  # (time, y, x), uint8
        times_ns = np.asarray(cast("zarr.Array", level["time"])).tolist()  # int64 ns since epoch
        for i, t_ns in enumerate(times_ns):
            sl = mask[i]
            coverage = float(np.count_nonzero(sl) / sl.size)
            out.append(Slice(orbit, dt.datetime.fromtimestamp(t_ns / 1e9, tz=dt.UTC), coverage))
    return out


def acquisition_id(tile_id: str, when: dt.datetime) -> str:
    """Per-acquisition item id, e.g. ``s1-rtc-31TCH-20260607t055248``."""
    return f"s1-rtc-{tile_id}-{when.strftime('%Y%m%dt%H%M%S')}"


def _normalize_platform(raw: object) -> str | None:
    """Map the store's short platform code (e.g. ``s1a``) to the STAC convention (``sentinel-1a``).

    Mirrors the Sentinel-2 reference (``sentinel-2a``). Unknown values are returned lower-cased.
    """
    s = str(raw).strip().lower()
    if not s:
        return None
    if len(s) == 3 and s.startswith("s1"):
        return f"sentinel-1{s[2]}"
    return s


def build_s1_rtc_per_acquisition_items(
    zarr_store: str, *, orbit: str, collection_id: str
) -> list[pystac.Item]:
    """Build one queryable STAC item per cube ``time`` slice of a single orbit group.

    Each item is a single-``datetime`` view into the shared cube (no data duplication): it keeps the
    cube's geometry/SAR/projection metadata and the orbit's γ⁰ asset, drops the temporal range and the
    datacube structure (a single acquisition is not a cube), and is reoriented to ``orbit``. The item
    is deployment-agnostic — it carries the render config + datetime, and the registration layer derives
    the TiTiler links (which point at the cube endpoint with ``sel=time={datetime}``) from it.

    Parameters
    ----------
    zarr_store:
        Local path or ``s3://`` URI to the per-tile cube Zarr store.
    orbit:
        Orbit group to emit items for (``"ascending"`` or ``"descending"``).
    collection_id:
        Target (per-acquisition) STAC collection ID.

    Raises
    ------
    ValueError
        If the store has no acquisitions, or ``orbit`` is not present in the store.
    """
    if orbit not in _ORBIT_PREFERENCE:
        raise ValueError(f"orbit must be one of {_ORBIT_PREFERENCE}, got {orbit!r}")

    tile_id = Path(zarr_store).name.removeprefix("s1-rtc-").removesuffix(".zarr")

    root = _open_root(zarr_store)
    if orbit not in root:
        raise ValueError(f"Orbit group {orbit!r} not found in Zarr store: {zarr_store}")
    r10m = cast("zarr.Group", cast("zarr.Group", root[orbit])["r10m"])
    times_ns = np.array(cast("zarr.Array", r10m["time"])).tolist()
    platforms = np.array(cast("zarr.Array", r10m["platform"])).tolist()
    if not times_ns:
        raise ValueError(f"No acquisitions found for orbit {orbit!r} in: {zarr_store}")

    base = build_s1_rtc_stac_item(zarr_store, collection_id)
    base_dict = base.to_dict(include_self_link=False)

    # Assets to drop from each per-acquisition clone: the *other* orbit's groups (a per-acq item
    # represents one orbit). The datacube structure is dropped too — a single acquisition is not a cube.
    other_assets = {
        key
        for o in _ORBIT_PREFERENCE
        if o != orbit
        for key in (f"gamma0-rtc-backscatter-{_ORBIT_SHORT[o]}", f"border-mask-{_ORBIT_SHORT[o]}")
    }

    # A per-acquisition item covers only its run orbit's footprint — not the cube's union of both
    # orbits' extents (which the base item carries). Recompute bbox/geometry/proj:bbox from this orbit.
    # proj:code/shape/transform describe the shared MGRS grid (identical across orbits), so the values
    # inherited from the base (preferred orbit) are correct and are intentionally not recomputed here.
    og_attrs = dict(cast("zarr.Group", root[orbit]).attrs)
    orbit_utm_bbox = cast("list[float]", og_attrs["spatial:bbox"])
    orbit_wgs84_bbox = list(_utm_to_wgs84(cast("str", og_attrs["proj:code"]), orbit_utm_bbox))
    orbit_geometry = _bbox_to_geometry(orbit_wgs84_bbox)

    items: list[pystac.Item] = []
    for t_ns, platform in zip(times_ns, platforms, strict=True):
        when = dt.datetime.fromtimestamp(t_ns / 1e9, tz=dt.UTC)
        item_dict = {**base_dict}
        item_dict["id"] = acquisition_id(tile_id, when)
        item_dict["bbox"] = orbit_wgs84_bbox
        item_dict["geometry"] = orbit_geometry

        props = {
            k: v
            for k, v in base_dict["properties"].items()
            if k not in ("start_datetime", "end_datetime", "cube:dimensions", "cube:variables")
        }
        props["datetime"] = when.isoformat()
        props["sat:orbit_state"] = orbit
        props["proj:bbox"] = orbit_utm_bbox
        # Per-acquisition title carries the datetime + orbit so sibling scenes are distinguishable
        # (the inherited cube title "… — tile {id}" is identical across all acquisitions).
        props["title"] = (
            f"Sentinel-1 GRD RTC γ⁰ — tile {tile_id}, "
            f"{when.strftime('%Y-%m-%dT%H:%M:%SZ')} ({orbit})"
        )
        props["description"] = (
            "Radiometric-terrain-corrected (RTC) γ⁰ backscatter from a single Sentinel-1 GRD "
            "acquisition, reprojected onto the Sentinel-2 MGRS/UTM grid."
        )
        normalized = _normalize_platform(platform)
        if normalized:
            props["platform"] = normalized
        props["renders"] = {"rgb": _rgb_render(orbit)}
        item_dict["properties"] = props

        # Drop the datacube ext (a single acquisition is not a cube). Ensure the SAT ext is declared:
        # a per-acq item always sets sat:orbit_state, but a dual-orbit cube base omits both.
        exts = [e for e in base_dict.get("stac_extensions", []) if e != DATACUBE_EXT]
        if SAT_EXT not in exts:
            exts.append(SAT_EXT)
        item_dict["stac_extensions"] = exts
        item_dict["assets"] = {
            k: v for k, v in base_dict["assets"].items() if k not in other_assets
        }
        item_dict["links"] = []

        items.append(pystac.Item.from_dict(item_dict))
    return items
