"""STAC item builder for S1 GRD RTC Zarr V3 stores."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
import pyproj
import pystac
import zarr

from eopf_geozarr.conversion import fs_utils
from eopf_geozarr.types import BoundingBox2D, CRSCode, make_bounding_box, make_crs_code

SAR_EXT = "https://stac-extensions.github.io/sar/v1.0.0/schema.json"
SAT_EXT = "https://stac-extensions.github.io/sat/v1.0.0/schema.json"
PROJ_EXT = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
RENDER_EXT = "https://stac-extensions.github.io/render/v1.0.0/schema.json"
DATACUBE_EXT = "https://stac-extensions.github.io/datacube/v2.2.0/schema.json"
GRID_EXT = "https://stac-extensions.github.io/grid/v1.1.0/schema.json"
# No TIMESTAMPS_EXT: `created`/`updated` are STAC *Common Metadata*, present in the core Item spec.
# The timestamps extension only adds `published`/`expires`/`unpublished`, none of which this builder
# emits, so declaring it was inert — a validator downloaded a schema that constrained nothing.

ZARR_MEDIA_TYPE = "application/vnd.zarr; version=3"

# MGRS tile id as written by S1Tiling / the Sentinel-2 grid: two-digit UTM zone (01-60), latitude
# band, then the 100 km square column/row letters. `I` and `O` are excluded throughout (they read as
# 1/0), and row letters stop at V. Anchored, because the id becomes both the STAC item id and the
# queryable `grid:code` — see `_tile_id_from_store`.
_MGRS_TILE_RE = re.compile(r"^(0[1-9]|[1-5][0-9]|60)[C-HJ-NP-X][A-HJ-NP-Z][A-HJ-NP-V]$")

# Store-name prefixes the tile id may hide behind. "s1-rtc-" is the current one; "s1-grd-rtc-" is
# what #246 reverts to once titiler-eopf#108 lands. Accepting both means that rename does not turn
# every store into a hard build failure the day it happens (see `_tile_id_from_store`).
_STORE_PREFIXES = ("s1-rtc-", "s1-grd-rtc-")

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
    slice from each multi-band variable. Each band gets its own ``rescale`` pair:
    VV and VH are linear gamma0 (low values) while the VV/VH ratio spans ~1-15, so
    a single shared pair blew the ratio band out to a flat blue/purple wash (and
    dropped low-cross-pol water to transparent, making swaths look mislocated).
    Per-band stretches keep the composite natural and the swath readable.
    """
    vv = f"/{orbit}:vv"
    vh = f"/{orbit}:vh"
    return {
        # `assets` is the ONE required field of a Render Object (render v1.0.0,
        # `definitions/fields.required`). Omitting it made every emitted item fail the extension it
        # declared — `'assets' is a required property` — so `generate-stac-s1 | stac-validator`
        # failed and a validating STAC API refused the item. It names the orbit-keyed γ⁰ asset this
        # composite reads from, which is exactly the asset built below.
        "assets": [f"gamma0-rtc-backscatter-{_ORBIT_SHORT[orbit]}"],
        "title": "VV, VH, VV/VH composite",
        "expression": f"{vv};{vh};({vv})/({vh})",
        # Per-band linear stretch: VV/VH are low-valued gamma0; the VV/VH ratio spans ~1-15.
        # One shared pair saturated the ratio band (purple wash) and dropped low-cross-pol water.
        "rescale": [[0.0, 0.4], [0.0, 0.1], [1.0, 15.0]],
        "bidx": [1],
        # Not a render-extension field, but the schema sets `additionalProperties: true` on the
        # Render Object, so it validates. titiler reads it to size the tiles it renders from this
        # config; dropping it would silently fall back to titiler's default tile size.
        "tilesize": 256,
    }


def _providers() -> list[dict[str, object]]:
    """Attribution for the γ⁰ RTC product, copied down from the collection block.

    A STAC API search returns bare items, without the collection that would otherwise supply this,
    so an item that omits `providers` loses its attribution entirely for every consumer that does
    not follow the collection link. Kept in sync by hand with the collection templates
    (`stac/sentinel-1-grd-rtc*.json` in data-pipeline), which stay the source of truth — the
    collection is registered by the operator, not by this builder.

    Returned fresh per call: the dicts land in `item.properties` and must not be shared between
    items (mutating one item's providers would rewrite every other item's).
    """
    return [
        {
            "name": "European Commission",
            "roles": ["licensor"],
            "url": "https://commission.europa.eu/",
        },
        {
            "name": "ESA",
            "roles": ["producer", "processor"],
            "url": "https://sentinel.esa.int/web/sentinel/missions/sentinel-1",
        },
        {
            "name": "EOPF Sentinel Zarr Samples Service",
            "roles": ["host", "processor"],
            "url": "https://zarr.eopf.copernicus.eu/",
        },
    ]


def _tile_id_from_store(zarr_store: str) -> str:
    """Derive (and validate) the MGRS tile id from the store name.

    The store is written as ``s1-rtc-{tile}.zarr`` so its basename equals the item id, which
    titiler-eopf reconstructs as the render path (it ignores the asset href) — see the TEMPORARY
    note on #246 in :func:`build_s1_rtc_stac_item`.

    Stripping the prefix/suffix and *trusting* the remainder silently minted malformed ids from any
    other store name: ``cube.zarr`` produced item id ``s1-rtc-cube`` and ``grid:code`` ``MGRS-cube``,
    ``s1-rtc-.zarr`` produced ``s1-rtc-`` / ``MGRS-``, and (before ``s1-grd-rtc-`` was accepted here)
    ``s1-grd-rtc-31TCH.zarr`` produced ``s1-rtc-s1-grd-rtc-31TCH`` / ``MGRS-s1-grd-rtc-31TCH``. None
    of these fail anywhere downstream: they register cleanly and poison the catalogue, because
    ``grid:code`` is the field tile-filtered searches and the cube↔acquisition cross-links join on.
    Validate instead, and fail the build where the mistake is still cheap to fix.

    Raises
    ------
    ValueError
        If the store name does not carry a well-formed MGRS tile id.
    """
    name = Path(zarr_store).name
    tile_id = name.removesuffix(".zarr")
    for prefix in _STORE_PREFIXES:
        tile_id = tile_id.removeprefix(prefix)
    if not _MGRS_TILE_RE.match(tile_id):
        raise ValueError(
            f"Cannot derive an MGRS tile id from store name {name!r} (got {tile_id!r}). "
            f"Expected a store named {{{'|'.join(_STORE_PREFIXES)}}}<MGRS tile>.zarr, "
            "e.g. s1-rtc-31TCH.zarr."
        )
    return tile_id


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


def _utm_to_wgs84(proj_code: CRSCode, utm_bbox: BoundingBox2D) -> BoundingBox2D:
    """Convert UTM (xmin, ymin, xmax, ymax) to WGS84 (west, south, east, north).

    Transforming only the four corners is wrong twice over. It understates the footprint everywhere
    (projected edges curve in lon/lat, so the extreme lat/lon is mid-edge, not at a corner), and in
    UTM zones 1 and 60 it produces a near-global box: tile 01VCK at 300000-409800E in EPSG:32601
    corner-transforms to ``(-178.41, 54.11, 179.94, 55.13)`` — 358.4 degrees wide, spanning almost
    the whole planet — because the west edge lands just *east* of the antimeridian and the east edge
    just *west* of it, and min/max then straddle the wrap instead of the tile.

    ``Transformer.transform_bounds(..., densify_pts=21)`` fixes both: it samples along the edges and
    returns the antimeridian-crossing box ``(179.87, 54.11, -178.38, 55.13)``, where ``west > east``
    — the representation STAC and GeoJSON (RFC 7946 §5.2) prescribe for a crossing bbox.
    Callers must therefore not assume ``west <= east``; see ``_union_wgs84`` and
    ``_bbox_to_geometry``, which both handle the crossing case.

    Duplicated (deliberately, in the densification only) from
    :func:`eopf_geozarr.conversion.utils.write_store_root_geo_metadata`, which reprojects the
    store-root footprint the same way. Not extracted into a shared helper: the shared part is the
    one ``transform_bounds`` call, and the two call sites want *opposite* crossing behaviour — the
    store-root writer widens a crossing union to the full longitude range, while a STAC item must
    keep the narrow crossing box so spatial search stays selective.
    """
    transformer = pyproj.Transformer.from_crs(proj_code, "EPSG:4326", always_xy=True)
    return BoundingBox2D(transformer.transform_bounds(*utm_bbox, densify_pts=21))


def _union_wgs84(bboxes: list[BoundingBox2D]) -> BoundingBox2D:
    """Union WGS84 bboxes, keeping an antimeridian crossing narrow rather than wrapping the globe.

    A plain ``min(west)/max(east)`` is only correct while no input crosses the antimeridian; on a
    crossing box (``west > east``, see :func:`_utm_to_wgs84`) it inverts the box straight back into
    the 358-degree monster the densification just removed. Instead every box is unwrapped into one
    continuous frame anchored on the first box's west edge — a crossing box gets ``east += 360``,
    and a box a whole turn away from the anchor is shifted by 360 so the two are comparable — then
    the union is folded back into [-180, 180], keeping ``west > east`` if it still crosses.
    """
    anchor = bboxes[0][0]
    spans: list[tuple[float, float]] = []
    for west_i, _south, east_i, _north in bboxes:
        if east_i < west_i:
            east_i += 360.0
        offset = 360.0 * round((anchor - west_i) / 360.0)
        spans.append((west_i + offset, east_i + offset))

    west = min(s[0] for s in spans)
    east = max(s[1] for s in spans)
    if east > 180.0:
        east -= 360.0
    if west > 180.0:
        west -= 360.0
    elif west < -180.0:
        west += 360.0
    return BoundingBox2D((west, min(b[1] for b in bboxes), east, max(b[3] for b in bboxes)))


def _bbox_to_geometry(bbox: BoundingBox2D) -> dict[str, object]:
    """A closed rectangular Polygon for a WGS84 [west, south, east, north] bbox.

    A bbox that crosses the antimeridian (``west > east``) becomes a two-part MultiPolygon split at
    ±180: a single ring from ``west`` to ``east`` would run the *long* way round the globe, drawing
    a footprint covering everything except the tile. GeoJSON (RFC 7946 §3.1.9) requires the split.
    """
    west, south, east, north = bbox
    if west > east:
        # Drop a part with no width. An edge landing exactly on ±180 makes one half of the split
        # degenerate, and a zero-area ring is rejected by geometry stacks that check validity
        # (PostGIS ST_IsValid, shapely-based ingest) even though RFC 7946 does not forbid it.
        parts = [[_ring(w, south, e, north)] for w, e in ((west, 180.0), (-180.0, east)) if w != e]
        if len(parts) == 1:
            return {"type": "Polygon", "coordinates": parts[0]}
        return {"type": "MultiPolygon", "coordinates": parts}
    return {"type": "Polygon", "coordinates": [_ring(west, south, east, north)]}


def _ring(west: float, south: float, east: float, north: float) -> list[list[float]]:
    """A closed counter-clockwise rectangular ring."""
    return [[west, south], [east, south], [east, north], [west, north], [west, south]]


def _open_root(zarr_store: str) -> zarr.Group:
    """Open the cube root read-only, using consolidated metadata when it is present.

    A cube mid-ingest normally has no root consolidated block: `ingest_s1tiling_acquisition`
    strips it after every append so external readers cannot act on a stale one. The builder must
    therefore read the hierarchy directly when the block is absent, exactly as titiler does.

    ``mode="r"`` is load-bearing. ``zarr.open_consolidated`` is an alias for ``open_group``, whose
    default is ``mode="a"``, so the previous implementation *created* a store at any path that did
    not resolve — ``generate-stac-s1 --store s3://bucket/typo.zarr`` wrote into the bucket before
    failing. ``use_consolidated=None`` is zarr's documented use-if-present-else-list mode, which
    removes the need to discriminate a missing consolidated block from a missing store by
    exception type (both surface as ``ValueError`` on zarr 3.2.0).
    """
    return zarr.open_group(
        zarr_store,
        mode="r",
        zarr_format=3,
        use_consolidated=None,
        storage_options=cast("dict[str, object] | None", fs_utils.get_storage_options(zarr_store)),
    )


def _orbit_numbers(r10m: zarr.Group, name: str) -> list[int | None]:
    """Per-acquisition ``absolute_orbit`` / ``relative_orbit`` values from an r10m group.

    ``s1_ingest.py`` writes both as int32 coordinates on the ``time`` axis at native resolution
    only. Read best-effort — the same posture the r10m ``spatial:*`` attrs get above — so minimal
    and pre-#216 stores that predate these coordinates still build an item, just without the
    ``sat:*_orbit`` fields.
    """
    if name not in r10m:
        return []
    # `0` means "not recorded", never orbit zero. Both arrays are created with `fill_value=0`, and
    # the append resizes all three metadata coordinates before writing them, so a torn append
    # leaves a correctly-shaped slice holding 0 — and there is a known upstream bug putting 0 on
    # some live slices. The sat extension declares both fields `{"type": "integer", "minimum": 1}`,
    # so emitting a 0 makes a validating STAC API reject the item outright. Map it to None here
    # rather than dropping it, so the per-slice zip below stays aligned with `time`.
    return [
        int(v) if int(v) > 0 else None for v in np.asarray(cast("zarr.Array", r10m[name])).tolist()
    ]


def _single(values: list[int | None], expected: int) -> int | None:
    """The one distinct value in *values*, or ``None`` unless all *expected* slices agree.

    ``sat:relative_orbit`` / ``sat:absolute_orbit`` are single-valued STAC fields, so a cube can only
    carry them when its acquisitions agree. In practice a single-orbit tile cube shares one relative
    orbit (one track) and so gets the field, while absolute orbit is unique per acquisition and so is
    dropped from any multi-acquisition cube — the per-acquisition items are where it is exact.

    ``expected`` is the cube's total slice count, and requiring it guards a subtler case than
    disagreement: values are pooled only from orbit groups that actually carry the coordinate, so a
    dual-orbit cube with one group written before these coordinates existed would otherwise pool
    just the other group's values and confidently assert one track for acquisitions from *both*
    orbit directions. A `None` anywhere (an unrecorded slice) likewise means the cube cannot claim
    a single value.
    """
    if len(values) != expected or any(v is None for v in values):
        return None
    distinct = set(values)
    return distinct.pop() if len(distinct) == 1 else None


class _OrbitInfo(NamedTuple):
    """Per-orbit-group metadata driving assets, projection fields and the datacube extension."""

    orbit: str
    proj_code: CRSCode
    utm_bbox: BoundingBox2D
    shape: object | None
    transform: object | None


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
        If the store contains no acquisitions, or its name carries no valid MGRS tile id.
    """
    # TEMPORARY (#246): the store is written as s1-rtc-{tile}.zarr so its filename equals
    # the item id, which titiler-eopf reconstructs as the render path (it ignores the asset
    # href). `_STORE_PREFIXES` accepts both this name and the "s1-grd-rtc-" one it reverts to
    # once titiler-eopf#108 lands; drop the stale entry then.
    tile_id = _tile_id_from_store(zarr_store)

    root = _open_root(zarr_store)

    all_times_ns: list[int] = []
    # Per-acquisition orbit numbers, pooled across orbit groups; see the single-value guard below.
    all_absolute_orbits: list[int | None] = []
    all_relative_orbits: list[int | None] = []
    wgs84_bboxes: list[BoundingBox2D] = []
    # Per present orbit, in preference order: the metadata needed for assets, projection and datacube.
    present: list[_OrbitInfo] = []

    for orbit_dir in _ORBIT_PREFERENCE:
        if orbit_dir not in root:
            continue
        og = cast("zarr.Group", root[orbit_dir])
        attrs = dict(og.attrs)
        proj_code = make_crs_code(attrs["proj:code"])
        utm_bbox = make_bounding_box(attrs["spatial:bbox"])

        r10m = cast("zarr.Group", og["r10m"])
        times = np.array(cast("zarr.Array", r10m["time"])).tolist()
        if not times:
            continue

        # proj:shape / proj:transform live on the r10m group attrs in real stores; read best-effort so
        # minimal/legacy stores without them still build (just without those projection refinements).
        r10m_attrs = dict(r10m.attrs)
        all_times_ns.extend(times)
        all_absolute_orbits.extend(_orbit_numbers(r10m, "absolute_orbit"))
        all_relative_orbits.extend(_orbit_numbers(r10m, "relative_orbit"))
        wgs84_bboxes.append(_utm_to_wgs84(proj_code, utm_bbox))
        present.append(
            _OrbitInfo(
                orbit=orbit_dir,
                proj_code=proj_code,
                utm_bbox=utm_bbox,
                shape=r10m_attrs.get("spatial:shape"),
                transform=r10m_attrs.get("spatial:transform"),
            )
        )

    if not all_times_ns:
        raise ValueError(f"No acquisitions found in Zarr store: {zarr_store}")

    # Temporal range
    start_dt = dt.datetime.fromtimestamp(min(all_times_ns) / 1e9, tz=dt.UTC)
    end_dt = dt.datetime.fromtimestamp(max(all_times_ns) / 1e9, tz=dt.UTC)

    # WGS84 bbox union across all present orbit directions (antimeridian-aware — see _union_wgs84)
    wgs84_bbox = _union_wgs84(wgs84_bboxes)

    geometry = _bbox_to_geometry(wgs84_bbox)

    # The preferred orbit (ascending if present) drives the single-valued projection fields and the
    # default render/preview; every present orbit gets its own first-class asset below.
    preferred = present[0]
    preferred_orbit = preferred.orbit
    preferred_proj_code = preferred.proj_code
    preferred_bbox = preferred.utm_bbox

    build_time = dt.datetime.now(tz=dt.UTC).isoformat()
    # Built once and shared by the item root and the `properties` mirror below, so the two cannot
    # drift apart. Nothing mutates a render config after construction.
    cube_render = {"rgb": _rgb_render(preferred_orbit)}
    properties: dict[str, object] = {
        "start_datetime": start_dt.isoformat(),
        "end_datetime": end_dt.isoformat(),
        "title": f"Sentinel-1 GRD RTC γ⁰ — tile {tile_id}",
        "description": (
            "Radiometric-terrain-corrected (RTC) γ⁰ backscatter datacube from Sentinel-1 GRD, "
            "reprojected onto the Sentinel-2 MGRS/UTM grid."
        ),
        # STAC Common Metadata (NOT the timestamps extension — that only adds published/expires/
        # unpublished, which this builder never sets). Both mark this metadata build: `created` is
        # required by the S1 RTC collection, and the store records no separate item-creation instant,
        # so build time is the only honest value available. It does churn when a cube is rebuilt
        # after an append; that is the accepted cost of emitting the field at all.
        "created": build_time,
        "updated": build_time,
        # Attribution, copied down from the collection block: a STAC API search returns items
        # without their collection, so an item that omits `providers` has no attribution at all.
        "providers": _providers(),
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
        "proj:bbox": list(preferred_bbox),
        # Grid extension: the Sentinel-2 MGRS tile this cube is gridded onto — a queryable tile id
        # (enables tile-filtering the acquisitions collection and cube↔acquisition cross-links).
        "grid:code": f"MGRS-{tile_id}",
        # BOTH COPIES ARE LOAD-BEARING -- do NOT delete this one. `renders` belongs at the item
        # root (see the note on `extra_fields` below), but the STAC API this feeds drops root-level
        # keys it does not model: its `Item-Input` write schema declares exactly ten root fields
        # and no `additionalProperties`, while `ItemProperties` and `Asset` in the same OpenAPI
        # document do allow extras. A registered item is therefore expected to come back with
        # `properties.renders` and nothing at the root, making this the copy a consumer reading a
        # render config back OUT of the catalogue gets. Both are built from the same
        # `_rgb_render(preferred_orbit)` call, so they cannot disagree.
        "renders": cube_render,
    }
    if preferred.shape is not None:
        properties["proj:shape"] = preferred.shape
    if preferred.transform is not None:
        properties["proj:transform"] = preferred.transform

    stac_extensions = [SAR_EXT, PROJ_EXT, RENDER_EXT, DATACUBE_EXT, GRID_EXT]

    # Every sat:* field is single-valued, so each is set only where the cube actually agrees on it,
    # and the SAT extension is declared only if at least one landed.
    #   - sat:orbit_state: a dual-orbit cube would mislabel half its slices, so omit it there.
    #   - sat:relative_orbit: constant while the tile is covered by one track (the usual case).
    #   - sat:absolute_orbit: unique per acquisition, so a multi-acquisition cube drops it.
    # The per-acquisition items are single-orbit and single-slice, so they carry all three exactly.
    if len(present) == 1:
        properties["sat:orbit_state"] = preferred_orbit
    for field, values in (
        ("sat:relative_orbit", all_relative_orbits),
        ("sat:absolute_orbit", all_absolute_orbits),
    ):
        single = _single(values, len(all_times_ns))
        if single is not None:
            properties[field] = single
    if any(k.startswith("sat:") for k in properties):
        stac_extensions.append(SAT_EXT)

    # Datacube extension. The time axis is irregularly sampled, so it lists its discrete `values` (the
    # acquisition instants across all orbit groups, sorted) — their count is the number of time steps,
    # and the list stays modest (bounded by the tile's acquisitions). The regular x/y axes instead carry
    # extent + step (their element count is derivable, and the exact pixel count is in proj:shape);
    # enumerating their ~10⁴ coordinates would not scale.
    # `to_epsg()` returns None for any CRS with no EPSG match — a WKT2/PROJJSON `proj:code`, or a
    # custom projection — and that None flowed straight into `reference_system`, emitting a literal
    # `null` that tells a reader nothing about the grid. datacube v2.2.0 accepts an EPSG integer, a
    # WKT2 string or a PROJJSON object there, so fall back to WKT2 rather than to null.
    preferred_crs = pyproj.CRS.from_user_input(preferred_proj_code)
    epsg_or_wkt: object = preferred_crs.to_epsg()
    if epsg_or_wkt is None:
        epsg_or_wkt = preferred_crs.to_wkt()
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
        "reference_system": epsg_or_wkt,
    }
    y_dim: dict[str, object] = {
        "type": "spatial",
        "axis": "y",
        "extent": [ymin, ymax],
        "reference_system": epsg_or_wkt,
    }
    if preferred.transform is not None:
        transform = cast("list[float]", preferred.transform)
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
        bbox=list(wgs84_bbox),
        datetime=None,
        properties=properties,
        stac_extensions=stac_extensions,
        collection=collection_id,
        # `renders` belongs at the ITEM ROOT, not in `properties`. The render v1.0.0 Item branch
        # is `required: ["type", "assets", "renders"]` against the item object itself, and the
        # extension's own examples/item-landsat8.json puts it at the root — only its README's
        # "Item Properties" table says otherwise, and that contradicts both. Emitting it under
        # `properties` made every item fail the extension it declared, with the misleading
        # message "'renders' is a required property".
        #
        # It is ALSO mirrored into `properties` above, and that mirror is NOT temporary. The
        # deployed STAC API models the write body as `Item-Input`, whose ten root fields do not
        # include `renders` and which sets no `additionalProperties` -- while `ItemProperties` and
        # `Asset` in the same OpenAPI document do allow extras. Pydantic drops what it does not
        # model, so a POST/PUT is expected to keep only `properties.renders` and an item read back
        # from the catalogue will carry that copy alone. (Basis: the deployed API's published
        # OpenAPI, plus an offline repro through `stac_pydantic.Item`. NOT yet confirmed against a
        # real POST -- no item carrying a root copy has been registered, so the fact that live
        # items show only `properties.renders` is not evidence either way.)
        #
        # Hence both: the root copy is what makes the item satisfy the extension it declares, the
        # mirror is what is expected to survive registration. Deleting the mirror would silently
        # remove every render/viewer/tilejson/thumbnail link the pipeline derives, and would make
        # `register_per_acquisition` raise outright. Consumers should read the root first and fall
        # back to `properties` -- EOPF-Explorer/data-pipeline#413 does.
        extra_fields={"renders": cube_render},
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
        orbit = info.orbit
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

# Default the cube preview to the most recent acquisition that fills most of the preview FRAME, so a
# browser shows fresh near-full data rather than the oldest slice. Not 80% of the tile's native valid
# area -- see `slice_coverages` for why the two differ and why the frame is the right quantity here.
COVERAGE_THRESHOLD = 0.80


class Slice(NamedTuple):
    """One cube time slice: its orbit group, acquisition instant, and preview-frame fill (0..1).

    ``coverage`` is the fraction of the r720m preview image that renders as data -- NOT the native
    valid-data fraction, which it over-estimates. See :func:`slice_coverages`.
    """

    orbit: str
    dt: dt.datetime
    coverage: float


def pick_slice(slices: list[Slice]) -> Slice | None:
    """Choose the slice the cube preview should default to.

    The most recent acquisition filling more than ``COVERAGE_THRESHOLD`` of the preview frame; if
    none clears it, the fullest slice (ties broken by most recent). Spans both orbit groups. Returns
    ``None`` for an empty cube.

    Note the two regimes: once any slice clears the gate, selection is purely by recency and the
    fill number stops mattering; below the gate it is purely relative, where a bias shared by all
    slices largely cancels. Only a slice sitting just under the gate is sensitive to the exact value.
    """
    if not slices:
        return None
    good = [s for s in slices if s.coverage > COVERAGE_THRESHOLD]
    if good:
        return max(good, key=lambda s: s.dt)
    return max(slices, key=lambda s: (s.coverage, s.dt))


def slice_coverages(zarr_store: str) -> list[Slice]:
    """Per-slice PREVIEW-FRAME FILL from the cube, across both orbit groups.

    Reads ``border_mask`` at the cheap ``r720m`` overview only (~150x150). The value is the fraction
    of **valid** pixels there; the S1Tiling border mask stores ``0`` for the border, so valid =
    non-zero. ``time`` is raw int64 ns (as :func:`build_s1_rtc_stac_item` reads it) -> UTC datetime.

    What this number IS: the fraction of the preview image that renders as data. Overview masks are
    built with block-max while vv/vh are block-averaged with ``nanmean``, and those agree exactly --
    ``(mask != 0) == isfinite(vv)`` is array-identical at every overview level, verified in
    ``test_mask_agrees_with_backscatter_at_every_level``. So this is precisely the right input for
    the only decision it feeds: which slice the cube thumbnail should default to.

    What it is NOT: a data-quality metric, and not the native valid-data fraction, which it
    OVER-ESTIMATES. A block counts valid if any one of its native pixels is, and r720m is a 72x
    reduction (5184 pixels per block), so interior no-data holes narrower than a block are filled in
    and the swath edge is rounded outward. Measured on a solid swath edge the effect is small
    (0.8749 native -> 0.8783 here); on a mask with fine interior holes it is not. Do not reuse this
    as a STAC field or a quality gate, and do not "fix" it by reading a finer level -- that would
    cost 36x the bytes and move the metric away from the quantity the decision actually needs.

    Speckle cannot arise upstream: S1Tiling's ``SmoothBorderMask`` applies a binary morphological
    opening with a ball of radius 5, so the valid set is a union of ~110 m disks. The residual
    fine-grained invalidity is interior radar-shadow holes, which the opening preserves.
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
        If the store has no acquisitions, ``orbit`` is not present in the store, or the store name
        carries no valid MGRS tile id.
    """
    if orbit not in _ORBIT_PREFERENCE:
        raise ValueError(f"orbit must be one of {_ORBIT_PREFERENCE}, got {orbit!r}")

    tile_id = _tile_id_from_store(zarr_store)

    root = _open_root(zarr_store)
    if orbit not in root:
        raise ValueError(f"Orbit group {orbit!r} not found in Zarr store: {zarr_store}")
    r10m = cast("zarr.Group", cast("zarr.Group", root[orbit])["r10m"])
    times_ns = np.array(cast("zarr.Array", r10m["time"])).tolist()
    platforms = np.array(cast("zarr.Array", r10m["platform"])).tolist()
    if not times_ns:
        raise ValueError(f"No acquisitions found for orbit {orbit!r} in: {zarr_store}")

    # Per-slice orbit numbers. Unlike the cube item, a per-acquisition item is single-valued by
    # construction, so it always carries both when the store records them. `None` per slice when the
    # coordinates are absent (pre-#216 stores) — see `_orbit_numbers`.
    absolute_orbits = _orbit_numbers(r10m, "absolute_orbit") or [None] * len(times_ns)
    relative_orbits = _orbit_numbers(r10m, "relative_orbit") or [None] * len(times_ns)

    # A half-completed resize can leave a metadata coordinate shorter than `time`. Say so, rather
    # than letting the `zip(..., strict=True)` below fail with "zip() argument 4 is shorter than
    # arguments 1-3", which names neither the store nor the array.
    for name, values in (
        ("platform", platforms),
        ("absolute_orbit", absolute_orbits),
        ("relative_orbit", relative_orbits),
    ):
        if len(values) != len(times_ns):
            raise ValueError(
                f"Cannot build per-acquisition items for {orbit!r} in {zarr_store}: "
                f"r10m/{name} has {len(values)} entries but r10m/time has {len(times_ns)}. "
                "The cube is half-built -- re-run the interrupted append or wipe + reingest."
            )

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
    orbit_utm_bbox = make_bounding_box(og_attrs["spatial:bbox"])
    orbit_wgs84_bbox = _utm_to_wgs84(make_crs_code(og_attrs["proj:code"]), orbit_utm_bbox)
    orbit_geometry = _bbox_to_geometry(orbit_wgs84_bbox)

    items: list[pystac.Item] = []
    for t_ns, platform, abs_orbit, rel_orbit in zip(
        times_ns, platforms, absolute_orbits, relative_orbits, strict=True
    ):
        when = dt.datetime.fromtimestamp(t_ns / 1e9, tz=dt.UTC)
        item_dict = {**base_dict}
        item_dict["id"] = acquisition_id(tile_id, when)
        item_dict["bbox"] = list(orbit_wgs84_bbox)
        item_dict["geometry"] = orbit_geometry

        props = {
            k: v
            for k, v in base_dict["properties"].items()
            if k not in ("start_datetime", "end_datetime", "cube:dimensions", "cube:variables")
        }
        props["datetime"] = when.isoformat()
        props["sat:orbit_state"] = orbit
        props["proj:bbox"] = list(orbit_utm_bbox)
        # Own attribution objects per item: the base's list would otherwise be shared by reference
        # across every item built from this cube (see `_providers`).
        props["providers"] = _providers()
        # Override, never inherit: the cube's single-valued sat:*_orbit (if any) describes the whole
        # cube. Pop when this orbit group records no orbit numbers, so a value the *other* orbit
        # contributed to the base cannot leak onto these items.
        for field, value in (
            ("sat:absolute_orbit", abs_orbit),
            ("sat:relative_orbit", rel_orbit),
        ):
            if value is None:
                props.pop(field, None)
            else:
                props[field] = value
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
        # At the item ROOT, not in properties — see the note on the cube item's `extra_fields`.
        # The base item is cloned per acquisition, so this also overrides whatever orbit the cube
        # chose as preferred. Both copies MUST be overwritten: `props` is inherited from the cube
        # base, so leaving the mirror alone would leave the *preferred* orbit's render sitting in
        # `properties` while the root carried this item's — exactly the disagreement the mirror
        # exists to avoid. Build once and share the object between the two: they are the same
        # render config, and nothing mutates an item after this loop.
        render = {"rgb": _rgb_render(orbit)}
        props["renders"] = render
        item_dict["properties"] = props
        item_dict["renders"] = render

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
