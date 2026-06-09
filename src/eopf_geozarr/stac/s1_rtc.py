"""STAC item builder for S1 GRD RTC Zarr V3 stores."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import cast

import numpy as np
import pyproj
import pystac
import zarr

SAR_EXT = "https://stac-extensions.github.io/sar/v1.0.0/schema.json"
SAT_EXT = "https://stac-extensions.github.io/sat/v1.0.0/schema.json"
PROJ_EXT = "https://stac-extensions.github.io/projection/v2.0.0/schema.json"
RENDER_EXT = "https://stac-extensions.github.io/render/v1.0.0/schema.json"

_ORBIT_PREFERENCE = ("ascending", "descending")


def _rgb_render(orbit: str) -> dict[str, object]:
    """Build the dual-pol RGB composite render config for the given orbit group.

    Produces a 3-band false-colour composite (R=VV, G=VH, B=VV/VH ratio) that
    titiler renders into previews/tiles. ``bidx=[1]`` selects the single time
    slice from each multi-band variable; ``rescale`` is in linear gamma0 units.
    """
    vv = f"/{orbit}:vv"
    vh = f"/{orbit}:vh"
    return {
        "title": "VV, VH, VV/VH composite",
        "expression": f"{vv};{vh};({vv})/({vh})",
        "rescale": [[0.0, 0.1], [0.0, 0.1], [0.0, 0.1]],
        "bidx": [1],
        "tilesize": 256,
    }


def _utm_to_wgs84(
    proj_code: str, utm_bbox: list[float]
) -> tuple[float, float, float, float]:
    """Convert UTM [xmin, ymin, xmax, ymax] to WGS84 (west, south, east, north)."""
    xmin, ymin, xmax, ymax = utm_bbox
    transformer = pyproj.Transformer.from_crs(proj_code, "EPSG:4326", always_xy=True)
    xs = [xmin, xmax, xmin, xmax]
    ys = [ymin, ymin, ymax, ymax]
    lons, lats = transformer.transform(xs, ys)
    return min(lons), min(lats), max(lons), max(lats)


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
    tile_id = Path(zarr_store).name.removeprefix("s1-grd-rtc-").removesuffix(".zarr")

    root = zarr.open_consolidated(zarr_store, zarr_format=3)

    all_times_ns: list[int] = []
    wgs84_bboxes: list[tuple[float, float, float, float]] = []
    preferred_orbit: str | None = None

    for orbit_dir in _ORBIT_PREFERENCE:
        if orbit_dir not in root:
            continue
        og = cast(zarr.Group, root[orbit_dir])
        attrs = dict(og.attrs)
        proj_code = cast(str, attrs["proj:code"])
        utm_bbox = cast(list[float], attrs["spatial:bbox"])

        r10m = cast(zarr.Group, og["r10m"])
        times = np.array(cast(zarr.Array, r10m["time"])).tolist()
        if not times:
            continue

        all_times_ns.extend(times)
        wgs84_bboxes.append(_utm_to_wgs84(proj_code, utm_bbox))
        if preferred_orbit is None:
            preferred_orbit = orbit_dir

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

    geometry = {
        "type": "Polygon",
        "coordinates": [
            [[west, south], [east, south], [east, north], [west, north], [west, south]]
        ],
    }

    # preferred_orbit is guaranteed non-None here (ValueError raised above if no acquisitions)
    assert preferred_orbit is not None
    preferred_proj_code = cast(str, dict(cast(zarr.Group, root[preferred_orbit]).attrs)["proj:code"])

    item = pystac.Item(
        id=f"s1-rtc-{tile_id}",
        geometry=geometry,
        bbox=wgs84_bbox,
        datetime=None,
        properties={
            "start_datetime": start_dt.isoformat(),
            "end_datetime": end_dt.isoformat(),
            # SAR extension
            "sar:instrument_mode": "IW",
            "sar:frequency_band": "C",
            "sar:center_frequency": 5.405,
            "sar:polarizations": ["VV", "VH"],
            "sar:product_type": "GRD",
            # SAT extension
            "sat:orbit_state": preferred_orbit,
            # Projection extension
            "proj:code": preferred_proj_code,
            # Render extension: dual-pol RGB composite for previews/tiles
            "renders": {"rgb": _rgb_render(preferred_orbit)},
        },
        stac_extensions=[SAR_EXT, SAT_EXT, PROJ_EXT, RENDER_EXT],
        collection=collection_id,
    )

    store_str = str(zarr_store)
    item.add_asset(
        "zarr-store",
        pystac.Asset(
            href=store_str,
            media_type="application/vnd.zarr; version=3",
            roles=["data"],
            title="S1 GRD RTC Zarr store",
        ),
    )
    for pol in ("vv", "vh"):
        item.add_asset(
            pol,
            pystac.Asset(
                href=f"{store_str}/{preferred_orbit}",
                media_type="application/vnd.zarr; version=3",
                roles=["data"],
                title=f"{pol.upper()} polarisation",
            ),
        )

    return item
