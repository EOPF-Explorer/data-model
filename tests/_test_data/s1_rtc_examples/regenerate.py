"""Regenerate the S1 GRD RTC example JSON from a store built by the current ingester.

Run from the repository root::

    .venv/bin/python tests/_test_data/s1_rtc_examples/regenerate.py

The fixture in this directory is the only thing ``tests/test_data_api/test_s1_rtc.py``
validates the ``S1RtcRoot`` model against, so a hand-maintained fixture lets the model and
the writer drift apart silently. That is exactly what happened: the checked-in JSON predated
both the ``x``/``y``/``spatial_ref`` coordinate arrays and the per-level ``time`` coordinate
(data-model #192), so ``S1RtcRoot.from_zarr`` raised 23 ``extra_forbidden`` errors on any
store this repo's own writer produced while every model test stayed green.

This script closes that loop: it drives ``ingest_s1tiling_acquisition`` /
``ingest_s1tiling_conditions`` / ``consolidate_s1_store`` over synthetic GeoTIFFs and dumps
the resulting hierarchy with the *generic* ``GroupSpec`` (never ``S1RtcRoot``) — the fixture
must record what the writer emits, not what the model already accepts.

The synthetic tile mimics the real 31TCH descending cube the original fixture came from
(EPSG:32631, origin at the MGRS tile corner, 3 acquisitions on relative orbit 110, gamma_area
for the three orbits that cover the tile) at 1/10 the linear size, so regeneration stays cheap.
Only metadata reaches the JSON, so the pixel values are irrelevant beyond being well-formed.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import zarr
from pydantic_zarr.v3 import GroupSpec
from rasterio.transform import from_bounds

from eopf_geozarr.conversion.s1_ingest import (
    consolidate_s1_store,
    ingest_s1tiling_acquisition,
    ingest_s1tiling_conditions,
)

# 31TCH is UTM zone 31N with its north-west corner at (500000, 5000000). The real tile is
# 10980 px at 10 m; 1080 is a decade-smaller stand-in sharing the corner and the pixel size.
# The exact value matters: `calculate_aligned_chunk_size` only accepts an inner chunk that
# divides the level's edge, so every level in the r10m→r720m chain must keep a divisor near
# 512 (1080 → 540 → 180 → 90 → 30 → 15 does; 1098 → 549 does not, and the shard write raises).
SIZE = 1080
PIXEL = 10.0
CRS = "EPSG:32631"
XMIN, YMAX = 500000.0, 5000000.0
TRANSFORM = from_bounds(XMIN, YMAX - SIZE * PIXEL, XMIN + SIZE * PIXEL, YMAX, SIZE, SIZE)

ORBIT_DIRECTION = "descending"
RELATIVE_ORBIT = 110
# The three relative orbits whose gamma-area maps cover 31TCH; the original fixture carried
# all three, and a multi-orbit conditions group is the case worth keeping under test.
CONDITION_ORBITS = (8, 37, 110)

ACQUISITIONS = (
    ("2025:02:05T06:01:10Z", 57895),
    ("2025:02:17T06:01:10Z", 58070),
    ("2025:03:01T06:01:09Z", 58245),
)

OUTPUT = Path(__file__).with_name("s1-grd-rtc-31TCH.json")


def _tags(acquisition_datetime: str, absolute_orbit: int) -> dict[str, str]:
    return {
        "ACQUISITION_DATETIME": acquisition_datetime,
        "ORBIT_NUMBER": str(absolute_orbit),
        "RELATIVE_ORBIT_NUMBER": f"{RELATIVE_ORBIT:03d}",
        "FLYING_UNIT_CODE": "S1A",
        "CALIBRATION": "gamma_naught",
        "INPUT_S1_IMAGES": f"S1A_IW_GRDH_1SDV_{acquisition_datetime[:10].replace(':', '')}",
    }


def _write_geotiff(path: Path, data: np.ndarray, tags: dict[str, str] | None = None) -> Path:
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=CRS,
        transform=TRANSFORM,
    ) as dst:
        if tags:
            dst.update_tags(**tags)
        dst.write(data, 1)
    return path


def build_store(work_dir: Path) -> Path:
    """Ingest synthetic 31TCH GeoTIFFs into a consolidated GeoZarr V3 store."""
    rng = np.random.default_rng(31031)
    store_path = work_dir / "s1-grd-rtc-31TCH.zarr"

    for index, (stamp, absolute_orbit) in enumerate(ACQUISITIONS):
        tags = _tags(stamp, absolute_orbit)
        prefix = f"s1a_31TCH_DES_{RELATIVE_ORBIT:03d}_{index}"
        border_mask = np.ones((SIZE, SIZE), dtype=np.uint8)
        border_mask[:32, :] = 0
        vv, vh = (
            _write_geotiff(
                work_dir / f"{prefix}_{pol}.tif",
                rng.uniform(0.0, 1.0, (SIZE, SIZE)).astype(np.float32),
                tags,
            )
            for pol in ("vv", "vh")
        )
        mask = _write_geotiff(work_dir / f"{prefix}_mask.tif", border_mask, tags)
        ingest_s1tiling_acquisition(vv, vh, mask, store_path, ORBIT_DIRECTION)

    for orbit in CONDITION_ORBITS:
        gamma_area = _write_geotiff(
            work_dir / f"GAMMA_AREA_31TCH_{orbit:03d}.tif",
            rng.uniform(0.5, 2.0, (SIZE, SIZE)).astype(np.float32),
        )
        ingest_s1tiling_conditions(
            store_path=store_path,
            orbit_direction=ORBIT_DIRECTION,
            relative_orbit=orbit,
            gamma_area_path=gamma_area,
        )

    consolidate_s1_store(store_path, ORBIT_DIRECTION)
    return store_path


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store_path = build_store(Path(tmp))
        root = zarr.open_group(str(store_path), mode="r", zarr_format=3)
        # The *generic* GroupSpec, never S1RtcRoot: the fixture must record what the writer
        # emits, not what the model already accepts, or the drift this closes reopens.
        spec = GroupSpec.from_zarr(root)

    OUTPUT.write_text(json.dumps(spec.model_dump(mode="json"), indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
