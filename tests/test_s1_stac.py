"""Tests for build_s1_rtc_stac_item — STAC item builder for S1 GRD RTC Zarr stores."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
import pytest
import zarr

from eopf_geozarr.stac.s1_rtc import build_s1_rtc_stac_item

# =============================================================================
# Constants
# =============================================================================

CRS = "EPSG:32631"
UTM_BBOX = [300000.0, 4900000.0, 400000.0, 5000000.0]  # [xmin, ymin, xmax, ymax]

# Nanoseconds since epoch for two acquisitions
T1_NS = int(np.datetime64("2023-01-15T06:12:34", "ns").astype(np.int64))
T2_NS = int(np.datetime64("2023-01-27T06:12:35", "ns").astype(np.int64))


# =============================================================================
# Fixture helper
# =============================================================================


def _make_s1_store(
    tmp_path: Path,
    orbits: dict[str, list[tuple[int, str]]],
    tile_id: str = "31TCH",
    crs: str = CRS,
    utm_bbox: list[float] | None = None,
) -> Path:
    """Create a minimal consolidated S1 Zarr store.

    ``orbits`` maps orbit_direction -> list of (time_ns, platform) tuples.
    Creates only the attrs and coordinate arrays needed by build_s1_rtc_stac_item.
    """
    if utm_bbox is None:
        utm_bbox = UTM_BBOX
    store_path = tmp_path / f"s1-grd-rtc-{tile_id}.zarr"
    root = zarr.open_group(str(store_path), mode="w", zarr_format=3)
    for orbit_dir, acquisitions in orbits.items():
        og = root.create_group(orbit_dir)
        og.attrs.update({"proj:code": crs, "spatial:bbox": utm_bbox})
        r10m = og.create_group("r10m")
        times = np.array([t for t, _ in acquisitions], dtype="int64")
        platforms = np.array([p for _, p in acquisitions], dtype="<U4")
        t_arr = r10m.create_array("time", shape=times.shape, dtype="int64", chunks=(512,))
        t_arr[:] = times
        p_arr = r10m.create_array("platform", shape=platforms.shape, dtype="<U4", chunks=(512,))
        p_arr[:] = platforms
    zarr.consolidate_metadata(str(store_path), zarr_format=3)
    return store_path


# =============================================================================
# Tests
# =============================================================================


def test_item_id_matches_tile_id(tmp_path: Path) -> None:
    """Item id must be s1-rtc-{tile_id} derived from the store basename."""
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")
    assert item.id == "s1-rtc-31TCH"


def test_temporal_range_min_max(tmp_path: Path) -> None:
    """start_datetime/end_datetime must span the full time range across all acquisitions."""
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A"), (T2_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    start = dt.datetime.fromisoformat(item.properties["start_datetime"])
    end = dt.datetime.fromisoformat(item.properties["end_datetime"])

    expected_start = dt.datetime(2023, 1, 15, 6, 12, 34, tzinfo=dt.UTC)
    expected_end = dt.datetime(2023, 1, 27, 6, 12, 35, tzinfo=dt.UTC)

    assert abs((start - expected_start).total_seconds()) < 1
    assert abs((end - expected_end).total_seconds()) < 1
    assert item.datetime is None


def test_bbox_wgs84_from_utm(tmp_path: Path) -> None:
    """UTM bbox must be converted to WGS84 and stored as item bbox."""
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    west, south, east, north = item.bbox  # type: ignore[misc]
    # EPSG:32631 [300000,4900000,400000,5000000] -> approx 0.46E-1.75E, 44.2N-45.1N
    assert 0.0 < west < 1.0
    assert 44.0 < south < 45.0
    assert 1.0 < east < 2.0
    assert 45.0 < north < 46.0


def test_both_orbits_bbox_union(tmp_path: Path) -> None:
    """When ascending and descending are both present, the WGS84 bbox is the union."""
    # Give ascending a different UTM bbox (shifted east)
    store_path = tmp_path / "s1-grd-rtc-31TCH.zarr"
    root = zarr.open_group(str(store_path), mode="w", zarr_format=3)

    for orbit_dir, bbox in [
        ("descending", [300000.0, 4900000.0, 400000.0, 5000000.0]),
        ("ascending", [400000.0, 4900000.0, 500000.0, 5000000.0]),
    ]:
        og = root.create_group(orbit_dir)
        og.attrs.update({"proj:code": CRS, "spatial:bbox": bbox})
        r10m = og.create_group("r10m")
        t_arr = r10m.create_array("time", shape=(1,), dtype="int64", chunks=(512,))
        t_arr[:] = [T1_NS]
        p_arr = r10m.create_array("platform", shape=(1,), dtype="<U4", chunks=(512,))
        p_arr[:] = ["S1A"]

    zarr.consolidate_metadata(str(store_path), zarr_format=3)
    item = build_s1_rtc_stac_item(str(store_path), "sentinel-1-grd-rtc-staging")

    # Union must be wider than either individual bbox
    west, _south, east, _north = item.bbox  # type: ignore[misc]
    assert west < 1.0   # left edge from descending
    assert east > 2.5   # right edge from ascending (shifted ~1° further east)


def test_ascending_preferred_for_assets(tmp_path: Path) -> None:
    """When both orbits present, vv/vh assets must point to the ascending group."""
    store_path = tmp_path / "s1-grd-rtc-31TCH.zarr"
    root = zarr.open_group(str(store_path), mode="w", zarr_format=3)
    for orbit_dir in ("descending", "ascending"):
        og = root.create_group(orbit_dir)
        og.attrs.update({"proj:code": CRS, "spatial:bbox": UTM_BBOX})
        r10m = og.create_group("r10m")
        t_arr = r10m.create_array("time", shape=(1,), dtype="int64", chunks=(512,))
        t_arr[:] = [T1_NS]
        p_arr = r10m.create_array("platform", shape=(1,), dtype="<U4", chunks=(512,))
        p_arr[:] = ["S1A"]
    zarr.consolidate_metadata(str(store_path), zarr_format=3)

    item = build_s1_rtc_stac_item(str(store_path), "sentinel-1-grd-rtc-staging")
    assert "ascending" in item.assets["vv"].href
    assert "ascending" in item.assets["vh"].href


def test_empty_store_raises(tmp_path: Path) -> None:
    """A store with an orbit group but no acquisitions must raise ValueError."""
    store_path = tmp_path / "s1-grd-rtc-31TCH.zarr"
    root = zarr.open_group(str(store_path), mode="w", zarr_format=3)
    og = root.create_group("descending")
    og.attrs.update({"proj:code": CRS, "spatial:bbox": UTM_BBOX})
    r10m = og.create_group("r10m")
    t_arr = r10m.create_array("time", shape=(0,), dtype="int64", chunks=(512,))
    p_arr = r10m.create_array("platform", shape=(0,), dtype="<U4", chunks=(512,))
    del t_arr, p_arr
    zarr.consolidate_metadata(str(store_path), zarr_format=3)

    with pytest.raises(ValueError, match="No acquisitions"):
        build_s1_rtc_stac_item(str(store_path), "sentinel-1-grd-rtc-staging")


def test_asset_hrefs(tmp_path: Path) -> None:
    """zarr-store href = store URI; vv/vh hrefs = {store}/{orbit} (orbit group root, per geozarr spec)."""
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    store_str = str(store)
    assert item.assets["zarr-store"].href == store_str
    assert item.assets["vv"].href == f"{store_str}/descending"
    assert item.assets["vh"].href == f"{store_str}/descending"


def test_sar_extension_fields(tmp_path: Path) -> None:
    """SAR extension fields must be set with correct values for S1 IW GRD."""
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    props = item.properties
    assert props["sar:instrument_mode"] == "IW"
    assert props["sar:frequency_band"] == "C"
    assert props["sar:center_frequency"] == pytest.approx(5.405)
    assert props["sar:polarizations"] == ["VV", "VH"]
    assert props["sar:product_type"] == "GRD"

    sar_ext_uri = "https://stac-extensions.github.io/sar/v1.0.0/schema.json"
    assert sar_ext_uri in item.stac_extensions
