"""Tests for the per-acquisition + coverage construction moved into eopf_geozarr.stac.s1_rtc.

Ported from the data-pipeline construction tests (``test_pick_slice``, ``test_slice_coverage`` and the
construction subset of ``test_register_per_acquisition``). The TiTiler-link / thumbnail / alternate-asset
behaviour stays in data-pipeline — those are registration concerns, not item construction.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from eopf_geozarr.stac.s1_rtc import (
    DATACUBE_EXT,
    Slice,
    acquisition_id,
    build_s1_rtc_per_acquisition_items,
    pick_slice,
    slice_coverages,
)

if TYPE_CHECKING:
    from pathlib import Path

CRS = "EPSG:32631"
UTM_BBOX = [300000.0, 4900000.0, 400000.0, 5000000.0]

# Two acquisitions in PHYSICAL (append) order — deliberately NOT chronological, to prove each item's
# datetime follows its own slice, not its list position.
T_LATER = int(dt.datetime(2026, 6, 7, 5, 52, 48, tzinfo=dt.UTC).timestamp() * 1e9)
T_EARLY = int(dt.datetime(2026, 6, 5, 6, 9, 7, tzinfo=dt.UTC).timestamp() * 1e9)


# =============================================================================
# pick_slice — pure preview-slice selector
# =============================================================================


def _s(orbit: str, day: int, coverage: float) -> Slice:
    return Slice(orbit=orbit, dt=dt.datetime(2026, 6, day, 6, 0, tzinfo=dt.UTC), coverage=coverage)


def test_most_recent_above_threshold_wins_even_if_older_has_more_coverage() -> None:
    chosen = pick_slice(
        [_s("ascending", 4, 0.99), _s("descending", 7, 0.85), _s("ascending", 6, 0.90)]
    )
    assert chosen is not None
    assert (chosen.orbit, chosen.dt.day, chosen.coverage) == ("descending", 7, 0.85)


def test_falls_back_to_max_coverage_when_none_above_threshold() -> None:
    chosen = pick_slice(
        [_s("ascending", 7, 0.40), _s("descending", 5, 0.75), _s("ascending", 4, 0.50)]
    )
    assert chosen is not None
    assert (chosen.orbit, chosen.dt.day, chosen.coverage) == ("descending", 5, 0.75)


def test_exact_80_percent_is_not_above_threshold() -> None:
    # 0.80 is NOT > 0.80 -> excluded from the "good" set; the 0.81 older slice wins instead.
    chosen = pick_slice([_s("ascending", 7, 0.80), _s("descending", 4, 0.81)])
    assert chosen is not None
    assert (chosen.dt.day, chosen.coverage) == (4, 0.81)


def test_max_coverage_tie_broken_by_most_recent() -> None:
    chosen = pick_slice(
        [_s("ascending", 4, 0.60), _s("descending", 8, 0.60), _s("ascending", 6, 0.60)]
    )
    assert chosen is not None
    assert chosen.dt.day == 8


def test_single_slice_returned() -> None:
    only = _s("ascending", 5, 0.10)
    assert pick_slice([only]) == only


def test_empty_returns_none() -> None:
    assert pick_slice([]) is None


# =============================================================================
# slice_coverages — per-slice tile coverage from the cube's r720m border_mask
# =============================================================================


def _ns(day: int) -> int:
    return int(dt.datetime(2026, 6, day, 6, 0, tzinfo=dt.UTC).timestamp() * 1e9)


def _write_r720m(root: zarr.Group, orbit: str, masks: list[np.ndarray], days: list[int]) -> None:
    """Write {orbit}/r720m with a border_mask (time,y,x) + time (int64 ns), like the real cube."""
    lvl = root.create_group(orbit).create_group("r720m")
    n = len(masks)
    y, x = masks[0].shape
    bm = lvl.create_array("border_mask", shape=(n, y, x), dtype="uint8", fill_value=0)
    bm[:] = np.stack(masks).astype("uint8")
    t = lvl.create_array("time", shape=(n,), dtype="int64")
    t[:] = np.array([_ns(d) for d in days], dtype="int64")


def _make_coverage_cube(tmp_path: Path) -> str:
    store = str(tmp_path / "s1-rtc-31TCH.zarr")
    root = zarr.open_group(store, mode="w", zarr_format=3)
    full = np.ones((4, 4))  # coverage 1.0
    half = np.zeros((4, 4))
    half[:2, :] = 1  # coverage 0.5
    quarter = np.zeros((4, 4))
    quarter[0, :] = 1  # coverage 0.25
    _write_r720m(root, "ascending", [full, half], [4, 6])
    _write_r720m(root, "descending", [quarter], [5])
    return store


def test_slice_coverages_reads_both_orbits_with_correct_fractions(tmp_path: Path) -> None:
    by_key = {
        (s.orbit, s.dt.day): s.coverage for s in slice_coverages(_make_coverage_cube(tmp_path))
    }
    assert by_key == {
        ("ascending", 4): 1.0,  # full mask -> polarity check (non-zero = valid)
        ("ascending", 6): 0.5,
        ("descending", 5): 0.25,
    }


def test_slice_coverages_times_are_utc_datetimes(tmp_path: Path) -> None:
    s = next(s for s in slice_coverages(_make_coverage_cube(tmp_path)) if s.orbit == "descending")
    assert s.dt == dt.datetime(2026, 6, 5, 6, 0, tzinfo=dt.UTC)


def test_slice_coverages_skips_missing_orbit(tmp_path: Path) -> None:
    store = str(tmp_path / "s1-rtc-asc-only.zarr")
    root = zarr.open_group(store, mode="w", zarr_format=3)
    _write_r720m(root, "ascending", [np.ones((4, 4))], [7])
    assert {s.orbit for s in slice_coverages(store)} == {"ascending"}


# =============================================================================
# build_s1_rtc_per_acquisition_items — one item per cube time slice
# =============================================================================


def _make_acq_cube(
    tmp_path: Path, orbits: dict[str, list[tuple[int, str]]], tile_id: str = "31TCH"
) -> str:
    """A cube with r10m time/platform (+ tiny data arrays) per orbit — enough to build per-acq items."""
    store = str(tmp_path / f"s1-rtc-{tile_id}.zarr")
    root = zarr.open_group(store, mode="w", zarr_format=3)
    ny = nx = 4
    for orbit, acqs in orbits.items():
        og = root.create_group(orbit)
        og.attrs.update({"proj:code": CRS, "spatial:bbox": UTM_BBOX})
        r10m = og.create_group("r10m")
        r10m.attrs.update(
            {
                "spatial:shape": [ny, nx],
                "spatial:transform": [10.0, 0.0, UTM_BBOX[0], 0.0, -10.0, UTM_BBOX[3]],
            }
        )
        times = np.array([t for t, _ in acqs], dtype="int64")
        platforms = np.array([p for _, p in acqs], dtype="<U4")
        nt = times.shape[0]
        r10m.create_array("time", shape=times.shape, dtype="int64", chunks=(512,))[:] = times
        r10m.create_array("platform", shape=platforms.shape, dtype="<U4", chunks=(512,))[:] = (
            platforms
        )
        for name, dtype in (("vv", "float32"), ("vh", "float32"), ("border_mask", "uint8")):
            r10m.create_array(name, shape=(nt, ny, nx), dtype=dtype, chunks=(1, ny, nx))[:] = 0
    zarr.consolidate_metadata(store, zarr_format=3)
    return store


def test_acquisition_id_format() -> None:
    when = dt.datetime(2026, 6, 7, 5, 52, 48, tzinfo=dt.UTC)
    assert acquisition_id("31TCH", when) == "s1-rtc-31TCH-20260607t055248"


def test_per_acq_title_carries_datetime_and_orbit(tmp_path: Path) -> None:
    """Per-acq titles must distinguish sibling scenes (not inherit the cube's '— tile {id}' title)."""
    store = _make_acq_cube(tmp_path, {"descending": [(T_EARLY, "s1a")]})
    item = build_s1_rtc_per_acquisition_items(store, orbit="descending", collection_id="acq")[0]
    title = item.properties["title"]
    assert "2026-06-05T06:09:07Z" in title
    assert "descending" in title
    assert title != "Sentinel-1 GRD RTC γ⁰ — tile 31TCH"


def test_single_datetime_no_range_and_targets_collection(tmp_path: Path) -> None:
    store = _make_acq_cube(tmp_path, {"descending": [(T_EARLY, "S1A")]})
    item = build_s1_rtc_per_acquisition_items(
        store, orbit="descending", collection_id="sentinel-1-grd-rtc-acquisitions"
    )[0]
    props = item.properties
    assert item.collection_id == "sentinel-1-grd-rtc-acquisitions"
    assert props["datetime"] == "2026-06-05T06:09:07+00:00"
    assert "start_datetime" not in props
    assert "end_datetime" not in props
    assert props["sar:product_type"] == "GRD"


def test_reorients_orbit_metadata_and_keeps_only_run_orbit_asset(tmp_path: Path) -> None:
    """A descending run carries /descending metadata + only the descending γ⁰/mask assets — never the
    cube's preferred /ascending."""
    store = _make_acq_cube(
        tmp_path, {"ascending": [(T_EARLY, "S1A")], "descending": [(T_EARLY, "S1A")]}
    )
    item = build_s1_rtc_per_acquisition_items(store, orbit="descending", collection_id="acq")[0]
    assert item.properties["sat:orbit_state"] == "descending"
    # SAT ext must be declared even though the dual-orbit cube base omitted it.
    assert "https://stac-extensions.github.io/sat/v1.0.0/schema.json" in item.stac_extensions
    assert item.properties["renders"]["rgb"]["expression"].startswith("/descending:vv")
    assert "gamma0-rtc-backscatter-desc" in item.assets
    assert "gamma0-rtc-backscatter-asc" not in item.assets
    assert "border-mask-asc" not in item.assets
    assert "ascending" not in json.dumps(item.to_dict(include_self_link=False))


def test_per_slice_platform_and_no_datacube(tmp_path: Path) -> None:
    """Each item gets its own slice's platform (normalized to the STAC convention); a single
    acquisition is not a datacube, and `created` is not set."""
    store = _make_acq_cube(tmp_path, {"descending": [(T_LATER, "s1a"), (T_EARLY, "s1c")]})
    items = build_s1_rtc_per_acquisition_items(store, orbit="descending", collection_id="acq")
    assert [i.properties["platform"] for i in items] == ["sentinel-1a", "sentinel-1c"]
    for item in items:
        assert "cube:dimensions" not in item.properties
        assert DATACUBE_EXT not in item.stac_extensions
        assert "created" not in item.properties


def test_grid_code_inherited_from_cube(tmp_path: Path) -> None:
    """Per-acquisition items inherit the cube's grid extension + grid:code (the MGRS tile)."""
    store = _make_acq_cube(tmp_path, {"descending": [(T_EARLY, "S1A")]})
    item = build_s1_rtc_per_acquisition_items(store, orbit="descending", collection_id="acq")[0]
    assert item.properties["grid:code"] == "MGRS-31TCH"
    assert "https://stac-extensions.github.io/grid/v1.1.0/schema.json" in item.stac_extensions


def test_footprint_is_run_orbit_not_cube_union(tmp_path: Path) -> None:
    """A per-acq item must carry ITS orbit's footprint, not the cube's union of both orbits' extents."""
    # ascending shifted east so the cube union bbox is wider than the descending orbit alone.
    store = str(tmp_path / "s1-rtc-31TCH.zarr")
    root = zarr.open_group(store, mode="w", zarr_format=3)
    ny = nx = 4
    desc_bbox = [300000.0, 4900000.0, 400000.0, 5000000.0]
    asc_bbox = [400000.0, 4900000.0, 500000.0, 5000000.0]
    for orbit, bbox in (("descending", desc_bbox), ("ascending", asc_bbox)):
        og = root.create_group(orbit)
        og.attrs.update({"proj:code": CRS, "spatial:bbox": bbox})
        r10m = og.create_group("r10m")
        r10m.attrs.update(
            {
                "spatial:shape": [ny, nx],
                "spatial:transform": [10.0, 0.0, bbox[0], 0.0, -10.0, bbox[3]],
            }
        )
        r10m.create_array("time", shape=(1,), dtype="int64", chunks=(512,))[:] = [T_EARLY]
        r10m.create_array("platform", shape=(1,), dtype="<U4", chunks=(512,))[:] = ["S1A"]
        for name, dtype in (("vv", "float32"), ("vh", "float32"), ("border_mask", "uint8")):
            r10m.create_array(name, shape=(1, ny, nx), dtype=dtype, chunks=(1, ny, nx))[:] = 0
    zarr.consolidate_metadata(store, zarr_format=3)

    item = build_s1_rtc_per_acquisition_items(store, orbit="descending", collection_id="acq")[0]
    # proj:bbox is the descending orbit's UTM extent — not the ascending (preferred) orbit's.
    assert item.properties["proj:bbox"] == desc_bbox
    # WGS84 bbox east edge stays within the descending footprint (~2°E), not the union (~3°E+).
    assert item.bbox[2] < 2.5


def test_invalid_orbit_raises(tmp_path: Path) -> None:
    store = _make_acq_cube(tmp_path, {"descending": [(T_EARLY, "S1A")]})
    with pytest.raises(ValueError, match="orbit must be one of"):
        build_s1_rtc_per_acquisition_items(store, orbit="sideways", collection_id="acq")


def test_orbit_absent_from_store_raises(tmp_path: Path) -> None:
    store = _make_acq_cube(tmp_path, {"descending": [(T_EARLY, "S1A")]})
    with pytest.raises(ValueError, match="not found"):
        build_s1_rtc_per_acquisition_items(store, orbit="ascending", collection_id="acq")


def test_datetime_follows_slice_not_physical_position(tmp_path: Path) -> None:
    """Items are emitted in physical (append) order, each carrying its OWN datetime — so a
    non-monotonic cube still yields a correct item per slice."""
    store = _make_acq_cube(tmp_path, {"descending": [(T_LATER, "S1A"), (T_EARLY, "S1A")]})
    items = build_s1_rtc_per_acquisition_items(store, orbit="descending", collection_id="acq")
    assert items[0].properties["datetime"] == "2026-06-07T05:52:48+00:00"
    assert items[1].properties["datetime"] == "2026-06-05T06:09:07+00:00"
    assert items[0].id == "s1-rtc-31TCH-20260607t055248"
