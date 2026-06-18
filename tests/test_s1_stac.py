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
    consolidate: bool = True,
) -> Path:
    """Create a minimal S1 Zarr store.

    ``orbits`` maps orbit_direction -> list of (time_ns, platform) tuples.
    Creates only the attrs and coordinate arrays needed by build_s1_rtc_stac_item.
    ``consolidate=False`` skips writing root consolidated metadata, mirroring a cube that grew by
    appending a time-slice to an existing same-orbit group (the builder must still read it).
    """
    if utm_bbox is None:
        utm_bbox = UTM_BBOX
    # TEMPORARY (#246): store basename == item id (s1-rtc-{tile}) so titiler's reconstructed
    # render path resolves; revert to "s1-grd-rtc-" when titiler-eopf#108 lands.
    store_path = tmp_path / f"s1-rtc-{tile_id}.zarr"
    root = zarr.open_group(str(store_path), mode="w", zarr_format=3)
    ny = nx = 4  # tiny spatial grid: enough for the builder's metadata reads
    for orbit_dir, acquisitions in orbits.items():
        og = root.create_group(orbit_dir)
        og.attrs.update({"proj:code": crs, "spatial:bbox": utm_bbox})
        r10m = og.create_group("r10m")
        # proj:shape / proj:transform live on the r10m group attrs in real stores.
        r10m.attrs.update(
            {
                "spatial:shape": [ny, nx],
                "spatial:transform": [10.0, 0.0, utm_bbox[0], 0.0, -10.0, utm_bbox[3]],
            }
        )
        times = np.array([t for t, _ in acquisitions], dtype="int64")
        platforms = np.array([p for _, p in acquisitions], dtype="<U4")
        nt = times.shape[0]
        t_arr = r10m.create_array("time", shape=times.shape, dtype="int64", chunks=(512,))
        t_arr[:] = times
        p_arr = r10m.create_array("platform", shape=platforms.shape, dtype="<U4", chunks=(512,))
        p_arr[:] = platforms
        # Data variables (named bands the builder advertises); tiny so creation stays cheap.
        for name, dtype in (("vv", "float32"), ("vh", "float32"), ("border_mask", "uint8")):
            arr = r10m.create_array(name, shape=(nt, ny, nx), dtype=dtype, chunks=(1, ny, nx))
            arr[:] = 0
    if consolidate:
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


def test_builds_from_non_consolidated_store(tmp_path: Path) -> None:
    """Regression: the builder must read a store that lacks root consolidated metadata.

    A per-tile cube grown by appending a time-slice to an existing same-orbit group can end up
    without root consolidated metadata (re-consolidating an S3 append is unreliable), which made
    ``zarr.open_consolidated`` raise ``ValueError: Consolidated metadata ... not found`` and broke
    STAC registration in the live S1 RTC pipeline. The builder must fall back to a direct read.
    """
    store = _make_s1_store(
        tmp_path, {"ascending": [(T1_NS, "S1A"), (T2_NS, "S1A")]}, consolidate=False
    )
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")
    assert item.id == "s1-rtc-31TCH"
    assert item.properties["start_datetime"]
    assert item.properties["end_datetime"]


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
    store_path = tmp_path / "s1-rtc-31TCH.zarr"
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
    assert west < 1.0  # left edge from descending
    assert east > 2.5  # right edge from ascending (shifted ~1° further east)


def test_both_orbits_get_first_class_assets(tmp_path: Path) -> None:
    """A dual-orbit cube must expose a γ⁰ asset per orbit group (bug #2), each pointing at its group."""
    store = _make_s1_store(
        tmp_path, {"descending": [(T1_NS, "S1A")], "ascending": [(T1_NS, "S1A")]}
    )
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    asc = item.assets["gamma0-rtc-backscatter-asc"]
    desc = item.assets["gamma0-rtc-backscatter-desc"]
    assert asc.href.endswith("/ascending")
    assert desc.href.endswith("/descending")
    # The dual-pol href ambiguity (bug #1) is gone: VV/VH are named bands, not duplicate assets.
    assert [b["name"] for b in asc.extra_fields["bands"]] == ["vv", "vh"]
    assert "border-mask-asc" in item.assets
    assert "border-mask-desc" in item.assets


def test_empty_store_raises(tmp_path: Path) -> None:
    """A store with an orbit group but no acquisitions must raise ValueError."""
    store_path = tmp_path / "s1-rtc-31TCH.zarr"
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
    """zarr-store href = store URI; the γ⁰ asset href = {store}/{orbit} (orbit group, per geozarr spec)."""
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    store_str = str(store)
    assert item.assets["zarr-store"].href == store_str
    # Single-orbit cube → only the descending γ⁰/mask assets exist (no ascending).
    assert item.assets["gamma0-rtc-backscatter-desc"].href == f"{store_str}/descending"
    assert item.assets["border-mask-desc"].href == f"{store_str}/descending"
    assert "gamma0-rtc-backscatter-asc" not in item.assets


def test_gamma0_asset_band_and_invariant_metadata(tmp_path: Path) -> None:
    """The γ⁰ asset carries VV/VH bands + data_type/nodata/unit/gsd invariants."""
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    asset = item.assets["gamma0-rtc-backscatter-desc"]
    assert asset.extra_fields["data_type"] == "float32"
    assert asset.extra_fields["nodata"] == "nan"
    assert asset.extra_fields["gsd"] == 10
    bands = asset.extra_fields["bands"]
    assert {b["name"] for b in bands} == {"vv", "vh"}
    assert all(b["data_type"] == "float32" for b in bands)


def test_identity_and_projection_fields(tmp_path: Path) -> None:
    """Item-level identity invariants + projection detail are present (S2-parity gaps)."""
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    props = item.properties
    assert props["constellation"] == "sentinel-1"
    assert props["instruments"] == ["c-sar"]
    assert props["gsd"] == 10
    assert "platform" not in props  # per-acquisition; a cube can mix S1A/S1C
    assert props["proj:bbox"] == UTM_BBOX
    assert props["proj:shape"] == [4, 4]
    assert props["proj:transform"][0] == 10.0


def test_datacube_extension(tmp_path: Path) -> None:
    """Cube items carry the datacube extension. The irregular time axis lists its discrete `values`
    (count = number of acquisitions); the regular x/y axes carry extent + step only (no per-pixel
    enumeration — the exact pixel count is in proj:shape)."""
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A"), (T2_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    assert "https://stac-extensions.github.io/datacube/v2.2.0/schema.json" in item.stac_extensions
    dims = item.properties["cube:dimensions"]
    assert dims["time"]["type"] == "temporal"
    assert len(dims["time"]["extent"]) == 2
    assert dims["time"]["values"] == [  # two distinct acquisitions -> two time steps, sorted
        dt.datetime.fromtimestamp(T1_NS / 1e9, tz=dt.UTC).isoformat(),
        dt.datetime.fromtimestamp(T2_NS / 1e9, tz=dt.UTC).isoformat(),
    ]
    assert dims["x"]["reference_system"] == 32631
    assert dims["x"]["step"] == 10.0
    assert dims["y"]["step"] == -10.0
    assert "values" not in dims["x"]  # regular axis: extent + step, not ~10^4 coordinates
    assert item.properties["proj:shape"] == [4, 4]  # exact x/y element count
    variables = item.properties["cube:variables"]
    assert set(variables) == {"vv", "vh", "border_mask"}
    assert variables["vv"]["dimensions"] == ["time", "y", "x"]
    # datacube field is `variable_type` (not `type`); the border mask is auxiliary, not data.
    assert variables["vv"]["variable_type"] == "data"
    assert variables["border_mask"]["variable_type"] == "auxiliary"
    assert "type" not in variables["vv"]


def test_orbit_state_single_vs_dual(tmp_path: Path) -> None:
    """sat:orbit_state (single-valued) is set only for a single-orbit cube; a dual-orbit cube omits it
    (and the SAT extension) rather than mislabel half its slices."""
    single = _make_s1_store(tmp_path / "a", {"descending": [(T1_NS, "S1A")]})
    item1 = build_s1_rtc_stac_item(str(single), "sentinel-1-grd-rtc-staging")
    assert item1.properties["sat:orbit_state"] == "descending"
    assert "https://stac-extensions.github.io/sat/v1.0.0/schema.json" in item1.stac_extensions
    assert "description" not in item1.properties["cube:dimensions"]["time"]  # single orbit: no note

    dual = _make_s1_store(
        tmp_path / "b", {"ascending": [(T1_NS, "S1A")], "descending": [(T1_NS, "S1A")]}
    )
    item2 = build_s1_rtc_stac_item(str(dual), "sentinel-1-grd-rtc-staging")
    assert "sat:orbit_state" not in item2.properties
    assert "https://stac-extensions.github.io/sat/v1.0.0/schema.json" not in item2.stac_extensions
    # dual-orbit cube notes that the merged time axis spans both orbits (orbit is an asset-level split)
    assert "orbit" in item2.properties["cube:dimensions"]["time"]["description"].lower()


def test_timestamps_updated_only(tmp_path: Path) -> None:
    """`updated` (build time) is set; `created` is omitted — the store records no item-creation time,
    so an acquisition time would misuse the field and a build-time value would churn on every append."""
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A"), (T2_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    assert "created" not in item.properties
    assert dt.datetime.fromisoformat(item.properties["updated"])


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


def test_render_extension_rgb_composite(tmp_path: Path) -> None:
    """Item must declare a render-extension RGB composite using the preferred orbit."""
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    render_ext_uri = "https://stac-extensions.github.io/render/v1.0.0/schema.json"
    assert render_ext_uri in item.stac_extensions

    rgb = item.properties["renders"]["rgb"]
    assert rgb["expression"] == "/descending:vv;/descending:vh;(/descending:vv)/(/descending:vh)"
    assert rgb["rescale"] == [[0.0, 0.2]]
    assert rgb["bidx"] == [1]
    assert rgb["tilesize"] == 256


def test_render_uses_ascending_when_preferred(tmp_path: Path) -> None:
    """When ascending is the preferred orbit, the render expression must reference it."""
    store_path = tmp_path / "s1-rtc-31TCH.zarr"
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
    assert item.properties["renders"]["rgb"]["expression"].startswith("/ascending:vv")
