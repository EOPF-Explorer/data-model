"""Tests for build_s1_rtc_stac_item — STAC item builder for S1 GRD RTC Zarr stores."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
import pyproj
import pytest
import zarr

from eopf_geozarr.stac.s1_rtc import _open_root, build_s1_rtc_stac_item
from eopf_geozarr.types import make_bounding_box, make_crs_code

if TYPE_CHECKING:
    from eopf_geozarr.types import BoundingBox2D

# =============================================================================
# Constants
# =============================================================================

CRS = make_crs_code("EPSG:32631")
UTM_BBOX = make_bounding_box([300000.0, 4900000.0, 400000.0, 5000000.0])  # (xmin, ymin, xmax, ymax)

# Nanoseconds since epoch for two acquisitions
T1_NS = int(np.datetime64("2023-01-15T06:12:34", "ns").astype(np.int64))
T2_NS = int(np.datetime64("2023-01-27T06:12:35", "ns").astype(np.int64))

# (first absolute orbit, relative orbit) written per orbit direction by the fixture below. Ascending
# and descending passes over one tile are different tracks, so they differ in both.
_DEFAULT_ORBIT_NUMBERS = {"ascending": (58011, 37), "descending": (57050, 110)}


# =============================================================================
# Fixture helper
# =============================================================================


def _make_s1_store(
    tmp_path: Path,
    orbits: dict[str, list[tuple[int, str]]],
    tile_id: str = "31TCH",
    crs: str = CRS,
    utm_bbox: BoundingBox2D | None = None,
    consolidate: bool = True,
    orbit_numbers: bool = True,
    relative_orbits: list[int] | None = None,
) -> Path:
    """Create a minimal S1 Zarr store.

    ``orbits`` maps orbit_direction -> list of (time_ns, platform) tuples.
    Creates only the attrs and coordinate arrays needed by build_s1_rtc_stac_item.
    ``consolidate=False`` skips writing root consolidated metadata, mirroring a cube that grew by
    appending a time-slice to an existing same-orbit group (the builder must still read it).
    ``orbit_numbers=False`` omits the absolute/relative orbit coordinates entirely, mirroring a
    store written before ``s1_ingest`` recorded them; ``relative_orbits`` overrides the default
    single-track relative orbit (to build a cube covered by more than one track).
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
        if orbit_numbers:
            # int32 `time` coords, exactly as s1_ingest.py writes them at native resolution. The
            # defaults model one track per orbit direction (the two directions are different tracks,
            # hence different relative orbits), with an absolute orbit unique per acquisition —
            # which is why only the relative orbit survives into a multi-slice cube.
            base_abs, default_rel = _DEFAULT_ORBIT_NUMBERS[orbit_dir]
            for name, values in (
                ("absolute_orbit", [base_abs + i for i in range(nt)]),
                ("relative_orbit", list(relative_orbits or [default_rel] * nt)),
            ):
                arr = r10m.create_array(name, shape=(nt,), dtype="int32", chunks=(512,))
                arr[:] = np.array(values, dtype="int32")
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


def test_grid_code_and_extension(tmp_path: Path) -> None:
    """The cube declares the grid extension + grid:code = MGRS-{tile} (a queryable tile id)."""
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")
    assert item.properties["grid:code"] == "MGRS-31TCH"
    assert "https://stac-extensions.github.io/grid/v1.1.0/schema.json" in item.stac_extensions


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
    assert props["proj:bbox"] == list(UTM_BBOX)
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


def test_created_and_updated_are_common_metadata_not_the_timestamps_extension(
    tmp_path: Path,
) -> None:
    """`created`/`updated` are both set (STAC Common Metadata) and the timestamps extension is NOT
    declared.

    The extension was declared but inert: it only adds `published`/`expires`/`unpublished`, none of
    which this builder emits, so it made every consumer fetch a schema that constrained nothing
    while `created`/`updated` were validated by the core Item spec regardless. `created` was
    previously omitted altogether; the collection requires it, and the store records no separate
    item-creation instant, so it tracks the metadata build like `updated`.
    """
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A"), (T2_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    assert dt.datetime.fromisoformat(item.properties["created"])
    assert dt.datetime.fromisoformat(item.properties["updated"])
    assert "https://stac-extensions.github.io/timestamps/v1.1.0/schema.json" not in (
        item.stac_extensions
    )
    # No timestamps-extension field is emitted, which is why declaring it was pointless.
    assert not {"published", "expires", "unpublished"} & set(item.properties)


def test_providers_copied_down_from_the_collection(tmp_path: Path) -> None:
    """Each item carries its own `providers`: a STAC API search returns items without their
    collection, so an item that omits the block has no attribution for those consumers."""
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    providers = item.properties["providers"]
    assert {p["name"] for p in providers} == {
        "European Commission",
        "ESA",
        "EOPF Sentinel Zarr Samples Service",
    }
    licensor = next(p for p in providers if "licensor" in p["roles"])
    assert licensor["name"] == "European Commission"
    assert all(p["url"].startswith("https://") for p in providers)


def test_sat_orbit_numbers_from_the_store_coordinates(tmp_path: Path) -> None:
    """`sat:relative_orbit`/`sat:absolute_orbit` come from the int32 r10m coordinates, and each is
    emitted only where the cube agrees on a single value (both are single-valued STAC fields)."""
    one = _make_s1_store(tmp_path / "one", {"descending": [(T1_NS, "S1A")]})
    props = build_s1_rtc_stac_item(str(one), "sentinel-1-grd-rtc-staging").properties
    assert props["sat:relative_orbit"] == 110
    assert props["sat:absolute_orbit"] == 57050

    # Two acquisitions on one track: relative orbit still agrees, absolute orbit does not.
    two = _make_s1_store(tmp_path / "two", {"descending": [(T1_NS, "S1A"), (T2_NS, "S1A")]})
    props = build_s1_rtc_stac_item(str(two), "sentinel-1-grd-rtc-staging").properties
    assert props["sat:relative_orbit"] == 110
    assert "sat:absolute_orbit" not in props

    # A tile covered by two tracks: neither field can describe the whole cube.
    tracks = _make_s1_store(
        tmp_path / "tracks",
        {"descending": [(T1_NS, "S1A"), (T2_NS, "S1A")]},
        relative_orbits=[110, 37],
    )
    props = build_s1_rtc_stac_item(str(tracks), "sentinel-1-grd-rtc-staging").properties
    assert "sat:relative_orbit" not in props
    assert "sat:absolute_orbit" not in props


def test_orbit_numbers_are_read_best_effort(tmp_path: Path) -> None:
    """A store written before `s1_ingest` recorded the orbit coordinates must still build — without
    the sat:*_orbit fields, and (with no other sat:* field) without declaring the SAT extension."""
    store = _make_s1_store(
        tmp_path,
        {"ascending": [(T1_NS, "S1A")], "descending": [(T1_NS, "S1A")]},
        orbit_numbers=False,
    )
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    assert item.id == "s1-rtc-31TCH"
    assert not [k for k in item.properties if k.startswith("sat:")]
    assert "https://stac-extensions.github.io/sat/v1.0.0/schema.json" not in item.stac_extensions


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

    rgb = item.to_dict(include_self_link=False)["renders"]["rgb"]
    assert rgb["expression"] == "/descending:vv;/descending:vh;(/descending:vv)/(/descending:vh)"
    assert rgb["rescale"] == [[0.0, 0.4], [0.0, 0.1], [1.0, 15.0]]
    assert rgb["bidx"] == [1]
    # `tilesize` is not a render-extension field, but the Render Object sets
    # `additionalProperties: true`, so it validates; titiler reads it to size rendered tiles.
    assert rgb["tilesize"] == 256


def test_render_objects_carry_the_required_assets_field(tmp_path: Path) -> None:
    """Every Render Object must name the asset(s) it renders.

    `assets` is the ONE required field of a Render Object (render v1.0.0,
    `definitions/fields.required`). Emitting a render config without it made every item fail the
    extension it declared — `'assets' is a required property` — so `generate-stac-s1 |
    stac-validator` failed and a validating STAC API refused the item. The value must name an asset
    the item actually has, or the render config points at nothing.

    The PLACEMENT is pinned too. Render v1.0.0's Item branch is
    `required: ["type", "assets", "renders"]` against the item object itself, and the extension's
    own `examples/item-landsat8.json` puts `renders` at the root — only the README's "Item
    Properties" table says otherwise, and it contradicts both. Emitting it under `properties`
    still failed validation, with the misleading message `'renders' is a required property`.
    """
    store = _make_s1_store(
        tmp_path, {"ascending": [(T1_NS, "S1A")], "descending": [(T1_NS, "S1A")]}
    )
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")
    item_dict = item.to_dict(include_self_link=False)

    assert "renders" in item_dict, "render v1.0.0 requires `renders` at the item root"
    assert "renders" not in item_dict["properties"], (
        "`renders` under properties does not satisfy the render extension"
    )

    for render in item_dict["renders"].values():
        assert render["assets"], "render v1.0.0 requires a non-empty `assets`"
        assert all(name in item.assets for name in render["assets"])
    # Ascending is preferred, so the default render points at the ascending γ⁰ asset.
    assert item_dict["renders"]["rgb"]["assets"] == ["gamma0-rtc-backscatter-asc"]


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
    assert item.to_dict(include_self_link=False)["renders"]["rgb"]["expression"].startswith(
        "/ascending:vv"
    )


def test_open_root_never_creates_a_store(tmp_path: Path) -> None:
    """A read-only STAC build must not write.

    `zarr.open_consolidated` is an alias for `open_group`, whose default is `mode="a"`, so
    `generate-stac-s1 --store s3://bucket/typo.zarr` created a store at the typo'd path --
    writing into the production bucket -- before failing.
    """
    missing = tmp_path / "typo.zarr"

    with pytest.raises(FileNotFoundError):
        _open_root(str(missing))

    assert not missing.exists(), "opening a nonexistent store created it on disk"
    assert list(tmp_path.iterdir()) == [], f"left files behind: {list(tmp_path.iterdir())}"


def test_open_root_reads_both_consolidated_and_unconsolidated(tmp_path: Path) -> None:
    """The unconsolidated path is permanent, not transitional: appends now strip the block."""
    store_path = tmp_path / "cube.zarr"
    group = zarr.open_group(str(store_path), mode="w", zarr_format=3)
    group.create_array("marker", shape=(1,), dtype="int32")

    assert "marker" in dict(_open_root(str(store_path)).members())

    zarr.consolidate_metadata(str(store_path), zarr_format=3)
    assert "marker" in dict(_open_root(str(store_path)).members())


# =============================================================================
# Tile id derivation (item id + grid:code)
# =============================================================================


@pytest.mark.parametrize(
    ("store_name", "expected_remainder"),
    [
        # A store not following the naming convention at all: silently produced item id
        # "s1-rtc-cube" and grid:code "MGRS-cube".
        ("cube.zarr", "cube"),
        # A suffixed rerun: "MGRS-31TCH_v2" is not a tile any search will ever join on.
        ("s1-rtc-31TCH_v2.zarr", "31TCH_v2"),
        # An empty tile: item id "s1-rtc-", grid:code "MGRS-".
        ("s1-rtc-.zarr", ""),
        # Lower case is not an MGRS tile id, and would not match a tile-filtered search.
        ("s1-rtc-31tch.zarr", "31tch"),
        # `I` is excluded from the MGRS 100 km square letters (it reads as 1).
        ("s1-rtc-31TIH.zarr", "31TIH"),
        # Zone numbers stop at 60.
        ("s1-rtc-61TCH.zarr", "61TCH"),
    ],
)
def test_malformed_tile_id_raises_instead_of_registering(
    tmp_path: Path, store_name: str, expected_remainder: str
) -> None:
    """A store name that yields no valid MGRS tile must fail the build, not mint a malformed id.

    None of these failed anywhere downstream before: they registered cleanly and poisoned the
    catalogue, because `grid:code` is the field tile-filtered searches and the cube↔acquisition
    cross-links join on.
    """
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A")]})
    renamed = store.with_name(store_name)
    store.rename(renamed)

    with pytest.raises(ValueError, match="MGRS tile id") as excinfo:
        build_s1_rtc_stac_item(str(renamed), "sentinel-1-grd-rtc-staging")
    # The message must show what it actually derived, or the operator cannot see the typo.
    assert repr(expected_remainder) in str(excinfo.value)


def test_legacy_store_prefix_still_yields_the_bare_tile_id(tmp_path: Path) -> None:
    """`s1-grd-rtc-` is what #246 reverts the store prefix to once titiler-eopf#108 lands.

    Stripping only `s1-rtc-` left the whole name as the tile: item id `s1-rtc-s1-grd-rtc-31TCH`,
    grid:code `MGRS-s1-grd-rtc-31TCH`. Accepting both prefixes means that rename is not a fleet-wide
    build failure the day it happens.
    """
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A")]})
    renamed = store.with_name("s1-grd-rtc-31TCH.zarr")
    store.rename(renamed)

    item = build_s1_rtc_stac_item(str(renamed), "sentinel-1-grd-rtc-staging")
    assert item.id == "s1-rtc-31TCH"
    assert item.properties["grid:code"] == "MGRS-31TCH"


# =============================================================================
# Antimeridian / densified reprojection
# =============================================================================


def test_zone_1_bbox_stays_narrow_and_crosses_the_antimeridian(tmp_path: Path) -> None:
    """UTM zone 1 (and 60) must not produce a near-global footprint.

    Transforming only the corners put the west edge just *east* of the antimeridian and the east
    edge just *west* of it, so min/max straddled the wrap: tile 01VCK came out as
    (-178.41, 54.11, 179.94, 55.13) — 358.4 degrees wide, a footprint matching almost every spatial
    search on Earth. The densified `transform_bounds` returns the crossing box instead (west > east,
    per STAC/GeoJSON), which is ~1.75 degrees wide.
    """
    store = _make_s1_store(
        tmp_path,
        {"descending": [(T1_NS, "S1A")]},
        tile_id="01VCK",
        crs=make_crs_code("EPSG:32601"),
        utm_bbox=make_bounding_box([300000.0, 6000000.0, 409800.0, 6109800.0]),
    )
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    assert item.bbox is not None
    west, south, east, north = item.bbox
    assert west > east, "an antimeridian-crossing bbox is expressed as west > east"
    assert (east + 360.0) - west == pytest.approx(1.75, abs=0.1)
    assert (south, north) == pytest.approx((54.109, 55.127), abs=0.01)

    # A single ring from west to east would run the long way round the globe; GeoJSON requires the
    # footprint be split at ±180 instead.
    geometry = item.geometry
    assert geometry is not None
    assert geometry["type"] == "MultiPolygon"
    lons = [pt[0] for poly in geometry["coordinates"] for ring in poly for pt in ring]
    assert max(lons) == 180.0
    assert min(lons) == -180.0


def test_non_crossing_bbox_is_unchanged_and_stays_a_polygon(tmp_path: Path) -> None:
    """The antimeridian handling must not perturb the ordinary case (EPSG:32631, zone 31)."""
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    assert item.bbox is not None
    west, south, east, north = item.bbox
    assert west < east
    assert (west, south, east, north) == pytest.approx((0.457, 44.226, 1.748, 45.146), abs=0.01)
    assert item.geometry is not None
    assert item.geometry["type"] == "Polygon"


def test_bbox_union_across_orbits_does_not_wrap_the_globe(tmp_path: Path) -> None:
    """Unioning two crossing bboxes with a plain min(west)/max(east) reinstates the 358-degree box.

    Both orbit groups of a zone-1 tile cross the antimeridian, so the naive union picks the
    westmost west and the eastmost east straight back out of the crossing pair.
    """
    store_path = tmp_path / "s1-rtc-01VCK.zarr"
    root = zarr.open_group(str(store_path), mode="w", zarr_format=3)
    for orbit_dir, bbox in [
        ("descending", [300000.0, 6000000.0, 409800.0, 6109800.0]),
        ("ascending", [409800.0, 6000000.0, 509800.0, 6109800.0]),
    ]:
        og = root.create_group(orbit_dir)
        og.attrs.update({"proj:code": "EPSG:32601", "spatial:bbox": bbox})
        r10m = og.create_group("r10m")
        r10m.create_array("time", shape=(1,), dtype="int64", chunks=(512,))[:] = [T1_NS]
        r10m.create_array("platform", shape=(1,), dtype="<U4", chunks=(512,))[:] = ["S1A"]
    zarr.consolidate_metadata(str(store_path), zarr_format=3)

    item = build_s1_rtc_stac_item(str(store_path), "sentinel-1-grd-rtc-staging")

    assert item.bbox is not None
    west, _south, east, _north = item.bbox
    assert west > east  # still one crossing box, not an inverted global one
    assert (east + 360.0) - west < 5.0, f"union wrapped the globe: {item.bbox}"


# =============================================================================
# Projection edge cases
# =============================================================================


def test_reference_system_falls_back_to_wkt_when_there_is_no_epsg_code(tmp_path: Path) -> None:
    """`to_epsg()` returns None for a CRS with no EPSG match, and that None used to be emitted as a
    literal `null` reference_system — telling a reader nothing about the grid.

    datacube v2.2.0 accepts an EPSG integer, a WKT2 string or a PROJJSON object there, so fall back
    to WKT2. `proj:code` is only validated as a non-empty pyproj-parseable string, so a WKT2 value
    reaches the builder intact.
    """
    wkt = pyproj.CRS.from_user_input(
        "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +datum=WGS84 +units=m +no_defs"
    ).to_wkt()
    assert pyproj.CRS.from_user_input(wkt).to_epsg() is None  # the precondition being handled

    store = _make_s1_store(
        tmp_path,
        {"descending": [(T1_NS, "S1A")]},
        crs=make_crs_code(wkt),
        utm_bbox=make_bounding_box([4000000.0, 3000000.0, 4100000.0, 3100000.0]),
    )
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    dims = item.properties["cube:dimensions"]
    assert dims["x"]["reference_system"] == dims["y"]["reference_system"]
    assert dims["x"]["reference_system"].startswith("PROJCRS[")


def test_reference_system_is_the_epsg_int_when_there_is_one(tmp_path: Path) -> None:
    """The WKT fallback must not displace the plain EPSG integer for a normal UTM store."""
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A")]})
    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    assert item.properties["cube:dimensions"]["x"]["reference_system"] == 32631


def _set_orbit_coord(store: Path, orbit: str, name: str, values: list[int]) -> None:
    """Overwrite an orbit-number coordinate in place (test helper)."""
    root = zarr.open_group(str(store), mode="r+", zarr_format=3, use_consolidated=False)
    orbit_group = root[orbit]
    assert isinstance(orbit_group, zarr.Group)
    r10m = orbit_group["r10m"]
    assert isinstance(r10m, zarr.Group)
    arr = r10m[name]
    assert isinstance(arr, zarr.Array)
    arr.resize((len(values),))
    arr[:] = np.array(values, dtype="int32")


def test_zero_orbit_numbers_are_not_emitted(tmp_path: Path) -> None:
    """`0` means "not recorded", and the sat extension forbids it.

    Both coordinates are created with `fill_value=0` and the append resizes all three metadata
    coordinates before writing them, so a torn append leaves a correctly-shaped slice holding 0 —
    and there is a known upstream bug putting 0 on live slices. The sat v1.0.0 schema declares
    both fields `{"type": "integer", "minimum": 1}`, so a 0 makes a validating STAC API reject
    the item outright.
    """
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A"), (T2_NS, "S1A")]})
    _set_orbit_coord(store, "descending", "relative_orbit", [0, 0])
    _set_orbit_coord(store, "descending", "absolute_orbit", [0, 0])

    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    assert "sat:relative_orbit" not in item.properties
    assert "sat:absolute_orbit" not in item.properties


def test_orbit_number_needs_every_slice_recorded(tmp_path: Path) -> None:
    """A single-valued sat field may not be claimed from a partial pool.

    One unrecorded slice among otherwise-agreeing values must drop the field: the cube cannot
    assert a track it has not observed for every acquisition.
    """
    store = _make_s1_store(tmp_path, {"descending": [(T1_NS, "S1A"), (T2_NS, "S1A")]})
    _set_orbit_coord(store, "descending", "relative_orbit", [110, 0])

    item = build_s1_rtc_stac_item(str(store), "sentinel-1-grd-rtc-staging")

    assert "sat:relative_orbit" not in item.properties, (
        "one unrecorded slice means the cube cannot claim a single track"
    )
