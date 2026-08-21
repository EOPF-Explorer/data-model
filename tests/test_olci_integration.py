"""Integration tests for convert_olci_optimized."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
import zarr
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import GroupSpec

from eopf_geozarr.s3_olci_optimization.olci_converter import (
    _sanitize_olci_array_attrs_keep_fill,
)

if TYPE_CHECKING:
    import pathlib

from eopf_geozarr.s3_olci_optimization.olci_converter import convert_olci_optimized


def build_synthetic_olci(rows: int = 512, cols: int = 480) -> xr.DataTree:
    """Minimal synthetic OLCI L1 EFR datatree (measurements only).

    Geolocation is an axis-aligned ~300 m grid (0.003 deg spacing) so the
    dataset is genuinely warpable to a regular lat/lon grid.
    """
    rng = np.random.default_rng(0)
    lat_1d = np.linspace(45.0, 45.0 + 0.003 * (rows - 1), rows)
    lon_1d = np.linspace(10.0, 10.0 + 0.003 * (cols - 1), cols)
    lat = np.repeat(lat_1d[:, None], cols, axis=1)
    lon = np.repeat(lon_1d[None, :], rows, axis=0)
    alt = np.zeros((rows, cols), dtype="int16")
    data: dict[str, xr.DataArray] = {}
    for i in range(1, 22):
        name = f"oa{i:02d}_radiance"
        arr = xr.DataArray(
            rng.integers(0, 6000, (rows, cols)).astype("uint16"),
            dims=("rows", "columns"),
            attrs={
                "scale_factor": 0.0139,
                "add_offset": 0.0,
                "standard_name": "toa_upwelling_spectral_radiance",
                "coordinates": "latitude longitude altitude",
            },
        )
        data[name] = arr
    ds = xr.Dataset(
        data,
        coords={
            "latitude": (("rows", "columns"), lat, {"standard_name": "latitude"}),
            "longitude": (("rows", "columns"), lon, {"standard_name": "longitude"}),
            "altitude": (("rows", "columns"), alt, {"standard_name": "altitude"}),
        },
    )
    return xr.DataTree.from_dict({"/measurements": ds})


def test_convert_olci_writes_measurements(tmp_path: object) -> None:
    """Native-resolution measurements group must contain all 21 radiance bands."""
    import zarr

    dt = build_synthetic_olci()
    out = str(tmp_path / "olci_geozarr.zarr")  # type: ignore[operator]
    convert_olci_optimized(dt, output_path=out)

    g = zarr.open_group(out, mode="r")
    # native measurements present
    assert "measurements" in g
    # all 21 bands at native res, in the r0 base-level group
    meas = g["measurements"]
    assert isinstance(meas, zarr.Group)
    r0 = meas["r0"]
    assert isinstance(r0, zarr.Group)
    for i in range(1, 22):
        assert f"oa{i:02d}_radiance" in r0


def test_convert_olci_creates_overviews(tmp_path: object) -> None:
    """Overview subgroups must only be written when BOTH post-decimation dims >= min_dimension.

    Case 1 — 512x480 with min_dimension=256:
        480//2 = 240 < 256, so the guard fires immediately → zero overview levels.

    Case 2 — 1024x1024 with min_dimension=256:
        the warped grid stays close to 1024x1024, yielding exactly two
        overview levels; each halves the previous and the deepest would drop
        below min_dimension if halved again.
    """
    import zarr

    # --- Case 1: 512x480, min_dimension=256 → zero overview levels ---
    dt1 = build_synthetic_olci(rows=512, cols=480)
    out1 = str(tmp_path / "olci_case1.zarr")  # type: ignore[operator]
    convert_olci_optimized(dt1, output_path=out1, min_dimension=256)

    g1 = zarr.open_group(out1, mode="r")
    meas1 = g1["measurements"]
    assert isinstance(meas1, zarr.Group)
    subgroups1 = sorted(meas1.group_keys())
    assert subgroups1 == ["r0"], (
        f"Expected only the r0 base level for 512x480 at min_dimension=256, got {subgroups1}"
    )

    # --- Case 2: 1024x1024, min_dimension=256 → exactly two valid levels ---
    dt2 = build_synthetic_olci(rows=1024, cols=1024)
    out2 = str(tmp_path / "olci_case2.zarr")  # type: ignore[operator]
    convert_olci_optimized(dt2, output_path=out2, min_dimension=256)

    g2 = zarr.open_group(out2, mode="r")
    meas2 = g2["measurements"]
    assert isinstance(meas2, zarr.Group)
    subgroups2 = sorted(meas2.group_keys())
    assert subgroups2 == ["r0", "r2", "r4"], (
        f"Expected r0 + exactly 2 overview levels for 1024x1024 at min_dimension=256, "
        f"got {subgroups2}"
    )

    # Every level must have BOTH spatial dims >= min_dimension, and each
    # overview must be exactly half the previous level (floor division).
    r0_2 = meas2["r0"]
    assert isinstance(r0_2, zarr.Group)
    band0 = r0_2["oa01_radiance"]
    assert isinstance(band0, zarr.Array)
    prev_shape = band0.shape
    for sg_name in [k for k in subgroups2 if k != "r0"]:
        sg = meas2[sg_name]
        assert isinstance(sg, zarr.Group)
        band = sg["oa01_radiance"]
        assert isinstance(band, zarr.Array)
        assert band.shape[0] == prev_shape[0] // 2
        assert band.shape[1] == prev_shape[1] // 2
        assert band.shape[0] >= 256
        assert band.shape[1] >= 256
        prev_shape = band.shape

    # The deepest level would violate min_dimension if halved again.
    assert prev_shape[0] // 2 < 256 or prev_shape[1] // 2 < 256


def test_convert_olci_returns_datatree(tmp_path: object) -> None:
    """convert_olci_optimized must return an xr.DataTree backed by the output store."""
    dt = build_synthetic_olci(rows=256, cols=256)
    out = str(tmp_path / "olci_geozarr.zarr")  # type: ignore[operator]
    result = convert_olci_optimized(dt, output_path=out, min_dimension=256)
    assert isinstance(result, xr.DataTree)
    assert "/measurements" in result.groups


def test_convert_olci_native_output_opens_as_datatree(tmp_path: object) -> None:
    """Acceptance for the default (native) mode: swath geometry, no CRS.

    The store keeps instrument geometry: r0 holds raw bands with 2-D
    lat/lon over (rows, columns) at exact input size, overview siblings
    halve it, the whole store opens with xr.open_datatree, and nothing
    fabricates a CRS (no spatial_ref, no grid_mapping, no proj: attrs).
    """
    dt = build_synthetic_olci(rows=512, cols=512)
    out = str(tmp_path / "olci_native.zarr")  # type: ignore[operator]
    result = convert_olci_optimized(dt, output_path=out, min_dimension=128)

    opened = xr.open_datatree(out, engine="zarr", consolidated=False, chunks={})

    r0 = opened["/measurements/r0"].to_dataset()
    assert dict(r0.sizes) == {"rows": 512, "columns": 512}  # exact: no warp
    for level, size in (("r0", 512), ("r2", 256), ("r4", 128)):
        ds = opened[f"/measurements/{level}"].to_dataset()
        assert dict(ds.sizes) == {"rows": size, "columns": size}
        assert ds["latitude"].dims == ("rows", "columns")
        assert ds["longitude"].dims == ("rows", "columns")
        assert "spatial_ref" not in ds.variables
        assert "grid_mapping" not in ds["oa01_radiance"].attrs
        assert ds.rio.crs is None

    meas_group = zarr.open_group(out, mode="r")["measurements"]
    assert isinstance(meas_group, zarr.Group)
    meas_attrs = dict(meas_group.attrs)
    assert meas_attrs["spatial:dimensions"] == ["rows", "columns"]
    assert not any(k.startswith("proj:") for k in meas_attrs)
    multiscales = meas_attrs["multiscales"]
    assert isinstance(multiscales, dict)
    layout = multiscales["layout"]
    assert isinstance(layout, list)
    assert layout[0] == {"asset": "r0"}
    r0_group = meas_group["r0"]
    assert isinstance(r0_group, zarr.Group)
    r0_attrs = dict(r0_group.attrs)
    assert r0_attrs["spatial:dimensions"] == ["rows", "columns"]
    assert not any(k.startswith("proj:") for k in r0_attrs)

    assert "/measurements/r0" in result.groups
    assert "/measurements/r2" in result.groups


def test_convert_olci_invalid_output_grid_raises(tmp_path: object) -> None:
    dt = build_synthetic_olci(rows=64, cols=64)
    out = str(tmp_path / "bad.zarr")  # type: ignore[operator]
    with pytest.raises(ValueError, match="output_grid"):
        convert_olci_optimized(dt, output_path=out, output_grid="not-a-crs")


def test_convert_olci_regridded_output_opens_as_datatree(tmp_path: object) -> None:
    """Acceptance for the opt-in regridded mode: the exported store is a regular grid with CRS at every level.

    xr.open_datatree must open the whole store; measurements/r0 holds the
    warped native-resolution grid with 1-D y/x coordinates and a declared
    CRS, overview siblings halve it, and the multiscales layout references
    the base level by name.
    """
    dt = build_synthetic_olci(rows=1024, cols=1024)
    out = str(tmp_path / "olci_geozarr.zarr")  # type: ignore[operator]
    result = convert_olci_optimized(dt, output_path=out, min_dimension=256, output_grid="EPSG:4326")

    opened = xr.open_datatree(out, engine="zarr", consolidated=False, chunks={})

    r0 = opened["/measurements/r0"].to_dataset()
    assert set(r0.sizes) == {"y", "x"}
    for i in range(1, 22):
        assert f"oa{i:02d}_radiance" in r0
    # CRS declared at every level, 1-D regular coordinates
    for level in ("r0", "r2", "r4"):
        ds = opened[f"/measurements/{level}"].to_dataset()
        assert ds.rio.crs is not None, f"{level}: no CRS"
        assert ds.rio.crs.to_epsg() == 4326
        for dim in ("y", "x"):
            coord = ds[dim]
            assert coord.ndim == 1
            steps = np.diff(coord.values)
            assert np.allclose(steps, steps[0])
        band = ds["oa01_radiance"]
        assert band.attrs["grid_mapping"] == "spatial_ref"
    # halving structure relative to observed r0
    r2 = opened["/measurements/r2"].to_dataset()
    assert r2.sizes["y"] == r0.sizes["y"] // 2
    assert r2.sizes["x"] == r0.sizes["x"] // 2

    # measurements itself holds only convention metadata
    meas = opened["/measurements"].to_dataset()
    assert len(meas.data_vars) == 0

    meas_attrs = dict(zarr.open_group(out, mode="r")["measurements"].attrs)
    multiscales = meas_attrs["multiscales"]
    assert isinstance(multiscales, dict)
    layout = multiscales["layout"]
    assert isinstance(layout, list)
    assert layout[0] == {"asset": "r0"}
    # per-level geo-proj convention present
    meas_group = zarr.open_group(out, mode="r")["measurements"]
    assert isinstance(meas_group, zarr.Group)
    r0_group = meas_group["r0"]
    assert isinstance(r0_group, zarr.Group)
    r0_attrs = dict(r0_group.attrs)
    assert "proj:code" in r0_attrs

    assert "/measurements/r0" in result.groups
    assert "/measurements/r2" in result.groups


def test_convert_olci_conditions_quality_passthrough(tmp_path: object) -> None:
    """conditions and quality groups, when present, are copied through unchanged."""
    import zarr

    # Build a tree with conditions and quality groups
    rng = np.random.default_rng(1)
    rows, cols = 128, 128
    lat_1d = np.linspace(45.0, 45.0 + 0.003 * (rows - 1), rows)
    lon_1d = np.linspace(10.0, 10.0 + 0.003 * (cols - 1), cols)
    lat = np.repeat(lat_1d[:, None], cols, axis=1)
    lon = np.repeat(lon_1d[None, :], rows, axis=0)
    alt = np.zeros((rows, cols), dtype="int16")

    meas_data: dict[str, xr.DataArray] = {
        "oa01_radiance": xr.DataArray(
            rng.integers(0, 6000, (rows, cols)).astype("uint16"),
            dims=("rows", "columns"),
        )
    }
    meas_ds = xr.Dataset(
        meas_data,
        coords={
            "latitude": (("rows", "columns"), lat),
            "longitude": (("rows", "columns"), lon),
            "altitude": (("rows", "columns"), alt),
        },
    )
    cond_ds = xr.Dataset(
        {
            "wind_speed": xr.DataArray(
                rng.random((rows, cols)).astype("float32"), dims=("rows", "columns")
            )
        }
    )
    quality_ds = xr.Dataset(
        {
            "flags": xr.DataArray(
                rng.integers(0, 255, (rows, cols)).astype("uint8"), dims=("rows", "columns")
            )
        }
    )
    # Build a measurements/orphans sub-dataset to verify child subgroup copy.
    orphans_ds = xr.Dataset(
        {
            "removed_count": xr.DataArray(
                rng.integers(0, 10, (rows,)).astype("int32"), dims=("rows",)
            )
        }
    )
    dt = xr.DataTree.from_dict(
        {
            "/measurements": meas_ds,
            "/measurements/orphans": orphans_ds,
            "/conditions": cond_ds,
            "/quality": quality_ds,
        }
    )

    out = str(tmp_path / "olci_geozarr.zarr")  # type: ignore[operator]
    convert_olci_optimized(dt, output_path=out, min_dimension=64)

    g = zarr.open_group(out, mode="r")
    assert "conditions" in g
    assert "quality" in g
    # measurements/orphans subgroup must have been copied through.
    assert "orphans" in g["measurements"]


def test_cli_convert_s3_olci_optimized(tmp_path: pathlib.Path) -> None:
    """convert-s3-olci-optimized subcommand must write a GeoZarr measurements group."""
    import zarr

    # materialise a synthetic OLCI product to a zarr v2 store on disk
    dt = build_synthetic_olci(rows=300, cols=300)
    src = tmp_path / "olci_src.zarr"
    dt.to_zarr(src, mode="w", consolidated=False)
    out = tmp_path / "olci_out.zarr"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "eopf_geozarr",
            "convert-s3-olci-optimized",
            str(src),
            str(out),
            "--spatial-chunk",
            "256",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    g = zarr.open_group(str(out), mode="r")
    assert "measurements" in g

    # Default (no flag) produced the native instrument grid.
    meas_default = g["measurements"]
    assert isinstance(meas_default, zarr.Group)
    r0_default = meas_default["r0"]
    assert isinstance(r0_default, zarr.Group)
    band_default = r0_default["oa01_radiance"]
    assert isinstance(band_default, zarr.Array)
    assert band_default.metadata.dimension_names == ("rows", "columns")  # type: ignore[attr-defined]

    # Opt-in regridding via --output-grid.
    out2 = tmp_path / "olci_out_gridded.zarr"
    result2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "eopf_geozarr",
            "convert-s3-olci-optimized",
            str(src),
            str(out2),
            "--output-grid",
            "EPSG:4326",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result2.returncode == 0, result2.stdout + result2.stderr
    g2 = zarr.open_group(str(out2), mode="r")
    meas2 = g2["measurements"]
    assert isinstance(meas2, zarr.Group)
    r0_2 = meas2["r0"]
    assert isinstance(r0_2, zarr.Group)
    band2 = r0_2["oa01_radiance"]
    assert isinstance(band2, zarr.Array)
    assert band2.metadata.dimension_names == ("y", "x")  # type: ignore[attr-defined]


def _assert_radiance_dtype_and_attrs(
    group: zarr.Group, band_name: str, *, level_label: str
) -> None:
    """Assert that *band_name* in *group* is uint16 with scale_factor and no stale attrs.

    This is a regression guard: the converter must preserve raw integer storage
    and CF scale/offset, and must strip source-only attrs (_eopf_attrs, dtype,
    valid_min, valid_max).
    """
    band = group[band_name]
    assert isinstance(band, zarr.Array), f"{level_label}/{band_name} is not a zarr.Array"
    assert band.dtype == np.dtype("uint16"), (
        f"{level_label}/{band_name}: expected uint16, got {band.dtype}"
    )
    attrs = dict(band.attrs)
    assert "scale_factor" in attrs, (
        f"{level_label}/{band_name}: scale_factor missing from attrs (got {list(attrs)})"
    )
    for stale_key in ("_eopf_attrs", "dtype", "valid_min", "valid_max"):
        assert stale_key not in attrs, (
            f"{level_label}/{band_name}: stale attr '{stale_key}' present in output attrs"
        )


@pytest.mark.parametrize(
    ("output_grid", "golden_suffix"),
    [("native", "native"), ("EPSG:4326", "epsg4326")],
    ids=("native", "epsg4326"),
)
def test_olci_conversion_matches_snapshot(
    s3_olci_group_example: pathlib.Path,
    tmp_path: pathlib.Path,
    output_grid: str,
    golden_suffix: str,
) -> None:
    """Snapshot test: converted OLCI structure must match committed golden file.

    The fixture is a Zarr v2 store representing a real OLCI L1 EFR product.
    We open the whole DataTree so that conditions, quality, and
    measurements/orphans subgroups are included in the conversion.
    The fixture uses the real product's dimension names: tie-point grids reuse
    'columns' (at size 4) while measurement grids also use 'columns' (at size 16);
    orphan arrays use removed_pixels=4.

    ``min_dimension=8`` is used so that the 16x16 measurements grid generates
    one overview level (r2 at 8x8).

    Parametrized over both output_grid modes (native swath, regridded
    EPSG:4326), each with its own committed golden file.

    To (re)generate the snapshot, uncomment the regeneration block below,
    run the test once, then re-comment before committing.
    """
    # The JSON fixture materializes arrays as zeros; zero lat/lon is a
    # degenerate geolocation the warp rejects. Seed a plausible ~300 m
    # (0.003 deg) grid the way the real product stores it: the fixture's
    # latitude/longitude arrays are raw int32 microdegrees with a CF
    # scale_factor of 1e-6 (real degrees = raw * scale_factor), so write
    # RAW values of degrees / 1e-6. Both pipelines unpack the packing
    # (reproject_olci for the warp, _geodesic_block_mean for native
    # centroids), so the seeded geolocation decodes to real degrees.
    fixture_group = zarr.open_group(str(s3_olci_group_example), mode="a")
    fixture_meas = fixture_group["measurements"]
    assert isinstance(fixture_meas, zarr.Group)
    lat_arr = fixture_meas["latitude"]
    assert isinstance(lat_arr, zarr.Array)
    ny, nx = lat_arr.shape
    lat_deg = np.linspace(45.0, 45.0 + 0.003 * (ny - 1), ny)
    lon_deg = np.linspace(10.0, 10.0 + 0.003 * (nx - 1), nx)
    lat_arr[:] = np.round(np.repeat(lat_deg[:, None], nx, axis=1) / 1e-6).astype("int32")
    lon_arr = fixture_meas["longitude"]
    assert isinstance(lon_arr, zarr.Array)
    lon_arr[:] = np.round(np.repeat(lon_deg[None, :], ny, axis=0) / 1e-6).astype("int32")

    dt_in = xr.open_datatree(
        str(s3_olci_group_example),
        engine="zarr",
        consolidated=False,
        chunks={},
        mask_and_scale=False,
    )
    out = str(tmp_path / "out.zarr")
    convert_olci_optimized(dt_in, output_path=out, min_dimension=8, output_grid=output_grid)

    observed_group = zarr.open_group(out, use_consolidated=False)
    observed_structure_json = GroupSpec.from_zarr(observed_group).model_dump()

    expected_path = Path("tests/_test_data/optimized_olci_examples") / (
        f"{s3_olci_group_example.stem}-{golden_suffix}.json"
    )

    # Uncomment this block to (re)generate the snapshot from the observed structure.
    # expected_path.parent.mkdir(parents=True, exist_ok=True)
    # expected_path.write_text(json.dumps(observed_structure_json, indent=2, sort_keys=True))

    observed_structure = GroupSpec(**tuplify_json(observed_structure_json))
    observed_structure_flat = observed_structure.to_flat()
    expected_structure_json = tuplify_json(json.loads(expected_path.read_text()))
    expected_structure = GroupSpec(**expected_structure_json)
    expected_structure_flat = expected_structure.to_flat()

    o_keys = set(observed_structure_flat.keys())
    e_keys = set(expected_structure_flat.keys())
    assert o_keys == e_keys
    assert [k for k in o_keys if observed_structure_flat[k] != expected_structure_flat[k]] == []

    # Dtype/attrs regression guard: radiance must be stored as uint16 with
    # scale_factor preserved and stale source attrs absent.
    meas_g = observed_group["measurements"]
    assert isinstance(meas_g, zarr.Group)
    r0_g = meas_g["r0"]
    assert isinstance(r0_g, zarr.Group)
    _assert_radiance_dtype_and_attrs(r0_g, "oa01_radiance", level_label="r0")
    if "r2" in meas_g:
        r2_g = meas_g["r2"]
        assert isinstance(r2_g, zarr.Group)
        _assert_radiance_dtype_and_attrs(r2_g, "oa01_radiance", level_label="r2")


def test_convert_olci_odd_dims_overview_no_conflicting_sizes(tmp_path: object) -> None:
    """Overview groups written from an odd-dimensioned swath must open without errors.

    Regression test for the off-by-one bug in reduce_swath where coordinate
    decimation via [::factor] produced ceil(N/factor) elements while
    coarsen(boundary="trim") produced floor(N/factor) for the radiance data.
    On an odd-column real OLCI product (4865 cols) this caused xr.open_dataset
    to raise ``ValueError: conflicting sizes for dimension 'columns'``.

    We use rows=10, cols=9 (odd cols) with min_dimension=4 in the default
    (native) mode, so this asserts coord/data length agreement and
    floor-halving structure at every level over the swath ``rows``/``columns``
    dims rather than a specific (5, 4) shape.
    """
    dt = build_synthetic_olci(rows=10, cols=9)
    out = str(tmp_path / "odd_olci.zarr")  # type: ignore[operator]
    convert_olci_optimized(dt, output_path=out, min_dimension=4)

    # Determine which overview groups were written.
    import zarr as _zarr

    g = _zarr.open_group(out, mode="r")
    meas = g["measurements"]
    assert isinstance(meas, _zarr.Group)
    level_keys = sorted(meas.group_keys())
    assert level_keys[0] == "r0"

    # Open each level (base + overviews); this must NOT raise a
    # conflicting-sizes error, and each level must floor-halve the previous.
    prev_shape: tuple[int, ...] | None = None
    for lvl in level_keys:
        ds = xr.open_dataset(out, engine="zarr", group=f"measurements/{lvl}", consolidated=False)
        rad_shape = ds["oa01_radiance"].shape
        assert rad_shape == (ds.sizes["rows"], ds.sizes["columns"]), (
            f"measurements/{lvl}: data shape {rad_shape} != dim sizes "
            f"(rows={ds.sizes['rows']}, columns={ds.sizes['columns']})"
        )
        if prev_shape is not None:
            assert rad_shape[0] == prev_shape[0] // 2
            assert rad_shape[1] == prev_shape[1] // 2
        prev_shape = rad_shape
        ds.close()


def test_sanitize_olci_array_attrs_strips_stale_keeps_fill_value() -> None:
    """_sanitize_olci_array_attrs_keep_fill must strip stale source attrs and preserve _FillValue.

    Unlike the shared sanitize_array_attrs (which always strips _FillValue),
    the OLCI-local helper must preserve _FillValue so that downstream readers
    and reduce_swath can identify fill pixels on raw uint16 data opened with
    mask_and_scale=False.
    """
    attrs: dict[str, object] = {
        "_eopf_attrs": {"source": "some blob"},
        "dtype": "uint16",
        "valid_min": 0,
        "valid_max": 65534,
        "scale_factor": 0.0139,
        "add_offset": 0.0,
        "_FillValue": 65535,
        "units": "W m-2 sr-1 um-1",
        "standard_name": "toa_upwelling_spectral_radiance",
        "coordinates": "latitude longitude altitude",
    }
    result = _sanitize_olci_array_attrs_keep_fill(attrs)
    # Stale source-only attrs must be removed.
    assert "_eopf_attrs" not in result, "_eopf_attrs must be stripped"
    assert "dtype" not in result, "dtype must be stripped"
    assert "valid_min" not in result, "valid_min must be stripped"
    assert "valid_max" not in result, "valid_max must be stripped"
    # CF and fill attrs must be preserved.
    assert result.get("scale_factor") == 0.0139, "scale_factor must be preserved"
    assert result.get("add_offset") == 0.0, "add_offset must be preserved"
    assert result.get("_FillValue") == 65535, "_FillValue must be preserved for OLCI raw uint16"
    assert result.get("units") == "W m-2 sr-1 um-1", "units must be preserved"
    assert result.get("standard_name") == "toa_upwelling_spectral_radiance"
    assert result.get("coordinates") == "latitude longitude altitude"


def test_convert_olci_rerun_removes_stale_overview_groups(tmp_path: object) -> None:
    """Re-running against an existing output path must not leave stale groups.

    The first run (small min_dimension) produces more overview levels than the
    second; without up-front store truncation the extra r{N} groups from run
    one would survive run two.
    """
    import zarr

    out = str(tmp_path / "olci_geozarr.zarr")  # type: ignore[operator]
    convert_olci_optimized(
        build_synthetic_olci(rows=256, cols=256), output_path=out, min_dimension=64
    )
    root = zarr.open_group(out, mode="r")
    meas = root["measurements"]
    assert isinstance(meas, zarr.Group)
    levels_first = {k for k in meas.group_keys() if k.startswith("r")}
    assert "r4" in levels_first

    convert_olci_optimized(
        build_synthetic_olci(rows=256, cols=256), output_path=out, min_dimension=128
    )
    root = zarr.open_group(out, mode="r")
    meas = root["measurements"]
    assert isinstance(meas, zarr.Group)
    levels_second = {k for k in meas.group_keys() if k.startswith("r")}
    assert levels_second == {"r0", "r2"}, f"stale overview groups survived re-run: {levels_second}"


def test_convert_olci_nonpositive_min_dimension_raises(tmp_path: object) -> None:
    """min_dimension < 1 must fail fast, before the output store is touched.

    Regression: _overview_levels loops `while min(r, c) // 2 >= min_dimension`,
    which never terminates once r and c decay to 0 — so min_dimension <= 0
    previously hung the converter forever (after truncating the store).
    """
    dt = build_synthetic_olci(rows=64, cols=64)
    for bad in (0, -1):
        out = str(tmp_path / f"bad_{bad}.zarr")  # type: ignore[operator]
        with pytest.raises(ValueError, match="min_dimension"):
            convert_olci_optimized(dt, output_path=out, min_dimension=bad)
        assert not Path(out).exists(), "store must not be created on invalid min_dimension"


def test_convert_olci_regridded_overviews_share_common_origin(tmp_path: object) -> None:
    """Regridded overview levels must be edge-aligned /2 reductions of r0.

    Radiance is block-averaged, so an overview coordinate is the CENTER of
    the block it aggregates. Stride-decimating the 1-D coords instead put
    every level's declared transform/bbox (2^l - 1)/2 fine pixels up/left of
    the data and contradicted the multiscales layout's
    {scale: [2, 2], translation: [0, 0]}.
    """
    dt = build_synthetic_olci(rows=1024, cols=1024)
    out = str(tmp_path / "olci_reg.zarr")  # type: ignore[operator]
    convert_olci_optimized(dt, output_path=out, min_dimension=256, output_grid="EPSG:4326")

    levels = {}
    for level in ("r0", "r2", "r4"):
        levels[level] = xr.open_dataset(
            out,
            engine="zarr",
            group=f"measurements/{level}",
            consolidated=False,
            decode_coords="all",
        )
    t0 = levels["r0"].rio.transform(recalc=True)
    t2 = levels["r2"].rio.transform(recalc=True)
    t4 = levels["r4"].rio.transform(recalc=True)

    # All levels share one origin (edge-aligned pyramid)...
    for t in (t2, t4):
        assert abs(t.c - t0.c) < 1e-9, f"x-origin drifted: {t.c} vs {t0.c}"
        assert abs(t.f - t0.f) < 1e-9, f"y-origin drifted: {t.f} vs {t0.f}"
    # ...with pixel size doubling per level.
    assert np.isclose(t2.a, 2 * t0.a)
    assert np.isclose(t2.e, 2 * t0.e)
    assert np.isclose(t4.a, 4 * t0.a)
    assert np.isclose(t4.e, 4 * t0.e)

    # Overview coordinate = center of the aggregated 2x2 block, not the
    # first fine pixel's center.
    x0, y0 = levels["r0"]["x"].values, levels["r0"]["y"].values
    assert np.isclose(levels["r2"]["x"].values[0], (x0[0] + x0[1]) / 2)
    assert np.isclose(levels["r2"]["y"].values[0], (y0[0] + y0[1]) / 2)
