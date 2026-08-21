"""Tests for olci_multiscale: decimate_swath, reduce_swath, grid_spatial_attrs."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio.transform
import xarray as xr

from eopf_geozarr.s3_olci_optimization.olci_multiscale import (
    decimate_swath,
    grid_spatial_attrs,
    reduce_swath,
    swath_spatial_attrs,
)


def _geodesic_center(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[float, float]:
    """Reference spherical centroid of a set of lat/lon positions, in degrees."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype="float64"))
    lon = np.deg2rad(np.asarray(lon_deg, dtype="float64"))
    x = float((np.cos(lat) * np.cos(lon)).mean())
    y = float((np.cos(lat) * np.sin(lon)).mean())
    z = float(np.sin(lat).mean())
    return (
        float(np.rad2deg(np.arctan2(z, np.hypot(x, y)))),
        float(np.rad2deg(np.arctan2(y, x))),
    )


def _swath(rows: int = 8, cols: int = 6) -> xr.Dataset:
    """Minimal synthetic swath dataset with one radiance band and two coords."""
    rad = xr.DataArray(
        np.arange(rows * cols, dtype="uint16").reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={
            "scale_factor": 0.5,
            "units": "mW.m-2.sr-1.nm-1",
            "_FillValue": 65535,
        },
    )
    lat = xr.DataArray(
        np.linspace(0, 1, rows * cols).reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={"standard_name": "latitude"},
    )
    lon = xr.DataArray(
        np.linspace(10, 11, rows * cols).reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={"standard_name": "longitude"},
    )
    return xr.Dataset(
        {"oa01_radiance": rad},
        coords={"latitude": lat, "longitude": lon},
    )


# ---------------------------------------------------------------------------
# decimate_swath tests
# ---------------------------------------------------------------------------


def test_decimate_halves_each_axis() -> None:
    out = decimate_swath(_swath(8, 6), factor=2)
    assert out["oa01_radiance"].shape == (4, 3)
    assert out["latitude"].shape == (4, 3)
    assert out["longitude"].shape == (4, 3)


def test_decimate_takes_every_other_pixel() -> None:
    ds = _swath(8, 6)
    out = decimate_swath(ds, factor=2)
    # top-left pixel is preserved exactly (no averaging)
    assert int(out["oa01_radiance"].values[0, 0]) == 0
    assert float(out["latitude"].values[0, 0]) == 0.0
    # interior pixel: stride-2 decimation means out[1, 1] comes from original [2, 2]
    assert int(out["oa01_radiance"].values[1, 1]) == int(ds["oa01_radiance"].values[2, 2])


def test_decimate_preserves_attrs() -> None:
    out = decimate_swath(_swath(8, 6), factor=2)
    assert out["oa01_radiance"].attrs["scale_factor"] == 0.5
    assert out["latitude"].attrs["standard_name"] == "latitude"


def test_decimate_factor_1_returns_unchanged() -> None:
    ds = _swath(8, 6)
    out = decimate_swath(ds, factor=1)
    assert out["oa01_radiance"].shape == (8, 6)


def test_decimate_invalid_factor_raises() -> None:
    with pytest.raises(ValueError, match="factor must be >= 1"):
        decimate_swath(_swath(), factor=0)


# ---------------------------------------------------------------------------
# reduce_swath tests
# ---------------------------------------------------------------------------


def test_reduce_swath_halves_each_axis() -> None:
    """reduce_swath must produce output with halved spatial dims."""
    out = reduce_swath(_swath(8, 6), factor=2)
    assert out["oa01_radiance"].shape == (4, 3)
    assert out["latitude"].shape == (4, 3)
    assert out["longitude"].shape == (4, 3)


def test_reduce_swath_radiance_is_averaged_not_decimated() -> None:
    """Radiance must be block-averaged; top-left output != top-left input (unless accident)."""
    rng = np.random.default_rng(42)
    rad_data = rng.integers(100, 200, (8, 6)).astype("uint16")
    rad = xr.DataArray(
        rad_data,
        dims=("rows", "columns"),
        attrs={"_FillValue": 65535},
    )
    ds = xr.Dataset({"oa01_radiance": rad})
    out = reduce_swath(ds, factor=2)
    # block [0:2, 0:2] averages to a value; verify it's a rounded mean
    expected_block = int(np.round(rad_data[0:2, 0:2].astype("float64").mean()))
    assert int(out["oa01_radiance"].values[0, 0]) == expected_block


def test_reduce_swath_altitude_is_block_averaged() -> None:
    """Altitude gets the fill-aware block mean, not stride decimation.

    A stride sample would locate the overview cell's altitude ~half a block
    away from the geodesic-centroid lat/lon describing the same cell.
    """
    fill = -32768
    alt = xr.DataArray(
        np.array([[100, 200], [300, fill]], dtype="int16"),
        dims=("rows", "columns"),
        attrs={"standard_name": "altitude", "_FillValue": fill},
    )
    lat = xr.DataArray(
        np.zeros((2, 2)), dims=("rows", "columns"), attrs={"standard_name": "latitude"}
    )
    lon = xr.DataArray(
        np.zeros((2, 2)), dims=("rows", "columns"), attrs={"standard_name": "longitude"}
    )
    ds = xr.Dataset(coords={"latitude": lat, "longitude": lon, "altitude": alt})
    out = reduce_swath(ds, factor=2)
    # Mean of the three valid pixels (fill excluded), not alt[0, 0] == 100.
    assert int(out["altitude"].values[0, 0]) == 200
    assert out["altitude"].dtype == np.dtype("int16")
    assert out["altitude"].attrs["standard_name"] == "altitude"


def test_reduce_swath_fill_value_preserved_in_all_fill_block() -> None:
    """A block where all pixels are fill must produce fill output, not 65535.0 average."""
    fill = 65535
    rad_data = np.ones((4, 4), dtype="uint16") * fill
    # put some non-fill values only in lower-right block
    rad_data[2:4, 2:4] = 100
    rad = xr.DataArray(
        rad_data,
        dims=("rows", "columns"),
        attrs={"_FillValue": fill},
    )
    ds = xr.Dataset({"oa01_radiance": rad})
    out = reduce_swath(ds, factor=2)
    # top-left block: all fill -> output must be fill
    assert int(out["oa01_radiance"].values[0, 0]) == fill
    # bottom-right block: all 100 -> output must be 100
    assert int(out["oa01_radiance"].values[1, 1]) == 100


def test_reduce_swath_preserves_attrs() -> None:
    """reduce_swath must carry over variable attributes."""
    out = reduce_swath(_swath(8, 6), factor=2)
    assert out["oa01_radiance"].attrs["scale_factor"] == 0.5
    assert out["latitude"].attrs["standard_name"] == "latitude"


def test_reduce_swath_factor_1_returns_unchanged() -> None:
    ds = _swath(8, 6)
    out = reduce_swath(ds, factor=1)
    assert out["oa01_radiance"].shape == (8, 6)


def test_reduce_swath_invalid_factor_raises() -> None:
    with pytest.raises(ValueError, match="factor must be >= 1"):
        reduce_swath(_swath(), factor=0)


def test_reduce_swath_non_swath_var_passthrough() -> None:
    """Variables that don't span (rows, columns) must pass through unchanged."""
    ds = _swath(8, 6)
    scalar = xr.DataArray(42.0, attrs={"info": "scalar"})
    ds = ds.assign({"extra": scalar})
    out = reduce_swath(ds, factor=2)
    assert "extra" in out
    assert float(out["extra"].values) == 42.0


# ---------------------------------------------------------------------------
# odd-dimension regression tests (real OLCI: 4865 columns is odd)
# ---------------------------------------------------------------------------


def _swath_odd(rows: int = 7, cols: int = 5) -> xr.Dataset:
    """Synthetic swath with ODD spatial dimensions.

    Matches the real-world scenario where OLCI products have 4865 columns
    (odd).  Before the fix, reduce_swath on an odd-sized dimension produced
    coordinate arrays one element longer than the corresponding radiance data
    (ceil vs floor of N/factor), causing xr.open_dataset to raise a
    conflicting-sizes error.
    """
    fill = 65535
    rad = xr.DataArray(
        np.arange(rows * cols, dtype="uint16").reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={
            "scale_factor": 0.5,
            "units": "mW.m-2.sr-1.nm-1",
            "_FillValue": fill,
        },
    )
    lat = xr.DataArray(
        np.linspace(0, 1, rows * cols).reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={"standard_name": "latitude"},
    )
    lon = xr.DataArray(
        np.linspace(10, 11, rows * cols).reshape(rows, cols),
        dims=("rows", "columns"),
        attrs={"standard_name": "longitude"},
    )
    return xr.Dataset(
        {"oa01_radiance": rad},
        coords={"latitude": lat, "longitude": lon},
    )


def test_reduce_swath_odd_dims_consistent_shape() -> None:
    """reduce_swath must produce identical shapes for radiance AND coordinates on odd dims.

    Regression test for the off-by-one bug where coordinate decimation via
    [::factor] yields ceil(N/factor) but coarsen(boundary="trim") yields
    floor(N/factor).  For rows=7, cols=5, factor=2 the expected output shape
    is (floor(7/2), floor(5/2)) = (3, 2).
    """
    ds = _swath_odd(rows=7, cols=5)
    out = reduce_swath(ds, factor=2)

    expected_rows = 7 // 2  # 3
    expected_cols = 5 // 2  # 2

    assert out["oa01_radiance"].shape == (expected_rows, expected_cols), (
        f"radiance shape {out['oa01_radiance'].shape} != ({expected_rows}, {expected_cols})"
    )
    assert out["latitude"].shape == (expected_rows, expected_cols), (
        f"latitude shape {out['latitude'].shape} != ({expected_rows}, {expected_cols})"
    )
    assert out["longitude"].shape == (expected_rows, expected_cols), (
        f"longitude shape {out['longitude'].shape} != ({expected_rows}, {expected_cols})"
    )


def test_reduce_swath_odd_dims_radiance_is_block_averaged() -> None:
    """Radiance values must be block-averaged (not decimated) on odd-dim inputs."""
    rng = np.random.default_rng(7)
    rows, cols = 7, 5
    rad_data = rng.integers(100, 200, (rows, cols)).astype("uint16")
    rad = xr.DataArray(
        rad_data,
        dims=("rows", "columns"),
        attrs={"_FillValue": 65535},
    )
    ds = xr.Dataset({"oa01_radiance": rad})
    out = reduce_swath(ds, factor=2)
    # The top-left output pixel must be the rounded mean of the 2x2 input block.
    expected = int(np.round(rad_data[0:2, 0:2].astype("float64").mean()))
    assert int(out["oa01_radiance"].values[0, 0]) == expected


def test_reduce_swath_odd_simulates_real_olci_columns() -> None:
    """Simulate the real-world OLCI case: 4090x4865 (odd cols) -> both 2432 cols.

    Uses smaller proxy dimensions that are proportionally odd to avoid
    heavy memory use: rows=10, cols=9 with factor=2 must yield (5, 4) for
    both radiance and coordinates.  This specifically guards floor vs ceil
    on the cols dimension (9 // 2 = 4, not 5).
    """
    ds = _swath_odd(rows=10, cols=9)
    out = reduce_swath(ds, factor=2)
    expected = (10 // 2, 9 // 2)  # (5, 4)
    assert out["oa01_radiance"].shape == expected, (
        f"radiance shape {out['oa01_radiance'].shape} != {expected}"
    )
    assert out["latitude"].shape == expected, (
        f"latitude shape {out['latitude'].shape} != {expected}"
    )
    assert out["longitude"].shape == expected, (
        f"longitude shape {out['longitude'].shape} != {expected}"
    )


# ---------------------------------------------------------------------------
# grid dims tests (for Task 4: reprojected pyramid)
# ---------------------------------------------------------------------------


def test_reduce_swath_on_grid_dims() -> None:
    """reduce_swath(dims=("y","x")) block-averages bands and strides 1-D coords.

    After reprojection the pyramid dims are (y, x): bands average fill-aware,
    the 1-D dimension coordinates decimate by trimmed stride, and non-spatial
    variables (spatial_ref) pass through unchanged.
    """
    ny, nx = 6, 5  # odd x exercises the coarsen-trim alignment
    band = np.arange(ny * nx, dtype="uint16").reshape(ny, nx)
    ds = xr.Dataset(
        {"oa01_radiance": (("y", "x"), band, {"_FillValue": 65535})},
        coords={
            "y": ("y", np.linspace(46.0, 45.0, ny)),
            "x": ("x", np.linspace(10.0, 11.0, nx)),
            "spatial_ref": ((), 0, {"crs_wkt": "stub"}),
        },
    )
    out = reduce_swath(ds, factor=2, dims=("y", "x"))
    assert dict(out.sizes) == {"y": 3, "x": 2}
    # block mean of the top-left 2x2 block, rounded
    expected00 = round((band[0, 0] + band[0, 1] + band[1, 0] + band[1, 1]) / 4)
    assert int(out["oa01_radiance"].values[0, 0]) == expected00
    # 1-D coords: trimmed stride, lengths match the data
    assert out["y"].size == 3
    assert out["x"].size == 2
    np.testing.assert_allclose(out["x"].values, ds["x"].values[0:4:2])
    # scalar passthrough survives
    assert "spatial_ref" in out.coords
    assert out["spatial_ref"].attrs["crs_wkt"] == "stub"


# ---------------------------------------------------------------------------
# grid_spatial_attrs tests
# ---------------------------------------------------------------------------


def test_grid_spatial_attrs() -> None:
    """grid_spatial_attrs derives dimensions, bbox, and 6-element transform."""
    transform = rasterio.transform.from_origin(10.0, 46.0, 0.01, 0.01)
    attrs = grid_spatial_attrs(transform, (100, 200))
    assert attrs["spatial:dimensions"] == ["y", "x"]
    assert attrs["spatial:registration"] == "pixel"  # type: ignore[index]
    assert attrs["spatial:transform"] == [  # type: ignore[index]
        0.01,
        0.0,
        10.0,
        0.0,
        -0.01,
        46.0,
    ]
    # bbox is [xmin, ymin, xmax, ymax] from array_bounds
    assert attrs["spatial:bbox"] == [10.0, 45.0, 12.0, 46.0]  # type: ignore[index]


def test_reduce_fill_collision_nudged_not_recoded_as_fill() -> None:
    """A valid block whose mean rounds to the fill sentinel must not become fill.

    Uses a sentinel interior to the data range (a bound sentinel such as 0 or
    65535 cannot be reached by a mean of valid values that all sit on one side
    of it): values [999, 1001, 999, 1001] average exactly to _FillValue=1000
    and must be nudged to the neighboring in-range value instead.
    """
    rows, cols = 2, 2
    rad = xr.DataArray(
        np.array([[999, 1001], [999, 1001]], dtype="uint16"),
        dims=("rows", "columns"),
        attrs={"_FillValue": 1000},
    )
    lat = xr.DataArray(np.zeros((rows, cols)), dims=("rows", "columns"))
    lon = xr.DataArray(np.zeros((rows, cols)), dims=("rows", "columns"))
    ds = xr.Dataset({"oa01_radiance": rad}, coords={"latitude": lat, "longitude": lon})

    out = reduce_swath(ds, factor=2)
    value = int(out["oa01_radiance"].values[0, 0])
    assert value != 1000, "valid block was recoded as fill"
    assert value == 999  # unrounded mean == sentinel → nudged one step down


def test_reduce_all_fill_block_stays_fill() -> None:
    """An all-fill block keeps the sentinel value in the overview."""
    rad = xr.DataArray(
        np.full((2, 2), 1000, dtype="uint16"),
        dims=("rows", "columns"),
        attrs={"_FillValue": 1000},
    )
    lat = xr.DataArray(np.zeros((2, 2)), dims=("rows", "columns"))
    lon = xr.DataArray(np.zeros((2, 2)), dims=("rows", "columns"))
    ds = xr.Dataset({"oa01_radiance": rad}, coords={"latitude": lat, "longitude": lon})

    out = reduce_swath(ds, factor=2)
    assert int(out["oa01_radiance"].values[0, 0]) == 1000


def test_reduce_partially_filled_block_averages_valid_pixels_only() -> None:
    """A block mixing fill and valid pixels averages only the valid pixels.

    Block [[100, 200], [300, fill]] with _FillValue=65535 must average the
    three valid pixels (200), not collapse to fill or include the sentinel.
    """
    rad = xr.DataArray(
        np.array([[100, 200], [300, 65535]], dtype="uint16"),
        dims=("rows", "columns"),
        attrs={"_FillValue": 65535},
    )
    lat = xr.DataArray(np.zeros((2, 2)), dims=("rows", "columns"))
    lon = xr.DataArray(np.zeros((2, 2)), dims=("rows", "columns"))
    ds = xr.Dataset({"oa01_radiance": rad}, coords={"latitude": lat, "longitude": lon})

    out = reduce_swath(ds, factor=2)
    assert int(out["oa01_radiance"].values[0, 0]) == 200


def test_reduce_transposed_band_still_block_averaged() -> None:
    """A radiance band stored as (columns, rows) is averaged, not decimated.

    Swath detection is order-insensitive: the transposed band is normalized to
    (rows, columns) and fill-aware block-averaged. Stride decimation would
    keep the block's origin value (10) instead of the block mean (25).
    """
    rad = xr.DataArray(
        np.array([[10, 20], [30, 40]], dtype="uint16"),
        dims=("columns", "rows"),
        attrs={"_FillValue": 65535},
    )
    lat = xr.DataArray(np.zeros((2, 2)), dims=("rows", "columns"))
    lon = xr.DataArray(np.zeros((2, 2)), dims=("rows", "columns"))
    ds = xr.Dataset({"oa01_radiance": rad}, coords={"latitude": lat, "longitude": lon})

    out = reduce_swath(ds, factor=2)
    assert tuple(out["oa01_radiance"].dims) == ("rows", "columns")
    assert int(out["oa01_radiance"].values[0, 0]) == 25


# ---------------------------------------------------------------------------
# restored geodesic-centroid geolocation tests (upstream d69bd45)
# ---------------------------------------------------------------------------


def test_reduce_swath_coordinates_are_block_geodesic_centroids() -> None:
    """Each output coordinate is the geodesic centroid of its pixel block."""
    ds = _swath(8, 6)
    out = reduce_swath(ds, factor=2)
    lat_block = ds["latitude"].values[0:2, 0:2]
    lon_block = ds["longitude"].values[0:2, 0:2]
    exp_lat, exp_lon = _geodesic_center(lat_block, lon_block)
    np.testing.assert_allclose(float(out["latitude"].values[0, 0]), exp_lat, rtol=1e-12)
    np.testing.assert_allclose(float(out["longitude"].values[0, 0]), exp_lon, rtol=1e-12)


def test_reduce_swath_whole_image_collapses_to_geodesic_center() -> None:
    """Reducing the full swath to one cell must yield its geodesic center."""
    rows = cols = 4
    ds = _swath(rows, cols)
    out = reduce_swath(ds, factor=rows)
    assert out["latitude"].shape == (1, 1)
    exp_lat, exp_lon = _geodesic_center(ds["latitude"].values, ds["longitude"].values)
    np.testing.assert_allclose(float(out["latitude"].values[0, 0]), exp_lat, rtol=1e-12)
    np.testing.assert_allclose(float(out["longitude"].values[0, 0]), exp_lon, rtol=1e-12)


def test_reduce_swath_longitude_stable_across_antimeridian() -> None:
    """Lon values straddling +/-180 must average to ~180, not ~0.

    A planar mean of [179.5, -179.5] is 0 (the wrong side of the planet);
    the unit-vector mean lands on the antimeridian.
    """
    lat = xr.DataArray(
        np.zeros((2, 2)), dims=("rows", "columns"), attrs={"standard_name": "latitude"}
    )
    lon = xr.DataArray(
        np.array([[179.5, -179.5], [179.5, -179.5]]),
        dims=("rows", "columns"),
        attrs={"standard_name": "longitude"},
    )
    rad = xr.DataArray(
        np.full((2, 2), 100, dtype="uint16"),
        dims=("rows", "columns"),
        attrs={"_FillValue": 65535},
    )
    ds = xr.Dataset({"oa01_radiance": rad}, coords={"latitude": lat, "longitude": lon})
    out = reduce_swath(ds, factor=2)
    assert abs(abs(float(out["longitude"].values[0, 0])) - 180.0) < 1e-9


def test_reduce_swath_geolocation_fill_excluded_from_centroid() -> None:
    """Fill pixels in lat OR lon are excluded from the block centroid; all-fill stays fill."""
    fill = -999.0
    lat_data = np.array([[10.0, 20.0], [30.0, fill]])
    lon_data = np.array([[5.0, fill], [6.0, 7.0]])
    lat = xr.DataArray(
        lat_data,
        dims=("rows", "columns"),
        attrs={"standard_name": "latitude", "_FillValue": fill},
    )
    lon = xr.DataArray(
        lon_data,
        dims=("rows", "columns"),
        attrs={"standard_name": "longitude", "_FillValue": fill},
    )
    ds = xr.Dataset(coords={"latitude": lat, "longitude": lon})
    out = reduce_swath(ds, factor=2)
    # Pixels (0,1) and (1,1) are fill in one of the pair -> only (0,0) and (1,0) count.
    exp_lat, exp_lon = _geodesic_center(np.array([10.0, 30.0]), np.array([5.0, 6.0]))
    np.testing.assert_allclose(float(out["latitude"].values[0, 0]), exp_lat, rtol=1e-12)
    np.testing.assert_allclose(float(out["longitude"].values[0, 0]), exp_lon, rtol=1e-12)

    all_fill = xr.Dataset(
        coords={
            "latitude": xr.full_like(lat, fill),
            "longitude": xr.full_like(lon, fill),
        }
    )
    out_fill = reduce_swath(all_fill, factor=2)
    assert float(out_fill["latitude"].values[0, 0]) == fill
    assert float(out_fill["longitude"].values[0, 0]) == fill


def test_reduce_swath_packed_integer_geolocation_unpacked_for_centroid() -> None:
    """CF-packed int32 microdegree lat/lon is decoded before the spherical mean.

    Real OLCI products store geolocation as int32 with scale_factor=1e-06
    (and the converter runs on un-decoded data).  Regression: feeding the raw
    packed integers (45_000_000 for 45 deg) into the trigonometry collapsed
    every overview coordinate to ~(0, 0).
    """
    scale = 1e-6
    fill = np.iinfo("int32").min
    lat_deg = np.array([[45.0, 45.001], [45.002, 45.003]])
    lon_deg = np.array([[10.0, 10.001], [10.002, 10.003]])
    lat = xr.DataArray(
        np.round(lat_deg / scale).astype("int32"),
        dims=("rows", "columns"),
        attrs={"standard_name": "latitude", "scale_factor": scale, "_FillValue": fill},
    )
    lon = xr.DataArray(
        np.round(lon_deg / scale).astype("int32"),
        dims=("rows", "columns"),
        attrs={"standard_name": "longitude", "scale_factor": scale, "_FillValue": fill},
    )
    ds = xr.Dataset(coords={"latitude": lat, "longitude": lon})
    out = reduce_swath(ds, factor=2)

    assert out["latitude"].dtype == np.dtype("int32")
    assert out["latitude"].attrs["scale_factor"] == scale
    exp_lat, exp_lon = _geodesic_center(lat_deg, lon_deg)
    # Output is re-packed, so compare decoded values to within one quantum.
    np.testing.assert_allclose(float(out["latitude"].values[0, 0]) * scale, exp_lat, atol=scale)
    np.testing.assert_allclose(float(out["longitude"].values[0, 0]) * scale, exp_lon, atol=scale)


def test_reduce_swath_odd_dims_coords_are_block_centroids() -> None:
    """Coordinates on odd-dim inputs use the same trimmed blocks as radiance."""
    ds = _swath_odd(rows=7, cols=5)
    out = reduce_swath(ds, factor=2)
    # Block [2:4, 2:4] feeds output cell (1, 1); the trailing odd row/column
    # is trimmed exactly as coarsen(boundary="trim") trims the radiance.
    exp_lat, exp_lon = _geodesic_center(
        ds["latitude"].values[2:4, 2:4], ds["longitude"].values[2:4, 2:4]
    )
    np.testing.assert_allclose(float(out["latitude"].values[1, 1]), exp_lat, rtol=1e-12)
    np.testing.assert_allclose(float(out["longitude"].values[1, 1]), exp_lon, rtol=1e-12)


def test_swath_spatial_attrs_has_no_transform() -> None:
    attrs = swath_spatial_attrs()
    assert attrs["spatial:dimensions"] == ["rows", "columns"]
    assert attrs.get("spatial:registration") == "pixel"
    assert "spatial:transform" not in attrs
    assert "spatial:bbox" not in attrs
