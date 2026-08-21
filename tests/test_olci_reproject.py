"""Tests for OLCI swath -> regular grid reprojection."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.s3_olci_optimization.olci_reproject import reproject_olci

FILL = 65535


def build_rotated_swath(rows: int = 64, cols: int = 60, angle_deg: float = 30.0) -> xr.Dataset:
    """Synthetic OLCI-like swath: regular grid rotated by *angle_deg* in lon/lat.

    Rotation matters: it makes the dst bounding-box corners fall outside the
    swath, exercising real geolocation warping and off-swath fill.
    """
    rr, cc = np.meshgrid(
        np.arange(rows, dtype="float64"), np.arange(cols, dtype="float64"), indexing="ij"
    )
    theta = np.deg2rad(angle_deg)
    lon = 10.0 + 0.01 * (cc * np.cos(theta) - rr * np.sin(theta))
    lat = 45.0 + 0.01 * (cc * np.sin(theta) + rr * np.cos(theta))

    band = (100.0 * rr + cc).astype("uint16")
    band[:4, :4] = FILL  # a filled block inside the swath
    da = xr.DataArray(
        band,
        dims=("rows", "columns"),
        attrs={"scale_factor": 0.0139, "add_offset": 0.0, "_FillValue": FILL},
    )
    # float32 band with no pre-existing _FillValue: nodata defaults to NaN.
    solar_flux = xr.DataArray(
        (rr + cc).astype("float32"),
        dims=("rows", "columns"),
    )
    time_stamp = xr.DataArray(np.arange(rows).astype("datetime64[ns]"), dims=("rows",))
    return xr.Dataset(
        {"oa01_radiance": da, "solar_flux_proxy": solar_flux},
        coords={
            "latitude": (("rows", "columns"), lat),
            "longitude": (("rows", "columns"), lon),
            "altitude": (("rows", "columns"), np.full((rows, cols), 7, dtype="int16")),
            "time_stamp": time_stamp,
        },
    )


def test_reproject_olci_produces_regular_grid_with_crs() -> None:
    """One behavior test: grid shape, CRS, dtype, attrs, fill, and dropped vars."""
    ds = build_rotated_swath()
    out = reproject_olci(ds)

    # Regular 1-D coordinate grid over (y, x)
    assert set(out.sizes) == {"y", "x"}
    for dim in ("y", "x"):
        coord = out[dim]
        assert coord.ndim == 1
        steps = np.diff(coord.values)
        assert np.allclose(steps, steps[0])
    # y descends (north-up grid)
    assert out["y"].values[0] > out["y"].values[-1]

    # CRS declared in both idioms
    assert out.rio.crs is not None
    assert out.rio.crs.to_epsg() == 4326
    assert "spatial_ref" in out.coords or "spatial_ref" in out.variables
    assert out["oa01_radiance"].attrs["grid_mapping"] == "spatial_ref"

    # dtype and CF scaling preserved; fill recorded
    band = out["oa01_radiance"]
    assert band.dtype == np.dtype("uint16")
    assert band.attrs["scale_factor"] == 0.0139
    assert int(band.attrs["_FillValue"]) == FILL

    # Off-swath cells (bbox corners of a rotated swath) are fill
    vals = band.values
    assert vals[0, 0] == FILL
    assert vals[-1, -1] == FILL
    # …and real data survived the warp
    valid = vals[vals != FILL]
    assert valid.size > 0
    assert valid.max() <= (100.0 * 64 + 60)

    # altitude warped onto the grid; per-scan-line time_stamp dropped
    assert "altitude" in out.data_vars
    assert out["altitude"].dims == ("y", "x")
    assert "time_stamp" not in out.variables

    # float variable with no pre-existing _FillValue: nodata is NaN, but it
    # is still recorded, and off-swath corners are NaN.
    flux = out["solar_flux_proxy"]
    assert np.isnan(flux.attrs["_FillValue"])
    flux_vals = flux.values
    assert np.isnan(flux_vals[0, 0])
    assert np.isnan(flux_vals[-1, -1])


def test_reproject_olci_missing_geolocation_raises() -> None:
    ds = build_rotated_swath().drop_vars("latitude")
    with pytest.raises(ValueError, match="latitude"):
        reproject_olci(ds)


def test_reproject_olci_wrong_geolocation_dims_raises() -> None:
    ds = build_rotated_swath()
    ds = ds.assign_coords(latitude=("rows", np.linspace(45, 46, 64)))
    with pytest.raises(ValueError, match="2-D"):
        reproject_olci(ds)


def test_reproject_olci_degenerate_extent_raises() -> None:
    ds = build_rotated_swath()
    ds = ds.assign_coords(
        latitude=(("rows", "columns"), np.full((64, 60), 45.0)),
        longitude=(("rows", "columns"), np.full((64, 60), 10.0)),
    )
    with pytest.raises(ValueError, match="degenerate"):
        reproject_olci(ds)


def test_reproject_olci_projected_target_crs() -> None:
    """Warping to a projected target CRS yields a metric grid with matching coord attrs.

    Exercises the non-geographic branch end-to-end: the CLI-advertised
    --target-crs flag must produce a regular grid in the requested CRS, with
    projection_x/y_coordinate standard names and metre units on the 1-D coords.
    """
    ds = build_rotated_swath()
    out = reproject_olci(ds, target_crs="EPSG:3857")

    assert out.rio.crs is not None
    assert out.rio.crs.to_epsg() == 3857
    for dim, std_name in (
        ("y", "projection_y_coordinate"),
        ("x", "projection_x_coordinate"),
    ):
        coord = out[dim]
        assert coord.ndim == 1
        steps = np.diff(coord.values)
        assert np.allclose(steps, steps[0])
        assert coord.attrs["standard_name"] == std_name
        assert coord.attrs["units"] == "m"
    band = out["oa01_radiance"]
    assert band.dtype == np.dtype("uint16")
    assert band.attrs["grid_mapping"] == "spatial_ref"


def test_reproject_olci_unpacks_cf_packed_geolocation() -> None:
    """CF-packed geolocation (raw int32 microdegrees) must be unpacked before warping.

    The pipeline contract is mask_and_scale=False: real OLCI stores lat/lon
    as int32 with scale_factor=1e-6. Feeding raw values to the warp as if
    they were degrees georeferences the output a factor of 1e6 off.
    The packed dataset must produce the same grid as its unpacked twin.
    """
    ds_float = build_rotated_swath()
    lat_deg = np.asarray(ds_float["latitude"].values)
    lon_deg = np.asarray(ds_float["longitude"].values)
    ds_packed = ds_float.assign_coords(
        latitude=(
            ("rows", "columns"),
            np.round(lat_deg / 1e-6).astype("int32"),
            {"standard_name": "latitude", "scale_factor": 1e-6, "add_offset": 0.0},
        ),
        longitude=(
            ("rows", "columns"),
            np.round(lon_deg / 1e-6).astype("int32"),
            {"standard_name": "longitude", "scale_factor": 1e-6, "add_offset": 0.0},
        ),
    )

    out_packed = reproject_olci(ds_packed)
    out_float = reproject_olci(ds_float)

    # Coordinates land in real degree space, not microdegree-as-degree space.
    assert 9.0 < float(out_packed["x"].values.min()) < 12.0
    assert 44.0 < float(out_packed["y"].values.min()) < 47.0
    # Same grid as the unpacked twin (int32 microdegree rounding is ~1e-6 deg).
    np.testing.assert_allclose(out_packed["x"].values, out_float["x"].values, atol=1e-5)
    np.testing.assert_allclose(out_packed["y"].values, out_float["y"].values, atol=1e-5)
    assert out_packed.sizes == out_float.sizes


def test_reproject_olci_transposes_swath_bands() -> None:
    """A band stored as (columns, rows) must be warped, not silently dropped.

    The native pipeline normalizes transposed bands before block averaging;
    the warp path must do the same rather than discarding them via the
    swath-dim fallthrough.
    """
    ds = build_rotated_swath()
    transposed = ds["oa01_radiance"].transpose("columns", "rows")
    ds = ds.assign(oa02_radiance=transposed)

    out = reproject_olci(ds)

    assert "oa02_radiance" in out.data_vars
    assert out["oa02_radiance"].dims == ("y", "x")
    # Same underlying data as oa01 -> identical warped values.
    np.testing.assert_array_equal(out["oa02_radiance"].values, out["oa01_radiance"].values)
