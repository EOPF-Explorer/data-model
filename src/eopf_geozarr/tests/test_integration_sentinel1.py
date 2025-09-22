"""Integration tests for Sentinel-1 GeoZarr conversion."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyproj
import pytest
import rioxarray  # noqa: F401  # Import to enable .rio accessor
import xarray as xr

from eopf_geozarr.conversion import create_geozarr_dataset
from eopf_geozarr.tests import _verify_basic_structure


class MockSentinel1L1GRDBuilder:
    """Builder class to generate a sample EOPF Sentinel-1 Level 1 GRD data product for testing purpose."""

    def __init__(self, product_id):
        self.product_title = "S01SIWGRD"
        self.product_id = product_id

        self.az_dim = "azimuth_time"
        self.gr_dim = "ground_range"
        self.data_dims = (self.az_dim, self.gr_dim)

        self.nlines = 552
        self.npixels = 1131

    def create_coordinates(self, az_dim_size, gr_dim_size) -> xr.Coordinates:
        coords = {
            self.az_dim: pd.date_range(
                start="2017-05-08T16:48:30",
                end="2017-05-08T16:48:55",
                periods=az_dim_size,
            ).values,
            self.gr_dim: np.floor(np.linspace(0.0, 262380.0, num=gr_dim_size)),
            "line": (
                self.az_dim,
                np.linspace(0, self.nlines, num=az_dim_size, dtype=np.int64),
            ),
            "pixel": (
                self.gr_dim,
                np.linspace(0, self.npixels, num=gr_dim_size, dtype=np.int64),
            ),
        }
        return xr.Coordinates(coords)

    def build_conditions_group(self) -> xr.DataTree:
        """Create a sample Sentinel-1 'conditions' group.

        Only create 'orbit' and 'gcp' subgroups.

        """
        dt = xr.DataTree()

        az_dim_size = 17
        dt["orbit"] = xr.Dataset(
            coords={
                self.az_dim: pd.date_range(
                    start="2017-05-08T16:47",
                    end="2017-05-08T16:50",
                    periods=az_dim_size,
                ).values,
            },
            data_vars={
                "position": (
                    (self.az_dim, "axis"),
                    np.random.uniform(size=(az_dim_size, 3)),
                ),
                "velocity": (
                    (self.az_dim, "axis"),
                    np.random.uniform(size=(az_dim_size, 3)),
                ),
            },
        )

        # gridded GCPs (no rotation here)
        data_shape = (10, 21)
        lat, lon = np.meshgrid(
            np.linspace(39.0, 41.0, num=data_shape[0]),
            np.linspace(15.0, 18.0, num=data_shape[1]),
            indexing="ij",
        )
        dt["gcp"] = xr.Dataset(
            coords=self.create_coordinates(*data_shape),
            data_vars={
                "height": (self.data_dims, np.zeros(data_shape)),
                "latitude": (self.data_dims, lat),
                "longitude": (self.data_dims, lon),
            },
        )

        return dt

    def build_quality_group(self) -> xr.DataTree:
        """Create a sample Sentinel-1 'quality' group.

        Only creates the 'calibration' subgroup.

        """
        dt = xr.DataTree()

        data_shape = (27, 657)
        dt["calibration"] = xr.Dataset(
            coords=self.create_coordinates(*data_shape),
            data_vars={
                "beta_nought": (self.data_dims, np.full(data_shape, 474.0)),
                "dn": (self.data_dims, np.full(data_shape, 474.0)),
                "gamma": (
                    self.data_dims,
                    np.random.uniform(615.0, 462.0, size=data_shape),
                ),
                "sigma_nought": (
                    self.data_dims,
                    np.random.uniform(615.0, 462.0, size=data_shape),
                ),
            },
        )

        return dt

    def build_measurements_group(self) -> xr.Dataset:
        """Create a sample Sentinel-1 'measurements' group."""

        data_shape = (self.nlines, self.npixels)
        return xr.Dataset(
            coords=self.create_coordinates(*data_shape),
            data_vars={
                "grd": (
                    self.data_dims,
                    np.random.randint(0, 200, size=data_shape, dtype=np.uint16),
                )
            },
        )

    def build(self) -> xr.DataTree:
        dt = xr.DataTree()

        common_groups = {
            "conditions": self.build_conditions_group(),
            "quality": self.build_quality_group(),
        }

        dt_vh = xr.DataTree.from_dict(common_groups)
        dt_vh["measurements"] = self.build_measurements_group()
        dt_vv = xr.DataTree.from_dict(common_groups)
        dt_vv["measurements"] = self.build_measurements_group()

        dt[f"{self.product_title}_{self.product_id}_VH"] = dt_vh
        dt[f"{self.product_title}_{self.product_id}_VV"] = dt_vv

        dt.attrs["other_metadata"] = {"title": "S01SIWGRH"}
        dt.attrs["stac_discovery"] = {
            "properties": {
                "product:type": "S01SIWGRH",
                "platform": "sentinel-1a",
            },
        }

        return dt


@pytest.fixture
def sample_sentinel1_datatree():
    """Create a sample Sentinel-1 datatree with GCPs."""

    builder = MockSentinel1L1GRDBuilder("20170508T164830_0025_A094_8604_01B54C")
    return builder.build()


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_no_gcp_group(temp_output_dir, sample_sentinel1_datatree) -> None:
    output_path = Path(temp_output_dir) / "temp.zarr"

    with pytest.raises(ValueError, match="Detected Sentinel-1.*GCP group not provided"):
        create_geozarr_dataset(
            sample_sentinel1_datatree,
            groups=["measurements"],
            output_path=str(output_path),
        )


def test_invalid_gcp_group_raises_error(
    temp_output_dir, sample_sentinel1_datatree
) -> None:
    """Test that specifying a non-existent GCP group raises an error."""
    output_path = Path(temp_output_dir) / "test_s1_invalid_gcp.zarr"
    groups = ["measurements"]

    # Try with invalid GCP group
    with pytest.raises(ValueError, match="GCP group.*not found"):
        create_geozarr_dataset(
            sample_sentinel1_datatree,
            groups=groups,
            output_path=str(output_path),
            gcp_group="invalid/gcp",
        )


@pytest.mark.parametrize(
    "polarization_group",
    [
        "S01SIWGRD_20170508T164830_0025_A094_8604_01B54C_VH",
        "S01SIWGRD_20170508T164830_0025_A094_8604_01B54C_VV",
    ],
)
def test_sentinel1_gcp_conversion(
    temp_output_dir, sample_sentinel1_datatree, polarization_group
) -> None:
    """Test conversion of Sentinel-1 data with GCPs."""
    # Prepare test
    output_path = Path(temp_output_dir) / "test_s1_gcp.zarr"
    groups = ["measurements"]
    gcp_group = "conditions/gcp"

    # Execute conversion
    with patch("eopf_geozarr.conversion.geozarr.print"):
        dt_geozarr = create_geozarr_dataset(
            sample_sentinel1_datatree,
            groups=groups,
            output_path=str(output_path),
            spatial_chunk=1024,
            min_dimension=256,
            tile_width=256,
            max_retries=3,
            gcp_group=gcp_group,
        )

    # Verify the conversion was successful
    assert dt_geozarr is not None
    assert output_path.exists()

    _verify_basic_structure(output_path / polarization_group, groups)

    # Load the result for validation
    dt = xr.open_datatree(output_path, group=polarization_group)

    # Check basic structure (multiscale)
    assert "measurements" in dt
    assert "0" in dt["measurements"]
    assert "grd" not in dt["measurements"].to_dataset().data_vars
    assert "spatial_ref" in dt["measurements/0"]

    # Verify Sentinel-1 GRD specific metadata - now reprojected to x/y coordinates
    grd = dt["measurements/0"].grd
    assert grd.dims == ("y", "x")  # Now reprojected to geographic coordinates
    assert (
        grd.attrs["standard_name"]
        == "surface_backwards_scattering_coefficient_of_radar_wave"
    )
    assert grd.attrs["units"] == "1"
    assert grd.attrs["grid_mapping"] == "spatial_ref"

    # Verify reprojected data has proper CRS
    ds_measurements = dt["measurements/0"].to_dataset()
    spatial_ref = ds_measurements.spatial_ref
    assert "crs_wkt" in spatial_ref.attrs
    assert "spatial_ref" in spatial_ref.attrs
    actual_crs = pyproj.CRS.from_wkt(spatial_ref.attrs["crs_wkt"])
    assert actual_crs == pyproj.CRS.from_epsg(4326)  # Data is now in lat/lon WGS84
    assert actual_crs == pyproj.CRS.from_wkt(spatial_ref.attrs["spatial_ref"])

    # Verify coordinate attributes for reprojected data
    assert "x" in ds_measurements.coords
    assert "y" in ds_measurements.coords
    assert ds_measurements.x.attrs["standard_name"] == "longitude"
    assert ds_measurements.x.attrs["units"] == "degrees_east"
    assert ds_measurements.y.attrs["standard_name"] == "latitude"
    assert ds_measurements.y.attrs["units"] == "degrees_north"

    # Verify data bounds are reasonable (should be within the GCP bounds)
    x_bounds = (ds_measurements.x.min().values, ds_measurements.x.max().values)
    y_bounds = (ds_measurements.y.min().values, ds_measurements.y.max().values)

    # Should be within the original GCP bounds (15-18 lon, 39-41 lat)
    assert (
        14.5 <= x_bounds[0] <= 15.5
    ), f"X min bound {x_bounds[0]} outside expected range"
    assert (
        17.5 <= x_bounds[1] <= 18.5
    ), f"X max bound {x_bounds[1]} outside expected range"
    assert (
        38.5 <= y_bounds[0] <= 39.5
    ), f"Y min bound {y_bounds[0]} outside expected range"
    assert (
        40.5 <= y_bounds[1] <= 41.5
    ), f"Y max bound {y_bounds[1]} outside expected range"

    # Check multiscales 2 levels created: 0 (native, checked above) and 1
    assert "1" in dt["measurements"]
    ds_measurements1 = dt["measurements/1"].to_dataset()
    grd1 = ds_measurements1.grd
    assert grd1.dims == ("y", "x")  # Overview also has reprojected coordinates
    spatial_ref1 = ds_measurements1.spatial_ref
    assert "crs_wkt" in spatial_ref1.attrs
    assert "spatial_ref" in spatial_ref1.attrs
    actual_crs = pyproj.CRS.from_wkt(spatial_ref1.attrs["crs_wkt"])
    assert actual_crs == pyproj.CRS.from_epsg(4326)  # Overview also in lat/lon WGS84
    assert actual_crs == pyproj.CRS.from_wkt(spatial_ref1.attrs["spatial_ref"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
