"""Integration tests for Sentinel-1 GeoZarr conversion."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401  # Import to enable .rio accessor
import xarray as xr

from eopf_geozarr.conversion import create_geozarr_dataset


class MockSentinel1L1GRDBuilder:
    """Builder class to generate a sample EOPF Sentinel-1 Level 1 GRD data product for testing purpose."""

    def __init__(self, product_id):
        self.product_title = "S01SIWGRD"
        self.product_id = product_id

        self.az_dim = "azimuth_time"
        self.gr_dim = "ground_range"
        self.data_dims = (self.az_dim, self.gr_dim)

        self.nlines = 160
        self.npixels = 260

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


def test_invalid_gcp_group_raises_error(temp_output_dir, sample_sentinel1_datatree):
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
):
    """Test conversion of Sentinel-1 data with GCPs."""
    # Prepare test
    output_path = Path(temp_output_dir) / "test_s1_gcp.zarr"
    groups = ["measurements"]

    # Execute conversion
    create_geozarr_dataset(
        sample_sentinel1_datatree,
        groups=groups,
        output_path=str(output_path),
        gcp_group="conditions/gcp",
    )

    # Load the result for validation
    dt = xr.open_datatree(output_path, group=polarization_group)

    # Check basic structure
    print(dt.measurements)
    print(dt.measurements.grd)
    assert "measurements" in dt

    # assert "spatial_ref" in dt.measurements

    # Verify Sentinel-1 specific metadata
    grd = dt.measurements.grd
    assert (
        grd.attrs["standard_name"]
        == "surface_backwards_scattering_coefficient_of_radar_wave"
    )
    assert grd.attrs["units"] == "1"
    # assert meas.attrs["grid_mapping"] == "spatial_ref"

    # # Verify GCP handling
    # spatial_ref = dt.measurements.spatial_ref
    # assert "gcps" in spatial_ref.attrs  # rioxarray adds GCP info here
    # assert spatial_ref.attrs["crs"] == "EPSG:4326"  # GCPs are in lat/lon

    # # Verify GCP storage format and coordinates
    # gcps = spatial_ref.attrs["gcps"]
    # assert len(gcps) == 4  # We created 4 GCPs
    # assert all(hasattr(gcp, "row") for gcp in gcps)  # Check GCP attributes
    # assert all(hasattr(gcp, "col") for gcp in gcps)
    # assert all(hasattr(gcp, "x") for gcp in gcps)
    # assert all(hasattr(gcp, "y") for gcp in gcps)
    # assert all(hasattr(gcp, "z") for gcp in gcps)
    # assert all(isinstance(gcp.id, str) for gcp in gcps)

    # # Check first GCP coordinates
    # gcp0 = gcps[0]
    # np.testing.assert_allclose(gcp0.x, 10.0)  # longitude
    # np.testing.assert_allclose(gcp0.y, 45.0)  # latitude
    # np.testing.assert_allclose(gcp0.z, 0.0)  # height

    # # Check that row/col values map to correct grid positions
    # assert gcp0.row == 0
    # assert gcp0.col == 0

    # # Check last GCP to verify grid mapping
    # gcp3 = gcps[3]
    # np.testing.assert_allclose(gcp3.x, 10.1)  # longitude
    # np.testing.assert_allclose(gcp3.y, 45.1)  # latitude
    # np.testing.assert_allclose(
    #     gcp3.col, sample_sentinel1_datatree["measurements"].dims["x"] - 1
    # )
    # np.testing.assert_allclose(
    #     gcp3.row, sample_sentinel1_datatree["measurements"].dims["y"] - 1
    # )

    # # Verify no multiscales were created (as per ADR-102)
    # measurements_group = xr.open_zarr(output_path).measurements
    # assert not any(k.startswith("1") for k in measurements_group.data_vars.keys())
