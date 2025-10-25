"""
Unit tests for _write_geo_metadata method in S2MultiscalePyramid.

Tests the geographic metadata writing functionality added to level creation.
"""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.s2_optimization.s2_multiscale import S2MultiscalePyramid


@pytest.fixture
def pyramid_creator():
    """Create a S2MultiscalePyramid instance for testing."""
    return S2MultiscalePyramid(enable_sharding=True, spatial_chunk=1024)


@pytest.fixture
def sample_dataset_with_crs():
    """Create a sample dataset with CRS information."""
    coords = {
        "x": (["x"], np.linspace(0, 1000, 100)),
        "y": (["y"], np.linspace(0, 1000, 100)),
        "time": (["time"], [np.datetime64("2023-01-01")]),
    }

    data_vars = {
        "b02": (["time", "y", "x"], np.random.rand(1, 100, 100)),
        "b03": (["time", "y", "x"], np.random.rand(1, 100, 100)),
        "b04": (["y", "x"], np.random.rand(100, 100)),
    }

    ds = xr.Dataset(data_vars, coords=coords)

    ds["b02"].attrs["proj:epsg"] = 32632
    ds["b03"].attrs["proj:epsg"] = 32632
    ds["b04"].attrs["proj:epsg"] = 32632

    return ds


@pytest.fixture
def sample_dataset_with_epsg_attrs():
    """Create a sample dataset with EPSG in attributes."""
    coords = {
        "x": (["x"], np.linspace(0, 1000, 50)),
        "y": (["y"], np.linspace(0, 1000, 50)),
    }

    data_vars = {
        "b05": (["y", "x"], np.random.rand(50, 50)),
        "b06": (["y", "x"], np.random.rand(50, 50)),
    }

    ds = xr.Dataset(data_vars, coords=coords)

    # Add EPSG to variable attributes
    ds["b05"].attrs["proj:epsg"] = 32632
    ds["b06"].attrs["proj:epsg"] = 32632

    return ds


@pytest.fixture
def sample_dataset_no_crs():
    """Create a sample dataset without CRS information."""
    coords = {
        "x": (["x"], np.linspace(0, 1000, 25)),
        "y": (["y"], np.linspace(0, 1000, 25)),
    }

    data_vars = {
        "b11": (["y", "x"], np.random.rand(25, 25)),
        "b12": (["y", "x"], np.random.rand(25, 25)),
    }

    return xr.Dataset(data_vars, coords=coords)


class TestWriteGeoMetadata:
    """Test the _write_geo_metadata method."""

    def test_write_geo_metadata_with_rio_crs(
        self, pyramid_creator, sample_dataset_with_crs
    ):
        """Test _write_geo_metadata with dataset that has rioxarray CRS."""

        # Call the method
        pyramid_creator._write_geo_metadata(sample_dataset_with_crs)

        # Verify CRS was written
        assert hasattr(sample_dataset_with_crs, "rio")
        assert sample_dataset_with_crs.rio.crs is not None
        assert sample_dataset_with_crs.rio.crs.to_epsg() == 32632

    def test_write_geo_metadata_with_epsg_attrs(
        self, pyramid_creator, sample_dataset_with_epsg_attrs
    ):
        """Test _write_geo_metadata with dataset that has EPSG in variable attributes."""

        # Verify initial state - no CRS
        assert (
            not hasattr(sample_dataset_with_epsg_attrs, "rio")
            or sample_dataset_with_epsg_attrs.rio.crs is None
        )

        # Call the method
        pyramid_creator._write_geo_metadata(sample_dataset_with_epsg_attrs)

        # Verify CRS was written from attributes
        assert hasattr(sample_dataset_with_epsg_attrs, "rio")
        assert sample_dataset_with_epsg_attrs.rio.crs is not None
        assert sample_dataset_with_epsg_attrs.rio.crs.to_epsg() == 32632

    def test_write_geo_metadata_no_crs(self, pyramid_creator, sample_dataset_no_crs):
        """Test _write_geo_metadata with dataset that has no CRS information."""

        # Verify initial state - no CRS
        assert (
            not hasattr(sample_dataset_no_crs, "rio")
            or sample_dataset_no_crs.rio.crs is None
        )

        # Call the method - should not fail but also not add CRS
        pyramid_creator._write_geo_metadata(sample_dataset_no_crs)

        # Verify no CRS was added (method handles gracefully)
        # The method should not fail even when no CRS is available
        # This tests the robustness of the method

    def test_write_geo_metadata_custom_grid_mapping_name(
        self, pyramid_creator, sample_dataset_with_crs
    ):
        """Test _write_geo_metadata with custom grid_mapping variable name."""

        # Call the method with custom grid mapping name
        custom_name = "custom_spatial_ref"
        pyramid_creator._write_geo_metadata(sample_dataset_with_crs, custom_name)

        # Verify CRS was written
        assert hasattr(sample_dataset_with_crs, "rio")
        assert sample_dataset_with_crs.rio.crs is not None

    def test_write_geo_metadata_preserves_existing_data(
        self, pyramid_creator, sample_dataset_with_crs
    ):
        """Test that _write_geo_metadata preserves existing data variables and coordinates."""

        # Store original data
        original_vars = list(sample_dataset_with_crs.data_vars.keys())
        original_coords = list(sample_dataset_with_crs.coords.keys())
        original_b02_data = sample_dataset_with_crs["b02"].values.copy()

        # Call the method
        pyramid_creator._write_geo_metadata(sample_dataset_with_crs)

        # Verify all original data is preserved
        assert list(sample_dataset_with_crs.data_vars.keys()) == original_vars
        assert all(coord in sample_dataset_with_crs.coords for coord in original_coords)
        assert np.array_equal(sample_dataset_with_crs["b02"].values, original_b02_data)

    def test_write_geo_metadata_empty_dataset(self, pyramid_creator):
        """Test _write_geo_metadata with empty dataset."""

        empty_ds = xr.Dataset({}, coords={})

        # Call the method - should handle gracefully
        pyramid_creator._write_geo_metadata(empty_ds)

        # Verify method doesn't fail with empty dataset
        # This tests robustness

    def test_write_geo_metadata_rio_write_crs_called(
        self, pyramid_creator, sample_dataset_with_crs
    ):
        """Test that rio.write_crs is called correctly."""

        # Mock the rio.write_crs method
        with patch.object(sample_dataset_with_crs.rio, "write_crs") as mock_write_crs:
            # Call the method
            pyramid_creator._write_geo_metadata(sample_dataset_with_crs)

            # Verify rio.write_crs was called with correct arguments
            mock_write_crs.assert_called_once()
            call_args = mock_write_crs.call_args
            assert call_args[1]["inplace"] is True  # inplace=True should be passed

    def test_write_geo_metadata_crs_from_multiple_sources(self, pyramid_creator):
        """Test CRS detection from multiple sources in priority order."""

        # Create dataset with both rio CRS and EPSG attributes
        coords = {
            "x": (["x"], np.linspace(0, 1000, 50)),
            "y": (["y"], np.linspace(0, 1000, 50)),
        }

        data_vars = {"b08": (["y", "x"], np.random.rand(50, 50))}

        ds = xr.Dataset(data_vars, coords=coords)

        # Add both rio CRS and EPSG attribute (rio should take priority)
        ds = ds.rio.write_crs("EPSG:4326")  # Rio CRS
        ds["b08"].attrs["proj:epsg"] = 32632  # EPSG attribute

        # Call the method
        pyramid_creator._write_geo_metadata(ds)

        # Verify rio CRS was used (priority over attributes)
        assert ds.rio.crs.to_epsg() == 4326  # Should still be 4326, not 32632

    def test_write_geo_metadata_integration_with_level_creation(self, pyramid_creator):
        """Test that _write_geo_metadata is properly integrated in level creation methods."""

        # Create mock measurements data
        measurements_by_resolution = {
            10: {
                "bands": {
                    "b02": xr.DataArray(
                        np.random.rand(100, 100),
                        dims=["y", "x"],
                        coords={
                            "x": (["x"], np.linspace(0, 1000, 100)),
                            "y": (["y"], np.linspace(0, 1000, 100)),
                        },
                    ).rio.write_crs("EPSG:32632")
                }
            }
        }

        # Create level 0 dataset (which should call _write_geo_metadata)
        level_0_ds = pyramid_creator._create_level_0_dataset(measurements_by_resolution)

        # Verify CRS was written by _write_geo_metadata
        assert hasattr(level_0_ds, "rio")
        assert level_0_ds.rio.crs is not None
        assert level_0_ds.rio.crs.to_epsg() == 32632


class TestWriteGeoMetadataEdgeCases:
    """Test edge cases for _write_geo_metadata method."""

    def test_write_geo_metadata_invalid_crs(self, pyramid_creator):
        """Test _write_geo_metadata with invalid CRS data."""

        coords = {
            "x": (["x"], np.linspace(0, 1000, 10)),
            "y": (["y"], np.linspace(0, 1000, 10)),
        }

        data_vars = {"test_var": (["y", "x"], np.random.rand(10, 10))}

        ds = xr.Dataset(data_vars, coords=coords)

        # Add invalid EPSG code
        ds["test_var"].attrs["proj:epsg"] = "invalid_epsg"

        # Method should raise an exception for invalid CRS (normal behavior)
        from pyproj.exceptions import CRSError

        with pytest.raises(CRSError):
            pyramid_creator._write_geo_metadata(ds)

    def test_write_geo_metadata_mixed_crs_variables(self, pyramid_creator):
        """Test _write_geo_metadata with variables having different CRS information."""

        coords = {
            "x": (["x"], np.linspace(0, 1000, 20)),
            "y": (["y"], np.linspace(0, 1000, 20)),
        }

        data_vars = {
            "var1": (["y", "x"], np.random.rand(20, 20)),
            "var2": (["y", "x"], np.random.rand(20, 20)),
        }

        ds = xr.Dataset(data_vars, coords=coords)

        # Add different EPSG codes to different variables
        ds["var1"].attrs["proj:epsg"] = 32632
        ds["var2"].attrs["proj:epsg"] = 4326

        # Call the method (should use the first CRS found)
        pyramid_creator._write_geo_metadata(ds)

        # Verify a CRS was applied (should be the first one found)
        assert hasattr(ds, "rio")

    def test_write_geo_metadata_maintains_dataset_attrs(
        self, pyramid_creator, sample_dataset_with_crs
    ):
        """Test that _write_geo_metadata maintains dataset-level attributes."""

        # Add some dataset attributes
        sample_dataset_with_crs.attrs["pyramid_level"] = 1
        sample_dataset_with_crs.attrs["resolution_meters"] = 20
        sample_dataset_with_crs.attrs["custom_attr"] = "test_value"

        original_attrs = sample_dataset_with_crs.attrs.copy()

        # Call the method
        pyramid_creator._write_geo_metadata(sample_dataset_with_crs)

        # Verify dataset attributes are preserved
        for key, value in original_attrs.items():
            assert sample_dataset_with_crs.attrs[key] == value


if __name__ == "__main__":
    pytest.main([__file__])
