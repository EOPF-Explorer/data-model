"""
Unit tests for _write_geo_metadata method in S2MultiscalePyramid.

Tests the geographic metadata writing functionality added to level creation.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import zarr

from eopf_geozarr.data_api.geozarr.geoproj import ProjConventionMetadata
from eopf_geozarr.data_api.geozarr.spatial import SpatialConventionMetadata
from eopf_geozarr.s2_optimization.s2_multiscale import (
    create_measurements_encoding,
    stream_write_dataset,
    write_geo_metadata,
)


@pytest.fixture
def sample_dataset_with_crs() -> xr.Dataset:
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
def sample_dataset_with_epsg_attrs() -> xr.Dataset:
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
def sample_dataset_no_crs() -> xr.Dataset:
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

    def test_write_geo_metadata_with_rio_crs(self, sample_dataset_with_crs: xr.Dataset) -> None:
        """Test _write_geo_metadata with dataset that has rioxarray CRS."""

        # Call the method
        write_geo_metadata(sample_dataset_with_crs)

        # Verify CRS was written
        assert hasattr(sample_dataset_with_crs, "rio")
        assert sample_dataset_with_crs.rio.crs is not None
        assert sample_dataset_with_crs.rio.crs.to_epsg() == 32632

    def test_write_geo_metadata_with_epsg_attrs(
        self, sample_dataset_with_epsg_attrs: xr.Dataset
    ) -> None:
        """Test _write_geo_metadata with dataset that has EPSG in variable attributes."""

        # Verify initial state - no CRS
        assert (
            not hasattr(sample_dataset_with_epsg_attrs, "rio")
            or sample_dataset_with_epsg_attrs.rio.crs is None
        )

        # Call the method
        write_geo_metadata(sample_dataset_with_epsg_attrs)

        # Verify CRS was written from attributes
        assert hasattr(sample_dataset_with_epsg_attrs, "rio")
        assert sample_dataset_with_epsg_attrs.rio.crs is not None
        assert sample_dataset_with_epsg_attrs.rio.crs.to_epsg() == 32632

    def test_write_geo_metadata_no_crs(self, sample_dataset_no_crs: xr.Dataset) -> None:
        """Test _write_geo_metadata with dataset that has no CRS information."""

        # Verify initial state - no CRS
        assert not hasattr(sample_dataset_no_crs, "rio") or sample_dataset_no_crs.rio.crs is None

        # Call the method - should not fail but also not add CRS
        write_geo_metadata(sample_dataset_no_crs)

        # Verify no CRS was added (method handles gracefully)
        # The method should not fail even when no CRS is available
        # This tests the robustness of the method

    def test_write_geo_metadata_custom_grid_mapping_name(
        self, sample_dataset_with_crs: xr.Dataset
    ) -> None:
        """Test _write_geo_metadata with custom grid_mapping variable name."""

        # Call the method with custom grid mapping name
        custom_name = "custom_spatial_ref"
        write_geo_metadata(sample_dataset_with_crs, custom_name)

        # Verify CRS was written
        assert hasattr(sample_dataset_with_crs, "rio")
        assert sample_dataset_with_crs.rio.crs is not None

    def test_write_geo_metadata_preserves_existing_data(
        self, sample_dataset_with_crs: xr.Dataset
    ) -> None:
        """Test that _write_geo_metadata preserves existing data variables and coordinates."""

        # Store original data
        original_vars = list(sample_dataset_with_crs.data_vars.keys())
        original_coords = list(sample_dataset_with_crs.coords.keys())
        original_b02_data = sample_dataset_with_crs["b02"].values.copy()

        # Call the method
        write_geo_metadata(sample_dataset_with_crs)

        # Verify all original data is preserved
        assert list(sample_dataset_with_crs.data_vars.keys()) == original_vars
        assert all(coord in sample_dataset_with_crs.coords for coord in original_coords)
        assert np.array_equal(sample_dataset_with_crs["b02"].values, original_b02_data)

    def test_write_geo_metadata_empty_dataset(self) -> None:
        """Test _write_geo_metadata with empty dataset."""

        empty_ds = xr.Dataset({}, coords={})

        # Call the method - should handle gracefully
        write_geo_metadata(empty_ds)

        # Verify method doesn't fail with empty dataset
        # This tests robustness

    def test_write_geo_metadata_rio_write_crs_called(
        self, sample_dataset_with_crs: xr.Dataset
    ) -> None:
        """Test that rio.write_crs is called correctly."""

        # Mock the rio.write_crs method
        with patch.object(sample_dataset_with_crs.rio, "write_crs") as mock_write_crs:
            # Call the method
            write_geo_metadata(sample_dataset_with_crs)

            # Verify rio.write_crs was called with correct arguments
            mock_write_crs.assert_called_once()
            call_args = mock_write_crs.call_args
            assert call_args[1]["inplace"] is True  # inplace=True should be passed

    def test_write_geo_metadata_crs_from_multiple_sources(self) -> None:
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
        write_geo_metadata(ds)

        # Verify rio CRS was used (priority over attributes)
        assert ds.rio.crs.to_epsg() == 4326  # Should still be 4326, not 32632

    def test_write_geo_metadata_integration_with_stream_write(self, tmp_path: Path) -> None:
        """Test that _write_geo_metadata is properly integrated in _stream_write_dataset."""

        # Create a simple dataset with CRS
        coords = {
            "x": (["x"], np.linspace(0, 1000, 100)),
            "y": (["y"], np.linspace(0, 1000, 100)),
        }

        data_vars = {
            "b02": (["y", "x"], np.random.rand(100, 100)),
        }

        ds = xr.Dataset(data_vars, coords=coords)
        ds = ds.rio.write_crs("EPSG:32632")

        # Create encoding for the dataset
        encoding = create_measurements_encoding(ds, spatial_chunk=1024, enable_sharding=True)

        # Call _stream_write_dataset (which should call _write_geo_metadata internally)
        # Use a measurements path to trigger geo metadata writing
        dataset_path = "/measurements/reflectance/r10m"
        stream_write_dataset(
            ds,
            path=dataset_path,
            group=zarr.create_group(tmp_path),
            encoding=encoding,
            enable_sharding=True,
        )

        # Re-open the written dataset to verify CRS was persisted
        written_ds = xr.open_dataset(
            tmp_path, engine="zarr", chunks={}, decode_coords="all", group=dataset_path
        )

        # Verify CRS was written and persisted
        assert hasattr(written_ds, "rio")
        assert written_ds.rio.crs is not None
        assert written_ds.rio.crs.to_epsg() == 32632


class TestWriteGeoMetadataEdgeCases:
    """Test edge cases for _write_geo_metadata method."""

    def test_write_geo_metadata_invalid_crs(
        self,
    ) -> None:
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
            write_geo_metadata(ds)

    def test_write_geo_metadata_mixed_crs_variables(
        self,
    ) -> None:
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
        write_geo_metadata(ds)

        # Verify a CRS was applied (should be the first one found)
        assert hasattr(ds, "rio")

    def test_write_geo_metadata_maintains_dataset_attrs(
        self, sample_dataset_with_crs: xr.Dataset
    ) -> None:
        """Test that _write_geo_metadata maintains dataset-level attributes."""

        # Add some dataset attributes
        sample_dataset_with_crs.attrs["pyramid_level"] = 1
        sample_dataset_with_crs.attrs["resolution_meters"] = 20
        sample_dataset_with_crs.attrs["custom_attr"] = "test_value"

        original_attrs = sample_dataset_with_crs.attrs.copy()

        # Call the method
        write_geo_metadata(sample_dataset_with_crs)

        # Verify dataset attributes are preserved
        for key, value in original_attrs.items():
            assert sample_dataset_with_crs.attrs[key] == value

    def test_write_geo_metadata_adds_zarr_conventions(
        self, sample_dataset_with_crs: xr.Dataset
    ) -> None:
        """Test that zarr_conventions are properly added to dataset attributes."""

        # Verify zarr_conventions is not initially present
        assert "zarr_conventions" not in sample_dataset_with_crs.attrs

        # Call the method
        write_geo_metadata(sample_dataset_with_crs)

        # Verify zarr_conventions was added
        assert "zarr_conventions" in sample_dataset_with_crs.attrs
        zarr_conventions = sample_dataset_with_crs.attrs["zarr_conventions"]

        # Verify it's a list with 2 conventions
        assert isinstance(zarr_conventions, list)
        assert len(zarr_conventions) == 2

        # Verify the conventions contain the expected metadata
        spatial_convention = None
        proj_convention = None

        for convention in zarr_conventions:
            if convention.get("name") == "spatial:":
                spatial_convention = convention
            elif convention.get("name") == "proj:":
                proj_convention = convention

        # Verify spatial convention
        assert spatial_convention is not None
        expected_spatial = SpatialConventionMetadata().model_dump()
        assert spatial_convention == expected_spatial

        # Verify proj convention
        assert proj_convention is not None
        expected_proj = ProjConventionMetadata().model_dump()
        assert proj_convention == expected_proj

    def test_write_geo_metadata_adds_spatial_and_proj_attributes(
        self, sample_dataset_with_crs: xr.Dataset
    ) -> None:
        """Test that spatial and proj attributes are added along with conventions."""

        # Call the method
        write_geo_metadata(sample_dataset_with_crs)

        # Verify spatial attributes
        assert "spatial:dimensions" in sample_dataset_with_crs.attrs
        assert sample_dataset_with_crs.attrs["spatial:dimensions"] == ["y", "x"]
        assert "spatial:registration" in sample_dataset_with_crs.attrs
        assert sample_dataset_with_crs.attrs["spatial:registration"] == "pixel"
        assert "spatial:bbox" in sample_dataset_with_crs.attrs
        assert "spatial:shape" in sample_dataset_with_crs.attrs

        # Verify proj attributes (should have either proj:code or proj:wkt2)
        has_proj_code = "proj:code" in sample_dataset_with_crs.attrs
        has_proj_wkt2 = "proj:wkt2" in sample_dataset_with_crs.attrs
        assert has_proj_code or has_proj_wkt2, "Should have either proj:code or proj:wkt2"

        # Verify zarr_conventions includes both spatial and proj
        zarr_conventions = sample_dataset_with_crs.attrs["zarr_conventions"]
        convention_names = [conv.get("name") for conv in zarr_conventions]
        assert "spatial:" in convention_names
        assert "proj:" in convention_names

    def test_write_geo_metadata_no_crs_no_conventions(
        self, sample_dataset_no_crs: xr.Dataset
    ) -> None:
        """Test that zarr_conventions are not added when no CRS is available."""

        # Call the method with no CRS
        write_geo_metadata(sample_dataset_no_crs)

        # Verify zarr_conventions was not added since no CRS was available
        assert "zarr_conventions" not in sample_dataset_no_crs.attrs
