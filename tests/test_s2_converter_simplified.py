"""
Unit tests for simplified S2 converter implementation.

Tests the new simplified approach that uses only xarray and proper metadata consolidation.
"""

import os
import shutil
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr
import zarr

from eopf_geozarr.s2_optimization.s2_converter import S2OptimizedConverter


@pytest.fixture
def mock_s2_dataset():
    """Create a mock S2 dataset for testing."""
    # Create test data arrays
    coords = {
        "x": (["x"], np.linspace(0, 1000, 100)),
        "y": (["y"], np.linspace(0, 1000, 100)),
        "time": (["time"], [np.datetime64("2023-01-01")]),
    }

    # Create test variables
    data_vars = {
        "b02": (["time", "y", "x"], np.random.rand(1, 100, 100)),
        "b03": (["time", "y", "x"], np.random.rand(1, 100, 100)),
        "b04": (["time", "y", "x"], np.random.rand(1, 100, 100)),
    }

    ds = xr.Dataset(data_vars, coords=coords)

    # Add rioxarray CRS
    ds = ds.rio.write_crs("EPSG:32632")

    # Create datatree
    dt = xr.DataTree(ds)
    dt.attrs = {"stac_discovery": {"properties": {"mission": "sentinel-2"}}}

    return dt


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestS2OptimizedConverter:
    """Test the S2OptimizedConverter class."""

    def test_init(self):
        """Test converter initialization."""
        converter = S2OptimizedConverter(
            enable_sharding=True, spatial_chunk=512, compression_level=5, max_retries=2
        )

        assert converter.enable_sharding is True
        assert converter.spatial_chunk == 512
        assert converter.compression_level == 5
        assert converter.max_retries == 2
        assert converter.pyramid_creator is not None
        assert converter.validator is not None

    def test_is_sentinel2_dataset_with_mission(self):
        """Test S2 detection via mission attribute."""
        converter = S2OptimizedConverter()

        # Test with S2 mission
        dt = xr.DataTree()
        dt.attrs = {"stac_discovery": {"properties": {"mission": "sentinel-2a"}}}

        assert converter._is_sentinel2_dataset(dt) is True

        # Test with non-S2 mission
        dt.attrs["stac_discovery"]["properties"]["mission"] = "sentinel-1"
        assert converter._is_sentinel2_dataset(dt) is False

    def test_is_sentinel2_dataset_with_groups(self):
        """Test S2 detection via characteristic groups."""
        converter = S2OptimizedConverter()

        dt = xr.DataTree()
        dt.attrs = {}

        # Mock groups property using patch
        with patch.object(
            type(dt),
            "groups",
            new_callable=lambda: property(
                lambda self: [
                    "/measurements/reflectance",
                    "/conditions/geometry",
                    "/quality/atmosphere",
                ]
            ),
        ):
            assert converter._is_sentinel2_dataset(dt) is True

        # Test with insufficient indicators
        with patch.object(
            type(dt),
            "groups",
            new_callable=lambda: property(lambda self: ["/measurements/reflectance"]),
        ):
            assert converter._is_sentinel2_dataset(dt) is False


class TestMetadataConsolidation:
    """Test metadata consolidation functionality."""

    @patch("xarray.DataTree.to_zarr")
    def test_simple_root_consolidation_success(self, mock_to_zarr, temp_output_dir):
        """Test successful root consolidation with DataTree."""
        converter = S2OptimizedConverter()

        # Call the method with empty datasets
        converter._simple_root_consolidation(temp_output_dir, {})

        # Verify to_zarr was called multiple times to create root group
        assert mock_to_zarr.call_count >= 2

        # Verify the calls included consolidated=True
        calls = mock_to_zarr.call_args_list
        assert any(kwargs.get("consolidated", False) for args, kwargs in calls)

    @patch("xarray.DataTree.to_zarr")
    def test_simple_root_consolidation_with_groups(
        self, mock_to_zarr, temp_output_dir
    ):
        """Test root consolidation with nested groups."""
        converter = S2OptimizedConverter()

        # Create test datasets dict with nested groups
        datasets = {
            "/measurements/reflectance/r10m": Mock(),
            "/quality/atmosphere": Mock(),
        }

        converter._simple_root_consolidation(temp_output_dir, datasets)

        # Verify to_zarr was called (for root + intermediary groups)
        assert mock_to_zarr.call_count >= 2


class TestConvenienceFunction:
    """Test the convenience function."""

    @patch("eopf_geozarr.s2_optimization.s2_converter.S2OptimizedConverter")
    def test_convert_s2_optimized_convenience_function(self, mock_converter_class):
        """Test the convenience function parameter separation."""
        from eopf_geozarr.s2_optimization.s2_converter import convert_s2_optimized

        mock_converter_instance = Mock()
        mock_converter_class.return_value = mock_converter_instance
        mock_converter_instance.convert_s2_optimized.return_value = Mock()

        # Test parameter separation
        dt_input = Mock()
        output_path = "/test/path"

        result = convert_s2_optimized(
            dt_input,
            output_path,
            enable_sharding=False,
            spatial_chunk=512,
            compression_level=2,
            max_retries=5,
            create_geometry_group=False,
            validate_output=False,
            verbose=True,
        )

        # Verify constructor was called with correct args
        mock_converter_class.assert_called_once_with(
            enable_sharding=False, spatial_chunk=512, compression_level=2, max_retries=5
        )

        # Verify method was called with remaining args
        mock_converter_instance.convert_s2_optimized.assert_called_once_with(
            dt_input,
            output_path,
            create_geometry_group=False,
            validate_output=False,
            verbose=True,
        )


if __name__ == "__main__":
    pytest.main([__file__])
