"""Tests for the conversion module."""

from unittest.mock import patch

import numpy as np
import pytest
import rioxarray  # noqa: F401  # Import to enable .rio accessor
import xarray as xr

from eopf_geozarr.conversion import (
    calculate_aligned_chunk_size,
    calculate_overview_levels,
    downsample_2d_array,
    is_grid_mapping_variable,
    setup_datatree_metadata_geozarr_spec_compliant,
    validate_existing_band_data,
)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_downsample_2d_array_block_averaging(self) -> None:
        """Test downsampling with block averaging."""
        # Create a 4x4 array
        source_data = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=float
        )

        # Downsample to 2x2
        result = downsample_2d_array(source_data, 2, 2)

        # Expected result: average of 2x2 blocks
        expected = np.array(
            [
                [3.5, 5.5],  # (1+2+5+6)/4, (3+4+7+8)/4
                [11.5, 13.5],  # (9+10+13+14)/4, (11+12+15+16)/4
            ]
        )

        np.testing.assert_array_equal(result, expected)

    def test_downsample_2d_array_subsampling(self) -> None:
        """Test downsampling with subsampling when block size is 1."""
        source_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

        # Downsample to 2x2 (will use subsampling)
        result = downsample_2d_array(source_data, 2, 2)

        # Should subsample at indices [0, 2] for both dimensions
        expected = np.array([[1, 3], [7, 9]])

        np.testing.assert_array_equal(result, expected)

    def test_calculate_aligned_chunk_size_perfect_divisor(self) -> None:
        """Test chunk size calculation with perfect divisor."""
        # 1000 dimension, want 256 chunks
        result = calculate_aligned_chunk_size(1000, 256)
        # Should find 250 as the largest divisor <= 256
        assert result == 250
        assert 1000 % result == 0

    def test_calculate_aligned_chunk_size_larger_than_dimension(self) -> None:
        """Test chunk size calculation when desired size is larger than dimension."""
        result = calculate_aligned_chunk_size(100, 256)
        assert result == 100

    def test_calculate_aligned_chunk_size_no_perfect_divisor(self) -> None:
        """Test chunk size calculation when no perfect divisor exists."""
        # Prime number dimension
        result = calculate_aligned_chunk_size(97, 50)
        # Should return 1 as the only divisor when no good divisor is found
        assert result == 1

    def test_is_grid_mapping_variable(self) -> None:
        """Test grid mapping variable detection."""
        # Create a dataset with a grid mapping variable
        ds = xr.Dataset(
            {
                "temperature": (
                    ["y", "x"],
                    np.random.rand(10, 10),
                    {"grid_mapping": "spatial_ref"},
                ),
                "spatial_ref": ([], 0, {"grid_mapping_name": "latitude_longitude"}),
            }
        )

        assert is_grid_mapping_variable(ds, "spatial_ref") is True
        assert is_grid_mapping_variable(ds, "temperature") is False

    def test_validate_existing_band_data_valid(self) -> None:
        """Test validation of existing valid band data."""
        # Create datasets
        existing_ds = xr.Dataset(
            {
                "B04": (
                    ["y", "x"],
                    np.random.rand(100, 100),
                    {
                        "_ARRAY_DIMENSIONS": ["y", "x"],
                        "standard_name": "toa_bidirectional_reflectance",
                        "grid_mapping": "spatial_ref",
                    },
                )
            }
        )

        expected_ds = xr.Dataset({"B04": (["y", "x"], np.random.rand(100, 100))})

        assert validate_existing_band_data(existing_ds, "B04", expected_ds) is True

    def test_validate_existing_band_data_missing(self) -> None:
        """Test validation of missing band data."""
        existing_ds = xr.Dataset({})
        expected_ds = xr.Dataset({"B04": (["y", "x"], np.random.rand(100, 100))})

        assert validate_existing_band_data(existing_ds, "B04", expected_ds) is False

    def test_calculate_overview_levels(self) -> None:
        """Test overview levels calculation."""
        levels = calculate_overview_levels(1024, 1024, min_dimension=256, tile_width=256)

        # Should have levels 0, 1, 2 (1024 -> 512 -> 256)
        assert len(levels) == 3
        assert levels[0]["level"] == 0
        assert levels[0]["width"] == 1024
        assert levels[0]["height"] == 1024
        assert levels[0]["scale_factor"] == 1

        assert levels[1]["level"] == 1
        assert levels[1]["width"] == 512
        assert levels[1]["height"] == 512
        assert levels[1]["scale_factor"] == 2

        assert levels[2]["level"] == 2
        assert levels[2]["width"] == 256
        assert levels[2]["height"] == 256
        assert levels[2]["scale_factor"] == 4


class TestMetadataSetup:
    """Test metadata setup functions."""

    def test_setup_datatree_metadata_geozarr_spec_compliant(self) -> None:
        """Test GeoZarr metadata setup."""
        # Create a real DataTree with measurement groups
        # Create datasets for different resolution groups
        r10m_ds = xr.Dataset(
            {
                "B04": (["y", "x"], np.random.rand(100, 100), {"proj:epsg": 32633}),
                "B03": (["y", "x"], np.random.rand(100, 100), {"proj:epsg": 32633}),
            },
            coords={
                "x": (["x"], np.linspace(0, 1000, 100)),
                "y": (["y"], np.linspace(0, 1000, 100)),
            },
        )

        # Create a DataTree structure
        dt = xr.DataTree()
        dt["measurements/r10m"] = r10m_ds

        groups = ["/measurements/r10m"]

        with patch("eopf_geozarr.conversion.geozarr.print"):
            result = setup_datatree_metadata_geozarr_spec_compliant(dt, groups)

        # Should return a dictionary with the processed group
        assert isinstance(result, dict)
        assert "/measurements/r10m" in result

        # Check that the dataset has the required attributes
        processed_ds = result["/measurements/r10m"]

        # Check that bands have required GeoZarr attributes
        for band in ["B04", "B03"]:
            assert "standard_name" in processed_ds[band].attrs
            assert "_ARRAY_DIMENSIONS" in processed_ds[band].attrs
            assert "grid_mapping" in processed_ds[band].attrs
            assert processed_ds[band].attrs["standard_name"] == "toa_bidirectional_reflectance"

        # Check coordinate attributes
        for coord in ["x", "y"]:
            assert "_ARRAY_DIMENSIONS" in processed_ds[coord].attrs
            assert "standard_name" in processed_ds[coord].attrs


class TestIntegration:
    """Integration tests."""

    def test_create_simple_geozarr_metadata(self) -> None:
        """Test creating simple GeoZarr metadata structure."""
        # Create a simple dataset
        data = np.random.rand(10, 10)
        ds = xr.Dataset(
            {"temperature": (["y", "x"], data, {"proj:epsg": 4326})},
            coords={
                "x": (["x"], np.linspace(-180, 180, 10)),
                "y": (["y"], np.linspace(-90, 90, 10)),
            },
        )

        # Create a DataTree structure
        dt = xr.DataTree()
        dt["test_group"] = ds

        groups = ["/test_group"]

        with patch("eopf_geozarr.conversion.geozarr.print"):
            result = setup_datatree_metadata_geozarr_spec_compliant(dt, groups)

        assert "/test_group" in result
        processed_ds = result["/test_group"]

        # Verify GeoZarr compliance
        assert "standard_name" in processed_ds["temperature"].attrs
        assert "_ARRAY_DIMENSIONS" in processed_ds["temperature"].attrs
        assert "grid_mapping" in processed_ds["temperature"].attrs

        # Verify coordinate metadata
        for coord in ["x", "y"]:
            assert "_ARRAY_DIMENSIONS" in processed_ds[coord].attrs
            assert "standard_name" in processed_ds[coord].attrs


if __name__ == "__main__":
    pytest.main([__file__])
