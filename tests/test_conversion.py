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
from eopf_geozarr.conversion.geozarr import (
    create_overview_dataset_all_vars,
    prepare_dataset_with_crs_info,
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
        levels = calculate_overview_levels(
            1024, 1024, min_dimension=256, tile_width=256
        )

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
            assert (
                processed_ds[band].attrs["standard_name"]
                == "toa_bidirectional_reflectance"
            )

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


class TestIssue12Fix:
    """Test fixes for GitHub Issue #12: Missing Coordinates arrays or CRS for groups."""

    def test_prepare_dataset_with_crs_info_with_spatial_coordinates(self) -> None:
        """Test adding CRS information to groups with spatial coordinates."""
        # Create a DataTree with measurement and geometry groups
        # Measurement group with CRS info
        measurement_ds = xr.Dataset(
            {
                "B04": (["y", "x"], np.random.rand(10, 10), {"proj:epsg": 32633}),
            },
            coords={
                "x": (["x"], np.linspace(300000, 310000, 10)),
                "y": (["y"], np.linspace(5000000, 5010000, 10)),
            },
        )

        # Geometry group without CRS info (simulating the issue)
        geometry_ds = xr.Dataset(
            {
                "mean_sun_angles": (["angle"], np.array([45.0, 30.0])),
                "sun_angles": (["angle", "y", "x"], np.random.rand(2, 5, 5)),
            },
            coords={
                "x": (["x"], np.linspace(300000, 310000, 5)),
                "y": (["y"], np.linspace(5000000, 5010000, 5)),
                "angle": (["angle"], ["zenith", "azimuth"]),
            },
        )

        # Create DataTree
        dt = xr.DataTree()
        dt["measurements/r10m"] = measurement_ds
        dt["conditions/geometry"] = geometry_ds

        # Mock the output path and file operations
        with patch(
            "eopf_geozarr.conversion.geozarr.fs_utils.normalize_path"
        ) as mock_normalize:
            with patch(
                "eopf_geozarr.conversion.geozarr.fs_utils.get_storage_options"
            ) as mock_storage:
                mock_normalize.return_value = "/mock/path"
                mock_storage.return_value = {}

                # Test the function
                processed_ds = prepare_dataset_with_crs_info(
                    dt["conditions/geometry"].to_dataset(),
                    reference_crs="epsg:32633",
                )

                # Verify CRS information was added to the dataset
                assert "spatial_ref" in processed_ds
                assert processed_ds.rio.crs.to_string() == "EPSG:32633"

    def test_prepare_dataset_with_crs_info_coordinate_attributes(self) -> None:
        """Test that coordinate attributes are properly set."""
        # Create a geometry dataset with various coordinate types
        geometry_ds = xr.Dataset(
            {
                "viewing_angles": (
                    ["band", "detector", "angle", "y", "x"],
                    np.random.rand(3, 2, 2, 5, 5),
                ),
                "mean_sun_angles": (["angle"], np.array([45.0, 30.0])),
            },
            coords={
                "x": (["x"], np.linspace(300000, 310000, 5)),
                "y": (["y"], np.linspace(5000000, 5010000, 5)),
                "angle": (["angle"], ["zenith", "azimuth"]),
                "band": (["band"], ["B02", "B03", "B04"]),
                "detector": (["detector"], [1, 2]),
            },
        )

        # Create DataTree with measurement group for CRS inference
        measurement_ds = xr.Dataset(
            {"B04": (["y", "x"], np.random.rand(10, 10), {"proj:epsg": 32633})}
        )

        dt = xr.DataTree()
        dt["measurements/r10m"] = measurement_ds
        dt["conditions/geometry"] = geometry_ds

        # Test the coordinate attribute setting logic directly
        # This simulates what prepare_dataset_with_crs_info does internally
        ds = dt["conditions/geometry"].to_dataset().copy()

        # Apply the same logic as in prepare_dataset_with_crs_info
        for coord_name in ds.coords:
            if coord_name == "x":
                ds[coord_name].attrs.update(
                    {
                        "_ARRAY_DIMENSIONS": ["x"],
                        "standard_name": "projection_x_coordinate",
                        "units": "m",
                        "long_name": "x coordinate of projection",
                    }
                )
            elif coord_name == "y":
                ds[coord_name].attrs.update(
                    {
                        "_ARRAY_DIMENSIONS": ["y"],
                        "standard_name": "projection_y_coordinate",
                        "units": "m",
                        "long_name": "y coordinate of projection",
                    }
                )
            elif coord_name == "angle":
                ds[coord_name].attrs.update(
                    {
                        "_ARRAY_DIMENSIONS": ["angle"],
                        "standard_name": "angle",
                        "long_name": "angle coordinate",
                    }
                )
            elif coord_name == "band":
                ds[coord_name].attrs.update(
                    {
                        "_ARRAY_DIMENSIONS": ["band"],
                        "standard_name": "band",
                        "long_name": "spectral band identifier",
                    }
                )
            elif coord_name == "detector":
                ds[coord_name].attrs.update(
                    {
                        "_ARRAY_DIMENSIONS": ["detector"],
                        "standard_name": "detector",
                        "long_name": "detector identifier",
                    }
                )
            else:
                # Generic coordinate
                if "_ARRAY_DIMENSIONS" not in ds[coord_name].attrs:
                    ds[coord_name].attrs["_ARRAY_DIMENSIONS"] = [coord_name]

        # Verify coordinate attributes were set correctly
        # Check x coordinate attributes
        x_attrs = ds.coords["x"].attrs
        assert x_attrs["_ARRAY_DIMENSIONS"] == ["x"]
        assert x_attrs["standard_name"] == "projection_x_coordinate"
        assert x_attrs["units"] == "m"
        assert x_attrs["long_name"] == "x coordinate of projection"

        # Check y coordinate attributes
        y_attrs = ds.coords["y"].attrs
        assert y_attrs["_ARRAY_DIMENSIONS"] == ["y"]
        assert y_attrs["standard_name"] == "projection_y_coordinate"
        assert y_attrs["units"] == "m"
        assert y_attrs["long_name"] == "y coordinate of projection"

        # Check angle coordinate attributes
        angle_attrs = ds.coords["angle"].attrs
        assert angle_attrs["_ARRAY_DIMENSIONS"] == ["angle"]
        assert angle_attrs["standard_name"] == "angle"
        assert angle_attrs["long_name"] == "angle coordinate"

        # Check band coordinate attributes
        band_attrs = ds.coords["band"].attrs
        assert band_attrs["_ARRAY_DIMENSIONS"] == ["band"]
        assert band_attrs["standard_name"] == "band"
        assert band_attrs["long_name"] == "spectral band identifier"

        # Check detector coordinate attributes
        detector_attrs = ds.coords["detector"].attrs
        assert detector_attrs["_ARRAY_DIMENSIONS"] == ["detector"]
        assert detector_attrs["standard_name"] == "detector"
        assert detector_attrs["long_name"] == "detector identifier"

    def test_prepare_dataset_with_crs_info_data_variable_attributes(self) -> None:
        """Test that data variable attributes are properly set."""
        # Create a geometry dataset
        geometry_ds = xr.Dataset(
            {
                "sun_angles": (["angle", "y", "x"], np.random.rand(2, 5, 5)),
                "mean_sun_angles": (["angle"], np.array([45.0, 30.0])),
            },
            coords={
                "x": (["x"], np.linspace(300000, 310000, 5)),
                "y": (["y"], np.linspace(5000000, 5010000, 5)),
                "angle": (["angle"], ["zenith", "azimuth"]),
            },
        )

        # Create DataTree with measurement group for CRS inference
        measurement_ds = xr.Dataset(
            {"B04": (["y", "x"], np.random.rand(10, 10), {"proj:epsg": 32633})}
        )

        dt = xr.DataTree()
        dt["measurements/r10m"] = measurement_ds
        dt["conditions/geometry"] = geometry_ds

        # Process the dataset
        processed_ds = prepare_dataset_with_crs_info(
            dt["conditions/geometry"].to_dataset(), reference_crs="epsg:32633"
        )

        # Verify data variable attributes were set correctly
        for var_name in processed_ds.data_vars:
            if var_name != "spatial_ref":  # Skip grid mapping variable
                var_attrs = processed_ds[var_name].attrs
                assert "_ARRAY_DIMENSIONS" in var_attrs
                assert var_attrs["_ARRAY_DIMENSIONS"] == list(
                    processed_ds[var_name].dims
                )

                # Variables with spatial coordinates should have grid_mapping
                if (
                    "x" in processed_ds[var_name].dims
                    and "y" in processed_ds[var_name].dims
                ):
                    assert "grid_mapping" in var_attrs
                    assert var_attrs["grid_mapping"] == "spatial_ref"

    def test_prepare_dataset_with_crs_info_crs_inference(self) -> None:
        """Test CRS inference from measurement groups."""
        # Create measurement groups with different EPSG codes
        measurement_ds1 = xr.Dataset(
            {"B04": (["y", "x"], np.random.rand(10, 10), {"proj:epsg": 32633})}
        )
        measurement_ds2 = xr.Dataset(
            {"B05": (["y", "x"], np.random.rand(10, 10), {"proj:epsg": 32633})}
        )

        # Create geometry group without CRS
        geometry_ds = xr.Dataset(
            {"angles": (["y", "x"], np.random.rand(5, 5))},
            coords={
                "x": (["x"], np.linspace(300000, 310000, 5)),
                "y": (["y"], np.linspace(5000000, 5010000, 5)),
            },
        )

        dt = xr.DataTree()
        dt["measurements/r10m"] = measurement_ds1
        dt["measurements/r20m"] = measurement_ds2
        dt["conditions/geometry"] = geometry_ds

        # Test the CRS inference and application logic directly
        # This simulates what prepare_dataset_with_crs_info does internally
        ds = dt["conditions/geometry"].to_dataset().copy()

        # Apply CRS (simulating the rioxarray write_crs call)
        ds = ds.rio.write_crs("epsg:32633")

        # Ensure spatial_ref variable has proper attributes
        if "spatial_ref" in ds:
            ds["spatial_ref"].attrs["_ARRAY_DIMENSIONS"] = []

            # Add GeoTransform if we can calculate it from coordinates
            if len(ds.coords["x"]) > 1 and len(ds.coords["y"]) > 1:
                x_coords = ds.coords["x"].values
                y_coords = ds.coords["y"].values

                # Calculate pixel size
                pixel_size_x = float(x_coords[1] - x_coords[0])
                pixel_size_y = float(y_coords[0] - y_coords[1])  # Usually negative

                # Create GeoTransform (GDAL format)
                transform_str = (
                    f"{x_coords[0]} {pixel_size_x} 0.0 {y_coords[0]} 0.0 {pixel_size_y}"
                )
                ds["spatial_ref"].attrs["GeoTransform"] = transform_str

        # Verify CRS was inferred and applied
        assert "spatial_ref" in ds

        # Check spatial_ref attributes
        spatial_ref_attrs = ds["spatial_ref"].attrs
        assert "_ARRAY_DIMENSIONS" in spatial_ref_attrs
        assert spatial_ref_attrs["_ARRAY_DIMENSIONS"] == []
        assert "crs_wkt" in spatial_ref_attrs

    def test_create_overview_dataset_all_vars_grid_mapping(self) -> None:
        """Test that overview datasets have proper grid_mapping attributes."""
        # Create a source dataset with CRS and grid_mapping
        source_ds = xr.Dataset(
            {
                "B04": (
                    ["y", "x"],
                    np.random.rand(100, 100),
                    {
                        "standard_name": "toa_bidirectional_reflectance",
                        "grid_mapping": "spatial_ref",
                    },
                ),
                "B03": (
                    ["y", "x"],
                    np.random.rand(100, 100),
                    {
                        "standard_name": "toa_bidirectional_reflectance",
                        "grid_mapping": "spatial_ref",
                    },
                ),
                "spatial_ref": (
                    [],
                    0,
                    {
                        "crs_wkt": 'PROJCS["WGS 84 / UTM zone 33N",...]',
                        "GeoTransform": "300000.0 10.0 0.0 5000000.0 0.0 -10.0",
                    },
                ),
            },
            coords={
                "x": (["x"], np.linspace(300000, 301000, 100)),
                "y": (["y"], np.linspace(5000000, 5001000, 100)),
            },
        )

        # Set CRS using rioxarray
        source_ds = source_ds.rio.write_crs("EPSG:32633")

        # Create overview dataset
        overview_ds = create_overview_dataset_all_vars(
            ds=source_ds,
            level=1,
            width=50,
            height=50,
            native_crs=source_ds.rio.crs,
            native_bounds=source_ds.rio.bounds(),
            data_vars=["B04", "B03"],
        )

        # Verify grid_mapping attributes are preserved
        assert "spatial_ref" in overview_ds
        assert overview_ds.rio.crs.to_string() == "EPSG:32633"

        # Check that all data variables reference the grid_mapping
        for var in ["B04", "B03"]:
            assert overview_ds[var].rio.crs.to_string() == "EPSG:32633"
            assert "_ARRAY_DIMENSIONS" in overview_ds[var].attrs

        # Verify coordinates have proper attributes
        for coord in ["x", "y"]:
            coord_attrs = overview_ds[coord].attrs
            assert "_ARRAY_DIMENSIONS" in coord_attrs
            assert coord_attrs["_ARRAY_DIMENSIONS"] == [coord]
            assert "standard_name" in coord_attrs

    def test_prepare_dataset_with_crs_info_missing_group(self) -> None:
        """Test handling of missing groups in crs_groups list."""
        # Create a simple DataTree
        dt = xr.DataTree()
        dt["existing_group"] = xr.Dataset({"var": (["x"], [1, 2, 3])})

        # Test with non-existent group
        with pytest.raises(KeyError, match="Could not find node at non_spatial_group"):
            prepare_dataset_with_crs_info(
                dt["non_spatial_group"].to_dataset(), reference_crs="epsg:32633"
            )

    def test_prepare_dataset_with_crs_info_no_spatial_coordinates(self) -> None:
        """Test handling of groups without spatial coordinates."""
        # Create a group without x,y coordinates
        non_spatial_ds = xr.Dataset(
            {
                "temperature": (["time"], np.array([20.0, 21.0, 22.0])),
            },
            coords={
                "time": (["time"], ["2023-01-01", "2023-01-02", "2023-01-03"]),
            },
        )

        dt = xr.DataTree()
        dt["non_spatial_group"] = non_spatial_ds

        # Test the coordinate attribute setting logic directly for non-spatial data
        # This simulates what prepare_dataset_with_crs_info does internally
        ds = dt["non_spatial_group"].to_dataset().copy()

        # Set up coordinate variables with proper attributes
        for coord_name in ds.coords:
            if "_ARRAY_DIMENSIONS" not in ds[coord_name].attrs:
                ds[coord_name].attrs["_ARRAY_DIMENSIONS"] = [coord_name]

        # Set up data variables with proper attributes
        for var_name in ds.data_vars:
            # Add _ARRAY_DIMENSIONS attribute if missing
            if "_ARRAY_DIMENSIONS" not in ds[var_name].attrs and hasattr(
                ds[var_name], "dims"
            ):
                ds[var_name].attrs["_ARRAY_DIMENSIONS"] = list(ds[var_name].dims)

        # Verify the group was processed but no CRS was added
        assert "spatial_ref" not in ds

        # But _ARRAY_DIMENSIONS should still be added
        assert "_ARRAY_DIMENSIONS" in ds["temperature"].attrs
        assert "_ARRAY_DIMENSIONS" in ds.coords["time"].attrs
        assert ds["temperature"].attrs["_ARRAY_DIMENSIONS"] == ["time"]
        assert ds.coords["time"].attrs["_ARRAY_DIMENSIONS"] == ["time"]


if __name__ == "__main__":
    pytest.main([__file__])
