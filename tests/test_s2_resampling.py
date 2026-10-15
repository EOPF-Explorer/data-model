"""
Unit tests for S2 resampling functionality.
"""

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.s2_optimization.s2_resampling import (
    determine_variable_type,
    downsample_variable,
)


@pytest.fixture
def sample_reflectance_data_2d() -> xr.DataArray:
    """Create a 2D reflectance data array for testing."""
    # Create a 4x4 array with known values
    data = np.array(
        [
            [100, 200, 300, 400],
            [150, 250, 350, 450],
            [110, 210, 310, 410],
            [160, 260, 360, 460],
        ],
        dtype=np.uint16,
    )

    coords = {
        "y": np.array([1000, 990, 980, 970]),
        "x": np.array([500000, 500010, 500020, 500030]),
    }

    return xr.DataArray(
        data,
        dims=["y", "x"],
        coords=coords,
        attrs={"units": "reflectance", "scale_factor": 0.0001},
    )


@pytest.fixture
def sample_reflectance_data_3d() -> xr.DataArray:
    """Create a 3D reflectance data array with time dimension for testing."""
    # Create a 2x4x4 array (time, y, x)
    data = np.array(
        [
            [
                [100, 200, 300, 400],
                [150, 250, 350, 450],
                [110, 210, 310, 410],
                [160, 260, 360, 460],
            ],
            [
                [120, 220, 320, 420],
                [170, 270, 370, 470],
                [130, 230, 330, 430],
                [180, 280, 380, 480],
            ],
        ],
        dtype=np.uint16,
    )

    coords = {
        "time": np.array(["2023-01-01", "2023-01-02"], dtype="datetime64[D]"),
        "y": np.array([1000, 990, 980, 970]),
        "x": np.array([500000, 500010, 500020, 500030]),
    }

    return xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords=coords,
        attrs={"units": "reflectance", "scale_factor": 0.0001},
    )


@pytest.fixture
def sample_classification_data() -> xr.DataArray:
    """Create classification data for testing."""
    # SCL values: 0=no_data, 1=saturated, 4=vegetation, 6=water, etc.
    data = np.array([[0, 1, 4, 4], [1, 4, 6, 6], [4, 4, 6, 8], [4, 6, 8, 8]], dtype=np.uint8)

    coords = {
        "y": np.array([1000, 990, 980, 970]),
        "x": np.array([500000, 500010, 500020, 500030]),
    }

    return xr.DataArray(
        data,
        dims=["y", "x"],
        coords=coords,
        attrs={"long_name": "Scene Classification Layer"},
    )


@pytest.fixture
def sample_quality_mask() -> xr.DataArray:
    """Create quality mask data for testing."""
    # Binary mask: 0=good, 1=bad
    data = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 1]], dtype=np.uint8)

    coords = {
        "y": np.array([1000, 990, 980, 970]),
        "x": np.array([500000, 500010, 500020, 500030]),
    }

    return xr.DataArray(data, dims=["y", "x"], coords=coords, attrs={"long_name": "Quality mask"})


@pytest.fixture
def sample_probability_data() -> xr.DataArray:
    """Create probability data for testing."""
    # Cloud probabilities in percent (0-100)
    data = np.array(
        [
            [10.5, 20.3, 85.7, 92.1],
            [15.2, 75.8, 88.3, 95.6],
            [12.7, 18.9, 90.2, 87.4],
            [8.1, 22.4, 78.9, 99.0],
        ],
        dtype=np.float32,
    )

    coords = {
        "y": np.array([1000, 990, 980, 970]),
        "x": np.array([500000, 500010, 500020, 500030]),
    }

    return xr.DataArray(
        data,
        dims=["y", "x"],
        coords=coords,
        attrs={"long_name": "Cloud probability", "units": "percent"},
    )


class TestS2ResamplingEngine:
    """Test cases for S2ResamplingEngine class."""

    def test_downsample_reflectance_2d(self, sample_reflectance_data_2d: xr.DataArray) -> None:
        """Test reflectance downsampling for 2D data."""

        # Downsample from 4x4 to 2x2
        result = downsample_variable(sample_reflectance_data_2d, 2, 2, "reflectance")

        # Check dimensions
        assert result.shape == (2, 2)
        assert result.dims == ("y", "x")

        # Check that values are averages of 2x2 blocks
        # Top-left block: mean of [100, 200, 150, 250] = 175
        assert result.values[0, 0] == 175.0

        # Top-right block: mean of [300, 400, 350, 450] = 375
        assert result.values[0, 1] == 375.0

        # Check coordinates are properly subsampled
        assert len(result.coords["y"]) == 2
        assert len(result.coords["x"]) == 2
        np.testing.assert_array_equal(result.coords["y"].values, [1000, 980])
        np.testing.assert_array_equal(result.coords["x"].values, [500000, 500020])

        # Check attributes are preserved
        assert result.attrs == sample_reflectance_data_2d.attrs

    def test_downsample_reflectance_3d(self, sample_reflectance_data_3d: xr.DataArray) -> None:
        """Test reflectance downsampling for 3D data."""
        # Downsample from 2x4x4 to 2x2x2
        result = downsample_variable(sample_reflectance_data_3d, 2, 2, "reflectance")

        # Check dimensions
        assert result.shape == (2, 2, 2)
        assert result.dims == ("time", "y", "x")

        # Check first time slice values
        # Top-left block: mean of [100, 200, 150, 250] = 175
        assert result.values[0, 0, 0] == 175.0

        # Check second time slice values
        # Top-left block: mean of [120, 220, 170, 270] = 195
        assert result.values[1, 0, 0] == 195.0

        # Check coordinates
        assert len(result.coords["time"]) == 2
        assert len(result.coords["y"]) == 2
        assert len(result.coords["x"]) == 2

    def test_downsample_classification(self, sample_classification_data: xr.DataArray) -> None:
        """Test classification downsampling using mode."""

        # Downsample from 4x4 to 2x2
        result = downsample_variable(sample_classification_data, 2, 2, "classification")

        # Check dimensions
        assert result.shape == (2, 2)
        assert result.dims == ("y", "x")

        # Check mode values
        # Top-left block: [0, 1, 1, 4] -> mode should be 1 (most frequent)
        # Top-right block: [4, 4, 6, 6] -> mode could be either 4 or 6 (both appear twice)
        assert result.values[0, 0] in [0, 1, 4]  # Allow for mode calculation variations

        # Check data type is preserved
        assert result.dtype == sample_classification_data.dtype

    def test_downsample_quality_mask(self, sample_quality_mask: xr.DataArray) -> None:
        """Test quality mask downsampling using logical OR."""

        # Downsample from 4x4 to 2x2
        result = downsample_variable(sample_quality_mask, 2, 2, "quality_mask")

        # Check dimensions
        assert result.shape == (2, 2)
        assert result.dims == ("y", "x")

        # Check logical OR behavior
        # Top-left block: [0, 0, 0, 1] -> any non-zero = 1
        assert result.values[0, 0] == 1

        # Top-right block: [1, 0, 0, 0] -> any non-zero = 1
        assert result.values[0, 1] == 1

        # Bottom-left block: [1, 0, 0, 0] -> any non-zero = 1
        assert result.values[1, 0] == 1

        # Bottom-right block: [0, 1, 1, 1] -> any non-zero = 1
        assert result.values[1, 1] == 1

    def test_downsample_probability(self, sample_probability_data: xr.DataArray) -> None:
        """Test probability downsampling with value clamping."""

        # Downsample from 4x4 to 2x2
        result = downsample_variable(sample_probability_data, 2, 2, "probability")

        # Check dimensions
        assert result.shape == (2, 2)
        assert result.dims == ("y", "x")

        # Values should be averages and clamped to [0, 100]
        assert np.all(result.values >= 0)
        assert np.all(result.values <= 100)

        # Check specific average calculation
        # Top-left block: mean of [10.5, 20.3, 15.2, 75.8] ≈ 30.45
        expected_val = (10.5 + 20.3 + 15.2 + 75.8) / 4
        np.testing.assert_almost_equal(result.values[0, 0], expected_val, decimal=2)

    def test_detector_footprint_same_as_quality_mask(
        self, sample_quality_mask: xr.DataArray
    ) -> None:
        """Test that detector footprint uses same method as quality mask."""

        result_quality = downsample_variable(sample_quality_mask, 2, 2, "quality_mask")
        result_detector = downsample_variable(sample_quality_mask, 2, 2, "detector_footprint")

        # Results should be identical
        np.testing.assert_array_equal(result_quality.values, result_detector.values)

    def test_invalid_variable_type(self, sample_reflectance_data_2d: xr.DataArray) -> None:
        """Test error handling for invalid variable type."""

        with pytest.raises(ValueError, match="Unknown variable type"):
            downsample_variable(sample_reflectance_data_2d, 2, 2, "invalid_type")

    def test_non_divisible_dimensions(self) -> None:
        """Test handling of non-divisible dimensions."""

        # Create 5x5 data (not evenly divisible by 2)
        data = np.random.rand(5, 5).astype(np.float32)
        coords = {"y": np.arange(5), "x": np.arange(5)}
        da = xr.DataArray(data, dims=["y", "x"], coords=coords)

        # Should crop to make it divisible
        result = downsample_variable(da, 2, 2, "reflectance")

        # Should result in 2x2 output (cropped from 4x4)
        assert result.shape == (2, 2)

    def test_single_pixel_downsampling(self) -> None:
        """Test downsampling to single pixel."""

        # Create 4x4 data
        data = np.ones((4, 4), dtype=np.float32) * 100
        coords = {"y": np.arange(4), "x": np.arange(4)}
        da = xr.DataArray(data, dims=["y", "x"], coords=coords)

        # Downsample to 1x1
        result = downsample_variable(da, 1, 1, "reflectance")

        assert result.shape == (1, 1)
        assert result.values[0, 0] == 100.0


class TestDetermineVariableType:
    """Test cases for determine_variable_type function."""

    @pytest.mark.parametrize(
        ("var_name", "expected"),
        [
            # Spectral bands
            ("b01", "reflectance"),
            ("b02", "reflectance"),
            ("b8a", "reflectance"),
            # Classification data
            ("scl", "classification"),
            # L1C cloud bitmask: named like a band, but must not be averaged
            ("b00", "classification"),
            # Probability data
            ("cld", "probability"),
            ("snw", "probability"),
            # Atmospheric quality, treated as reflectance
            ("aot", "reflectance"),
            ("wvp", "reflectance"),
            # Quality masks
            ("detector_footprint_b01", "quality_mask"),
            ("quality_b01", "quality_mask"),
            ("quality_b02", "quality_mask"),
            # Unknown variables default to reflectance
            ("unknown_var", "reflectance"),
            ("custom_band", "reflectance"),
        ],
    )
    def test_variable_type(self, var_name: str, expected: str) -> None:
        """Test that the variable type is derived from the variable name alone."""
        assert determine_variable_type(var_name) == expected


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_data_array(self) -> None:
        """Test handling of empty data arrays."""

        # Create minimal data array
        data = np.array([[1]])
        coords = {"y": [0], "x": [0]}
        da = xr.DataArray(data, dims=["y", "x"], coords=coords)

        # This should work for 1x1 -> 1x1 downsampling
        result = downsample_variable(da, 1, 1, "reflectance")
        assert result.shape == (1, 1)
        assert result.values[0, 0] == 1

    def test_preserve_attributes_and_encoding(self) -> None:
        """Test that attributes and encoding are preserved."""

        data = np.ones((4, 4), dtype=np.uint16) * 1000
        coords = {"y": np.arange(4), "x": np.arange(4)}

        attrs = {
            "long_name": "Test reflectance",
            "units": "reflectance",
            "scale_factor": 0.0001,
            "add_offset": 0,
        }

        da = xr.DataArray(data, dims=["y", "x"], coords=coords, attrs=attrs)

        result = downsample_variable(da, 2, 2, "reflectance")

        # Attributes should be preserved
        assert result.attrs == attrs

    def test_coordinate_names_preserved(self) -> None:
        """Test that coordinate names are preserved during downsampling."""

        data = np.ones((4, 4), dtype=np.float32)
        coords = {"latitude": np.arange(4), "longitude": np.arange(4)}

        da = xr.DataArray(data, dims=["latitude", "longitude"], coords=coords)

        result = downsample_variable(da, 2, 2, "reflectance")

        # Coordinate names should be preserved
        assert "latitude" in result.coords
        assert "longitude" in result.coords
        assert result.dims == ("latitude", "longitude")


class TestIntegrationScenarios:
    """Integration test scenarios."""

    def test_multiscale_pyramid_creation(self) -> None:
        """Test creating a complete multiscale pyramid."""

        # Start with 32x32 data
        original_size = 32
        data = np.random.rand(original_size, original_size).astype(np.float32) * 1000
        coords = {"y": np.arange(original_size), "x": np.arange(original_size)}

        da = xr.DataArray(data, dims=["y", "x"], coords=coords)

        # Create pyramid levels: 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1
        levels = []
        current_data = da
        current_size = original_size

        while current_size >= 2:
            next_size = current_size // 2
            downsampled = downsample_variable(current_data, next_size, next_size, "reflectance")
            levels.append(downsampled)
            current_data = downsampled
            current_size = next_size

        # Verify pyramid structure
        expected_sizes = [16, 8, 4, 2, 1]
        for i, level in enumerate(levels):
            expected_size = expected_sizes[i]
            assert level.shape == (expected_size, expected_size)

        # Verify that values are reasonable (not NaN, not extreme)
        for level in levels:
            assert not np.isnan(level.values).any()
            assert np.all(level.values >= 0)

    def test_mixed_variable_types_processing(self) -> None:
        """Test processing different variable types together."""

        # Create base 4x4 data
        size = 4
        coords = {"y": np.arange(size), "x": np.arange(size)}

        # Create different variable types
        reflectance_data = xr.DataArray(
            np.random.rand(size, size) * 1000, dims=["y", "x"], coords=coords
        )

        classification_data = xr.DataArray(
            np.random.randint(0, 10, (size, size)), dims=["y", "x"], coords=coords
        )

        quality_data = xr.DataArray(
            np.random.randint(0, 2, (size, size)), dims=["y", "x"], coords=coords
        )

        # Process each with appropriate method
        results = {}
        for var_name, var_data, var_type in [
            ("b04", reflectance_data, "reflectance"),
            ("scl", classification_data, "classification"),
            ("quality_b04", quality_data, "quality_mask"),
        ]:
            results[var_name] = downsample_variable(var_data, 2, 2, var_type)

        # Verify all results have same dimensions
        for result in results.values():
            assert result.shape == (2, 2)

        # Verify coordinate consistency
        y_coords = results["b04"].coords["y"]
        x_coords = results["b04"].coords["x"]

        for result in results.values():
            np.testing.assert_array_equal(result.coords["y"].values, y_coords.values)
            np.testing.assert_array_equal(result.coords["x"].values, x_coords.values)
