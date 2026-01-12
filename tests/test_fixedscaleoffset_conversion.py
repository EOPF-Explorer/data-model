"""Unit tests for FixedScaleOffset codec conversion functionality."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.s2_optimization.s2_multiscale import (
    create_fixed_scale_offset_filter,
    create_measurements_encoding,
    detect_scale_offset_attributes,
)


class TestScaleOffsetDetection:
    """Test scale_factor and add_offset detection from various sources."""

    def test_detect_from_direct_attributes(self) -> None:
        """Test detection from direct DataArray attributes."""
        data = xr.DataArray(
            np.array([[1, 2], [3, 4]], dtype=np.uint16),
            attrs={"scale_factor": 0.0001, "add_offset": -0.1, "dtype": "<u2"},
        )

        result = detect_scale_offset_attributes(data)

        assert result is not None
        assert result["scale_factor"] == 0.0001
        assert result["add_offset"] == -0.1
        assert result["dtype"] == "<u2"

    def test_detect_from_eopf_attrs(self) -> None:
        """Test detection from _eopf_attrs nested attributes."""
        data = xr.DataArray(
            np.array([[1, 2], [3, 4]], dtype=np.uint16),
            attrs={"_eopf_attrs": {"scale_factor": 0.0001, "add_offset": -0.1, "dtype": "<u2"}},
        )

        result = detect_scale_offset_attributes(data)

        assert result is not None
        assert result["scale_factor"] == 0.0001
        assert result["add_offset"] == -0.1
        assert result["dtype"] == "<u2"

    def test_detect_from_encoding(self) -> None:
        """Test detection from xarray encoding."""
        data = xr.DataArray(np.array([[1, 2], [3, 4]], dtype=np.uint16), attrs={"dtype": "<u2"})
        # Manually set encoding
        data.encoding["scale_factor"] = 0.0001
        data.encoding["add_offset"] = -0.1

        result = detect_scale_offset_attributes(data)

        assert result is not None
        assert result["scale_factor"] == 0.0001
        assert result["add_offset"] == -0.1
        assert result["dtype"] == "<u2"

    def test_no_scale_offset_attributes(self) -> None:
        """Test case where no scale/offset attributes are present."""
        data = xr.DataArray(np.array([[1, 2], [3, 4]], dtype=np.uint16))

        result = detect_scale_offset_attributes(data)

        assert result is None

    def test_partial_attributes_scale_only(self) -> None:
        """Test case with only scale_factor present."""
        data = xr.DataArray(
            np.array([[1, 2], [3, 4]], dtype=np.uint16),
            attrs={"scale_factor": 0.0001, "dtype": "<u2"},
        )

        result = detect_scale_offset_attributes(data)

        assert result is not None
        assert result["scale_factor"] == 0.0001
        assert result["add_offset"] == 0.0  # Default value
        assert result["dtype"] == "<u2"

    def test_partial_attributes_offset_only(self) -> None:
        """Test case with only add_offset present."""
        data = xr.DataArray(
            np.array([[1, 2], [3, 4]], dtype=np.uint16), attrs={"add_offset": -0.1, "dtype": "<u2"}
        )

        result = detect_scale_offset_attributes(data)

        assert result is not None
        assert result["scale_factor"] == 1.0  # Default value
        assert result["add_offset"] == -0.1
        assert result["dtype"] == "<u2"


class TestFixedScaleOffsetCreation:
    """Test FixedScaleOffset filter creation."""

    def test_create_filter_success(self) -> None:
        """Test successful FixedScaleOffset filter creation."""
        scale_attrs = {
            "scale_factor": 0.0001,
            "add_offset": -0.1,
            "dtype": "<u2",  # Target storage type
        }

        filter_obj = create_fixed_scale_offset_filter(scale_attrs)

        if filter_obj is not None:  # Only test if numcodecs.zarr3 is available
            assert hasattr(filter_obj, "codec_config")
            config = filter_obj.codec_config

            # Verify parameter mapping for Sentinel-2 data
            assert config["offset"] == -0.1  # Same as add_offset
            assert abs(config["scale"] - 10000.0) < 0.001  # 1/0.0001
            assert config["dtype"] == "float64"  # Input type
            assert config["astype"] == "uint16"  # Storage type

    def test_create_filter_zero_scale_factor(self) -> None:
        """Test filter creation with zero scale_factor (edge case)."""
        scale_attrs = {"scale_factor": 0.0, "add_offset": -0.1, "dtype": "<u2"}

        filter_obj = create_fixed_scale_offset_filter(scale_attrs)

        if filter_obj is not None:
            config = filter_obj.codec_config
            assert config["scale"] == 1.0  # Fallback for zero scale_factor

    def test_dtype_mapping(self) -> None:
        """Test various dtype string mappings."""
        test_cases = [
            ("<u2", "uint16"),
            ("<u1", "uint8"),
            ("<i2", "int16"),
            ("<f4", "float32"),
            ("u2", "uint16"),  # Without endianness prefix
        ]

        for input_dtype, expected_storage in test_cases:
            scale_attrs = {"scale_factor": 0.0001, "add_offset": 0.0, "dtype": input_dtype}

            filter_obj = create_fixed_scale_offset_filter(scale_attrs)

            if filter_obj is not None:
                config = filter_obj.codec_config
                assert config["astype"] == expected_storage


class TestMeasurementsEncodingIntegration:
    """Test integration with create_measurements_encoding function."""

    def test_encoding_with_scale_offset(self) -> None:
        """Test that scale/offset attributes generate FixedScaleOffset filters."""
        # Create test dataset with scale/offset
        data = xr.DataArray(
            np.random.uniform(0.0, 1.0, size=(10, 10)).astype(np.float64),
            dims=["y", "x"],
            attrs={
                "scale_factor": 0.0001,
                "add_offset": -0.1,
                "dtype": "<u2",
                "long_name": "Test reflectance data",
            },
        )

        dataset = xr.Dataset({"test_var": data})

        encoding = create_measurements_encoding(dataset, spatial_chunk=5, enable_sharding=True)

        assert "test_var" in encoding
        var_encoding = encoding["test_var"]

        # Check required encoding keys
        assert "chunks" in var_encoding

        # Check if filters were used (when numcodecs.zarr3 is available)
        if "filters" in var_encoding:
            # FixedScaleOffset filter was applied
            assert len(var_encoding["filters"]) > 0
            assert "compressors" not in var_encoding
        else:
            # Fallback to standard compression
            assert "compressors" in var_encoding

        # Ensure scale_factor/add_offset are excluded from encoding
        assert "scale_factor" not in var_encoding
        assert "add_offset" not in var_encoding

    def test_encoding_without_scale_offset(self) -> None:
        """Test that variables without scale/offset use standard compression."""
        data = xr.DataArray(
            np.random.randint(0, 255, size=(10, 10), dtype=np.uint8),
            dims=["y", "x"],
            attrs={"long_name": "Test data without scaling"},
        )

        dataset = xr.Dataset({"test_var": data})

        encoding = create_measurements_encoding(dataset, spatial_chunk=5, enable_sharding=True)

        assert "test_var" in encoding
        var_encoding = encoding["test_var"]

        # Should use standard compressors, not filters
        assert "compressors" in var_encoding
        assert "filters" not in var_encoding
        assert "chunks" in var_encoding

    def test_coordinate_encoding(self) -> None:
        """Test that coordinates get proper encoding."""
        data = xr.DataArray(
            np.random.uniform(0.0, 1.0, size=(5, 5)).astype(np.float64),
            dims=["y", "x"],
            coords={"y": np.linspace(0, 100, 5), "x": np.linspace(0, 100, 5)},
            attrs={"scale_factor": 0.0001, "add_offset": -0.1, "dtype": "<u2"},
        )

        dataset = xr.Dataset({"test_var": data})

        encoding = create_measurements_encoding(dataset, spatial_chunk=5)

        # Check that coordinates are included in encoding
        assert "x" in encoding
        assert "y" in encoding

        # Coordinates should have empty compressors list
        assert encoding["x"]["compressors"] == []
        assert encoding["y"]["compressors"] == []


class TestSentinel2RealWorldScenario:
    """Test real-world Sentinel-2 data conversion scenarios."""

    @pytest.fixture
    def sentinel2_dataset(self) -> xr.Dataset:
        """Create a realistic Sentinel-2 dataset for testing."""
        # Simulate Sentinel-2 reflectance data (float64 values)
        reflectance_values = np.random.uniform(0.0, 1.0, size=(20, 20)).astype(np.float64)

        data_array = xr.DataArray(
            reflectance_values,  # Input: float64 reflectance values
            dims=["y", "x"],
            coords={"y": np.linspace(4000000, 4001000, 20), "x": np.linspace(300000, 301000, 20)},
            attrs={
                "scale_factor": 0.0001,
                "add_offset": -0.1,
                "dtype": "<u2",  # Target storage type: uint16
                "long_name": "BOA reflectance from MSI acquisition at spectral band 02 490 nm",
                "units": "digital_counts",
            },
        )

        return xr.Dataset({"b02": data_array})

    def test_sentinel2_encoding_generation(self, sentinel2_dataset) -> None:
        """Test encoding generation for Sentinel-2 data."""
        encoding = create_measurements_encoding(
            sentinel2_dataset, spatial_chunk=10, enable_sharding=True
        )

        assert "b02" in encoding
        var_encoding = encoding["b02"]

        # Check basic encoding structure
        assert "chunks" in var_encoding
        assert "shards" in var_encoding

        # Verify scale/offset exclusion
        assert "scale_factor" not in var_encoding
        assert "add_offset" not in var_encoding

    def test_sentinel2_fixedscaleoffset_parameters(self, sentinel2_dataset) -> None:
        """Test that Sentinel-2 data generates correct FixedScaleOffset parameters."""
        encoding = create_measurements_encoding(sentinel2_dataset, spatial_chunk=10)

        if "b02" in encoding and "filters" in encoding["b02"]:
            filters = encoding["b02"]["filters"]
            assert len(filters) > 0

            filter_obj = filters[0]
            if hasattr(filter_obj, "codec_config"):
                config = filter_obj.codec_config

                # Verify Sentinel-2 specific configuration
                assert config["offset"] == -0.1
                assert abs(config["scale"] - 10000.0) < 0.001
                assert config["dtype"] == "float64"  # Input: reflectance values
                assert config["astype"] == "uint16"  # Storage: compressed

    @pytest.mark.skipif(
        True,  # Skip by default as it requires zarr writing
        reason="Integration test requiring file I/O - run manually if needed",
    )
    def test_sentinel2_zarr_v3_writing(self, sentinel2_dataset) -> None:
        """Test writing Sentinel-2 data to zarr v3 with FixedScaleOffset."""
        encoding = create_measurements_encoding(sentinel2_dataset, spatial_chunk=10)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "s2_test.zarr"

            # This would require xarray with zarr v3 support
            sentinel2_dataset.to_zarr(output_path, encoding=encoding, zarr_format=3, mode="w")

            # Verify zarr.json structure
            zarr_json_path = output_path / "b02" / "zarr.json"
            assert zarr_json_path.exists()

            with open(zarr_json_path) as f:
                zarr_metadata = json.load(f)

            # Check data type and codec structure
            assert zarr_metadata["data_type"] == "float64"

            # Look for FixedScaleOffset codec
            codecs = zarr_metadata.get("codecs", [])
            fixed_scale_codec = next(
                (c for c in codecs if c.get("name") == "numcodecs.fixedscaleoffset"), None
            )

            if fixed_scale_codec:
                config = fixed_scale_codec["configuration"]
                assert config["offset"] == -0.1
                assert config["scale"] == 10000.0
                assert config["dtype"] == "float64"
                assert config["astype"] == "uint16"
