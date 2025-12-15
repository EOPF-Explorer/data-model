"""
Tests for S2 multiscale pyramid creation with xy-aligned sharding.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from structlog.testing import capture_logs

from eopf_geozarr.s2_optimization.s2_multiscale import (
    calculate_aligned_chunk_size,
    calculate_simple_shard_dimensions,
    create_downsampled_resolution_group,
    create_measurements_encoding,
    create_multiscale_from_datatree,
)


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    """Create a sample xarray dataset for testing."""
    x = np.linspace(0, 1000, 100)
    y = np.linspace(0, 1000, 100)
    time = np.array(["2023-01-01", "2023-01-02"], dtype="datetime64[ns]")

    # Create sample variables with different dimensions
    b02 = xr.DataArray(
        np.random.randint(0, 4000, (2, 100, 100)),
        dims=["time", "y", "x"],
        coords={"time": time, "y": y, "x": x},
        name="b02",
    )

    b05 = xr.DataArray(
        np.random.randint(0, 4000, (2, 100, 100)),
        dims=["time", "y", "x"],
        coords={"time": time, "y": y, "x": x},
        name="b05",
    )

    scl = xr.DataArray(
        np.random.randint(0, 11, (2, 100, 100)),
        dims=["time", "y", "x"],
        coords={"time": time, "y": y, "x": x},
        name="scl",
    )

    return xr.Dataset({"b02": b02, "b05": b05, "scl": scl})


class TestS2MultiscaleFunctions:
    """Test suite for S2 multiscale functions."""

    def test_create_downsampled_resolution_group_quality_mask(self) -> None:
        """Quality-mask downsampling should not crash and should preserve dtype."""
        x = np.arange(8)
        y = np.arange(6)
        quality = xr.DataArray(
            np.random.randint(0, 2, (6, 8), dtype=np.uint8),
            dims=["y", "x"],
            coords={"y": y, "x": x},
            name="quality_clouds",
        )
        ds = xr.Dataset({"quality_clouds": quality})

        out = create_downsampled_resolution_group(ds, factor=2)

        assert "quality_clouds" in out.data_vars
        assert out["quality_clouds"].dtype == np.uint8
        assert out["quality_clouds"].shape == (3, 4)

    def test_calculate_simple_shard_dimensions(self) -> None:
        """Test simplified shard dimensions calculation."""
        # Test 3D data (time, y, x) - shards are multiples of chunks
        data_shape = (5, 1024, 1024)
        chunks = (1, 256, 256)

        shard_dims = calculate_simple_shard_dimensions(data_shape, chunks)

        assert len(shard_dims) == 3
        assert shard_dims[0] == 1  # Time dimension should be 1
        assert shard_dims[1] == 1024  # Y dimension matches exactly (divisible by 256)
        assert shard_dims[2] == 1024  # X dimension matches exactly (divisible by 256)

        # Test 2D data (y, x) with non-divisible dimensions
        data_shape = (1000, 1000)
        chunks = (256, 256)

        shard_dims = calculate_simple_shard_dimensions(data_shape, chunks)

        assert len(shard_dims) == 2
        # Should use largest multiple of chunk_size that fits
        assert shard_dims[0] == 768  # 3 * 256 = 768 (largest multiple that fits in 1000)
        assert shard_dims[1] == 768  # 3 * 256 = 768

    def test_create_measurements_encoding(self, sample_dataset: xr.Dataset) -> None:
        """Test measurements encoding creation with xy-aligned sharding."""
        encoding = create_measurements_encoding(
            sample_dataset, enable_sharding=True, spatial_chunk=1024
        )

        # Check that encoding is created for all variables
        for var_name in sample_dataset.data_vars:
            assert var_name in encoding
            var_encoding = encoding[var_name]

            # Check basic encoding structure
            assert "chunks" in var_encoding
            # Zarr v3 uses 'compressors' (plural)
            assert "compressors" in var_encoding or "compressor" in var_encoding

            # Check sharding is included when enabled
            assert "shards" in var_encoding

        # Check coordinate encoding
        for coord_name in sample_dataset.coords:
            if coord_name in encoding:
                # Coordinates may have either compressor or compressors set to None
                assert (
                    encoding[coord_name].get("compressor") is None
                    or encoding[coord_name].get("compressors") is None
                )

    def test_create_measurements_encoding_time_chunking(self, sample_dataset: xr.Dataset) -> None:
        """Test that time dimension is chunked to 1 for single file per time."""
        encoding = create_measurements_encoding(
            sample_dataset, enable_sharding=True, spatial_chunk=1024
        )

        for var_name in sample_dataset.data_vars:
            if sample_dataset[var_name].ndim == 3:  # 3D variable with time
                chunks = encoding[var_name]["chunks"]
                assert chunks[0] == 1  # Time dimension should be chunked to 1

    def test_calculate_aligned_chunk_size(self) -> None:
        """Test aligned chunk size calculation."""
        # Test with spatial_chunk that divides evenly
        chunk_size = calculate_aligned_chunk_size(1024, 256)
        assert chunk_size == 256

        # Test with spatial_chunk that doesn't divide evenly
        chunk_size = calculate_aligned_chunk_size(1000, 256)
        # Should return a value that divides evenly into 1000
        assert 1000 % chunk_size == 0


class TestS2MultiscaleIntegration:
    """Integration tests for S2 multiscale functions."""

    @pytest.fixture
    def simple_datatree(self) -> xr.DataTree:
        """Create a simple DataTree for integration testing."""
        # Create sample data
        x = np.linspace(0, 1000, 100)
        y = np.linspace(0, 1000, 100)
        time = np.array(["2023-01-01"], dtype="datetime64[ns]")

        # Create sample band
        b02 = xr.DataArray(
            np.random.randint(0, 4000, (1, 100, 100)),
            dims=["time", "y", "x"],
            coords={"time": time, "y": y, "x": x},
            name="b02",
            attrs={"long_name": "Blue band", "units": "digital_number"},
        )

        # Create dataset
        ds = xr.Dataset({"b02": b02})

        # Create DataTree
        dt = xr.DataTree(name="root")
        dt["/measurements/reflectance/r10m"] = xr.DataTree(ds)

        return dt

    @patch("eopf_geozarr.s2_optimization.s2_multiscale.add_multiscales_metadata_to_parent")
    @patch("eopf_geozarr.s2_optimization.s2_multiscale.stream_write_dataset")
    def test_create_multiscale_from_datatree(
        self,
        mock_write: Any,
        mock_add_multiscales: Any,
        simple_datatree: xr.DataTree,
        tmp_path: Path,
    ) -> None:
        """Test multiscale creation from DataTree."""
        output_path = str(tmp_path / "output.zarr")

        # Mock the write to avoid actual file I/O
        mock_write.return_value = xr.Dataset({"b02": xr.DataArray([1, 2, 3])})
        mock_add_multiscales.return_value = xr.DataTree()

        # Capture log output using structlog's testing context manager
        with capture_logs() as cap_logs:
            result = create_multiscale_from_datatree(
                simple_datatree,
                output_path,
                enable_sharding=True,
                spatial_chunk=256,
            )

        # Should process groups
        assert isinstance(result, dict)
        # At minimum, should write the original group
        assert mock_write.call_count >= 1

        # Optionally verify log messages (cap_logs contains all logged events)
        # With verbose=False, there won't be many logs, but we can check the structure
        assert isinstance(cap_logs, list)

    @patch("eopf_geozarr.s2_optimization.s2_multiscale.add_multiscales_metadata_to_parent")
    @patch("eopf_geozarr.s2_optimization.s2_multiscale.stream_write_dataset")
    def test_create_multiscale_from_datatree_raises_when_multiscales_metadata_missing(
        self,
        mock_write: Any,
        mock_add_multiscales: Any,
        simple_datatree: xr.DataTree,
        tmp_path: Path,
    ) -> None:
        output_path = str(tmp_path / "output.zarr")

        mock_write.return_value = xr.Dataset({"b02": xr.DataArray([1, 2, 3])})
        mock_add_multiscales.return_value = None

        result = create_multiscale_from_datatree(
            simple_datatree,
            output_path,
            enable_sharding=True,
            spatial_chunk=256,
        )

        assert "/measurements/reflectance" not in result


if __name__ == "__main__":
    pytest.main([__file__])
