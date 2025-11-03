"""
Tests for S2 multiscale pyramid creation with xy-aligned sharding.
"""

import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.s2_optimization.s2_multiscale import S2MultiscalePyramid


class TestS2MultiscalePyramid:
    """Test suite for S2MultiscalePyramid class."""

    @pytest.fixture
    def pyramid(self):
        """Create a basic S2MultiscalePyramid instance."""
        return S2MultiscalePyramid(enable_sharding=True, spatial_chunk=1024)

    @pytest.fixture
    def sample_dataset(self):
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

        dataset = xr.Dataset({"b02": b02, "b05": b05, "scl": scl})

        return dataset

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_init(self):
        """Test S2MultiscalePyramid initialization."""
        pyramid = S2MultiscalePyramid(enable_sharding=True, spatial_chunk=512)

        assert pyramid.enable_sharding is True
        assert pyramid.spatial_chunk == 512
        assert hasattr(pyramid, "resampler")
        assert len(pyramid.pyramid_levels) == 6
        assert pyramid.pyramid_levels[0] == 10
        assert pyramid.pyramid_levels[1] == 20
        assert pyramid.pyramid_levels[2] == 60
        assert pyramid.pyramid_levels[3] == 120
        assert pyramid.pyramid_levels[4] == 360
        assert pyramid.pyramid_levels[5] == 720

    def test_pyramid_levels_structure(self, pyramid):
        """Test the pyramid levels structure."""
        expected_levels = {
            0: 10,  # Level 0: 10m
            1: 20,  # Level 1: 20m
            2: 60,  # Level 2: 60m
            3: 120,  # Level 3: 120m
            4: 360,  # Level 4: 360m
            5: 720,  # Level 5: 720m
        }

        assert pyramid.pyramid_levels == expected_levels

    def test_calculate_simple_shard_dimensions(self, pyramid):
        """Test simplified shard dimensions calculation."""
        # Test 3D data (time, y, x) - shards are multiples of chunks
        data_shape = (5, 1024, 1024)
        chunks = (1, 256, 256)

        shard_dims = pyramid._calculate_simple_shard_dimensions(data_shape, chunks)

        assert len(shard_dims) == 3
        assert shard_dims[0] == 1  # Time dimension should be 1
        assert shard_dims[1] == 1024  # Y dimension matches exactly (divisible by 256)
        assert shard_dims[2] == 1024  # X dimension matches exactly (divisible by 256)

        # Test 2D data (y, x) with non-divisible dimensions
        data_shape = (1000, 1000)
        chunks = (256, 256)

        shard_dims = pyramid._calculate_simple_shard_dimensions(data_shape, chunks)

        assert len(shard_dims) == 2
        # Should use largest multiple of chunk_size that fits
        assert shard_dims[0] == 768  # 3 * 256 = 768 (largest multiple that fits in 1000)
        assert shard_dims[1] == 768  # 3 * 256 = 768

    def test_create_measurements_encoding(self, pyramid, sample_dataset):
        """Test measurements encoding creation with xy-aligned sharding."""
        encoding = pyramid._create_measurements_encoding(sample_dataset)

        # Check that encoding is created for all variables
        for var_name in sample_dataset.data_vars:
            assert var_name in encoding
            var_encoding = encoding[var_name]

            # Check basic encoding structure
            assert "chunks" in var_encoding
            # Zarr v3 uses 'compressors' (plural)
            assert "compressors" in var_encoding or "compressor" in var_encoding

            # Check sharding is included when enabled
            if pyramid.enable_sharding:
                assert "shards" in var_encoding

        # Check coordinate encoding
        for coord_name in sample_dataset.coords:
            if coord_name in encoding:
                # Coordinates may have either compressor or compressors set to None
                assert (
                    encoding[coord_name].get("compressor") is None
                    or encoding[coord_name].get("compressors") is None
                )

    def test_create_measurements_encoding_time_chunking(self, pyramid, sample_dataset):
        """Test that time dimension is chunked to 1 for single file per time."""
        encoding = pyramid._create_measurements_encoding(sample_dataset)

        for var_name in sample_dataset.data_vars:
            if sample_dataset[var_name].ndim == 3:  # 3D variable with time
                chunks = encoding[var_name]["chunks"]
                assert chunks[0] == 1  # Time dimension should be chunked to 1

    def test_calculate_aligned_chunk_size(self, pyramid):
        """Test aligned chunk size calculation."""
        # Test with spatial_chunk that divides evenly
        chunk_size = pyramid._calculate_aligned_chunk_size(1024, 256)
        assert chunk_size == 256

        # Test with spatial_chunk that doesn't divide evenly
        chunk_size = pyramid._calculate_aligned_chunk_size(1000, 256)
        # Should return a value that divides evenly into 1000
        assert 1000 % chunk_size == 0


class TestS2MultiscalePyramidIntegration:
    """Integration tests for S2MultiscalePyramid."""

    @pytest.fixture
    def simple_datatree(self):
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

    @patch("builtins.print")
    @patch.object(S2MultiscalePyramid, "_stream_write_dataset")
    def test_create_multiscale_from_datatree(
        self, mock_write, mock_print, simple_datatree, tmp_path
    ):
        """Test multiscale creation from DataTree."""
        pyramid = S2MultiscalePyramid(enable_sharding=True, spatial_chunk=256)

        output_path = str(tmp_path / "output.zarr")

        # Mock the write to avoid actual file I/O
        mock_write.return_value = xr.Dataset({"b02": xr.DataArray([1, 2, 3])})

        result = pyramid.create_multiscale_from_datatree(
            simple_datatree, output_path, verbose=False
        )

        # Should process groups
        assert isinstance(result, dict)
        # At minimum, should write the original group
        assert mock_write.call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__])
