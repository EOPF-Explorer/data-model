"""
Tests for S2 multiscale pyramid creation with xy-aligned sharding.
"""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
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
        time = np.array(['2023-01-01', '2023-01-02'], dtype='datetime64[ns]')
        
        # Create sample variables with different dimensions
        b02 = xr.DataArray(
            np.random.randint(0, 4000, (2, 100, 100)),
            dims=['time', 'y', 'x'],
            coords={'time': time, 'y': y, 'x': x},
            name='b02'
        )
        
        b05 = xr.DataArray(
            np.random.randint(0, 4000, (2, 100, 100)),
            dims=['time', 'y', 'x'],
            coords={'time': time, 'y': y, 'x': x},
            name='b05'
        )
        
        scl = xr.DataArray(
            np.random.randint(0, 11, (2, 100, 100)),
            dims=['time', 'y', 'x'],
            coords={'time': time, 'y': y, 'x': x},
            name='scl'
        )

        dataset = xr.Dataset({
            'b02': b02,
            'b05': b05,
            'scl': scl
        })
        
        return dataset

    @pytest.fixture
    def sample_measurements_by_resolution(self):
        """Create sample measurements organized by resolution."""
        x_10m = np.linspace(0, 1000, 200)
        y_10m = np.linspace(0, 1000, 200)
        x_20m = np.linspace(0, 1000, 100)
        y_20m = np.linspace(0, 1000, 100)
        x_60m = np.linspace(0, 1000, 50)
        y_60m = np.linspace(0, 1000, 50)
        time = np.array(['2023-01-01'], dtype='datetime64[ns]')

        # 10m data
        b02_10m = xr.DataArray(
            np.random.randint(0, 4000, (1, 200, 200)),
            dims=['time', 'y', 'x'],
            coords={'time': time, 'y': y_10m, 'x': x_10m},
            name='b02'
        )

        # 20m data
        b05_20m = xr.DataArray(
            np.random.randint(0, 4000, (1, 100, 100)),
            dims=['time', 'y', 'x'],
            coords={'time': time, 'y': y_20m, 'x': x_20m},
            name='b05'
        )

        scl_20m = xr.DataArray(
            np.random.randint(0, 11, (1, 100, 100)),
            dims=['time', 'y', 'x'],
            coords={'time': time, 'y': y_20m, 'x': x_20m},
            name='scl'
        )

        # 60m data
        b01_60m = xr.DataArray(
            np.random.randint(0, 4000, (1, 50, 50)),
            dims=['time', 'y', 'x'],
            coords={'time': time, 'y': y_60m, 'x': x_60m},
            name='b01'
        )

        return {
            10: {
                'reflectance': {'b02': b02_10m}
            },
            20: {
                'reflectance': {'b05': b05_20m},
                'quality': {'scl': scl_20m}
            },
            60: {
                'reflectance': {'b01': b01_60m}
            }
        }

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
        assert hasattr(pyramid, 'resampler')
        assert len(pyramid.pyramid_levels) == 7
        assert pyramid.pyramid_levels[0] == 10
        assert pyramid.pyramid_levels[1] == 20
        assert pyramid.pyramid_levels[2] == 60

    def test_pyramid_levels_structure(self, pyramid):
        """Test the pyramid levels structure."""
        expected_levels = {
            0: 10,    # Level 0: 10m
            1: 20,    # Level 1: 20m
            2: 60,    # Level 2: 60m
            3: 120,   # Level 3: 120m
            4: 240,   # Level 4: 240m
            5: 480,   # Level 5: 480m
            6: 960    # Level 6: 960m
        }
        
        assert pyramid.pyramid_levels == expected_levels

    def test_calculate_simple_shard_dimensions(self, pyramid):
        """Test simplified shard dimensions calculation."""
        # Test 3D data (time, y, x) - shards match dimensions exactly
        data_shape = (5, 1000, 1000)
        
        shard_dims = pyramid._calculate_simple_shard_dimensions(data_shape)
        
        assert len(shard_dims) == 3
        assert shard_dims[0] == 1     # Time dimension should be 1
        assert shard_dims[1] == 1000  # Y dimension matches exactly
        assert shard_dims[2] == 1000  # X dimension matches exactly
        
        # Test 2D data (y, x) - shards match dimensions exactly
        data_shape = (500, 800)
        
        shard_dims = pyramid._calculate_simple_shard_dimensions(data_shape)
        
        assert len(shard_dims) == 2
        assert shard_dims[0] == 500   # Y dimension matches exactly
        assert shard_dims[1] == 800   # X dimension matches exactly

    def test_create_level_encoding(self, pyramid, sample_dataset):
        """Test level encoding creation with xy-aligned sharding."""
        encoding = pyramid._create_level_encoding(sample_dataset, level=1)
        
        # Check that encoding is created for all variables
        for var_name in sample_dataset.data_vars:
            assert var_name in encoding
            var_encoding = encoding[var_name]
            
            # Check basic encoding structure
            assert 'chunks' in var_encoding
            assert 'compressor' in var_encoding
            
            # Check sharding is included when enabled
            if pyramid.enable_sharding:
                assert 'shards' in var_encoding
                
        # Check coordinate encoding
        for coord_name in sample_dataset.coords:
            if coord_name in encoding:
                assert encoding[coord_name]['compressor'] is None

    def test_create_level_encoding_time_chunking(self, pyramid, sample_dataset):
        """Test that time dimension is chunked to 1 for single file per time."""
        encoding = pyramid._create_level_encoding(sample_dataset, level=0)
        
        for var_name in sample_dataset.data_vars:
            if sample_dataset[var_name].ndim == 3:  # 3D variable with time
                chunks = encoding[var_name]['chunks']
                assert chunks[0] == 1  # Time dimension should be chunked to 1

    def test_should_separate_time_files(self, pyramid):
        """Test time file separation detection."""
        # Create dataset with multiple time points
        time = np.array(['2023-01-01', '2023-01-02'], dtype='datetime64[ns]')
        x = np.linspace(0, 100, 10)
        y = np.linspace(0, 100, 10)
        
        data_multi_time = xr.DataArray(
            np.random.rand(2, 10, 10),
            dims=['time', 'y', 'x'],
            coords={'time': time, 'y': y, 'x': x}
        )
        
        dataset_multi_time = xr.Dataset({'var1': data_multi_time})
        assert pyramid._should_separate_time_files(dataset_multi_time) is True
        
        # Create dataset with single time point
        data_single_time = xr.DataArray(
            np.random.rand(1, 10, 10),
            dims=['time', 'y', 'x'],
            coords={'time': time[:1], 'y': y, 'x': x}
        )
        
        dataset_single_time = xr.Dataset({'var1': data_single_time})
        assert pyramid._should_separate_time_files(dataset_single_time) is False
        
        # Create dataset with no time dimension
        data_no_time = xr.DataArray(
            np.random.rand(10, 10),
            dims=['y', 'x'],
            coords={'y': y, 'x': x}
        )
        
        dataset_no_time = xr.Dataset({'var1': data_no_time})
        assert pyramid._should_separate_time_files(dataset_no_time) is False

    def test_update_encoding_for_time_slice(self, pyramid):
        """Test encoding update for time slices."""
        # Original encoding with 3D chunks
        original_encoding = {
            'var1': {
                'chunks': (1, 100, 100),
                'shards': (1, 200, 200),
                'compressor': 'default'
            },
            'x': {'compressor': None},
            'y': {'compressor': None}
        }
        
        # Create a time slice dataset
        x = np.linspace(0, 100, 100)
        y = np.linspace(0, 100, 100)
        
        time_slice = xr.Dataset({
            'var1': xr.DataArray(
                np.random.rand(100, 100),
                dims=['y', 'x'],
                coords={'y': y, 'x': x}
            )
        })
        
        updated_encoding = pyramid._update_encoding_for_time_slice(original_encoding, time_slice)
        
        # Check that time dimension is removed from chunks and shards
        assert updated_encoding['var1']['chunks'] == (100, 100)
        assert updated_encoding['var1']['shards'] == (200, 200)
        assert updated_encoding['var1']['compressor'] == 'default'
        
        # Check coordinates are preserved
        assert updated_encoding['x']['compressor'] is None
        assert updated_encoding['y']['compressor'] is None

    @patch('builtins.print')
    @patch('xarray.Dataset.to_zarr')
    def test_write_level_dataset_no_time(self, mock_to_zarr, mock_print, pyramid, sample_dataset, temp_dir):
        """Test writing level dataset without time separation."""
        # Create dataset without multiple time points
        single_time_dataset = sample_dataset.isel(time=0)
        
        pyramid._write_level_dataset(single_time_dataset, temp_dir, level=0)
        
        # Should call to_zarr once (no time separation)
        mock_to_zarr.assert_called_once()
        args, kwargs = mock_to_zarr.call_args
        
        assert kwargs['mode'] == 'w'
        assert kwargs['consolidated'] is True
        assert kwargs['zarr_format'] == 3
        assert 'encoding' in kwargs

    @patch('builtins.print')
    def test_write_level_dataset_with_time_separation(self, mock_print, pyramid, sample_dataset, temp_dir):
        """Test writing level dataset with time separation."""
        with patch.object(pyramid, '_write_time_separated_dataset') as mock_time_sep:
            pyramid._write_level_dataset(sample_dataset, temp_dir, level=0)
            
            # Should call time separation method
            mock_time_sep.assert_called_once()

    def test_create_level_0_dataset(self, pyramid, sample_measurements_by_resolution):
        """Test level 0 dataset creation."""
        dataset = pyramid._create_level_0_dataset(sample_measurements_by_resolution)
        
        assert len(dataset.data_vars) > 0
        assert dataset.attrs['pyramid_level'] == 0
        assert dataset.attrs['resolution_meters'] == 10
        
        # Should only contain 10m native data
        assert 'b02' in dataset.data_vars

    def test_create_level_0_dataset_no_10m_data(self, pyramid):
        """Test level 0 dataset creation with no 10m data."""
        measurements_no_10m = {
            20: {'reflectance': {'b05': Mock()}},
            60: {'reflectance': {'b01': Mock()}}
        }
        
        dataset = pyramid._create_level_0_dataset(measurements_no_10m)
        assert len(dataset.data_vars) == 0

    @patch.object(S2MultiscalePyramid, '_create_level_0_dataset')
    @patch.object(S2MultiscalePyramid, '_create_level_1_dataset')
    @patch.object(S2MultiscalePyramid, '_create_level_2_dataset')
    @patch.object(S2MultiscalePyramid, '_create_downsampled_dataset')
    def test_create_level_dataset_routing(self, mock_downsampled, mock_level2, mock_level1, mock_level0, pyramid):
        """Test that _create_level_dataset routes to correct methods."""
        measurements = {}
        
        # Test level 0
        pyramid._create_level_dataset(0, 10, measurements)
        mock_level0.assert_called_once_with(measurements)
        
        # Test level 1
        pyramid._create_level_dataset(1, 20, measurements)
        mock_level1.assert_called_once_with(measurements)
        
        # Test level 2
        pyramid._create_level_dataset(2, 60, measurements)
        mock_level2.assert_called_once_with(measurements)
        
        # Test level 3+
        pyramid._create_level_dataset(3, 120, measurements)
        mock_downsampled.assert_called_once_with(3, 120, measurements)

    @patch('builtins.print')
    @patch.object(S2MultiscalePyramid, '_write_level_dataset')
    @patch.object(S2MultiscalePyramid, '_create_level_dataset')
    def test_create_multiscale_measurements(self, mock_create, mock_write, mock_print, pyramid, temp_dir):
        """Test multiscale measurements creation."""
        # Mock dataset creation
        mock_dataset = Mock()
        mock_dataset.data_vars = {'b02': Mock()}  # Non-empty dataset
        mock_create.return_value = mock_dataset
        
        measurements = {10: {'reflectance': {'b02': Mock()}}}
        
        result = pyramid.create_multiscale_measurements(measurements, temp_dir)
        
        # Should create all pyramid levels
        assert len(result) == len(pyramid.pyramid_levels)
        assert mock_create.call_count == len(pyramid.pyramid_levels)
        assert mock_write.call_count == len(pyramid.pyramid_levels)

    @patch('builtins.print')
    @patch.object(S2MultiscalePyramid, '_write_level_dataset')
    @patch.object(S2MultiscalePyramid, '_create_level_dataset')
    def test_create_multiscale_measurements_empty_dataset(self, mock_create, mock_write, mock_print, pyramid, temp_dir):
        """Test multiscale measurements creation with empty dataset."""
        # Mock empty dataset creation
        mock_dataset = Mock()
        mock_dataset.data_vars = {}  # Empty dataset
        mock_create.return_value = mock_dataset
        
        measurements = {}
        
        result = pyramid.create_multiscale_measurements(measurements, temp_dir)
        
        # Should not include empty datasets
        assert len(result) == 0
        assert mock_write.call_count == 0

    def test_create_level_1_dataset_with_downsampling(self, pyramid):
        """Test level 1 dataset creation with downsampling from 10m."""
        # Create mock measurements with both 10m and 20m data
        x_20m = np.linspace(0, 1000, 100)
        y_20m = np.linspace(0, 1000, 100)
        x_10m = np.linspace(0, 1000, 200)
        y_10m = np.linspace(0, 1000, 200)
        time = np.array(['2023-01-01'], dtype='datetime64[ns]')

        # 20m native data
        b05_20m = xr.DataArray(
            np.random.randint(0, 4000, (1, 100, 100)),
            dims=['time', 'y', 'x'],
            coords={'time': time, 'y': y_20m, 'x': x_20m},
            name='b05'
        )

        # 10m data to be downsampled
        b02_10m = xr.DataArray(
            np.random.randint(0, 4000, (1, 200, 200)),
            dims=['time', 'y', 'x'],
            coords={'time': time, 'y': y_10m, 'x': x_10m},
            name='b02'
        )

        measurements = {
            20: {'reflectance': {'b05': b05_20m}},
            10: {'reflectance': {'b02': b02_10m}}
        }

        with patch.object(pyramid.resampler, 'downsample_variable') as mock_downsample:
            # Mock the downsampling to return a properly shaped array
            mock_downsampled = xr.DataArray(
                np.random.randint(0, 4000, (1, 100, 100)),
                dims=['time', 'y', 'x'],
                coords={'time': time, 'y': y_20m, 'x': x_20m},
                name='b02'
            )
            mock_downsample.return_value = mock_downsampled

            dataset = pyramid._create_level_1_dataset(measurements)

            # Should call downsampling for 10m data
            mock_downsample.assert_called()
            
            # Should contain both native 20m and downsampled 10m data
            assert 'b05' in dataset.data_vars
            assert 'b02' in dataset.data_vars
            assert dataset.attrs['pyramid_level'] == 1
            assert dataset.attrs['resolution_meters'] == 20

    def test_create_level_2_dataset_structure(self, pyramid, sample_measurements_by_resolution):
        """Test level 2 dataset creation according to optimization plan."""
        dataset = pyramid._create_level_2_dataset(sample_measurements_by_resolution)
        
        # Check basic structure
        assert dataset.attrs['pyramid_level'] == 2
        assert dataset.attrs['resolution_meters'] == 60
        
        # Should contain 60m native data
        assert 'b01' in dataset.data_vars

    def test_create_level_2_dataset_with_downsampling(self, pyramid):
        """Test level 2 dataset creation with 20m data downsampling."""
        # Create measurements with 60m and 20m data
        x_60m = np.linspace(0, 1000, 50)
        y_60m = np.linspace(0, 1000, 50)
        x_20m = np.linspace(0, 1000, 100)
        y_20m = np.linspace(0, 1000, 100)
        time = np.array(['2023-01-01'], dtype='datetime64[ns]')

        # 60m native data
        b01_60m = xr.DataArray(
            np.random.randint(0, 4000, (1, 50, 50)),
            dims=['time', 'y', 'x'],
            coords={'time': time, 'y': y_60m, 'x': x_60m},
            name='b01'
        )

        # 20m data to be downsampled
        scl_20m = xr.DataArray(
            np.random.randint(0, 11, (1, 100, 100)),
            dims=['time', 'y', 'x'],
            coords={'time': time, 'y': y_20m, 'x': x_20m},
            name='scl'
        )

        measurements = {
            60: {'reflectance': {'b01': b01_60m}},
            20: {'quality': {'scl': scl_20m}}
        }

        with patch.object(pyramid.resampler, 'downsample_variable') as mock_downsample:
            # Mock the downsampling to return a properly shaped array
            mock_downsampled = xr.DataArray(
                np.random.randint(0, 11, (1, 50, 50)),
                dims=['time', 'y', 'x'],
                coords={'time': time, 'y': y_60m, 'x': x_60m},
                name='scl'
            )
            mock_downsample.return_value = mock_downsampled

            dataset = pyramid._create_level_2_dataset(measurements)

            # Should call downsampling for 20m data
            mock_downsample.assert_called()
            
            # Should contain both native 60m and downsampled 20m data
            assert 'b01' in dataset.data_vars
            assert 'scl' in dataset.data_vars
            assert dataset.attrs['pyramid_level'] == 2
            assert dataset.attrs['resolution_meters'] == 60

    def test_error_handling_invalid_level(self, pyramid):
        """Test error handling for invalid pyramid levels."""
        measurements = {}
        
        # Test with invalid level (should work but return empty dataset if no source data)
        dataset = pyramid._create_level_dataset(-1, 5, measurements)
        # Should create downsampled dataset (empty in this case)
        assert isinstance(dataset, xr.Dataset)


class TestS2MultiscalePyramidIntegration:
    """Integration tests for S2MultiscalePyramid."""

    @pytest.fixture
    def real_measurements_data(self):
        """Create realistic measurements data for integration testing."""
        time = np.array(['2023-06-15T10:30:00'], dtype='datetime64[ns]')
        
        # 10m resolution data (200x200 pixels)
        x_10m = np.linspace(300000, 310000, 200)  # UTM coordinates
        y_10m = np.linspace(4900000, 4910000, 200)
        
        # 20m resolution data (100x100 pixels)  
        x_20m = np.linspace(300000, 310000, 100)
        y_20m = np.linspace(4900000, 4910000, 100)
        
        # 60m resolution data (50x50 pixels)
        x_60m = np.linspace(300000, 310000, 50)
        y_60m = np.linspace(4900000, 4910000, 50)

        # Create realistic spectral bands
        measurements = {
            10: {
                'reflectance': {
                    'b02': xr.DataArray(
                        np.random.randint(500, 3000, (1, 200, 200), dtype=np.int16),
                        dims=['time', 'y', 'x'],
                        coords={'time': time, 'y': y_10m, 'x': x_10m},
                        attrs={'long_name': 'Blue band', 'units': 'digital_number'}
                    ),
                    'b03': xr.DataArray(
                        np.random.randint(600, 3500, (1, 200, 200), dtype=np.int16),
                        dims=['time', 'y', 'x'],
                        coords={'time': time, 'y': y_10m, 'x': x_10m},
                        attrs={'long_name': 'Green band', 'units': 'digital_number'}
                    ),
                    'b04': xr.DataArray(
                        np.random.randint(400, 3200, (1, 200, 200), dtype=np.int16),
                        dims=['time', 'y', 'x'],
                        coords={'time': time, 'y': y_10m, 'x': x_10m},
                        attrs={'long_name': 'Red band', 'units': 'digital_number'}
                    ),
                    'b08': xr.DataArray(
                        np.random.randint(3000, 6000, (1, 200, 200), dtype=np.int16),
                        dims=['time', 'y', 'x'],
                        coords={'time': time, 'y': y_10m, 'x': x_10m},
                        attrs={'long_name': 'NIR band', 'units': 'digital_number'}
                    )
                }
            },
            20: {
                'reflectance': {
                    'b05': xr.DataArray(
                        np.random.randint(2000, 4000, (1, 100, 100), dtype=np.int16),
                        dims=['time', 'y', 'x'],
                        coords={'time': time, 'y': y_20m, 'x': x_20m},
                        attrs={'long_name': 'Red edge 1', 'units': 'digital_number'}
                    ),
                    'b06': xr.DataArray(
                        np.random.randint(2500, 4500, (1, 100, 100), dtype=np.int16),
                        dims=['time', 'y', 'x'],
                        coords={'time': time, 'y': y_20m, 'x': x_20m},
                        attrs={'long_name': 'Red edge 2', 'units': 'digital_number'}
                    ),
                    'b07': xr.DataArray(
                        np.random.randint(2800, 4800, (1, 100, 100), dtype=np.int16),
                        dims=['time', 'y', 'x'],
                        coords={'time': time, 'y': y_20m, 'x': x_20m},
                        attrs={'long_name': 'Red edge 3', 'units': 'digital_number'}
                    ),
                    'b11': xr.DataArray(
                        np.random.randint(1000, 3000, (1, 100, 100), dtype=np.int16),
                        dims=['time', 'y', 'x'],
                        coords={'time': time, 'y': y_20m, 'x': x_20m},
                        attrs={'long_name': 'SWIR 1', 'units': 'digital_number'}
                    ),
                    'b12': xr.DataArray(
                        np.random.randint(500, 2500, (1, 100, 100), dtype=np.int16),
                        dims=['time', 'y', 'x'],
                        coords={'time': time, 'y': y_20m, 'x': x_20m},
                        attrs={'long_name': 'SWIR 2', 'units': 'digital_number'}
                    ),
                    'b8a': xr.DataArray(
                        np.random.randint(2800, 5500, (1, 100, 100), dtype=np.int16),
                        dims=['time', 'y', 'x'],
                        coords={'time': time, 'y': y_20m, 'x': x_20m},
                        attrs={'long_name': 'NIR narrow', 'units': 'digital_number'}
                    )
                },
                'quality': {
                    'scl': xr.DataArray(
                        np.random.randint(0, 11, (1, 100, 100), dtype=np.uint8),
                        dims=['time', 'y', 'x'],
                        coords={'time': time, 'y': y_20m, 'x': x_20m},
                        attrs={'long_name': 'Scene classification', 'units': 'class'}
                    ),
                    'aot': xr.DataArray(
                        np.random.randint(0, 1000, (1, 100, 100), dtype=np.uint16),
                        dims=['time', 'y', 'x'],
                        coords={'time': time, 'y': y_20m, 'x': x_20m},
                        attrs={'long_name': 'Aerosol optical thickness', 'units': 'dimensionless'}
                    ),
                    'wvp': xr.DataArray(
                        np.random.randint(0, 5000, (1, 100, 100), dtype=np.uint16),
                        dims=['time', 'y', 'x'],
                        coords={'time': time, 'y': y_20m, 'x': x_20m},
                        attrs={'long_name': 'Water vapor', 'units': 'kg/m^2'}
                    )
                }
            },
            60: {
                'reflectance': {
                    'b01': xr.DataArray(
                        np.random.randint(1500, 3500, (1, 50, 50), dtype=np.int16),
                        dims=['time', 'y', 'x'],
                        coords={'time': time, 'y': y_60m, 'x': x_60m},
                        attrs={'long_name': 'Coastal aerosol', 'units': 'digital_number'}
                    ),
                    'b09': xr.DataArray(
                        np.random.randint(100, 1000, (1, 50, 50), dtype=np.int16),
                        dims=['time', 'y', 'x'],
                        coords={'time': time, 'y': y_60m, 'x': x_60m},
                        attrs={'long_name': 'Water vapor', 'units': 'digital_number'}
                    )
                }
            }
        }
        
        return measurements

    @patch('builtins.print')  # Mock print to avoid test output
    def test_full_pyramid_creation(self, mock_print, real_measurements_data, tmp_path):
        """Test complete pyramid creation with realistic data."""
        pyramid = S2MultiscalePyramid(enable_sharding=True, spatial_chunk=512)
        
        output_path = str(tmp_path)
        
        with patch.object(pyramid, '_write_level_dataset') as mock_write:
            result = pyramid.create_multiscale_measurements(real_measurements_data, output_path)
            
            # Should create all 7 pyramid levels
            assert len(result) == 7
            
            # Check that each level has appropriate characteristics
            for level, dataset in result.items():
                assert dataset.attrs['pyramid_level'] == level
                assert dataset.attrs['resolution_meters'] == pyramid.pyramid_levels[level]
                assert len(dataset.data_vars) > 0
                
            # Verify write was called for each level
            assert mock_write.call_count == 7

    def test_level_specific_content(self, real_measurements_data):
        """Test that each pyramid level contains appropriate content."""
        pyramid = S2MultiscalePyramid(enable_sharding=False, spatial_chunk=256)  # Disable sharding for simpler testing
        
        # Test level 0 (10m native)
        level_0 = pyramid._create_level_0_dataset(real_measurements_data)
        level_0_vars = set(level_0.data_vars.keys())
        expected_10m_vars = {'b02', 'b03', 'b04', 'b08'}
        assert len(expected_10m_vars.intersection(level_0_vars)) > 0
        
        # Test level 1 (20m consolidated)
        level_1 = pyramid._create_level_1_dataset(real_measurements_data)
        # Should contain both native 20m and downsampled 10m data
        level_1_vars = set(level_1.data_vars.keys())
        # Check some expected variables are present
        expected_vars = {'b05', 'b06', 'b07', 'b11', 'b12', 'b8a', 'scl', 'aot', 'wvp'}
        assert len(expected_vars.intersection(level_1_vars)) > 0
        
        # Test level 2 (60m consolidated)  
        level_2 = pyramid._create_level_2_dataset(real_measurements_data)
        # Should contain native 60m and processed 20m data
        level_2_vars = set(level_2.data_vars.keys())
        expected_60m_vars = {'b01', 'b09'}
        assert len(expected_60m_vars.intersection(level_2_vars)) > 0

    def test_sharding_configuration_integration(self, real_measurements_data):
        """Test sharding configuration with realistic data."""
        pyramid = S2MultiscalePyramid(enable_sharding=True, spatial_chunk=256)
        
        # Create a test dataset
        level_0 = pyramid._create_level_0_dataset(real_measurements_data)
        
        if len(level_0.data_vars) > 0:
            encoding = pyramid._create_level_encoding(level_0, level=0)
            
            # Check encoding structure
            for var_name, var_data in level_0.data_vars.items():
                assert var_name in encoding
                var_encoding = encoding[var_name]
                
                # Check sharding configuration
                if var_data.ndim >= 2:
                    assert 'shards' in var_encoding
                    shards = var_encoding['shards']
                    
                    # Verify shard dimensions are reasonable
                    if var_data.ndim == 3:
                        assert shards[0] == 1  # Time dimension
                        assert shards[1] > 0   # Y dimension
                        assert shards[2] > 0   # X dimension
                    elif var_data.ndim == 2:
                        assert shards[0] > 0   # Y dimension
                        assert shards[1] > 0   # X dimension


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_measurements_data(self):
        """Test handling of empty measurements data."""
        pyramid = S2MultiscalePyramid()
        
        empty_measurements = {}
        
        with patch('builtins.print'):
            with patch.object(pyramid, '_write_level_dataset'):
                result = pyramid.create_multiscale_measurements(empty_measurements, "/tmp")
                
                # Should return empty results
                assert len(result) == 0

    def test_missing_resolution_data(self):
        """Test handling when specific resolution data is missing."""
        pyramid = S2MultiscalePyramid()
        
        # Only provide 20m data, missing 10m and 60m
        measurements_partial = {
            20: {
                'reflectance': {
                    'b05': xr.DataArray(
                        np.random.rand(1, 50, 50),
                        dims=['time', 'y', 'x'],
                        coords={
                            'time': ['2023-01-01'],
                            'y': np.arange(50),
                            'x': np.arange(50)
                        }
                    )
                }
            }
        }
        
        # Should handle gracefully
        level_0 = pyramid._create_level_0_dataset(measurements_partial)
        assert len(level_0.data_vars) == 0  # No 10m data available
        
        level_1 = pyramid._create_level_1_dataset(measurements_partial)
        assert len(level_1.data_vars) > 0  # Should have 20m data

    def test_coordinate_preservation(self):
        """Test that coordinate systems are preserved through processing."""
        pyramid = S2MultiscalePyramid()
        
        # Create data with specific coordinate attributes
        x = np.linspace(300000, 310000, 100)
        y = np.linspace(4900000, 4910000, 100)
        time = np.array(['2023-01-01'], dtype='datetime64[ns]')
        
        # Add coordinate attributes
        x_coord = xr.DataArray(x, dims=['x'], attrs={'units': 'm', 'crs': 'EPSG:32633'})
        y_coord = xr.DataArray(y, dims=['y'], attrs={'units': 'm', 'crs': 'EPSG:32633'})
        time_coord = xr.DataArray(time, dims=['time'], attrs={'calendar': 'gregorian'})
        
        test_data = xr.DataArray(
            np.random.rand(1, 100, 100),
            dims=['time', 'y', 'x'],
            coords={'time': time_coord, 'y': y_coord, 'x': x_coord},
            name='b05'
        )
        
        measurements = {
            20: {'reflectance': {'b05': test_data}}
        }
        
        dataset = pyramid._create_level_1_dataset(measurements)
        
        # Check that coordinate attributes are preserved
        if 'b05' in dataset.data_vars:
            assert 'x' in dataset.coords
            assert 'y' in dataset.coords
            assert 'time' in dataset.coords
            
            # Check coordinate attributes preservation
            assert dataset.coords['x'].attrs.get('units') == 'm'
            assert dataset.coords['y'].attrs.get('units') == 'm'
