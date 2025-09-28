"""
Unit tests for simplified S2 converter implementation.

Tests the new simplified approach that uses only xarray and proper metadata consolidation.
"""

import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import xarray as xr
import zarr
from rasterio.crs import CRS

from eopf_geozarr.s2_optimization.s2_converter import S2OptimizedConverter


@pytest.fixture
def mock_s2_dataset():
    """Create a mock S2 dataset for testing."""
    # Create test data arrays
    coords = {
        'x': (['x'], np.linspace(0, 1000, 100)),
        'y': (['y'], np.linspace(0, 1000, 100)),
        'time': (['time'], [np.datetime64('2023-01-01')])
    }
    
    # Create test variables
    data_vars = {
        'b02': (['time', 'y', 'x'], np.random.rand(1, 100, 100)),
        'b03': (['time', 'y', 'x'], np.random.rand(1, 100, 100)),
        'b04': (['time', 'y', 'x'], np.random.rand(1, 100, 100)),
    }
    
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Add rioxarray CRS
    ds = ds.rio.write_crs('EPSG:32632')
    
    # Create datatree
    dt = xr.DataTree(ds)
    dt.attrs = {
        'stac_discovery': {
            'properties': {
                'mission': 'sentinel-2'
            }
        }
    }
    
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
            enable_sharding=True,
            spatial_chunk=512,
            compression_level=5,
            max_retries=2
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
        dt.attrs = {
            'stac_discovery': {
                'properties': {
                    'mission': 'sentinel-2a'
                }
            }
        }
        
        assert converter._is_sentinel2_dataset(dt) is True
        
        # Test with non-S2 mission
        dt.attrs['stac_discovery']['properties']['mission'] = 'sentinel-1'
        assert converter._is_sentinel2_dataset(dt) is False
    
    def test_is_sentinel2_dataset_with_groups(self):
        """Test S2 detection via characteristic groups."""
        converter = S2OptimizedConverter()
        
        dt = xr.DataTree()
        dt.attrs = {}
        
        # Mock groups property using patch
        with patch.object(type(dt), 'groups', new_callable=lambda: property(lambda self: [
            '/measurements/reflectance',
            '/conditions/geometry',
            '/quality/atmosphere'
        ])):
            assert converter._is_sentinel2_dataset(dt) is True
        
        # Test with insufficient indicators
        with patch.object(type(dt), 'groups', new_callable=lambda: property(lambda self: ['/measurements/reflectance'])):
            assert converter._is_sentinel2_dataset(dt) is False


class TestMultiscalesMetadata:
    """Test multiscales metadata creation."""
    
    def test_create_multiscales_metadata_with_rio(self, temp_output_dir):
        """Test multiscales metadata creation using rioxarray."""
        converter = S2OptimizedConverter()
        
        # Create mock pyramid datasets with rioxarray
        pyramid_datasets = {}
        for level in [0, 1, 2]:
            # Create test dataset
            coords = {
                'x': (['x'], np.linspace(0, 1000, 100 // (2**level))),
                'y': (['y'], np.linspace(0, 1000, 100 // (2**level)))
            }
            data_vars = {
                'b02': (['y', 'x'], np.random.rand(100 // (2**level), 100 // (2**level)))
            }
            ds = xr.Dataset(data_vars, coords=coords)
            ds = ds.rio.write_crs('EPSG:32632')
            
            pyramid_datasets[level] = ds
        
        # Test metadata creation
        metadata = converter._create_multiscales_metadata_with_rio(pyramid_datasets)
        
        # Verify structure matches geozarr.py format
        assert 'tile_matrix_set' in metadata
        assert 'resampling_method' in metadata
        assert 'tile_matrix_limits' in metadata
        assert metadata['resampling_method'] == 'average'
        
        # Verify tile matrix set structure
        tms = metadata['tile_matrix_set']
        assert 'id' in tms
        assert 'crs' in tms
        assert 'tileMatrices' in tms
        assert len(tms['tileMatrices']) == 3  # 3 levels
    
    def test_create_multiscales_metadata_no_datasets(self):
        """Test metadata creation with no datasets."""
        converter = S2OptimizedConverter()
        
        metadata = converter._create_multiscales_metadata_with_rio({})
        assert metadata == {}
    
    def test_create_multiscales_metadata_no_crs(self):
        """Test metadata creation with datasets lacking CRS."""
        converter = S2OptimizedConverter()
        
        # Create dataset without CRS
        ds = xr.Dataset({'b02': (['y', 'x'], np.random.rand(10, 10))})
        pyramid_datasets = {0: ds}
        
        metadata = converter._create_multiscales_metadata_with_rio(pyramid_datasets)
        assert metadata == {}


class TestAuxiliaryGroupWriting:
    """Test auxiliary group writing functionality."""
    
    @patch('eopf_geozarr.s2_optimization.s2_converter.distributed')
    def test_write_auxiliary_group_with_distributed(self, mock_distributed, temp_output_dir):
        """Test auxiliary group writing with distributed available."""
        converter = S2OptimizedConverter()
        
        # Create test dataset
        data_vars = {
            'solar_zenith': (['y', 'x'], np.random.rand(50, 50)),
            'solar_azimuth': (['y', 'x'], np.random.rand(50, 50))
        }
        coords = {
            'x': (['x'], np.linspace(0, 1000, 50)),
            'y': (['y'], np.linspace(0, 1000, 50))
        }
        dataset = xr.Dataset(data_vars, coords=coords)
        
        group_path = os.path.join(temp_output_dir, 'geometry')
        
        # Mock distributed progress
        mock_progress = Mock()
        mock_distributed.progress = mock_progress
        
        # Test writing
        converter._write_auxiliary_group(dataset, group_path, 'geometry', verbose=True)
        
        # Verify zarr group was created
        assert os.path.exists(group_path)
        
        # Verify group can be opened
        zarr_group = zarr.open_group(group_path, mode='r')
        assert 'solar_zenith' in zarr_group
        assert 'solar_azimuth' in zarr_group
    
    def test_write_auxiliary_group_without_distributed(self, temp_output_dir):
        """Test auxiliary group writing without distributed."""
        converter = S2OptimizedConverter()
        
        # Create test dataset
        data_vars = {
            'temperature': (['y', 'x'], np.random.rand(30, 30)),
            'pressure': (['y', 'x'], np.random.rand(30, 30))
        }
        coords = {
            'x': (['x'], np.linspace(0, 1000, 30)),
            'y': (['y'], np.linspace(0, 1000, 30))
        }
        dataset = xr.Dataset(data_vars, coords=coords)
        
        group_path = os.path.join(temp_output_dir, 'meteorology')
        
        # Patch DISTRIBUTED_AVAILABLE to False
        with patch('eopf_geozarr.s2_optimization.s2_converter.DISTRIBUTED_AVAILABLE', False):
            converter._write_auxiliary_group(dataset, group_path, 'meteorology', verbose=False)
        
        # Verify zarr group was created
        assert os.path.exists(group_path)
        
        # Verify group can be opened
        zarr_group = zarr.open_group(group_path, mode='r')
        assert 'temperature' in zarr_group
        assert 'pressure' in zarr_group


class TestMetadataConsolidation:
    """Test metadata consolidation functionality."""
    
    def test_add_measurements_multiscales_metadata(self, temp_output_dir):
        """Test adding multiscales metadata to measurements group."""
        converter = S2OptimizedConverter()
        
        # Create measurements group structure
        measurements_path = os.path.join(temp_output_dir, 'measurements')
        os.makedirs(measurements_path)
        
        # Create a minimal zarr group
        zarr_group = zarr.open_group(measurements_path, mode='w')
        zarr_group.attrs['test'] = 'value'
        
        # Create mock pyramid datasets
        pyramid_datasets = {}
        for level in [0, 1]:
            coords = {
                'x': (['x'], np.linspace(0, 1000, 50 // (2**level))),
                'y': (['y'], np.linspace(0, 1000, 50 // (2**level)))
            }
            data_vars = {
                'b02': (['y', 'x'], np.random.rand(50 // (2**level), 50 // (2**level)))
            }
            ds = xr.Dataset(data_vars, coords=coords)
            ds = ds.rio.write_crs('EPSG:32632')
            pyramid_datasets[level] = ds
        
        # Test adding metadata
        converter._add_measurements_multiscales_metadata(temp_output_dir, pyramid_datasets)
        
        # Verify metadata was added
        zarr_group = zarr.open_group(measurements_path, mode='r')
        assert 'multiscales' in zarr_group.attrs
        
        multiscales = zarr_group.attrs['multiscales']
        assert 'tile_matrix_set' in multiscales
        assert 'resampling_method' in multiscales
        assert 'tile_matrix_limits' in multiscales
    
    def test_add_measurements_multiscales_metadata_error_handling(self, temp_output_dir):
        """Test error handling in multiscales metadata addition."""
        converter = S2OptimizedConverter()
        
        # Test with non-existent measurements path
        converter._add_measurements_multiscales_metadata(temp_output_dir, {})
        
        # Should not raise an exception, just print warnings
        # (We can't easily test print output in unit tests, but the method should handle errors gracefully)
    
    @patch('xarray.open_zarr')
    def test_simple_root_consolidation_success(self, mock_open_zarr, temp_output_dir):
        """Test successful root consolidation with xarray."""
        converter = S2OptimizedConverter()
        
        # Mock successful xarray consolidation
        mock_ds = Mock()
        mock_open_zarr.return_value.__enter__.return_value = mock_ds
        
        converter._simple_root_consolidation(temp_output_dir, {})
        
        # Verify xarray.open_zarr was called with correct parameters
        mock_open_zarr.assert_called_once()
        args, kwargs = mock_open_zarr.call_args
        assert args[0] == temp_output_dir
        assert kwargs['consolidated'] is True
        assert kwargs['chunks'] == {}
    
    @patch('zarr.consolidate_metadata')
    @patch('xarray.open_zarr')
    def test_simple_root_consolidation_fallback(self, mock_open_zarr, mock_consolidate, temp_output_dir):
        """Test fallback to zarr consolidation when xarray fails."""
        converter = S2OptimizedConverter()
        
        # Mock xarray failure
        mock_open_zarr.side_effect = Exception("xarray failed")
        
        converter._simple_root_consolidation(temp_output_dir, {})
        
        # Verify fallback to zarr.consolidate_metadata
        mock_consolidate.assert_called_once()


class TestEndToEndSimplified:
    """Test simplified end-to-end functionality with mocks."""
    
    @patch('eopf_geozarr.s2_optimization.s2_converter.S2DataConsolidator')
    @patch('eopf_geozarr.s2_optimization.s2_converter.S2MultiscalePyramid')
    @patch('eopf_geozarr.s2_optimization.s2_converter.S2OptimizationValidator')
    def test_convert_s2_optimized_simplified_flow(self, mock_validator, mock_pyramid, mock_consolidator, 
                                                  mock_s2_dataset, temp_output_dir):
        """Test the simplified conversion flow with all major components mocked."""
        converter = S2OptimizedConverter()
        
        # Mock consolidator
        mock_consolidator_instance = Mock()
        mock_consolidator.return_value = mock_consolidator_instance
        mock_consolidator_instance.consolidate_all_data.return_value = (
            {10: {'bands': {'b02': Mock(), 'b03': Mock()}}},  # measurements
            {'solar_zenith': Mock()},  # geometry  
            {'temperature': Mock()}  # meteorology
        )
        
        # Mock pyramid creator
        mock_pyramid_instance = Mock()
        mock_pyramid.return_value = mock_pyramid_instance
        
        # Create mock pyramid datasets with rioxarray
        pyramid_datasets = {}
        for level in [0, 1]:
            coords = {
                'x': (['x'], np.linspace(0, 1000, 50 // (2**level))),
                'y': (['y'], np.linspace(0, 1000, 50 // (2**level)))
            }
            data_vars = {
                'b02': (['y', 'x'], np.random.rand(50 // (2**level), 50 // (2**level)))
            }
            ds = xr.Dataset(data_vars, coords=coords)
            ds = ds.rio.write_crs('EPSG:32632')
            pyramid_datasets[level] = ds
        
        mock_pyramid_instance.create_multiscale_measurements.return_value = pyramid_datasets
        
        # Mock validator
        mock_validator_instance = Mock()
        mock_validator.return_value = mock_validator_instance
        mock_validator_instance.validate_optimized_dataset.return_value = {
            'is_valid': True,
            'issues': []
        }
        
        # Mock the multiscales metadata methods
        with patch.object(converter, '_add_measurements_multiscales_metadata') as mock_add_metadata, \
             patch.object(converter, '_simple_root_consolidation') as mock_consolidation, \
             patch.object(converter, '_write_auxiliary_group') as mock_write_aux, \
             patch.object(converter, '_create_result_datatree') as mock_create_result:
            
            mock_create_result.return_value = xr.DataTree()
            
            # Run conversion
            result = converter.convert_s2_optimized(
                mock_s2_dataset,
                temp_output_dir,
                create_geometry_group=True,
                create_meteorology_group=True,
                validate_output=True,
                verbose=True
            )
            
            # Verify all steps were called
            mock_consolidator_instance.consolidate_all_data.assert_called_once()
            mock_pyramid_instance.create_multiscale_measurements.assert_called_once()
            mock_write_aux.assert_called()  # Should be called twice (geometry + meteorology)
            mock_add_metadata.assert_called_once_with(temp_output_dir, pyramid_datasets)
            mock_consolidation.assert_called_once_with(temp_output_dir, pyramid_datasets)
            mock_validator_instance.validate_optimized_dataset.assert_called_once()
            
            assert result is not None


class TestConvenienceFunction:
    """Test the convenience function."""
    
    @patch('eopf_geozarr.s2_optimization.s2_converter.S2OptimizedConverter')
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
            verbose=True
        )
        
        # Verify constructor was called with correct args
        mock_converter_class.assert_called_once_with(
            enable_sharding=False,
            spatial_chunk=512,
            compression_level=2,
            max_retries=5
        )
        
        # Verify method was called with remaining args
        mock_converter_instance.convert_s2_optimized.assert_called_once_with(
            dt_input,
            output_path,
            create_geometry_group=False,
            validate_output=False,
            verbose=True
        )


if __name__ == '__main__':
    pytest.main([__file__])
