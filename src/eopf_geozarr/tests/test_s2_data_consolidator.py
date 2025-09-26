"""Tests for S2 data consolidator module."""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Tuple, Any

from eopf_geozarr.s2_optimization.s2_data_consolidator import (
    S2DataConsolidator,
    create_consolidated_dataset,
)


class TestS2DataConsolidator:
    """Test S2DataConsolidator class."""
    
    @pytest.fixture
    def sample_s2_datatree(self):
        """Create a sample S2 DataTree structure for testing."""
        # Create coordinate arrays for different resolutions
        x_10m = np.linspace(100000, 200000, 1098)
        y_10m = np.linspace(5000000, 5100000, 1098)
        x_20m = x_10m[::2]  # 549 points
        y_20m = y_10m[::2]
        x_60m = x_10m[::6]  # 183 points
        y_60m = y_10m[::6]
        time = np.array(['2023-01-15'], dtype='datetime64[ns]')
        
        # Create sample data arrays
        data_10m = np.random.randint(0, 10000, (1, 1098, 1098), dtype=np.uint16)
        data_20m = np.random.randint(0, 10000, (1, 549, 549), dtype=np.uint16)
        data_60m = np.random.randint(0, 10000, (1, 183, 183), dtype=np.uint16)
        
        # Create datasets for different resolution groups (using lowercase band names)
        ds_10m = xr.Dataset({
            'b02': (['time', 'y', 'x'], data_10m),
            'b03': (['time', 'y', 'x'], data_10m.copy()),
            'b04': (['time', 'y', 'x'], data_10m.copy()),
            'b08': (['time', 'y', 'x'], data_10m.copy()),
        }, coords={'time': time, 'x': x_10m, 'y': y_10m})
        
        ds_20m = xr.Dataset({
            'b05': (['time', 'y', 'x'], data_20m),
            'b06': (['time', 'y', 'x'], data_20m.copy()),
            'b07': (['time', 'y', 'x'], data_20m.copy()),
            'b8a': (['time', 'y', 'x'], data_20m.copy()),
            'b11': (['time', 'y', 'x'], data_20m.copy()),
            'b12': (['time', 'y', 'x'], data_20m.copy()),
            'aot': (['time', 'y', 'x'], data_20m.copy()),  # atmosphere
            'wvp': (['time', 'y', 'x'], data_20m.copy()),
            'scl': (['time', 'y', 'x'], data_20m.copy()),  # classification
            'cld': (['time', 'y', 'x'], data_20m.copy()),  # probability
            'snw': (['time', 'y', 'x'], data_20m.copy()),
        }, coords={'time': time, 'x': x_20m, 'y': y_20m})
        
        ds_60m = xr.Dataset({
            'b01': (['time', 'y', 'x'], data_60m),
            'b09': (['time', 'y', 'x'], data_60m.copy()),
        }, coords={'time': time, 'x': x_60m, 'y': y_60m})
        
        # Create quality datasets (using lowercase band names)
        quality_10m = xr.Dataset({
            'b02': (['time', 'y', 'x'], np.random.randint(0, 2, (1, 1098, 1098), dtype=np.uint8)),
            'b03': (['time', 'y', 'x'], np.random.randint(0, 2, (1, 1098, 1098), dtype=np.uint8)),
            'b04': (['time', 'y', 'x'], np.random.randint(0, 2, (1, 1098, 1098), dtype=np.uint8)),
            'b08': (['time', 'y', 'x'], np.random.randint(0, 2, (1, 1098, 1098), dtype=np.uint8)),
        }, coords={'time': time, 'x': x_10m, 'y': y_10m})
        
        # Create detector footprint datasets (using lowercase band names)
        detector_10m = xr.Dataset({
            'b02': (['time', 'y', 'x'], np.random.randint(0, 13, (1, 1098, 1098), dtype=np.uint8)),
            'b03': (['time', 'y', 'x'], np.random.randint(0, 13, (1, 1098, 1098), dtype=np.uint8)),
            'b04': (['time', 'y', 'x'], np.random.randint(0, 13, (1, 1098, 1098), dtype=np.uint8)),
            'b08': (['time', 'y', 'x'], np.random.randint(0, 13, (1, 1098, 1098), dtype=np.uint8)),
        }, coords={'time': time, 'x': x_10m, 'y': y_10m})
        
        # Create geometry data
        geometry_ds = xr.Dataset({
            'solar_zenith_angle': (['time', 'y', 'x'], np.random.uniform(0, 90, (1, 549, 549))),
            'solar_azimuth_angle': (['time', 'y', 'x'], np.random.uniform(0, 360, (1, 549, 549))),
            'view_zenith_angle': (['time', 'y', 'x'], np.random.uniform(0, 90, (1, 549, 549))),
            'view_azimuth_angle': (['time', 'y', 'x'], np.random.uniform(0, 360, (1, 549, 549))),
        }, coords={'time': time, 'x': x_20m, 'y': y_20m})
        
        # Create meteorology data
        cams_ds = xr.Dataset({
            'total_ozone': (['time', 'y', 'x'], np.random.uniform(200, 400, (1, 183, 183))),
            'relative_humidity': (['time', 'y', 'x'], np.random.uniform(0, 100, (1, 183, 183))),
        }, coords={'time': time, 'x': x_60m, 'y': y_60m})
        
        ecmwf_ds = xr.Dataset({
            'temperature': (['time', 'y', 'x'], np.random.uniform(250, 320, (1, 183, 183))),
            'pressure': (['time', 'y', 'x'], np.random.uniform(950, 1050, (1, 183, 183))),
        }, coords={'time': time, 'x': x_60m, 'y': y_60m})
        
        # Build the mock DataTree structure
        mock_dt = MagicMock()
        mock_dt.groups = {
            '/measurements/reflectance/r10m': Mock(),
            '/measurements/reflectance/r20m': Mock(), 
            '/measurements/reflectance/r60m': Mock(),
            '/quality/mask/r10m': Mock(),
            '/quality/mask/r20m': Mock(),
            '/quality/mask/r60m': Mock(),
            '/conditions/mask/detector_footprint/r10m': Mock(),
            '/conditions/mask/detector_footprint/r20m': Mock(),
            '/conditions/mask/detector_footprint/r60m': Mock(),
            '/quality/atmosphere/r20m': Mock(),
            '/conditions/mask/l2a_classification/r20m': Mock(),
            '/quality/probability/r20m': Mock(),
            '/conditions/geometry': Mock(),
            '/conditions/meteorology/cams': Mock(),
            '/conditions/meteorology/ecmwf': Mock(),
        }
        
        # Mock the dataset access
        def mock_getitem(self, path):
            mock_node = MagicMock()
            if 'r10m' in path:
                if 'reflectance' in path:
                    mock_node.to_dataset.return_value = ds_10m
                elif 'quality/mask' in path:
                    mock_node.to_dataset.return_value = quality_10m
                elif 'detector_footprint' in path:
                    mock_node.to_dataset.return_value = detector_10m
            elif 'r20m' in path:
                if 'reflectance' in path:
                    mock_node.to_dataset.return_value = ds_20m
                elif 'atmosphere' in path:
                    mock_node.to_dataset.return_value = ds_20m[['aot', 'wvp']]
                elif 'classification' in path:
                    mock_node.to_dataset.return_value = ds_20m[['scl']]
                elif 'probability' in path:
                    mock_node.to_dataset.return_value = ds_20m[['cld', 'snw']]
            elif 'r60m' in path:
                if 'reflectance' in path:
                    mock_node.to_dataset.return_value = ds_60m
            elif 'geometry' in path:
                mock_node.to_dataset.return_value = geometry_ds
            elif 'cams' in path:
                mock_node.to_dataset.return_value = cams_ds
            elif 'ecmwf' in path:
                mock_node.to_dataset.return_value = ecmwf_ds
            
            return mock_node
        
        mock_dt.__getitem__ = mock_getitem
        return mock_dt
    
    def test_init(self, sample_s2_datatree):
        """Test consolidator initialization."""
        consolidator = S2DataConsolidator(sample_s2_datatree)
        
        assert consolidator.dt_input == sample_s2_datatree
        assert consolidator.measurements_data == {}
        assert consolidator.geometry_data == {}
        assert consolidator.meteorology_data == {}
    
    def test_consolidate_all_data(self, sample_s2_datatree):
        """Test complete data consolidation."""
        consolidator = S2DataConsolidator(sample_s2_datatree)
        measurements, geometry, meteorology = consolidator.consolidate_all_data()
        
        # Check that all three categories are returned
        assert isinstance(measurements, dict)
        assert isinstance(geometry, dict)
        assert isinstance(meteorology, dict)
        
        # Check resolution groups in measurements
        assert 10 in measurements
        assert 20 in measurements
        assert 60 in measurements
        
        # Check data categories exist
        for resolution in [10, 20, 60]:
            assert 'bands' in measurements[resolution]
            assert 'quality' in measurements[resolution]
            assert 'detector_footprints' in measurements[resolution]
            assert 'classification' in measurements[resolution]
            assert 'atmosphere' in measurements[resolution]
            assert 'probability' in measurements[resolution]
    
    def test_extract_reflectance_bands(self, sample_s2_datatree):
        """Test reflectance band extraction."""
        consolidator = S2DataConsolidator(sample_s2_datatree)
        consolidator._extract_measurements_data()
        
        # Check 10m bands
        assert 'b02' in consolidator.measurements_data[10]['bands']
        assert 'b03' in consolidator.measurements_data[10]['bands']
        assert 'b04' in consolidator.measurements_data[10]['bands']
        assert 'b08' in consolidator.measurements_data[10]['bands']
        
        # Check 20m bands  
        assert 'b05' in consolidator.measurements_data[20]['bands']
        assert 'b06' in consolidator.measurements_data[20]['bands']
        assert 'b11' in consolidator.measurements_data[20]['bands']
        assert 'b12' in consolidator.measurements_data[20]['bands']
        
        # Check 60m bands
        assert 'b01' in consolidator.measurements_data[60]['bands']
        assert 'b09' in consolidator.measurements_data[60]['bands']
    
    def test_extract_quality_data(self, sample_s2_datatree):
        """Test quality data extraction."""
        consolidator = S2DataConsolidator(sample_s2_datatree)
        consolidator._extract_measurements_data()
        
        # Check quality data exists for native bands
        assert 'quality_b02' in consolidator.measurements_data[10]['quality']
        assert 'quality_b03' in consolidator.measurements_data[10]['quality']
    
    def test_extract_detector_footprints(self, sample_s2_datatree):
        """Test detector footprint extraction."""
        consolidator = S2DataConsolidator(sample_s2_datatree)
        consolidator._extract_measurements_data()
        
        # Check detector footprint data
        assert 'detector_footprint_b02' in consolidator.measurements_data[10]['detector_footprints']
        assert 'detector_footprint_b03' in consolidator.measurements_data[10]['detector_footprints']
    
    def test_extract_atmosphere_data(self, sample_s2_datatree):
        """Test atmosphere data extraction."""
        consolidator = S2DataConsolidator(sample_s2_datatree)
        consolidator._extract_measurements_data()
        
        # Atmosphere data should be at 20m resolution
        assert 'aot' in consolidator.measurements_data[20]['atmosphere']
        assert 'wvp' in consolidator.measurements_data[20]['atmosphere']
    
    def test_extract_classification_data(self, sample_s2_datatree):
        """Test classification data extraction."""
        consolidator = S2DataConsolidator(sample_s2_datatree)
        consolidator._extract_measurements_data()
        
        # Classification should be at 20m resolution
        assert 'scl' in consolidator.measurements_data[20]['classification']
    
    def test_extract_probability_data(self, sample_s2_datatree):
        """Test probability data extraction."""
        consolidator = S2DataConsolidator(sample_s2_datatree)
        consolidator._extract_measurements_data()
        
        # Probability data should be at 20m resolution
        assert 'cld' in consolidator.measurements_data[20]['probability']
        assert 'snw' in consolidator.measurements_data[20]['probability']
    
    def test_extract_geometry_data(self, sample_s2_datatree):
        """Test geometry data extraction."""
        consolidator = S2DataConsolidator(sample_s2_datatree)
        consolidator._extract_geometry_data()
        
        # Check that geometry variables are extracted
        assert 'solar_zenith_angle' in consolidator.geometry_data
        assert 'solar_azimuth_angle' in consolidator.geometry_data
        assert 'view_zenith_angle' in consolidator.geometry_data
        assert 'view_azimuth_angle' in consolidator.geometry_data
    
    def test_extract_meteorology_data(self, sample_s2_datatree):
        """Test meteorology data extraction."""
        consolidator = S2DataConsolidator(sample_s2_datatree)
        consolidator._extract_meteorology_data()
        
        # Check CAMS data
        assert 'cams_total_ozone' in consolidator.meteorology_data
        assert 'cams_relative_humidity' in consolidator.meteorology_data
        
        # Check ECMWF data
        assert 'ecmwf_temperature' in consolidator.meteorology_data
        assert 'ecmwf_pressure' in consolidator.meteorology_data
    
    def test_missing_groups_handling(self):
        """Test handling of missing data groups."""
        # Create DataTree with missing groups
        mock_dt = MagicMock()
        mock_dt.groups = {}  # No groups present
        
        consolidator = S2DataConsolidator(mock_dt)
        measurements, geometry, meteorology = consolidator.consolidate_all_data()
        
        # Should handle missing groups gracefully
        assert isinstance(measurements, dict)
        assert isinstance(geometry, dict)
        assert isinstance(meteorology, dict)
        
        # Data structures should be initialized but empty
        for resolution in [10, 20, 60]:
            assert resolution in measurements
            for category in ['bands', 'quality', 'detector_footprints', 'classification', 'atmosphere', 'probability']:
                assert category in measurements[resolution]
                assert len(measurements[resolution][category]) == 0


class TestCreateConsolidatedDataset:
    """Test the create_consolidated_dataset function."""
    
    @pytest.fixture
    def sample_data_dict(self):
        """Create sample consolidated data dictionary."""
        # Create coordinate arrays
        x = np.linspace(100000, 200000, 100)
        y = np.linspace(5000000, 5100000, 100)
        time = np.array(['2023-01-15'], dtype='datetime64[ns]')
        
        # Create sample data arrays
        data = np.random.randint(0, 10000, (1, 100, 100), dtype=np.uint16)
        
        return {
            'bands': {
                'b02': xr.DataArray(data, dims=['time', 'y', 'x'], 
                                   coords={'time': time, 'x': x, 'y': y}),
                'b03': xr.DataArray(data.copy(), dims=['time', 'y', 'x'], 
                                   coords={'time': time, 'x': x, 'y': y}),
            },
            'quality': {
                'quality_b02': xr.DataArray(np.random.randint(0, 2, (1, 100, 100), dtype=np.uint8),
                                          dims=['time', 'y', 'x'], 
                                          coords={'time': time, 'x': x, 'y': y}),
            },
            'atmosphere': {
                'aot': xr.DataArray(np.random.uniform(0.1, 0.5, (1, 100, 100)),
                                  dims=['time', 'y', 'x'], 
                                  coords={'time': time, 'x': x, 'y': y}),
            }
        }
    
    def test_create_consolidated_dataset_success(self, sample_data_dict):
        """Test successful dataset creation."""
        ds = create_consolidated_dataset(sample_data_dict, resolution=10)
        
        assert isinstance(ds, xr.Dataset)
        
        # Check that all variables are included
        expected_vars = {'b02', 'b03', 'quality_b02', 'aot'}
        assert set(ds.data_vars.keys()) == expected_vars
        
        # Check metadata
        assert ds.attrs['native_resolution_meters'] == 10
        assert ds.attrs['processing_level'] == 'L2A'
        assert ds.attrs['product_type'] == 'S2MSI2A'
        
        # Check coordinates
        assert 'x' in ds.coords
        assert 'y' in ds.coords
        assert 'time' in ds.coords
    
    def test_create_consolidated_dataset_empty_data(self):
        """Test dataset creation with empty data."""
        empty_data_dict = {'bands': {}, 'quality': {}, 'atmosphere': {}}
        ds = create_consolidated_dataset(empty_data_dict, resolution=20)
        
        # Should return empty dataset
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) == 0
    
    def test_create_consolidated_dataset_with_crs(self, sample_data_dict):
        """Test dataset creation with CRS information."""
        # Add CRS to one of the data arrays
        sample_data_dict['bands']['b02'] = sample_data_dict['bands']['b02'].rio.write_crs('EPSG:32632')
        
        ds = create_consolidated_dataset(sample_data_dict, resolution=10)
        
        assert isinstance(ds, xr.Dataset)
        # Check that CRS is propagated (assuming rio accessor is available)
        if hasattr(ds, 'rio'):
            assert ds.rio.crs is not None


class TestIntegration:
    """Integration tests combining consolidator and dataset creation."""
    
    @pytest.fixture
    def complete_s2_datatree(self):
        """Create a complete S2 DataTree for integration testing."""
        # This would be similar to the fixture in TestS2DataConsolidator
        # but with all data present for end-to-end testing
        x_10m = np.linspace(100000, 200000, 100)
        y_10m = np.linspace(5000000, 5100000, 100)
        x_20m = x_10m[::2]
        y_20m = y_10m[::2]
        time = np.array(['2023-01-15'], dtype='datetime64[ns]')
        
        # Create complete mock DataTree (simplified for integration test)
        mock_dt = MagicMock()
        mock_dt.groups = {
            '/measurements/reflectance/r10m': Mock(),
            '/conditions/geometry': Mock(),
            '/conditions/meteorology/cams': Mock(),
        }
        
        # Mock datasets
        reflectance_10m = xr.Dataset({
            'b02': (['time', 'y', 'x'], np.random.randint(0, 10000, (1, 100, 100), dtype=np.uint16)),
            'b03': (['time', 'y', 'x'], np.random.randint(0, 10000, (1, 100, 100), dtype=np.uint16)),
        }, coords={'time': time, 'x': x_10m, 'y': y_10m})
        
        geometry_ds = xr.Dataset({
            'solar_zenith_angle': (['time', 'y', 'x'], np.random.uniform(0, 90, (1, 50, 50))),
        }, coords={'time': time, 'x': x_20m, 'y': y_20m})
        
        cams_ds = xr.Dataset({
            'total_ozone': (['time', 'y', 'x'], np.random.uniform(200, 400, (1, 50, 50))),
        }, coords={'time': time, 'x': x_20m, 'y': y_20m})
        
        def mock_getitem(self, path):
            mock_node = MagicMock()
            if '/measurements/reflectance/r10m' in path:
                mock_node.to_dataset.return_value = reflectance_10m
            elif '/conditions/geometry' in path:
                mock_node.to_dataset.return_value = geometry_ds
            elif '/conditions/meteorology/cams' in path:
                mock_node.to_dataset.return_value = cams_ds
            return mock_node
        
        mock_dt.__getitem__ = mock_getitem
        return mock_dt
    
    def test_end_to_end_consolidation(self, complete_s2_datatree):
        """Test complete end-to-end consolidation and dataset creation."""
        # Step 1: Consolidate data
        consolidator = S2DataConsolidator(complete_s2_datatree)
        measurements, geometry, meteorology = consolidator.consolidate_all_data()
        
        # Step 2: Create consolidated datasets for each resolution
        consolidated_datasets = {}
        for resolution in [10, 20, 60]:
            if measurements[resolution]:  # Only create if data exists
                ds = create_consolidated_dataset(measurements[resolution], resolution)
                if len(ds.data_vars) > 0:  # Only keep non-empty datasets
                    consolidated_datasets[resolution] = ds
        
        # Step 3: Verify results
        assert len(consolidated_datasets) > 0
        
        # Check that 10m data is present (from our mock)
        if 10 in consolidated_datasets:
            ds_10m = consolidated_datasets[10]
            assert 'b02' in ds_10m.data_vars
            assert 'b03' in ds_10m.data_vars
            assert ds_10m.attrs['native_resolution_meters'] == 10
        
        # Verify geometry data
        assert len(geometry) > 0
        geometry_ds = create_consolidated_dataset({'geometry': geometry}, resolution=20)
        if len(geometry_ds.data_vars) > 0:
            assert 'solar_zenith_angle' in geometry_ds.data_vars
        
        # Verify meteorology data  
        assert len(meteorology) > 0
        met_ds = create_consolidated_dataset({'meteorology': meteorology}, resolution=60)
        if len(met_ds.data_vars) > 0:
            assert 'cams_total_ozone' in met_ds.data_vars


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_create_dataset_with_inconsistent_coordinates(self):
        """Test dataset creation with inconsistent coordinate systems."""
        # Create data with mismatched coordinates
        x1 = np.linspace(100000, 200000, 50)
        y1 = np.linspace(5000000, 5100000, 50) 
        x2 = np.linspace(100000, 200000, 100)  # Different size
        y2 = np.linspace(5000000, 5100000, 100)
        time = np.array(['2023-01-15'], dtype='datetime64[ns]')
        
        inconsistent_data = {
            'bands': {
                'b02': xr.DataArray(np.random.randint(0, 10000, (1, 50, 50), dtype=np.uint16),
                                   dims=['time', 'y', 'x'], 
                                   coords={'time': time, 'x': x1, 'y': y1}),
                'b03': xr.DataArray(np.random.randint(0, 10000, (1, 100, 100), dtype=np.uint16),
                                   dims=['time', 'y', 'x'], 
                                   coords={'time': time, 'x': x2, 'y': y2}),
            }
        }
        
        # Should handle inconsistent coordinates gracefully or raise appropriate error
        # The exact behavior depends on xarray's handling of mixed coordinates
        try:
            ds = create_consolidated_dataset(inconsistent_data, resolution=10)
            # If successful, verify it's a valid dataset
            assert isinstance(ds, xr.Dataset)
        except (ValueError, KeyError) as e:
            # Expected error due to coordinate mismatch
            assert "coordinate" in str(e).lower() or "dimension" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__])
