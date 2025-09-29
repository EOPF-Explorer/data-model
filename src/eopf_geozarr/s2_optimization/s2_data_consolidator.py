"""
Data consolidation logic for reorganizing S2 structure.
"""

import xarray as xr
from typing import Dict, List, Tuple

class S2DataConsolidator:
    """Consolidates S2 data from scattered structure into organized groups."""
    
    def __init__(self, dt_input: xr.DataTree):
        self.dt_input = dt_input
        self.measurements_data = {}
        self.geometry_data = {}
        self.meteorology_data = {}
    
    def consolidate_all_data(self) -> Tuple[Dict, Dict, Dict]:
        """
        Consolidate all data into three main categories.
        
        Returns:
            Tuple of (measurements, geometry, meteorology) data dictionaries
        """
        self._extract_measurements_data()
        self._extract_geometry_data()
        self._extract_meteorology_data()
        
        return self.measurements_data, self.geometry_data, self.meteorology_data
    
    def _extract_measurements_data(self) -> None:
        """Extract and organize all measurement-related data by native resolution."""
        
        # Initialize resolution groups
        for resolution in [10, 20, 60]:
            self.measurements_data[resolution] = {
                'bands': {},
                'quality': {},
                'detector_footprints': {},
                'classification': {},
                'atmosphere': {},
                'probability': {}
            }
        
        # Extract reflectance bands
        if any('/measurements/reflectance' in group for group in self.dt_input.groups):
            self._extract_reflectance_bands()
        
        # Extract quality data
        self._extract_quality_data()
        
        # Extract detector footprints
        self._extract_detector_footprints()
        
        # Extract atmosphere quality
        self._extract_atmosphere_data()
        
        # Extract classification data
        self._extract_classification_data()
        
        # Extract probability data
        self._extract_probability_data()
    
    def _extract_reflectance_bands(self) -> None:
        """Extract reflectance bands from measurements/reflectance groups."""
        for resolution in ['r10m', 'r20m', 'r60m']:
            res_num = int(resolution[1:-1])  # Extract number from 'r10m'
            group_path = f'/measurements/reflectance/{resolution}'
            
            if group_path in self.dt_input.groups:
                # Check if this is a multiscale group (has numeric subgroups)
                group_node = self.dt_input[group_path]
                if hasattr(group_node, 'children') and group_node.children:
                    # Take level 0 (native resolution)
                    native_path = f'{group_path}/0'
                    if native_path in self.dt_input.groups:
                        ds = self.dt_input[native_path].to_dataset()
                    else:
                        ds = group_node.to_dataset()
                else:
                    ds = group_node.to_dataset()
                
                # Extract only native bands for this resolution
                for band in ds.data_vars:
                    self.measurements_data[res_num]['bands'][band] = ds[band]
    
    def _extract_quality_data(self) -> None:
        """Extract quality mask data."""
        quality_base = '/quality/mask'
        
        for resolution in ['r10m', 'r20m', 'r60m']:
            res_num = int(resolution[1:-1])
            group_path = f'{quality_base}/{resolution}'
            
            if group_path in self.dt_input.groups:
                ds = self.dt_input[group_path].to_dataset()
                
                for band in ds.data_vars:
                    self.measurements_data[res_num]['quality'][f'quality_{band}'] = ds[band]
    
    def _extract_detector_footprints(self) -> None:
        """Extract detector footprint data."""
        footprint_base = '/conditions/mask/detector_footprint'
        
        for resolution in ['r10m', 'r20m', 'r60m']:
            res_num = int(resolution[1:-1])
            group_path = f'{footprint_base}/{resolution}'
            
            if group_path in self.dt_input.groups:
                ds = self.dt_input[group_path].to_dataset()
                
                for band in ds.data_vars:
                    var_name = f'detector_footprint_{band}'
                    self.measurements_data[res_num]['detector_footprints'][var_name] = ds[band]
    
    def _extract_atmosphere_data(self) -> None:
        """Extract atmosphere quality data (aot, wvp) - native at 20m."""
        atm_base = '/quality/atmosphere'
        
        # Atmosphere data is native at 20m resolution
        group_path = f'{atm_base}/r20m'
        if group_path in self.dt_input.groups:
            ds = self.dt_input[group_path].to_dataset()
            
            for var in ['aot', 'wvp']:
                if var in ds.data_vars:
                    self.measurements_data[20]['atmosphere'][var] = ds[var]
    
    def _extract_classification_data(self) -> None:
        """Extract scene classification data - native at 20m."""
        class_base = '/conditions/mask/l2a_classification'
        
        # Classification is native at 20m
        group_path = f'{class_base}/r20m'
        if group_path in self.dt_input.groups:
            ds = self.dt_input[group_path].to_dataset()
            
            if 'scl' in ds.data_vars:
                self.measurements_data[20]['classification']['scl'] = ds['scl']
    
    def _extract_probability_data(self) -> None:
        """Extract cloud and snow probability data - native at 20m."""
        prob_base = '/quality/probability/r20m'
        
        if prob_base in self.dt_input.groups:
            ds = self.dt_input[prob_base].to_dataset()
            
            for var in ['cld', 'snw']:
                if var in ds.data_vars:
                    self.measurements_data[20]['probability'][var] = ds[var]
    
    def _extract_geometry_data(self) -> None:
        """Extract all geometry-related data into single group."""
        geom_base = '/conditions/geometry'
        
        if geom_base in self.dt_input.groups:
            ds = self.dt_input[geom_base].to_dataset()
            
            # Consolidate all geometry variables
            for var_name in ds.data_vars:
                self.geometry_data[var_name] = ds[var_name]
    
    def _extract_meteorology_data(self) -> None:
        """Extract meteorological data (CAMS and ECMWF)."""
        # CAMS data
        cams_path = '/conditions/meteorology/cams'
        if cams_path in self.dt_input.groups:
            ds = self.dt_input[cams_path].to_dataset()
            for var_name in ds.data_vars:
                self.meteorology_data[f'cams_{var_name}'] = ds[var_name]
        
        # ECMWF data
        ecmwf_path = '/conditions/meteorology/ecmwf'
        if ecmwf_path in self.dt_input.groups:
            ds = self.dt_input[ecmwf_path].to_dataset()
            for var_name in ds.data_vars:
                self.meteorology_data[f'ecmwf_{var_name}'] = ds[var_name]

def create_consolidated_dataset(data_dict: Dict, resolution: int) -> xr.Dataset:
    """
    Create a consolidated dataset from categorized data.
    
    Args:
        data_dict: Dictionary with categorized data
        resolution: Target resolution in meters
        
    Returns:
        Consolidated xarray Dataset
    """
    all_vars = {}
    
    # Combine all data variables
    for category, vars_dict in data_dict.items():
        all_vars.update(vars_dict)
    
    if not all_vars:
        return xr.Dataset()
    
    # Create dataset
    ds = xr.Dataset(all_vars)
    
    # Set up coordinate system and metadata
    if 'x' in ds.coords and 'y' in ds.coords:
        # Ensure CRS information is present
        if ds.rio.crs is None:
            # Try to infer CRS from one of the variables
            for var_name, var_data in all_vars.items():
                if hasattr(var_data, 'rio') and var_data.rio.crs:
                    ds.rio.write_crs(var_data.rio.crs, inplace=True)
                    break
    
    # Add resolution metadata
    ds.attrs['native_resolution_meters'] = resolution
    ds.attrs['processing_level'] = 'L2A'
    ds.attrs['product_type'] = 'S2MSI2A'
    
    return ds