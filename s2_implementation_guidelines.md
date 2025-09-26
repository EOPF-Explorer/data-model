# Sentinel-2 Optimization - Detailed Implementation Guidelines

## Module Architecture

### Core Files to Create

```
eopf_geozarr/s2_optimization/
├── __init__.py
├── s2_converter.py          # Main conversion orchestration
├── s2_band_mapping.py       # Band resolution and availability mapping
├── s2_resampling.py         # Downsampling operations only
├── s2_data_consolidator.py  # Data reorganization logic
├── s2_multiscale.py         # Multiscale pyramid creation
├── s2_validation.py         # Validation and integrity checks
└── cli_integration.py       # CLI command integration
```

## Implementation Specifications

### 1. s2_band_mapping.py

```python
"""
Band mapping and resolution definitions for Sentinel-2 optimization.
"""

from typing import Dict, List, Set
from dataclasses import dataclass

@dataclass
class BandInfo:
    """Information about a spectral band."""
    name: str
    native_resolution: int  # meters
    data_type: str
    wavelength_center: float  # nanometers
    wavelength_width: float   # nanometers

# Native resolution definitions - CRITICAL: Only these bands exist at these resolutions
NATIVE_BANDS: Dict[int, List[str]] = {
    10: ['b02', 'b03', 'b04', 'b08'],        # Blue, Green, Red, NIR
    20: ['b05', 'b06', 'b07', 'b11', 'b12', 'b8a'],  # Red Edge, SWIR
    60: ['b01', 'b09']                        # Coastal, Water Vapor
}

# Complete band information
BAND_INFO: Dict[str, BandInfo] = {
    'b01': BandInfo('b01', 60, 'uint16', 443, 21),   # Coastal aerosol
    'b02': BandInfo('b02', 10, 'uint16', 490, 66),   # Blue
    'b03': BandInfo('b03', 10, 'uint16', 560, 36),   # Green
    'b04': BandInfo('b04', 10, 'uint16', 665, 31),   # Red
    'b05': BandInfo('b05', 20, 'uint16', 705, 15),   # Red Edge 1
    'b06': BandInfo('b06', 20, 'uint16', 740, 15),   # Red Edge 2
    'b07': BandInfo('b07', 20, 'uint16', 783, 20),   # Red Edge 3
    'b08': BandInfo('b08', 10, 'uint16', 842, 106),  # NIR
    'b8a': BandInfo('b8a', 20, 'uint16', 865, 21),   # NIR Narrow
    'b09': BandInfo('b09', 60, 'uint16', 945, 20),   # Water Vapor
    'b11': BandInfo('b11', 20, 'uint16', 1614, 91),  # SWIR 1
    'b12': BandInfo('b12', 20, 'uint16', 2202, 175), # SWIR 2
}

# Quality data mapping - defines which auxiliary data exists at which resolutions
QUALITY_DATA_NATIVE: Dict[str, int] = {
    'scl': 20,      # Scene Classification Layer - native 20m
    'aot': 20,      # Aerosol Optical Thickness - native 20m
    'wvp': 20,      # Water Vapor - native 20m
    'cld': 20,      # Cloud probability - native 20m
    'snw': 20,      # Snow probability - native 20m
}

# Detector footprint availability - matches spectral bands
DETECTOR_FOOTPRINT_NATIVE: Dict[int, List[str]] = {
    10: ['b02', 'b03', 'b04', 'b08'],
    20: ['b05', 'b06', 'b07', 'b11', 'b12', 'b8a'],
    60: ['b01', 'b09']
}

def get_bands_for_level(level: int) -> Set[str]:
    """
    Get all bands available at a given pyramid level.
    
    Args:
        level: Pyramid level (0=10m, 1=20m, 2=60m, 3+=downsampled)
    
    Returns:
        Set of band names available at this level
    """
    if level == 0:  # 10m - only native 10m bands
        return set(NATIVE_BANDS[10])
    elif level == 1:  # 20m - all bands (native + downsampled from 10m)
        return set(NATIVE_BANDS[10] + NATIVE_BANDS[20] + NATIVE_BANDS[60])
    elif level == 2:  # 60m - all bands downsampled
        return set(NATIVE_BANDS[10] + NATIVE_BANDS[20] + NATIVE_BANDS[60])
    else:  # Further downsampling - all bands
        return set(NATIVE_BANDS[10] + NATIVE_BANDS[20] + NATIVE_BANDS[60])

def get_quality_data_for_level(level: int) -> Set[str]:
    """Get quality data available at a given level (no upsampling)."""
    if level == 0:  # 10m - no quality data (would require upsampling)
        return set()
    elif level >= 1:  # 20m and below - all quality data available
        return set(QUALITY_DATA_NATIVE.keys())
```

### 2. s2_resampling.py

```python
"""
Downsampling operations for Sentinel-2 data (no upsampling).
"""

import numpy as np
import xarray as xr
from scipy.ndimage import zoom
from sklearn.preprocessing import mode
import warnings

class S2ResamplingEngine:
    """Handles downsampling operations for S2 multiscale creation."""
    
    def __init__(self):
        self.resampling_methods = {
            'reflectance': self._downsample_reflectance,
            'classification': self._downsample_classification,
            'quality_mask': self._downsample_quality_mask,
            'probability': self._downsample_probability,
            'detector_footprint': self._downsample_quality_mask,  # Same as quality mask
        }
    
    def downsample_variable(self, data: xr.DataArray, target_height: int, 
                          target_width: int, var_type: str) -> xr.DataArray:
        """
        Downsample a variable to target dimensions.
        
        Args:
            data: Input data array
            target_height: Target height in pixels
            target_width: Target width in pixels  
            var_type: Type of variable ('reflectance', 'classification', etc.)
        
        Returns:
            Downsampled data array
        """
        if var_type not in self.resampling_methods:
            raise ValueError(f"Unknown variable type: {var_type}")
        
        method = self.resampling_methods[var_type]
        return method(data, target_height, target_width)
    
    def _downsample_reflectance(self, data: xr.DataArray, target_height: int, 
                              target_width: int) -> xr.DataArray:
        """Block averaging for reflectance bands."""
        # Calculate block sizes
        current_height, current_width = data.shape[-2:]
        block_h = current_height // target_height
        block_w = current_width // target_width
        
        # Ensure exact divisibility
        if current_height % target_height != 0 or current_width % target_width != 0:
            # Crop to make it divisible
            new_height = (current_height // block_h) * block_h
            new_width = (current_width // block_w) * block_w
            data = data[..., :new_height, :new_width]
        
        # Perform block averaging
        if data.ndim == 3:  # (time, y, x) or similar
            reshaped = data.values.reshape(
                data.shape[0], target_height, block_h, target_width, block_w
            )
            downsampled = reshaped.mean(axis=(2, 4))
        else:  # (y, x)
            reshaped = data.values.reshape(target_height, block_h, target_width, block_w)
            downsampled = reshaped.mean(axis=(1, 3))
        
        # Create new coordinates
        y_coords = data.coords[data.dims[-2]][::block_h][:target_height]
        x_coords = data.coords[data.dims[-1]][::block_w][:target_width]
        
        # Create new DataArray
        if data.ndim == 3:
            coords = {
                data.dims[0]: data.coords[data.dims[0]],
                data.dims[-2]: y_coords,
                data.dims[-1]: x_coords
            }
        else:
            coords = {
                data.dims[-2]: y_coords,
                data.dims[-1]: x_coords
            }
        
        return xr.DataArray(
            downsampled,
            dims=data.dims,
            coords=coords,
            attrs=data.attrs.copy()
        )
    
    def _downsample_classification(self, data: xr.DataArray, target_height: int,
                                 target_width: int) -> xr.DataArray:
        """Mode-based downsampling for classification data."""
        from scipy import stats
        
        current_height, current_width = data.shape[-2:]
        block_h = current_height // target_height
        block_w = current_width // target_width
        
        # Crop to make divisible
        new_height = (current_height // block_h) * block_h
        new_width = (current_width // block_w) * block_w
        data = data[..., :new_height, :new_width]
        
        # Reshape for block processing
        if data.ndim == 3:
            reshaped = data.values.reshape(
                data.shape[0], target_height, block_h, target_width, block_w
            )
            # Compute mode for each block
            downsampled = np.zeros((data.shape[0], target_height, target_width), dtype=data.dtype)
            for t in range(data.shape[0]):
                for i in range(target_height):
                    for j in range(target_width):
                        block = reshaped[t, i, :, j, :].flatten()
                        mode_val = stats.mode(block, keepdims=False)[0]
                        downsampled[t, i, j] = mode_val
        else:
            reshaped = data.values.reshape(target_height, block_h, target_width, block_w)
            downsampled = np.zeros((target_height, target_width), dtype=data.dtype)
            for i in range(target_height):
                for j in range(target_width):
                    block = reshaped[i, :, j, :].flatten()
                    mode_val = stats.mode(block, keepdims=False)[0]
                    downsampled[i, j] = mode_val
        
        # Create coordinates
        y_coords = data.coords[data.dims[-2]][::block_h][:target_height]
        x_coords = data.coords[data.dims[-1]][::block_w][:target_width]
        
        if data.ndim == 3:
            coords = {
                data.dims[0]: data.coords[data.dims[0]],
                data.dims[-2]: y_coords,
                data.dims[-1]: x_coords
            }
        else:
            coords = {
                data.dims[-2]: y_coords,
                data.dims[-1]: x_coords
            }
        
        return xr.DataArray(
            downsampled,
            dims=data.dims,
            coords=coords,
            attrs=data.attrs.copy()
        )
    
    def _downsample_quality_mask(self, data: xr.DataArray, target_height: int,
                               target_width: int) -> xr.DataArray:
        """Logical OR downsampling for quality masks (any bad pixel = bad block)."""
        current_height, current_width = data.shape[-2:]
        block_h = current_height // target_height
        block_w = current_width // target_width
        
        # Crop to make divisible
        new_height = (current_height // block_h) * block_h
        new_width = (current_width // block_w) * block_w
        data = data[..., :new_height, :new_width]
        
        if data.ndim == 3:
            reshaped = data.values.reshape(
                data.shape[0], target_height, block_h, target_width, block_w
            )
            # Any non-zero value in block makes the downsampled pixel non-zero
            downsampled = (reshaped.sum(axis=(2, 4)) > 0).astype(data.dtype)
        else:
            reshaped = data.values.reshape(target_height, block_h, target_width, block_w)
            downsampled = (reshaped.sum(axis=(1, 3)) > 0).astype(data.dtype)
        
        # Create coordinates
        y_coords = data.coords[data.dims[-2]][::block_h][:target_height]
        x_coords = data.coords[data.dims[-1]][::block_w][:target_width]
        
        if data.ndim == 3:
            coords = {
                data.dims[0]: data.coords[data.dims[0]],
                data.dims[-2]: y_coords,
                data.dims[-1]: x_coords
            }
        else:
            coords = {
                data.dims[-2]: y_coords,
                data.dims[-1]: x_coords
            }
        
        return xr.DataArray(
            downsampled,
            dims=data.dims,
            coords=coords,
            attrs=data.attrs.copy()
        )
    
    def _downsample_probability(self, data: xr.DataArray, target_height: int,
                              target_width: int) -> xr.DataArray:
        """Average downsampling for probability data."""
        # Use same method as reflectance but ensure values stay in [0,1] or [0,100] range
        result = self._downsample_reflectance(data, target_height, target_width)
        
        # Clamp values to valid probability range
        if result.max() <= 1.0:  # [0,1] probabilities
            result.values = np.clip(result.values, 0, 1)
        else:  # [0,100] percentages
            result.values = np.clip(result.values, 0, 100)
        
        return result

def determine_variable_type(var_name: str, var_data: xr.DataArray) -> str:
    """
    Determine the type of a variable for appropriate resampling.
    
    Args:
        var_name: Name of the variable
        var_data: The data array
    
    Returns:
        Variable type string
    """
    # Spectral bands
    if var_name.startswith('b') and (var_name[1:].isdigit() or var_name == 'b8a'):
        return 'reflectance'
    
    # Quality data
    if var_name in ['scl']:  # Scene Classification Layer
        return 'classification'
    
    if var_name in ['cld', 'snw']:  # Probability data
        return 'probability'
    
    if var_name in ['aot', 'wvp']:  # Atmosphere quality - treat as reflectance
        return 'reflectance'
    
    if var_name.startswith('detector_footprint_') or var_name.startswith('quality_'):
        return 'quality_mask'
    
    # Default to reflectance for unknown variables
    return 'reflectance'
```

### 3. s2_data_consolidator.py

```python
"""
Data consolidation logic for reorganizing S2 structure.
"""

import xarray as xr
from typing import Dict, List, Tuple, Optional
from .s2_band_mapping import (
    NATIVE_BANDS, QUALITY_DATA_NATIVE, DETECTOR_FOOTPRINT_NATIVE,
    get_bands_for_level, get_quality_data_for_level
)

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
        if '/measurements/reflectance' in self.dt_input.groups:
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
                native_bands = NATIVE_BANDS.get(res_num, [])
                for band in native_bands:
                    if band in ds.data_vars:
                        self.measurements_data[res_num]['bands'][band] = ds[band]
    
    def _extract_quality_data(self) -> None:
        """Extract quality mask data."""
        quality_base = '/quality/mask'
        
        for resolution in ['r10m', 'r20m', 'r60m']:
            res_num = int(resolution[1:-1])
            group_path = f'{quality_base}/{resolution}'
            
            if group_path in self.dt_input.groups:
                ds = self.dt_input[group_path].to_dataset()
                
                # Only extract quality for native bands at this resolution
                native_bands = NATIVE_BANDS.get(res_num, [])
                for band in native_bands:
                    if band in ds.data_vars:
                        self.measurements_data[res_num]['quality'][f'quality_{band}'] = ds[band]
    
    def _extract_detector_footprints(self) -> None:
        """Extract detector footprint data."""
        footprint_base = '/conditions/mask/detector_footprint'
        
        for resolution in ['r10m', 'r20m', 'r60m']:
            res_num = int(resolution[1:-1])
            group_path = f'{footprint_base}/{resolution}'
            
            if group_path in self.dt_input.groups:
                ds = self.dt_input[group_path].to_dataset()
                
                # Only extract footprints for native bands
                native_bands = NATIVE_BANDS.get(res_num, [])
                for band in native_bands:
                    if band in ds.data_vars:
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
```

### 4. s2_multiscale.py

```python
"""
Multiscale pyramid creation for optimized S2 structure.
"""

import numpy as np
import xarray as xr
from typing import Dict, List, Tuple
from .s2_resampling import S2ResamplingEngine, determine_variable_type
from .s2_band_mapping import get_bands_for_level, get_quality_data_for_level

class S2MultiscalePyramid:
    """Creates multiscale pyramids for consolidated S2 data."""
    
    def __init__(self, enable_sharding: bool = True, spatial_chunk: int = 1024):
        self.enable_sharding = enable_sharding
        self.spatial_chunk = spatial_chunk
        self.resampler = S2ResamplingEngine()
        
        # Define pyramid levels: resolution in meters
        self.pyramid_levels = {
            0: 10,    # Level 0: 10m (native for b02,b03,b04,b08)
            1: 20,    # Level 1: 20m (native for b05,b06,b07,b11,b12,b8a + all quality)
            2: 40,    # Level 2: 40m (2x downsampling from 20m)
            3: 80,    # Level 3: 80m
            4: 160,   # Level 4: 160m
            5: 320,   # Level 5: 320m
            6: 640,   # Level 6: 640m
        }
    
    def create_multiscale_measurements(
        self,
        measurements_by_resolution: Dict[int, Dict],
        output_path: str
    ) -> Dict[int, xr.Dataset]:
        """
        Create multiscale pyramid from consolidated measurements.
        
        Args:
            measurements_by_resolution: Data organized by native resolution
            output_path: Base output path
            
        Returns:
            Dictionary of datasets by pyramid level
        """
        pyramid_datasets = {}
        
        # Create each pyramid level
        for level, target_resolution in self.pyramid_levels.items():
            print(f"Creating pyramid level {level} ({target_resolution}m)...")
            
            dataset = self._create_level_dataset(
                level, target_resolution, measurements_by_resolution
            )
            
            if dataset and len(dataset.data_vars) > 0:
                pyramid_datasets[level] = dataset
                
                # Write this level
                level_path = f"{output_path}/measurements/{level}"
                self._write_level_dataset(dataset, level_path, level)
        
        return pyramid_datasets
    
    def _create_level_dataset(
        self,
        level: int,
        target_resolution: int,
        measurements_by_resolution: Dict[int, Dict]
    ) -> xr.Dataset:
        """Create dataset for a specific pyramid level."""
        
        if level == 0:
            # Level 0: Only native 10m data
            return self._create_level_0_dataset(measurements_by_resolution)
        elif level == 1:
            # Level 1: All data at 20m (native + downsampled from 10m)
            return self._create_level_1_dataset(measurements_by_resolution)
        else:
            # Levels 2+: Downsample from level 1
            return self._create_downsampled_dataset(
                level, target_resolution, measurements_by_resolution
            )
    
    def _create_level_0_dataset(self, measurements_by_resolution: Dict) -> xr.Dataset:
        """Create level 0 dataset with only native 10m data."""
        if 10 not in measurements_by_resolution:
            return xr.Dataset()
        
        data_10m = measurements_by_resolution[10]
        all_vars = {}
        
        # Add only native 10m bands and their associated data
        for category, vars_dict in data_10m.items():
            all_vars.update(vars_dict)
        
        if not all_vars:
            return xr.Dataset()
        
        # Create consolidated dataset
        dataset = xr.Dataset(all_vars)
        dataset.attrs['pyramid_level'] = 0
        dataset.attrs['resolution_meters'] = 10
        
        return dataset
    
    def _create_level_1_dataset(self, measurements_by_resolution: Dict) -> xr.Dataset:
        """Create level 1 dataset with all data at 20m resolution."""
        all_vars = {}
        reference_coords = None
        
        # Start with native 20m data
        if 20 in measurements_by_resolution:
            data_20m = measurements_by_resolution[20]
            for category, vars_dict in data_20m.items():
                all_vars.update(vars_dict)
            
            # Get reference coordinates from 20m data
            if all_vars:
                first_var = next(iter(all_vars.values()))
                reference_coords = {
                    'x': first_var.coords['x'],
                    'y': first_var.coords['y']
                }
        
        # Add downsampled 10m data
        if 10 in measurements_by_resolution:
            data_10m = measurements_by_resolution[10]
            
            for category, vars_dict in data_10m.items():
                for var_name, var_data in vars_dict.items():
                    if reference_coords:
                        # Downsample to match 20m grid
                        target_height = len(reference_coords['y'])
                        target_width = len(reference_coords['x'])
                        
                        var_type = determine_variable_type(var_name, var_data)
                        downsampled = self.resampler.downsample_variable(
                            var_data, target_height, target_width, var_type
                        )
                        
                        # Align coordinates
                        downsampled = downsampled.assign_coords(reference_coords)
                        all_vars[var_name] = downsampled
        
        if not all_vars:
            return xr.Dataset()
        
        # Create consolidated dataset
        dataset = xr.Dataset(all_vars)
        dataset.attrs['pyramid_level'] = 1
        dataset.attrs['resolution_meters'] = 20
        
        return dataset
    
    def _create_downsampled_dataset(
        self,
        level: int,
        target_resolution: int,
        measurements_by_resolution: Dict
    ) -> xr.Dataset:
        """Create downsampled dataset for levels 2+."""
        # Start from level 1 data (20m) and downsample
        level_1_dataset = self._create_level_1_dataset(measurements_by_resolution)
        
        if len(level_1_dataset.data_vars) == 0:
            return xr.Dataset()
        
        # Calculate target dimensions (downsample by factor of 2^(level-1))
        downsample_factor = 2 ** (level - 1)
        
        # Get reference dimensions from level 1
        ref_var = next(iter(level_1_dataset.data_vars.values()))
        current_height, current_width = ref_var.shape[-2:]
        target_height = current_height // downsample_factor
        target_width = current_width // downsample_factor
        
        downsampled_vars = {}
        
        for var_name, var_data in level_1_dataset.data_vars.items():
            var_type = determine_variable_type(var_name, var_data)
            downsampled = self.resampler.downsample_variable(
                var_data, target_height, target_width, var_type
            )
            downsampled_vars[var_name] = downsampled
        
        # Create dataset
        dataset = xr.Dataset(downsampled_vars)
        dataset.attrs['pyramid_level'] = level
        dataset.attrs['resolution_meters'] = target_resolution
        
        return dataset
    
    def _write_level_dataset(self, dataset: xr.Dataset, level_path: str, level: int) -> None:
        """Write a pyramid level dataset to storage."""
        # Create encoding
        encoding = self._create_level_encoding(dataset, level)
        
        # Write dataset
        print(f"  Writing level {level} to {level_path}")
        dataset.to_zarr(
            level_path,
            mode='w',
            consolidated=True,
            zarr_format=3,
            encoding=encoding
        )
    
    def _create_level_encoding(self, dataset: xr.Dataset, level: int) -> Dict:
        """Create optimized encoding for a pyramid level."""
        encoding = {}
        
        # Calculate level-appropriate chunk sizes
        chunk_size = max(256, self.spatial_chunk // (2 ** level))
        
        for var_name, var_data in dataset.data_vars.items():
            if var_data.ndim >= 2:
                height, width = var_data.shape[-2:]
                
                # Adjust chunk size to data dimensions
                chunk_y = min(chunk_size, height)
                chunk_x = min(chunk_size, width)
                
                if var_data.ndim == 3:
                    chunks = (1, chunk_y, chunk_x)
                else:
                    chunks = (chunk_y, chunk_x)
            else:
                chunks = (min(chunk_size, var_data.shape[0]),)
            
            # Configure encoding
            var_encoding = {
                'chunks': chunks,
                'compressor': 'default'
            }
            
            # Add sharding if enabled
            if self.enable_sharding and var_data.ndim >= 2:
                shard_dims = self._calculate_shard_dimensions(var_data.shape, chunks)
                var_encoding['shards'] = shard_dims
            
            encoding[var_name] = var_encoding
        
        # Add coordinate encoding
        for coord_name in dataset.coords:
            encoding[coord_name] = {'compressor': None}
        
        return encoding
    
    def _calculate_shard_dimensions(self, data_shape: Tuple, chunks: Tuple) -> Tuple:
        """Calculate shard dimensions for Zarr v3 sharding."""
        shard_dims = []
        
        for i, (dim_size, chunk_size) in enumerate(zip(data_shape, chunks)):
            # Ensure shard dimension is evenly divisible by chunk dimension
            if chunk_size >= dim_size:
                shard_dim = dim_size
            else:
                # Calculate largest multiple of chunk_size that fits
                num_chunks = dim_size // chunk_size
                if num_chunks >= 4:  # Use 4 chunks per shard if possible
                    shard_dim = min(4 * chunk_size, dim_size)
                else:
                    shard_dim = num_chunks * chunk_size
            
            shard_dims.append(shard_dim)
        
        return tuple(shard_dims)
```

### 5. s2_converter.py (Main orchestration)

```python
"""
Main S2 optimization converter.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, List
import xarray as xr

from .s2_data_consolidator import S2DataConsolidator, create_consolidated_dataset
from .s2_multiscale import S2MultiscalePyramid
from .s2_validation import S2OptimizationValidator
from ..fs_utils import get_storage_options, normalize_path

class S2OptimizedConverter:
    """Optimized Sentinel-2 to GeoZarr converter."""
    
    def __init__(
        self,
        enable_sharding: bool = True,
        spatial_chunk: int = 1024,
        compression_level: int = 3,
        max_retries: int = 3
    ):
        self.enable_sharding = enable_sharding
        self.spatial_chunk = spatial_chunk
        self.compression_level = compression_level
        self.max_retries = max_retries
        
        # Initialize components
        self.pyramid_creator = S2MultiscalePyramid(enable_sharding, spatial_chunk)
        self.validator = S2OptimizationValidator()
    
    def convert_s2_optimized(
        self,
        dt_input: xr.DataTree,
        output_path: str,
        create_geometry_group: bool = True,
        create_meteorology_group: bool = True,
        validate_output: bool = True,
        verbose: bool = False
    ) -> xr.DataTree:
        """
        Convert S2 dataset to optimized structure.
        
        Args:
            dt_input: Input Sentinel-2 DataTree
            output_path: Output path for optimized dataset
            create_geometry_group: Whether to create geometry group
            create_meteorology_group: Whether to create meteorology group
            validate_output: Whether to validate the output
            verbose: Enable verbose logging
            
        Returns:
            Optimized DataTree
        """
        start_time = time.time()
        
        if verbose:
            print(f"Starting S2 optimization conversion...")
            print(f"Input: {len(dt_input.groups)} groups")
            print(f"Output: {output_path}")
        
        # Validate input is S2
        if not self._is_sentinel2_dataset(dt_input):
            raise ValueError("Input dataset is not a Sentinel-2 product")
        
        # Step 1: Consolidate data from scattered structure
        print("Step 1: Consolidating scattered data structure...")
        consolidator = S2DataConsolidator(dt_input)
        measurements_data, geometry_data, meteorology_data = consolidator.consolidate_all_data()
        
        if verbose:
            print(f"  Measurements data extracted: {sum(len(d['bands']) for d in measurements_data.values())} bands")
            print(f"  Geometry variables: {len(geometry_data)}")
            print(f"  Meteorology variables: {len(meteorology_data)}")
        
        # Step 2: Create multiscale measurements
        print("Step 2: Creating multiscale measurements pyramid...")
        pyramid_datasets = self.pyramid_creator.create_multiscale_measurements(
            measurements_data, output_path
        )
        
        print(f"  Created {len(pyramid_datasets)} pyramid levels")
        
        # Step 3: Create geometry group
        if create_geometry_group and geometry_data:
            print("Step 3: Creating consolidated geometry group...")
            geometry_ds = xr.Dataset(geometry_data)
            geometry_path = f"{output_path}/geometry"
            self._write_auxiliary_group(geometry_ds, geometry_path, "geometry", verbose)
        
        # Step 4: Create meteorology group
        if create_meteorology_group and meteorology_data:
            print("Step 4: Creating consolidated meteorology group...")
            meteorology_ds = xr.Dataset(meteorology_data)
            meteorology_path = f"{output_path}/meteorology"
            self._write_auxiliary_group(meteorology_ds, meteorology_path, "meteorology", verbose)
        
        # Step 5: Create root-level multiscales metadata
        print("Step 5: Adding multiscales metadata...")
        self._add_root_multiscales_metadata(output_path, pyramid_datasets)
        
        # Step 6: Consolidate metadata
        print("Step 6: Consolidating metadata...")
        self._consolidate_root_metadata(output_path)
        
        # Step 7: Validation
        if validate_output:
            print("Step 7: Validating optimized dataset...")
            validation_results = self.validator.validate_optimized_dataset(output_path)
            if not validation_results['is_valid']:
                print("  Warning: Validation issues found:")
                for issue in validation_results['issues']:
                    print(f"    - {issue}")
        
        # Create result DataTree
        result_dt = self._create_result_datatree(output_path)
        
        total_time = time.time() - start_time
        print(f"Optimization complete in {total_time:.2f}s")
        
        if verbose:
            self._print_optimization_summary(dt_input, result_dt, output_path)
        
        return result_dt
    
    def _is_sentinel2_dataset(self, dt: xr.DataTree) -> bool:
        """Check if dataset is Sentinel-2."""
        # Check STAC properties
        stac_props = dt.attrs.get('stac_discovery', {}).get('properties', {})
        mission = stac_props.get('mission', '')
        
        if mission.lower().startswith('sentinel-2'):
            return True
        
        # Check for characteristic S2 groups
        s2_indicators = [
            '/measurements/reflectance',
            '/conditions/geometry',
            '/quality/atmosphere'
        ]
        
        found_indicators = sum(1 for indicator in s2_indicators if indicator in dt.groups)
        return found_indicators >= 2
    
    def _write_auxiliary_group(
        self,
        dataset: xr.Dataset,
        group_path: str,
        group_type: str,
        verbose: bool
    ) -> None:
        """Write auxiliary group (geometry or meteorology)."""
        # Create simple encoding
        encoding = {}
        for var_name in dataset.data_vars:
            encoding[var_name] = {'compressor': 'default'}
        for coord_name in dataset.coords:
            encoding[coord_name] = {'compressor': None}
        
        # Write dataset
        storage_options = get_storage_options(group_path)
        dataset.to_zarr(
            group_path,
            mode='w',
            consolidated=True,
            zarr_format=3,
            encoding=encoding,
            storage_options=storage_options
        )
        
        if verbose:
            print(f"  {group_type.title()} group written: {len(dataset.data_vars)} variables")
    
    def _add_root_multiscales_metadata(
        self,
        output_path: str,
        pyramid_datasets: Dict[int, xr.Dataset]
    ) -> None:
        """Add multiscales metadata at root level."""
        from ..geozarr import create_native_crs_tile_matrix_set, calculate_overview_levels
        
        # Get information from level 0 dataset
        if 0 not in pyramid_datasets:
            return
        
        level_0_ds = pyramid_datasets[0]
        if not level_0_ds.data_vars:
            return
        
        # Get spatial info from first variable
        first_var = next(iter(level_0_ds.data_vars.values()))
        native_height, native_width = first_var.shape[-2:]
        native_crs = level_0_ds.rio.crs
        native_bounds = level_0_ds.rio.bounds()
        
        # Calculate overview levels
        overview_levels = []
        for level, resolution in self.pyramid_creator.pyramid_levels.items():
            if level in pyramid_datasets:
                level_ds = pyramid_datasets[level]
                level_var = next(iter(level_ds.data_vars.values()))
                level_height, level_width = level_var.shape[-2:]
                
                overview_levels.append({
                    'level': level,
                    'resolution': resolution,
                    'width': level_width,
                    'height': level_height,
                    'scale_factor': 2 ** level if level > 0 else 1
                })
        
        # Create tile matrix set
        tile_matrix_set = create_native_crs_tile_matrix_set(
            native_crs, native_bounds, overview_levels, "measurements"
        )
        
        # Add metadata to measurements group
        measurements_zarr_path = normalize_path(f"{output_path}/measurements/zarr.json")
        if os.path.exists(measurements_zarr_path):
            import json
            with open(measurements_zarr_path, 'r') as f:
                zarr_json = json.load(f)
            
            zarr_json.setdefault('attributes', {})
            zarr_json['attributes']['multiscales'] = {
                'tile_matrix_set': tile_matrix_set,
                'resampling_method': 'average',
                'datasets': [{'path': str(level)} for level in sorted(pyramid_datasets.keys())]
            }
            
            with open(measurements_zarr_path, 'w') as f:
                json.dump(zarr_json, f, indent=2)
    
    def _consolidate_root_metadata(self, output_path: str) -> None:
        """Consolidate metadata at root level."""
        try:
            from ..geozarr import consolidate_metadata
            from ..fs_utils import open_zarr_group
            
            zarr_group = open_zarr_group(output_path, mode="r+")
            consolidate_metadata(zarr_group.store)
        except Exception as e:
            print(f"  Warning: Root metadata consolidation failed: {e}")
    
    def _create_result_datatree(self, output_path: str) -> xr.DataTree:
        """Create result DataTree from written output."""
        try:
            storage_options = get_storage_options(output_path)
            return xr.open_datatree(
                output_path,
                engine='zarr',
                chunks='auto',
                storage_options=storage_options
            )
        except Exception as e:
            print(f"Warning: Could not open result DataTree: {e}")
            return xr.DataTree()
    
    def _print_optimization_summary(
        self,
        dt_input: xr.DataTree,
        dt_output: xr.DataTree,
        output_path: str
    ) -> None:
        """Print optimization summary statistics."""
        print("\n" + "="*50)
        print("OPTIMIZATION SUMMARY")
        print("="*50)
        
        # Count groups
        input_groups = len(dt_input.groups) if hasattr(dt_input, 'groups') else 0
        output_groups = len(dt_output.groups) if hasattr(dt_output, 'groups') else 0
        
        print(f"Groups: {input_groups} → {output_groups} ({((output_groups-input_groups)/input_groups*100):+.1f}%)")
        
        # Estimate file count reduction
        estimated_input_files = input_groups * 10  # Rough estimate
        estimated_output_files = output_groups * 5  # Fewer files per group
        print(f"Estimated files: {estimated_input_files} → {estimated_output_files} ({((estimated_output_files-estimated_input_files)/estimated_input_files*100):+.1f}%)")
        
        # Show structure
        print(f"\nNew structure:")
        print(f"  /measurements/  (multiscale: levels 0-6)")
        if f"{output_path}/geometry" in str(dt_output):
            print(f"  /geometry/      (consolidated)")
        if f"{output_path}/meteorology" in str(dt_output):
            print(f"  /meteorology/   (consolidated)")
        
        print("="*50)


def convert_s2_optimized(
    dt_input: xr.DataTree,
    output_path: str,
    **kwargs
) -> xr.DataTree:
    """
    Convenience function for S2 optimization.
    
    Args:
        dt_input: Input Sentinel-2 DataTree
        output_path: Output path
        **kwargs: Additional arguments for S2OptimizedConverter
    
    Returns:
        Optimized DataTree
    """
    converter = S2OptimizedConverter(**kwargs)
    return converter.convert_s2_optimized(dt_input, output_path, **kwargs)
```

### 6. s2_validation.py

```python
"""
Validation for S2 optimized datasets.
"""

import os
from typing import Dict, List, Any
import xarray as xr
from ..fs_utils import get_storage_options

class S2OptimizationValidator:
    """Validates S2 optimized dataset structure and integrity."""
    
    def validate_optimized_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Validate an optimized S2 dataset.
        
        Args:
            dataset_path: Path to the optimized dataset
            
        Returns:
            Validation results dictionary
        """
        results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'summary': {}
        }
        
        try:
            storage_options = get_storage_options(dataset_path)
            dt = xr.open_datatree(
                dataset_path,
                engine='zarr',
                chunks='auto',
                storage_options=storage_options
            )
            
            # Check required groups
            self._validate_group_structure(dt, results)
            
            # Check multiscale structure
            self._validate_multiscale_structure(dt, results)
            
            # Check data integrity
            self._validate_data_integrity(dt, results)
            
            # Check metadata compliance
            self._validate_metadata_compliance(dt, results)
            
        except Exception as e:
            results['is_valid'] = False
            results['issues'].append(f"Failed to open dataset: {e}")
        
        return results
    
    def _validate_group_structure(self, dt: xr.DataTree, results: Dict) -> None:
        """Validate the expected group structure."""
        required_groups = ['/measurements']
        optional_groups = ['/geometry', '/meteorology']
        
        existing_groups = set(dt.groups.keys()) if hasattr(dt, 'groups') else set()
        
        # Check required groups
        for group in required_groups:
            if group not in existing_groups:
                results['issues'].append(f"Missing required group: {group}")
                results['is_valid'] = False
        
        # Check for unexpected groups
        expected_groups = set(required_groups + optional_groups)
        unexpected = existing_groups - expected_groups - {'.'}  # Exclude root
        if unexpected:
            results['warnings'].append(f"Unexpected groups found: {list(unexpected)}")
        
        results['summary']['groups_found'] = len(existing_groups)
    
    def _validate_multiscale_structure(self, dt: xr.DataTree, results: Dict) -> None:
        """Validate multiscale pyramid structure in measurements."""
        if '/measurements' not in dt.groups:
            return
        
        measurements_group = dt['/measurements']
        
        # Check for numeric subgroups (pyramid levels)
        if not hasattr(measurements_group, 'children'):
            results['issues'].append("Measurements group has no pyramid levels")
            results['is_valid'] = False
            return
        
        pyramid_levels = []
        for child_name in measurements_group.children:
            if child_name.isdigit():
                pyramid_levels.append(int(child_name))
        
        pyramid_levels.sort()
        
        if not pyramid_levels:
            results['issues'].append("No pyramid levels found in measurements")
            results['is_valid'] = False
            return
        
        # Validate level 0 exists (native resolution)
        if 0 not in pyramid_levels:
            results['issues'].append("Missing pyramid level 0 (native resolution)")
            results['is_valid'] = False
        
        # Check for reasonable progression
        if len(pyramid_levels) < 2:
            results['warnings'].append("Only one pyramid level found - not truly multiscale")
        
        results['summary']['pyramid_levels'] = pyramid_levels
        results['summary']['max_pyramid_level'] = max(pyramid_levels) if pyramid_levels else 0
    
    def _validate_data_integrity(self, dt: xr.DataTree, results: Dict) -> None:
        """Validate data integrity across pyramid levels."""
        if '/measurements' not in dt.groups:
            return
        
        measurements = dt['/measurements']
        pyramid_levels = []
        
        # Get pyramid levels
        if hasattr(measurements, 'children'):
            for child_name in measurements.children:
                if child_name.isdigit():
                    pyramid_levels.append(int(child_name))
        
        if not pyramid_levels:
            return
        
        pyramid_levels.sort()
        
        # Check coordinate consistency across levels
        reference_crs = None
        for level in pyramid_levels:
            level_path = f'/measurements/{level}'
            if level_path in dt.groups:
                level_ds = dt[level_path].to_dataset()
                
                # Check CRS consistency
                if hasattr(level_ds, 'rio') and level_ds.rio.crs:
                    if reference_crs is None:
                        reference_crs = level_ds.rio.crs
                    elif reference_crs != level_ds.rio.crs:
                        results['issues'].append(f"CRS mismatch at level {level}")
                        results['is_valid'] = False
                
                # Check for data variables
                if not level_ds.data_vars:
                    results['warnings'].append(f"Pyramid level {level} has no data variables")
        
        results['summary']['reference_crs'] = str(reference_crs) if reference_crs else None
    
    def _validate_metadata_compliance(self, dt: xr.DataTree, results: Dict) -> None:
        """Validate GeoZarr and CF compliance."""
        compliance_issues = []
        
        # Check for multiscales metadata
        if '/measurements' in dt.groups:
            measurements = dt['/measurements']
            if hasattr(measurements, 'attrs'):
                if 'multiscales' not in measurements.attrs:
                    results['warnings'].append("Missing multiscales metadata in measurements group")
        
        # Check variable attributes
        total_vars = 0
        compliant_vars = 0
        
        for group_path in dt.groups:
            if group_path == '.':
                continue
            
            group_ds = dt[group_path].to_dataset()
            for var_name, var_data in group_ds.data_vars.items():
                total_vars += 1
                
                # Check required attributes
                var_issues = []
                if '_ARRAY_DIMENSIONS' not in var_data.attrs:
                    var_issues.append('Missing _ARRAY_DIMENSIONS')
                
                if 'standard_name' not in var_data.attrs:
                    var_issues.append('Missing standard_name')
                
                if not var_issues:
                    compliant_vars += 1
                else:
                    compliance_issues.append(f"{group_path}/{var_name}: {', '.join(var_issues)}")
        
        if compliance_issues:
            results['warnings'].extend(compliance_issues[:5])  # Show first 5
            if len(compliance_issues) > 5:
                results['warnings'].append(f"... and {len(compliance_issues) - 5} more compliance issues")
        
        results['summary']['total_variables'] = total_vars
        results['summary']['compliant_variables'] = compliant_vars
        results['summary']['compliance_rate'] = f"{compliant_vars/total_vars*100:.1f}%" if total_vars > 0 else "0%"
```

### 7. CLI Integration

```python
"""
CLI integration for S2 optimization.
"""

import argparse
from pathlib import Path
from .s2_converter import convert_s2_optimized
from ..fs_utils import get_storage_options
import xarray as xr

def add_s2_optimization_commands(subparsers):
    """Add S2 optimization commands to CLI parser."""
    
    # Convert S2 optimized command
    s2_parser = subparsers.add_parser(
        'convert-s2-optimized',
        help='Convert Sentinel-2 dataset to optimized structure'
    )
    s2_parser.add_argument(
        'input_path',
        type=str,
        help='Path to input Sentinel-2 dataset (Zarr format)'
    )
    s2_parser.add_argument(
        'output_path',
        type=str,
        help='Path for output optimized dataset'
    )
    s2_parser.add_argument(
        '--spatial-chunk',
        type=int,
        default=1024,
        help='Spatial chunk size (default: 1024)'
    )
    s2_parser.add_argument(
        '--enable-sharding',
        action='store_true',
        help='Enable Zarr v3 sharding'
    )
    s2_parser.add_argument(
        '--compression-level',
        type=int,
        default=3,
        choices=range(1, 10),
        help='Compression level 1-9 (default: 3)'
    )
    s2_parser.add_argument(
        '--skip-geometry',
        action='store_true',
        help='Skip creating geometry group'
    )
    s2_parser.add_argument(
        '--skip-meteorology',
        action='store_true',
        help='Skip creating meteorology group'
    )
    s2_parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip output validation'
    )
    s2_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    s2_parser.set_defaults(func=convert_s2_optimized_command)

def convert_s2_optimized_command(args):
    """Execute S2 optimized conversion command."""
    try:
        # Validate input
        input_path = Path(args.input_path)
        if not input_path.exists():
            print(f"Error: Input path {input_path} does not exist")
            return 1
        
        # Load input dataset
        print(f"Loading Sentinel-2 dataset from: {args.input_path}")
        storage_options = get_storage_options(str(input_path))
        dt_input = xr.open_datatree(
            str(input_path),
            engine='zarr',
            chunks='auto',
            storage_options=storage_options
        )
        
        # Convert
        dt_optimized = convert_s2_optimized(
            dt_input=dt_input,
            output_path=args.output_path,
            enable_sharding=args.enable_sharding,
            spatial_chunk=args.spatial_chunk,
            compression_level=args.compression_level,
            create_geometry_group=not args.skip_geometry,
            create_meteorology_group=not args.skip_meteorology,
            validate_output=not args.skip_validation,
            verbose=args.verbose
        )
        
        print(f"✅ S2 optimization completed: {args.output_path}")
        return 0
        
    except Exception as e:
        print(f"❌ Error during S2 optimization: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
```

## Testing Strategy

### Unit Tests Structure
```
tests/s2_optimization/
├── test_band_mapping.py
├── test_resampling.py  
├── test_consolidator.py
├── test_multiscale.py
├── test_converter.py
├── test_validation.py
└── fixtures/
    ├── sample_s2_structure.zarr/
    └── expected_outputs/
```

### Integration Test Scenarios
1. **Complete S2 L2A conversion** with all groups
2. **Minimal conversion** with measurements only  
3. **Large dataset handling** (>10GB)
4. **Error recovery** scenarios
5. **Performance benchmarking** vs. current implementation

### Validation Checklist
- [ ] All native resolution data preserved exactly
- [ ] Proper downsampling applied at each level
- [ ] Coordinate systems consistent across levels
- [ ] Metadata compliance maintained
- [ ] File count reduction achieved
- [ ] Access time improvements verified
- [ ] Memory usage optimized