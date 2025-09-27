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
            2: 60,    # Level 2: 60m (3x downsampling from 20m)
            3: 120,   # Level 3: 120m (2x downsampling from 60m)
            4: 240,   # Level 4: 240m (2x downsampling from 120m)
            5: 480,   # Level 5: 480m (2x downsampling from 240m)
            6: 960    # Level 6: 960m (2x downsampling from 480m)
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
        elif level == 2:
            # Level 2: All data at 60m (native + downsampled from 20m)
            return self._create_level_2_dataset(measurements_by_resolution)
        else:
            # Levels 3+: Downsample from level 2
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
    
    def _create_level_2_dataset(self, measurements_by_resolution: Dict) -> xr.Dataset:
        """Create level 2 dataset with all data at 60m resolution."""
        all_vars = {}
        reference_coords = None
        
        # Start with native 60m data
        if 60 in measurements_by_resolution:
            data_60m = measurements_by_resolution[60]
            for category, vars_dict in data_60m.items():
                all_vars.update(vars_dict)
            
            # Get reference coordinates from 60m data
            if all_vars:
                first_var = next(iter(all_vars.values()))
                reference_coords = {
                    'x': first_var.coords['x'],
                    'y': first_var.coords['y']
                }
        
        # Add downsampled 20m data
        if 20 in measurements_by_resolution:
            data_20m = measurements_by_resolution[20]
            
            for category, vars_dict in data_20m.items():
                for var_name, var_data in vars_dict.items():
                    if reference_coords:
                        # Downsample to match 60m grid
                        target_height = len(reference_coords['y'])
                        target_width = len(reference_coords['x'])
                        
                        var_type = determine_variable_type(var_name, var_data)
                        downsampled = self.resampler.downsample_variable(
                            var_data, target_height, target_width, var_type
                        )
                        
                        # Align coordinates
                        downsampled = downsampled.assign_coords(reference_coords)
                        all_vars[var_name] = downsampled
        
        # Add downsampled 10m data
        if 10 in measurements_by_resolution:
            data_10m = measurements_by_resolution[10]
            
            for category, vars_dict in data_10m.items():
                for var_name, var_data in vars_dict.items():
                    if reference_coords:
                        # Downsample to match 60m grid
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
        dataset.attrs['pyramid_level'] = 2
        dataset.attrs['resolution_meters'] = 60
        
        return dataset
    
    def _create_downsampled_dataset(
        self,
        level: int,
        target_resolution: int,
        measurements_by_resolution: Dict
    ) -> xr.Dataset:
        """Create downsampled dataset for levels 3+."""
        # Start from level 2 data (60m) which includes all bands, and downsample
        level_2_dataset = self._create_level_2_dataset(measurements_by_resolution)
        
        if len(level_2_dataset.data_vars) == 0:
            return xr.Dataset()
        
        # Calculate target dimensions (downsample by factor of 2^(level-2))
        downsample_factor = 2 ** (level - 2)
        
        # Get reference dimensions from level 2
        ref_var = next(iter(level_2_dataset.data_vars.values()))
        current_height, current_width = ref_var.shape[-2:]
        target_height = current_height // downsample_factor
        target_width = current_width // downsample_factor
        
        downsampled_vars = {}
        
        for var_name, var_data in level_2_dataset.data_vars.items():
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
        """
        Write a pyramid level dataset to storage with xy-aligned sharding.
        
        Ensures single file per variable per time point when time dimension exists.
        """
        # Create encoding with xy-aligned sharding
        encoding = self._create_level_encoding(dataset, level)
        
        # Check if we have time dimension for single file per time handling
        has_time_dim = any('time' in str(var.dims) for var in dataset.data_vars.values())
        
        if has_time_dim and self._should_separate_time_files(dataset):
            # Write each time slice separately to ensure single file per variable per time
            self._write_time_separated_dataset(dataset, level_path, level, encoding)
        else:
            # Write as single dataset with xy-aligned sharding
            print(f"  Writing level {level} to {level_path} (xy-aligned sharding)")
            
            # Rechunk the dataset to align with encoding chunks (following geozarr.py pattern)
            rechunked_dataset = self._rechunk_dataset_for_encoding(dataset, encoding)
            
            rechunked_dataset.to_zarr(
                level_path,
                mode='w',
                consolidated=True,
                zarr_format=3,
                encoding=encoding
            )
    
    def _should_separate_time_files(self, dataset: xr.Dataset) -> bool:
        """Determine if time files should be separated for single file per variable per time."""
        for var in dataset.data_vars.values():
            if 'time' in var.dims and len(var.coords.get('time', [])) > 1:
                return True
        return False
    
    def _write_time_separated_dataset(
        self, 
        dataset: xr.Dataset, 
        level_path: str, 
        level: int, 
        encoding: Dict
    ) -> None:
        """Write dataset with separate files for each time point."""
        import os
        
        # Get time coordinate
        time_coord = None
        for var in dataset.data_vars.values():
            if 'time' in var.dims:
                time_coord = var.coords['time']
                break
        
        if time_coord is None:
            # Fallback to regular writing if no time found
            print(f"  Writing level {level} to {level_path} (no time coord found)")
            dataset.to_zarr(
                level_path,
                mode='w',
                consolidated=True,
                zarr_format=3,
                encoding=encoding
            )
            return
        
        print(f"  Writing level {level} with time separation to {level_path}")
        
        # Write each time slice separately
        for t_idx, time_val in enumerate(time_coord.values):
            time_slice = dataset.isel(time=t_idx)
            time_path = os.path.join(level_path, f"time_{t_idx:04d}")
            
            # Update encoding for time slice (remove time dimension)
            time_encoding = self._update_encoding_for_time_slice(encoding, time_slice)
            
            print(f"    Writing time slice {t_idx} to {time_path}")
            time_slice.to_zarr(
                time_path,
                mode='w',
                consolidated=True,
                zarr_format=3,
                encoding=time_encoding
            )
    
    def _update_encoding_for_time_slice(self, encoding: Dict, time_slice: xr.Dataset) -> Dict:
        """Update encoding configuration for time slice data."""
        updated_encoding = {}
        
        for var_name, var_encoding in encoding.items():
            if var_name in time_slice.data_vars:
                var_data = time_slice[var_name]
                
                # Update chunks and shards for time slice (remove time dimension)
                if 'chunks' in var_encoding and len(var_encoding['chunks']) > 2:
                    # Remove time dimension from chunks (first dimension)
                    updated_chunks = var_encoding['chunks'][1:]
                    updated_encoding[var_name] = var_encoding.copy()
                    updated_encoding[var_name]['chunks'] = updated_chunks
                    
                    # Update shards if present
                    if 'shards' in var_encoding and len(var_encoding['shards']) > 2:
                        updated_shards = var_encoding['shards'][1:]
                        updated_encoding[var_name]['shards'] = updated_shards
                else:
                    updated_encoding[var_name] = var_encoding
            else:
                # Coordinate or other variable
                updated_encoding[var_name] = encoding[var_name]
        
        return updated_encoding
    
    def _create_level_encoding(self, dataset: xr.Dataset, level: int) -> Dict:
        """Create optimized encoding for a pyramid level with xy-aligned sharding."""
        encoding = {}
        
        # Calculate level-appropriate chunk sizes
        chunk_size = max(256, self.spatial_chunk // (2 ** level))
        
        for var_name, var_data in dataset.data_vars.items():
            if var_data.ndim >= 2:
                height, width = var_data.shape[-2:]
                
                # Use original geozarr.py chunk alignment logic
                spatial_chunk_aligned = min(
                    chunk_size,
                    self._calculate_aligned_chunk_size(width, chunk_size),
                    self._calculate_aligned_chunk_size(height, chunk_size),
                )
                
                if var_data.ndim == 3:
                    # Single file per variable per time: chunk time dimension to 1
                    chunks = (1, spatial_chunk_aligned, spatial_chunk_aligned)
                else:
                    chunks = (spatial_chunk_aligned, spatial_chunk_aligned)
            else:
                chunks = (min(chunk_size, var_data.shape[0]),)
            
            # Configure encoding - use proper compressor following geozarr.py pattern
            from zarr.codecs import BloscCodec
            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0)
            var_encoding = {
                'chunks': chunks,
                'compressors': [compressor]
            }
            
            # Add simplified sharding if enabled - shards match x/y dimensions exactly
            if self.enable_sharding and var_data.ndim >= 2:
                shard_dims = self._calculate_simple_shard_dimensions(var_data.shape, chunks)
                var_encoding['shards'] = shard_dims
            
            encoding[var_name] = var_encoding
        
        # Add coordinate encoding
        for coord_name in dataset.coords:
            encoding[coord_name] = {'compressors': None}
        
        return encoding

    def _calculate_aligned_chunk_size(self, dimension_size: int, target_chunk: int) -> int:
        """
        Calculate aligned chunk size following geozarr.py logic.
        
        This ensures good chunk alignment without complex calculations.
        """
        if target_chunk >= dimension_size:
            return dimension_size
            
        # Find the largest divisor of dimension_size that's close to target_chunk
        best_chunk = target_chunk
        for chunk_candidate in range(target_chunk, max(target_chunk // 2, 1), -1):
            if dimension_size % chunk_candidate == 0:
                best_chunk = chunk_candidate
                break
                
        return best_chunk

    def _calculate_simple_shard_dimensions(self, data_shape: Tuple, chunks: Tuple) -> Tuple:
        """
        Calculate shard dimensions that are compatible with chunk dimensions.
        
        Shard dimensions must be evenly divisible by chunk dimensions for Zarr v3.
        When possible, shards should match x/y dimensions exactly as required.
        """
        shard_dims = []
        
        for i, (dim_size, chunk_size) in enumerate(zip(data_shape, chunks)):
            if i == 0 and len(data_shape) == 3:
                # First dimension in 3D data (time) - use single time slice per shard
                shard_dims.append(1)
            else:
                # For x/y dimensions, try to use full dimension size
                # But ensure it's divisible by chunk size
                if dim_size % chunk_size == 0:
                    # Perfect: full dimension is divisible by chunk
                    shard_dims.append(dim_size)
                else:
                    # Find the largest multiple of chunk_size that fits
                    num_chunks = dim_size // chunk_size
                    if num_chunks > 0:
                        shard_size = num_chunks * chunk_size
                        shard_dims.append(shard_size)
                    else:
                        # Fallback: use chunk size itself
                        shard_dims.append(chunk_size)
        
        return tuple(shard_dims)

    def _rechunk_dataset_for_encoding(self, dataset: xr.Dataset, encoding: Dict) -> xr.Dataset:
        """
        Rechunk dataset variables to align with sharding dimensions when sharding is enabled.
        
        When using Zarr v3 sharding, Dask chunks must align with shard dimensions to avoid 
        checksum validation errors.
        """
        rechunked_vars = {}
        
        for var_name, var_data in dataset.data_vars.items():
            if var_name in encoding:
                var_encoding = encoding[var_name]
                
                # If sharding is enabled, rechunk based on shard dimensions
                if 'shards' in var_encoding and var_encoding['shards'] is not None:
                    target_chunks = var_encoding['shards']  # Use shard dimensions for rechunking
                elif 'chunks' in var_encoding:
                    target_chunks = var_encoding['chunks']  # Fallback to chunk dimensions
                else:
                    # No specific chunking needed, use original variable
                    rechunked_vars[var_name] = var_data
                    continue
                
                # Create chunk dict using the actual dimensions of the variable
                var_dims = var_data.dims
                chunk_dict = {}
                for i, dim in enumerate(var_dims):
                    if i < len(target_chunks):
                        chunk_dict[dim] = target_chunks[i]
                
                # Rechunk the variable to match the target dimensions
                rechunked_vars[var_name] = var_data.chunk(chunk_dict)
            else:
                # No specific chunking needed, use original variable
                rechunked_vars[var_name] = var_data
        
        # Create new dataset with rechunked variables, preserving coordinates
        rechunked_dataset = xr.Dataset(rechunked_vars, coords=dataset.coords, attrs=dataset.attrs)
        
        return rechunked_dataset
