"""
Streaming multiscale pyramid creation for optimized S2 structure.
Uses lazy evaluation to minimize memory usage during dataset preparation.
"""

from typing import Dict, Tuple, Any
import numpy as np

import xarray as xr
from pyproj import CRS

import os
import json

from eopf_geozarr.conversion import fs_utils
from eopf_geozarr.conversion.geozarr import (
    create_native_crs_tile_matrix_set,
    _create_tile_matrix_limits,
)

from .s2_resampling import S2ResamplingEngine, determine_variable_type

try:
    import distributed
    from dask import compute, delayed
    import dask.array as da

    DISTRIBUTED_AVAILABLE = True
    DASK_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    DASK_AVAILABLE = False

    # Create dummy delayed function for non-dask environments
    def delayed(func):
        return func

    def compute(*args, **kwargs):
        return args


class S2MultiscalePyramid:
    """Creates streaming multiscale pyramids with lazy evaluation."""
    
    def __init__(self, enable_sharding: bool = True, spatial_chunk: int = 256):
        self.enable_sharding = enable_sharding
        self.spatial_chunk = spatial_chunk
        self.resampler = S2ResamplingEngine()

        # Define pyramid levels: resolution in meters
        self.pyramid_levels = {
            0: 10,    # Level 0: 10m (native for b02,b03,b04,b08)
            1: 20,    # Level 1: 20m (native for b05,b06,b07,b11,b12,b8a + all quality)
            2: 60,    # Level 2: 60m (native for b01,b09,b10)
            3: 120,   # Level 3: 120m (2x downsampling from 60m)
            4: 360,   # Level 4: 360m (3x downsampling from 120m)
            5: 720,   # Level 5: 720m (2x downsampling from 360m)
        }

    def create_multiscale_from_datatree(
        self, dt_input: xr.DataTree, output_path: str, verbose: bool = False
    ) -> Dict[str, Dict]:
        """
        Create multiscale versions preserving original structure.
        Keeps all original groups, adds r120m, r360m, r720m downsampled versions.
        
        Args:
            dt_input: Input DataTree with original structure
            output_path: Base output path
            verbose: Enable verbose logging
            
        Returns:
            Dictionary of processed groups
        """
        processed_groups = {}
        
        # Step 1: Copy all original groups as-is
        for group_path in dt_input.groups:
            if group_path == ".":
                continue
            
            group_node = dt_input[group_path]
            
            # Skip parent groups that have children (only process leaf groups)
            if hasattr(group_node, 'children') and len(group_node.children) > 0:
                continue
            
            dataset = group_node.to_dataset()
            
            # Skip empty groups
            if not dataset.data_vars:
                if verbose:
                    print(f"  Skipping empty group: {group_path}")
                continue
            
            if verbose:
                print(f"  Copying original group: {group_path}")
            
            output_group_path = f"{output_path}{group_path}"
            
            # Determine if this is a measurement-related resolution group
            group_name = group_path.split("/")[-1]
            is_measurement_group = (
                group_name.startswith("r") and group_name.endswith("m") and
                "/measurements/" in group_path
            )
            
            if is_measurement_group:
                # Measurement groups: apply custom encoding
                encoding = self._create_measurements_encoding(dataset)
            else:
                # Non-measurement groups: preserve original chunking
                encoding = self._create_original_encoding(dataset)

            ds_out = self._stream_write_dataset(dataset, output_group_path, encoding)
            processed_groups[group_path] = ds_out

        # Step 2: Create downsampled resolution groups ONLY for measurements
        # Find all resolution-based groups under /measurements/ and organize by base path
        resolution_groups = {}
        base_path = "/measurements/reflectance"
        for group_path in processed_groups.keys():
            # Only process groups under /measurements/reflectance
            if not group_path.startswith(base_path):
                continue
                
            group_name = group_path.split("/")[-1]
            if group_name in ["r10m", "r20m", "r60m"]:
                resolution_groups[group_name] = processed_groups[group_path]

        # Find the coarsest resolution (r60m > r20m > r10m)
        source_dataset = None
        source_resolution = None
        
        for res in ["r60m", "r20m", "r10m"]:
            if res in resolution_groups:
                source_dataset = resolution_groups[res]
                source_resolution = int(res[1:-1])  # Extract number
                break
        
        if not source_dataset:
            if verbose:
                print(f"  No source resolution found for downsampling, skipping downsampled levels")
            return  # Stop processing if no valid source dataset is found

        if verbose:
            print(f"  Creating downsampled versions from: {source_dataset} ({source_resolution}m)")
        
        # Create r120m
        try:
            r120m_path = f"{base_path}/r120m"
            factor = 120 // source_resolution
            if verbose:
                print(f"    Creating r120m with factor {factor}")
            
            r120m_dataset = self._create_downsampled_resolution_group(
                source_dataset, factor=factor, verbose=verbose
            )
            
            if r120m_dataset and len(r120m_dataset.data_vars) > 0:
                
                output_path_120 = f"{output_path}{r120m_path}"
                if verbose:
                    print(f"    Writing r120m to {output_path_120}")
                encoding_120 = self._create_measurements_encoding(r120m_dataset)
                ds_120 =self._stream_write_dataset(r120m_dataset, output_path_120, encoding_120)
                processed_groups[r120m_path] = ds_120
                resolution_groups["r120m"] = ds_120
                
                # Create r360m from r120m
                try:
                    r360m_path = f"{base_path}/r360m"
                    if verbose:
                        print(f"    Creating r360m with factor 3")
                    
                    r360m_dataset = self._create_downsampled_resolution_group(
                        r120m_dataset, factor=3, verbose=verbose
                    )
                    
                    if r360m_dataset and len(r360m_dataset.data_vars) > 0:
                        output_path_360 = f"{output_path}{r360m_path}"
                        if verbose:
                            print(f"    Writing r360m to {output_path_360}")
                        encoding_360 = self._create_measurements_encoding(r360m_dataset)
                        ds_360 = self._stream_write_dataset(r360m_dataset, output_path_360, encoding_360)
                        processed_groups[r360m_path] = ds_360
                        resolution_groups["r360m"] = ds_360

                        # Create r720m from r360m
                        try:
                            r720m_path = f"{base_path}/r720m"
                            if verbose:
                                print(f"    Creating r720m with factor 2")
                            
                            r720m_dataset = self._create_downsampled_resolution_group(
                                r360m_dataset, factor=2, verbose=verbose
                            )
                            
                            if r720m_dataset and len(r720m_dataset.data_vars) > 0:
                                output_path_720 = f"{output_path}{r720m_path}"
                                if verbose:
                                    print(f"    Writing r720m to {output_path_720}")
                                encoding_720 = self._create_measurements_encoding(r720m_dataset)
                                ds_720 = self._stream_write_dataset(r720m_dataset, output_path_720, encoding_720)
                                processed_groups[r720m_path] = ds_720
                                resolution_groups["r720m"] = ds_720
                            else:
                                if verbose:
                                    print(f"    r720m dataset is empty, skipping")
                        except Exception as e:
                            print(f"  Warning: Could not create r720m for {base_path}: {e}")
                    else:
                        if verbose:
                            print(f"    r360m dataset is empty, skipping")
                except Exception as e:
                    print(f"  Warning: Could not create r360m for {base_path}: {e}")
                # Track r120m for multiscales if created
                if verbose:
                    print(f"  Tracking r120m for multiscales metadata")
            else:
                if verbose:
                    print(f"    r120m dataset is empty, skipping")
        except Exception as e:
            print(f"  Warning: Could not create r120m for {base_path}: {e}")
        
        # Step 3: Add multiscales metadata to parent groups
        if verbose:
            print("\n  Adding multiscales metadata to parent groups...")
            
        try:
            self._add_multiscales_metadata_to_parent(
                output_path, base_path, resolution_groups, verbose
            )
        except Exception as e:
            print(f"  Warning: Could not add multiscales metadata to {base_path}: {e}")
        
        return processed_groups

    def _create_original_encoding(
        self, dataset: xr.Dataset
    ) -> dict:
        """Write a group preserving its original chunking and encoding."""
        from zarr.codecs import BloscCodec
        
        # Simple encoding that preserves original structure
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0)
        encoding = {}
        
        for var_name in dataset.data_vars:
            var_data = dataset.data_vars[var_name]
            
            # Get original chunks if they exist
            if hasattr(var_data, 'encoding') and 'chunks' in var_data.encoding:
                original_chunks = var_data.encoding['chunks']
                
                # Set encoding with original chunks
                encoding[var_name] = {"chunks": original_chunks, "compressors": [compressor]}
            else:
                # No specific chunking - use as is
                encoding[var_name] = {"compressors": [compressor]}
        
        for coord_name in dataset.coords:
            encoding[coord_name] = {"compressors": None}

        return encoding

    def _create_downsampled_resolution_group(
        self, source_dataset: xr.Dataset, factor: int, verbose: bool = False
    ) -> xr.Dataset:
        """Create a downsampled version of a dataset by given factor."""
        if not source_dataset or len(source_dataset.data_vars) == 0:
            return xr.Dataset()
        
        # Get reference dimensions
        ref_var = next(iter(source_dataset.data_vars.values()))
        if ref_var.ndim < 2:
            return xr.Dataset()
        
        current_height, current_width = ref_var.shape[-2:]
        target_height = current_height // factor
        target_width = current_width // factor
        
        if target_height < 1 or target_width < 1:
            return xr.Dataset()
        
        # Create downsampled coordinates
        downsampled_coords = self._create_downsampled_coordinates(
            source_dataset, target_height, target_width, factor
        )
        
        # Downsample all variables using existing lazy operations
        lazy_vars = {}
        for var_name, var_data in source_dataset.data_vars.items():
            if var_data.ndim < 2:
                continue
            
            lazy_downsampled = self._create_lazy_downsample_operation_from_existing(
                var_data, target_height, target_width
            )
            lazy_vars[var_name] = lazy_downsampled
        
        if not lazy_vars:
            return xr.Dataset()
        
        # Create dataset with lazy variables and coordinates
        dataset = xr.Dataset(lazy_vars, coords=downsampled_coords)
        dataset.attrs.update(source_dataset.attrs)
        
        return dataset

    def _create_lazy_downsample_operation(
        self, 
        source_data: xr.DataArray, 
        target_height: int, 
        target_width: int, 
        reference_coords: dict
    ) -> xr.DataArray:
        """Create a lazy downsampling operation using Dask delayed."""
        
        @delayed
        def downsample_operation():
            var_type = determine_variable_type(source_data.name, source_data)
            downsampled = self.resampler.downsample_variable(
                source_data, target_height, target_width, var_type
            )
            # Align coordinates
            return downsampled.assign_coords(reference_coords)
        
        # Create delayed operation
        lazy_result = downsample_operation()
        
        # Convert to Dask array with proper shape and chunks
        # Estimate output shape based on target dimensions
        if source_data.ndim == 3:
            output_shape = (source_data.shape[0], target_height, target_width)
            chunks = (1, min(256, target_height), min(256, target_width))
        else:
            output_shape = (target_height, target_width)
            chunks = (min(256, target_height), min(256, target_width))
        
        # Create Dask array from delayed operation
        dask_array = da.from_delayed(
            lazy_result, 
            shape=output_shape, 
            dtype=source_data.dtype
        ).rechunk(chunks)
        
        # Create coordinates for the output
        if source_data.ndim == 3:
            coords = {
                source_data.dims[0]: source_data.coords[source_data.dims[0]],
                source_data.dims[-2]: reference_coords["y"],
                source_data.dims[-1]: reference_coords["x"],
            }
        else:
            coords = {
                source_data.dims[-2]: reference_coords["y"],
                source_data.dims[-1]: reference_coords["x"],
            }
        
        # Return as xarray DataArray with lazy data
        return xr.DataArray(
            dask_array,
            dims=source_data.dims,
            coords=coords,
            attrs=source_data.attrs.copy(),
            name=source_data.name
        )

    def _create_lazy_downsample_operation_from_existing(
        self, 
        source_data: xr.DataArray, 
        target_height: int, 
        target_width: int
    ) -> xr.DataArray:
        """Create lazy downsampling operation from existing data."""
        
        @delayed
        def downsample_operation():
            var_type = determine_variable_type(source_data.name, source_data)
            return self.resampler.downsample_variable(
                source_data, target_height, target_width, var_type
            )
        
        # Create delayed operation
        lazy_result = downsample_operation()
        
        # Estimate output shape and chunks
        if source_data.ndim == 3:
            output_shape = (source_data.shape[0], target_height, target_width)
            chunks = (1, min(256, target_height), min(256, target_width))
        else:
            output_shape = (target_height, target_width)
            chunks = (min(256, target_height), min(256, target_width))
        
        # Create Dask array from delayed operation
        dask_array = da.from_delayed(
            lazy_result, 
            shape=output_shape, 
            dtype=source_data.dtype
        ).rechunk(chunks)
        
        # Return as xarray DataArray with lazy data - no coords to avoid alignment issues
        # Coordinates will be set when the lazy operation is computed
        return xr.DataArray(
            dask_array,
            dims=source_data.dims,
            attrs=source_data.attrs.copy(),
            name=source_data.name
        )

    def _stream_write_dataset(
        self, dataset: xr.Dataset, dataset_path: str, encoding: Dict[str, Any]
    ) -> xr.Dataset:
        """
        Stream write a lazy dataset with advanced chunking and sharding.
        
        This is where the magic happens: all the lazy downsampling operations
        are executed as the data is streamed to storage with optimal performance.
        """
        
        # Check if level already exists
        if fs_utils.path_exists(dataset_path):
            print(f"    Level path {dataset_path} already exists. Skipping write.")
            existing_ds = xr.open_dataset(
                dataset_path,
                engine="zarr",
                chunks={},
                decode_coords="all",
            )
            return existing_ds
        
        print(f"    Streaming computation and write to {dataset_path}")
        print(f"    Variables: {list(dataset.data_vars.keys())}")
        
        # Rechunk dataset to align with encoding when sharding is enabled
        if self.enable_sharding:
            dataset = self._rechunk_dataset_for_encoding(dataset, encoding)
            
        # Add the geo metadata before writing
        self._write_geo_metadata(dataset)
        
        # Write with streaming computation and progress tracking
        # The to_zarr operation will trigger all lazy computations
        write_job = dataset.to_zarr(
            dataset_path,
            mode="w",
            consolidated=True,
            zarr_format=3,
            encoding=encoding,
            compute=False,  # Create job first for progress tracking
        )
        write_job = write_job.persist()
        
        ds_out = None

        # Show progress bar if distributed is available
        if DISTRIBUTED_AVAILABLE:
            try:
                distributed.progress(write_job, notebook=False)
            except Exception as e:
                print(f"    Warning: Could not display progress bar: {e}")
                ds_out = write_job.compute()
        else:
            print("    Writing zarr file...")
            ds_out = write_job.compute()

        print(f"    ✅ Streaming write complete for dataset {dataset_path}")
        return ds_out

    def _rechunk_dataset_for_encoding(
        self, dataset: xr.Dataset, encoding: Dict
    ) -> xr.Dataset:
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
                if "shards" in var_encoding and var_encoding["shards"] is not None:
                    target_chunks = var_encoding[
                        "shards"
                    ]  # Use shard dimensions for rechunking
                elif "chunks" in var_encoding:
                    target_chunks = var_encoding[
                        "chunks"
                    ]  # Fallback to chunk dimensions
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
        rechunked_dataset = xr.Dataset(
            rechunked_vars, coords=dataset.coords, attrs=dataset.attrs
        )

        return rechunked_dataset

    def _create_level_0_dataset(self, measurements_by_resolution: Dict) -> xr.Dataset:
        """Create level 0 dataset with only native 10m data (no lazy operations needed)."""
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
        dataset.attrs["pyramid_level"] = 0
        dataset.attrs["resolution_meters"] = 10

        self._write_geo_metadata(dataset)
        return dataset

    def _create_multiscale_measurements_sequential(
        self, measurements_by_resolution: Dict[int, Dict], output_path: str
    ) -> Dict[int, xr.Dataset]:
        """Fallback sequential processing for non-Dask environments."""
        print("Creating multiscale pyramid sequentially (no streaming)...")
        # Implementation would be similar to the original sequential approach
        # This is a fallback - the main value is in the streaming approach
        return {}

    def _create_measurements_encoding(self, dataset: xr.Dataset) -> Dict:
        """Create optimized encoding for a pyramid level with advanced chunking and sharding."""
        encoding = {}

        for var_name, var_data in dataset.data_vars.items():
            if var_data.ndim >= 2:
                height, width = var_data.shape[-2:]

                # Use advanced aligned chunk calculation
                spatial_chunk_aligned = min(
                    self.spatial_chunk,
                    self._calculate_aligned_chunk_size(width, self.spatial_chunk),
                    self._calculate_aligned_chunk_size(height, self.spatial_chunk),
                )

                if var_data.ndim == 3:
                    # Single file per variable per time: chunk time dimension to 1
                    chunks = (1, spatial_chunk_aligned, spatial_chunk_aligned)
                else:
                    chunks = (spatial_chunk_aligned, spatial_chunk_aligned)
            else:
                chunks = (min(self.spatial_chunk, var_data.shape[0]),)

            # Configure encoding - use proper compressor following geozarr.py pattern
            from zarr.codecs import BloscCodec

            compressor = BloscCodec(
                cname="zstd", clevel=3, shuffle="shuffle", blocksize=0
            )
            var_encoding = {"chunks": chunks, "compressors": [compressor]}

            # Add advanced sharding if enabled - shards match x/y dimensions exactly
            if self.enable_sharding and var_data.ndim >= 2:
                shard_dims = self._calculate_simple_shard_dimensions(
                    var_data.shape, chunks
                )
                var_encoding["shards"] = shard_dims

            encoding[var_name] = var_encoding

        # Add coordinate encoding
        for coord_name in dataset.coords:
            encoding[coord_name] = {"compressors": None}

        return encoding

    def _calculate_aligned_chunk_size(
        self, dimension_size: int, target_chunk: int
    ) -> int:
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

    def _calculate_simple_shard_dimensions(
        self, data_shape: tuple, chunks: tuple
    ) -> tuple:
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

    def _create_downsampled_coordinates(
        self, level_2_dataset: xr.Dataset, target_height: int, target_width: int, downsample_factor: int
    ) -> Dict:
        """Create downsampled coordinates for higher pyramid levels."""
        import numpy as np
        
        # Get original coordinates from level 2
        if 'x' not in level_2_dataset.coords or 'y' not in level_2_dataset.coords:
            return {}
        
        x_coords_orig = level_2_dataset.coords['x'].values
        y_coords_orig = level_2_dataset.coords['y'].values
        
        # Calculate downsampled coordinates by taking every nth point
        # where n is the downsample_factor
        x_coords_downsampled = x_coords_orig[::downsample_factor][:target_width]
        y_coords_downsampled = y_coords_orig[::downsample_factor][:target_height]
        
        # Create coordinate dictionary with proper attributes
        coords = {}
        
        # Copy x coordinate with attributes
        x_attrs = level_2_dataset.coords['x'].attrs.copy()
        coords['x'] = (['x'], x_coords_downsampled, x_attrs)
        
        # Copy y coordinate with attributes  
        y_attrs = level_2_dataset.coords['y'].attrs.copy()
        coords['y'] = (['y'], y_coords_downsampled, y_attrs)
        
        # Copy any other coordinates that might exist
        for coord_name, coord_data in level_2_dataset.coords.items():
            if coord_name not in ['x', 'y']:
                coords[coord_name] = coord_data
        
        return coords

    def _add_multiscales_metadata_to_parent(
        self, output_path: str, base_path: str, res_groups: dict, verbose: bool = False
    ) -> None:
        """Add GeoZarr-compliant multiscales metadata to parent group."""
        
        # Sort by resolution (finest to coarsest)
        res_order = {"r10m": 10, "r20m": 20, "r60m": 60, "r120m": 120, "r360m": 360, "r720m": 720}
        all_resolutions = sorted(set(res_groups.keys()), key=lambda x: res_order.get(x, 999))
        
        if len(all_resolutions) < 2:
            if verbose:
                print(f"    Skipping {base_path} - only one resolution available")
            return
        
        # Get CRS and bounds from first available dataset (load from output path)
        first_res = all_resolutions[0]
        first_dataset = res_groups[first_res]

        # Get CRS and bounds
        native_crs = first_dataset.rio.crs if hasattr(first_dataset, 'rio') else None
        if native_crs is None:
            if verbose:
                print(f"    No CRS found for {base_path}, skipping multiscales metadata")
            return
        
        native_bounds = first_dataset.rio.bounds() if hasattr(first_dataset, 'rio') else None
        if native_bounds is None:
            if verbose:
                print(f"    No bounds found for {base_path}, skipping multiscales metadata")
            return
        
        # Create overview_levels structure with string-based level names
        overview_levels = []
        for res_name in all_resolutions:
            res_meters = res_order[res_name]
        
            dataset = res_groups[res_name]
            
            if dataset is None:
                continue
            
            # Get first data variable to extract dimensions
            first_var = next(iter(dataset.data_vars.values()))
            height, width = first_var.shape[-2:]
            
            # Calculate zoom level (higher resolution = higher zoom)
            tile_width = 256
            zoom_for_width = max(0, int(np.ceil(np.log2(width / tile_width))))
            zoom_for_height = max(0, int(np.ceil(np.log2(height / tile_width))))
            zoom = max(zoom_for_width, zoom_for_height)
            
            # Calculate scale factor relative to finest resolution
            finest_res_meters = res_order[all_resolutions[0]]
            scale_factor = res_meters // finest_res_meters
            
            overview_levels.append({
                "level": res_name,  # Use string-based level name
                "zoom": zoom,
                "width": width,
                "height": height,
                "scale_factor": scale_factor,
                "chunks": dataset.data_vars[first_var.name].chunks
            })
        
        if len(overview_levels) < 2:
            if verbose:
                print(f"    Could not create overview levels for {base_path}")
            return
        
        # Create tile matrix set using geozarr function
        tile_matrix_set = create_native_crs_tile_matrix_set(
            native_crs, native_bounds, overview_levels, group_prefix=None
        )
        
        # Create tile matrix limits
        tile_matrix_limits = _create_tile_matrix_limits(overview_levels, tile_width=256)
        
        # Write multiscales metadata to parent group zarr.json
        parent_zarr_path = os.path.join(output_path, base_path.lstrip('/'), 'zarr.json')
        
        # Create parent group if it doesn't exist
        parent_group_path = os.path.join(output_path, base_path.lstrip('/'))
        os.makedirs(parent_group_path, exist_ok=True)
        
        # Read or create zarr.json
        if os.path.exists(parent_zarr_path):
            with open(parent_zarr_path, 'r') as f:
                zarr_json = json.load(f)
        else:
            zarr_json = {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {}
            }
        
        # Add multiscales metadata
        if "attributes" not in zarr_json:
            zarr_json["attributes"] = {}
        
        zarr_json["attributes"]["multiscales"] = {
            "tile_matrix_set": tile_matrix_set,
            "resampling_method": "average",
            "tile_matrix_limits": tile_matrix_limits,
        }
        
        # Write updated zarr.json
        with open(parent_zarr_path, 'w') as f:
            json.dump(zarr_json, f, indent=2)
        
        if verbose:
            print(f"    ✅ Added multiscales metadata to {base_path} ({len(overview_levels)} resolutions)")
    
    def _write_geo_metadata(
        self, dataset: xr.Dataset, grid_mapping_var_name: str = "spatial_ref"
    ) -> None:
        """Write geographic metadata to the dataset."""
        # Implementation same as original
        crs = None
        for var in dataset.data_vars.values():
            if hasattr(var, "rio") and var.rio.crs:
                crs = var.rio.crs
                break
            elif "proj:epsg" in var.attrs:
                epsg = var.attrs["proj:epsg"]
                crs = CRS.from_epsg(epsg)
                break

        if crs is not None:
            dataset.rio.write_crs(
                crs, grid_mapping_name=grid_mapping_var_name, inplace=True
            )
            dataset.rio.write_grid_mapping(grid_mapping_var_name, inplace=True)
            dataset.attrs["grid_mapping"] = grid_mapping_var_name

            for var in dataset.data_vars.values():
                var.rio.write_grid_mapping(grid_mapping_var_name, inplace=True)
                var.attrs["grid_mapping"] = grid_mapping_var_name
