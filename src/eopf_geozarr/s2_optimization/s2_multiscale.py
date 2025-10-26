"""
Streaming multiscale pyramid creation for optimized S2 structure.
Uses lazy evaluation to minimize memory usage during dataset preparation.
"""

from typing import Dict, Tuple

import xarray as xr
from pyproj import CRS

from eopf_geozarr.conversion import fs_utils

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
                self._stream_write_lazy_dataset(dataset, output_group_path, 0)
                processed_groups[group_path] = {"type": "original", "resolution": group_name}
            else:
                # Non-measurement groups: preserve original chunking
                self._write_group_preserving_original_encoding(dataset, output_group_path)
                processed_groups[group_path] = {"type": "original", "category": group_name}
        
        # Step 2: Create downsampled resolution groups from r60m
        # Find r60m groups to use as source
        r60m_groups = [g for g in dt_input.groups if g.endswith("/r60m")]
        
        for r60m_path in r60m_groups:
            base_path = r60m_path.rsplit("/", 1)[0]  # e.g., /measurements/reflectance
            
            # Get r60m dataset
            r60m_dataset = dt_input[r60m_path].to_dataset()
            if not r60m_dataset.data_vars:
                continue
            
            if verbose:
                print(f"  Creating downsampled versions from: {r60m_path}")
            
            # Create r120m (2x downsample from r60m)
            r120m_path = f"{base_path}/r120m"
            r120m_dataset = self._create_downsampled_resolution_group(
                r60m_dataset, factor=2, verbose=verbose
            )
            if r120m_dataset and len(r120m_dataset.data_vars) > 0:
                output_path_120 = f"{output_path}{r120m_path}"
                self._stream_write_lazy_dataset(r120m_dataset, output_path_120, 0)
                processed_groups[r120m_path] = {"type": "downsampled", "resolution": "r120m", "source": r60m_path}
            
            # Create r360m (6x downsample from r60m, or 3x from r120m)
            r360m_dataset = self._create_downsampled_resolution_group(
                r120m_dataset if r120m_dataset else r60m_dataset,
                factor=3 if r120m_dataset else 6,
                verbose=verbose
            )
            if r360m_dataset and len(r360m_dataset.data_vars) > 0:
                r360m_path = f"{base_path}/r360m"
                output_path_360 = f"{output_path}{r360m_path}"
                self._stream_write_lazy_dataset(r360m_dataset, output_path_360, 0)
                processed_groups[r360m_path] = {"type": "downsampled", "resolution": "r360m", "source": r60m_path}
            
            # Create r720m (12x downsample from r60m, or 2x from r360m)
            r720m_dataset = self._create_downsampled_resolution_group(
                r360m_dataset if r360m_dataset else r60m_dataset,
                factor=2 if r360m_dataset else 12,
                verbose=verbose
            )
            if r720m_dataset and len(r720m_dataset.data_vars) > 0:
                r720m_path = f"{base_path}/r720m"
                output_path_720 = f"{output_path}{r720m_path}"
                self._stream_write_lazy_dataset(r720m_dataset, output_path_720, 0)
                processed_groups[r720m_path] = {"type": "downsampled", "resolution": "r720m", "source": r60m_path}
        
        return processed_groups
    
    def _write_group_preserving_original_encoding(
        self, dataset: xr.Dataset, group_path: str
    ) -> None:
        """Write a group preserving its original chunking and encoding."""
        from zarr.codecs import BloscCodec
        
        # Simple encoding that preserves original structure
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0)
        encoding = {}
        
        # Rechunk each variable individually to match original encoding
        rechunked_vars = {}
        for var_name in dataset.data_vars:
            var_data = dataset.data_vars[var_name]
            
            # Get original chunks if they exist
            if hasattr(var_data, 'encoding') and 'chunks' in var_data.encoding:
                original_chunks = var_data.encoding['chunks']
                
                # Create rechunk dictionary using dimension names
                chunk_dict = {}
                for i, dim in enumerate(var_data.dims):
                    if i < len(original_chunks):
                        chunk_dict[dim] = original_chunks[i]
                
                # Rechunk this specific variable
                rechunked_vars[var_name] = var_data.chunk(chunk_dict)
                
                # Set encoding with original chunks
                encoding[var_name] = {"chunks": original_chunks, "compressors": [compressor]}
            else:
                # No specific chunking - use as is
                rechunked_vars[var_name] = var_data
                encoding[var_name] = {"compressors": [compressor]}
        
        for coord_name in dataset.coords:
            encoding[coord_name] = {"compressors": None}
        
        # Recreate dataset with rechunked variables
        if rechunked_vars:
            dataset = xr.Dataset(rechunked_vars, coords=dataset.coords, attrs=dataset.attrs)
        
        # Write dataset with original encoding preserved
        dataset.to_zarr(
            group_path,
            mode="w",
            consolidated=True,
            zarr_format=3,
            encoding=encoding,
        )
    
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

    def create_multiscale_measurements_streaming(
        self, measurements_by_resolution: Dict[int, Dict], output_path: str
    ) -> Dict[int, xr.Dataset]:
        """
        Create multiscale pyramid with streaming lazy evaluation.
        
        Key innovation: Downsampling operations are prepared as computation graphs
        but not executed until write time, enabling true streaming processing.
        """
        if DASK_AVAILABLE:
            return self._create_streaming_measurements_lazy(
                measurements_by_resolution, output_path
            )
        else:
            # Fallback to regular processing
            return self._create_multiscale_measurements_sequential(
                measurements_by_resolution, output_path
            )

    def _create_streaming_measurements_lazy(
        self, measurements_by_resolution: Dict[int, Dict], output_path: str
    ) -> Dict[int, xr.Dataset]:
        """
        Create multiscale pyramid with lazy evaluation and streaming writes.
        
        Strategy:
        1. Create lazy datasets with delayed downsampling operations
        2. Write each level with streaming execution
        3. Computation happens only during zarr write operations
        4. Minimal memory usage - no intermediate results stored
        """
        print("Creating streaming multiscale pyramid with lazy evaluation...")
        pyramid_datasets = {}

        # Process levels sequentially but prepare lazy operations
        for level in sorted(self.pyramid_levels.keys()):
            target_resolution = self.pyramid_levels[level]
            print(f"Preparing lazy operations for level {level} ({target_resolution}m)...")

            # Create lazy dataset with delayed operations
            if level <= 2:
                # Base levels: use source measurements data
                lazy_dataset = self._create_lazy_level_dataset(
                    level, target_resolution, measurements_by_resolution
                )
            else:
                # Higher levels: use level 2 data if available
                if 2 in pyramid_datasets:
                    lazy_dataset = self._create_lazy_downsampled_dataset_from_level2(
                        level, target_resolution, pyramid_datasets[2]
                    )
                else:
                    print(f"  Skipping level {level} - level 2 not available")
                    continue

            if lazy_dataset and len(lazy_dataset.data_vars) > 0:
                # Store lazy dataset for potential use by higher levels
                pyramid_datasets[level] = lazy_dataset
                
                # Stream write the lazy dataset (computation happens here)
                level_path = f"{output_path}/measurements/{level}"
                print(f"  Streaming write of level {level} to {level_path}")
                self._stream_write_lazy_dataset(lazy_dataset, level_path, level)
            else:
                print(f"  Skipping empty level {level}")

        print(f"✅ Streaming pyramid creation complete")
        return pyramid_datasets

    def _create_lazy_level_dataset(
        self,
        level: int,
        target_resolution: int,
        measurements_by_resolution: Dict[int, Dict],
    ) -> xr.Dataset:
        """Create dataset with lazy downsampling operations."""

        if level == 0:
            # Level 0: Only native 10m data (no downsampling needed)
            return self._create_level_0_dataset(measurements_by_resolution)
        elif level == 1:
            # Level 1: All data at 20m (native + lazy downsampled from 10m)
            return self._create_lazy_level_1_dataset(measurements_by_resolution)
        elif level == 2:
            # Level 2: All data at 60m (native + lazy downsampled from 20m/10m)
            return self._create_lazy_level_2_dataset(measurements_by_resolution)
        else:
            # Should not be called for levels 3+ in streaming approach
            raise ValueError(f"Use _create_lazy_downsampled_dataset_from_level2 for level {level}")

    def _create_lazy_level_1_dataset(
        self, measurements_by_resolution: Dict
    ) -> xr.Dataset:
        """Create level 1 dataset with lazy downsampling from 10m data."""
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
                    "x": first_var.coords["x"],
                    "y": first_var.coords["y"],
                }

        # Add lazy downsampled 10m data
        if 10 in measurements_by_resolution and reference_coords:
            data_10m = measurements_by_resolution[10]
            target_height = len(reference_coords["y"])
            target_width = len(reference_coords["x"])

            # Create lazy downsampling operations
            for category, vars_dict in data_10m.items():
                for var_name, var_data in vars_dict.items():
                    if var_name in all_vars:
                        continue
                    
                    # Create lazy downsampling operation
                    lazy_downsampled = self._create_lazy_downsample_operation(
                        var_data, target_height, target_width, reference_coords
                    )
                    all_vars[var_name] = lazy_downsampled

        if not all_vars:
            return xr.Dataset()

        # Create dataset with lazy variables
        dataset = xr.Dataset(all_vars)
        dataset.attrs["pyramid_level"] = 1
        dataset.attrs["resolution_meters"] = 20

        return dataset

    def _create_lazy_level_2_dataset(
        self, measurements_by_resolution: Dict
    ) -> xr.Dataset:
        """Create level 2 dataset with lazy downsampling to 60m."""
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
                    "x": first_var.coords["x"],
                    "y": first_var.coords["y"],
                }

        if reference_coords:
            target_height = len(reference_coords["y"])
            target_width = len(reference_coords["x"])

            # Add lazy downsampling from 20m data
            if 20 in measurements_by_resolution:
                data_20m = measurements_by_resolution[20]
                for category, vars_dict in data_20m.items():
                    for var_name, var_data in vars_dict.items():
                        if var_name not in all_vars:
                            lazy_downsampled = self._create_lazy_downsample_operation(
                                var_data, target_height, target_width, reference_coords
                            )
                            all_vars[var_name] = lazy_downsampled
            
            # Add lazy downsampling from 10m data
            if 10 in measurements_by_resolution:
                data_10m = measurements_by_resolution[10]
                for category, vars_dict in data_10m.items():
                    for var_name, var_data in vars_dict.items():
                        if var_name not in all_vars:
                            lazy_downsampled = self._create_lazy_downsample_operation(
                                var_data, target_height, target_width, reference_coords
                            )
                            all_vars[var_name] = lazy_downsampled

        if not all_vars:
            return xr.Dataset()

        # Create dataset with lazy variables
        dataset = xr.Dataset(all_vars)
        dataset.attrs["pyramid_level"] = 2
        dataset.attrs["resolution_meters"] = 60

        return dataset

    def _create_lazy_downsampled_dataset_from_level2(
        self, level: int, target_resolution: int, level_2_dataset: xr.Dataset
    ) -> xr.Dataset:
        """Create lazy downsampled dataset from level 2."""
        if len(level_2_dataset.data_vars) == 0:
            return xr.Dataset()

        # Calculate target dimensions based on resolution ratios from pyramid_levels
        level_2_resolution = self.pyramid_levels[2]  # 60m
        target_level_resolution = self.pyramid_levels[level]  # e.g., 120m, 360m, 720m
        downsample_factor = target_level_resolution // level_2_resolution  # 2x, 6x, 12x
        
        # Get reference dimensions from level 2
        ref_var = next(iter(level_2_dataset.data_vars.values()))
        current_height, current_width = ref_var.shape[-2:]
        target_height = current_height // downsample_factor
        target_width = current_width // downsample_factor

        # Create downsampled coordinates from level 2
        downsampled_coords = self._create_downsampled_coordinates(
            level_2_dataset, target_height, target_width, downsample_factor
        )

        # Create lazy downsampling operations for all variables
        lazy_vars = {}
        for var_name, var_data in level_2_dataset.data_vars.items():
            lazy_downsampled = self._create_lazy_downsample_operation_from_existing(
                var_data, target_height, target_width
            )
            lazy_vars[var_name] = lazy_downsampled

        # Create dataset with lazy variables AND proper coordinates
        dataset = xr.Dataset(lazy_vars, coords=downsampled_coords)
        dataset.attrs["pyramid_level"] = level
        dataset.attrs["resolution_meters"] = target_resolution

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

    def _stream_write_lazy_dataset(
        self, lazy_dataset: xr.Dataset, level_path: str, level: int
    ) -> None:
        """
        Stream write a lazy dataset with advanced chunking and sharding.
        
        This is where the magic happens: all the lazy downsampling operations
        are executed as the data is streamed to storage with optimal performance.
        """
        
        # Check if level already exists
        if fs_utils.path_exists(level_path):
            print(f"    Level path {level_path} already exists. Skipping write.")
            return

        # Create advanced encoding for streaming write
        encoding = self._create_level_encoding(lazy_dataset, level)
        
        print(f"    Streaming computation and write to {level_path}")
        print(f"    Variables: {list(lazy_dataset.data_vars.keys())}")
        
        # Rechunk dataset to align with encoding when sharding is enabled
        if self.enable_sharding:
            lazy_dataset = self._rechunk_dataset_for_encoding(lazy_dataset, encoding)
        
        # Write with streaming computation and progress tracking
        # The to_zarr operation will trigger all lazy computations
        write_job = lazy_dataset.to_zarr(
            level_path,
            mode="w",
            consolidated=True,
            zarr_format=3,
            encoding=encoding,
            compute=False,  # Create job first for progress tracking
        )
        write_job = write_job.persist()

        # Show progress bar if distributed is available
        if DISTRIBUTED_AVAILABLE:
            try:
                distributed.progress(write_job, notebook=False)
            except Exception as e:
                print(f"    Warning: Could not display progress bar: {e}")
                write_job.compute()
        else:
            print("    Writing zarr file...")
            write_job.compute()
        
        print(f"    ✅ Streaming write complete for level {level}")

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

    def _create_level_encoding(self, dataset: xr.Dataset, level: int) -> Dict:
        """Create optimized encoding for a pyramid level with advanced chunking and sharding."""
        encoding = {}

        # Calculate level-appropriate chunk sizes
        chunk_size = max(256, self.spatial_chunk // (2**level))

        for var_name, var_data in dataset.data_vars.items():
            if var_data.ndim >= 2:
                height, width = var_data.shape[-2:]

                # Use advanced aligned chunk calculation
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
