"""
Streaming multiscale pyramid creation for optimized S2 structure.
Uses lazy evaluation to minimize memory usage during dataset preparation.
"""

from typing import Dict, Tuple

import xarray as xr
from pyproj import CRS

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


class S2StreamingMultiscalePyramid:
    """Creates streaming multiscale pyramids with lazy evaluation."""
    
    def __init__(self, enable_sharding: bool = True, spatial_chunk: int = 256):
        self.enable_sharding = enable_sharding
        self.spatial_chunk = spatial_chunk
        self.resampler = S2ResamplingEngine()

        # Define pyramid levels: resolution in meters
        self.pyramid_levels = {
            0: 10,    # Level 0: 10m (native for b02,b03,b04,b08)
            1: 20,    # Level 1: 20m (native for b05,b06,b07,b11,b12,b8a + all quality)
            2: 60,    # Level 2: 60m (3x downsampling from 20m)
            3: 120,   # Level 3: 120m (2x downsampling from 60m)
            4: 360,   # Level 4: 360m (3x downsampling from 120m)
            5: 720,   # Level 5: 720m (2x downsampling from 360m)
        }

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
                
                # For levels 3+, we can discard after writing to save memory
                if level > 2:
                    pyramid_datasets[level] = None
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

        # Calculate target dimensions
        downsample_factor = 2 ** (level - 2)
        
        # Get reference dimensions from level 2
        ref_var = next(iter(level_2_dataset.data_vars.values()))
        current_height, current_width = ref_var.shape[-2:]
        target_height = current_height // downsample_factor
        target_width = current_width // downsample_factor

        # Create lazy downsampling operations for all variables
        lazy_vars = {}
        for var_name, var_data in level_2_dataset.data_vars.items():
            lazy_downsampled = self._create_lazy_downsample_operation_from_existing(
                var_data, target_height, target_width
            )
            lazy_vars[var_name] = lazy_downsampled

        # Create dataset with lazy variables - don't pass coords to avoid alignment issues
        # The coordinates will be computed when the lazy operations are executed
        dataset = xr.Dataset(lazy_vars)
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
        import os
        
        # Check if level already exists
        if os.path.exists(level_path):
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
        """Create encoding optimized for streaming writes."""
        encoding = {}
        
        # Calculate level-appropriate chunk sizes for streaming
        chunk_size = max(256, self.spatial_chunk // (2**level))

        for var_name, var_data in dataset.data_vars.items():
            if hasattr(var_data.data, 'chunks'):
                # Use existing chunks from Dask array
                chunks = var_data.data.chunks
                if len(chunks) >= 2:
                    # Convert chunk tuples to sizes
                    encoding_chunks = tuple(chunks[i][0] for i in range(len(chunks)))
                else:
                    encoding_chunks = (chunk_size,)
            else:
                # Fallback chunk calculation
                if var_data.ndim >= 2:
                    if var_data.ndim == 3:
                        encoding_chunks = (1, chunk_size, chunk_size)
                    else:
                        encoding_chunks = (chunk_size, chunk_size)
                else:
                    encoding_chunks = (min(chunk_size, var_data.shape[0]),)

            # Configure encoding for streaming
            from zarr.codecs import BloscCodec
            
            compressor = BloscCodec(
                cname="zstd", clevel=3, shuffle="shuffle", blocksize=0
            )
            encoding[var_name] = {
                "chunks": encoding_chunks, 
                "compressors": [compressor]
            }

        # Add coordinate encoding
        for coord_name in dataset.coords:
            encoding[coord_name] = {"compressors": None}

        return encoding

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

            for var in dataset.data_vars.values():
                var.rio.write_grid_mapping(grid_mapping_var_name, inplace=True)
