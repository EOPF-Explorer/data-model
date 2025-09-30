"""
Main S2 optimization converter.
"""

import time
from typing import Dict

import xarray as xr

from eopf_geozarr.conversion.fs_utils import get_storage_options
from eopf_geozarr.conversion.geozarr import (
    _create_tile_matrix_limits,
    create_native_crs_tile_matrix_set,
)

from .s2_data_consolidator import S2DataConsolidator
from .s2_multiscale import S2MultiscalePyramid
from .s2_multiscale_streaming import S2StreamingMultiscalePyramid
from .s2_validation import S2OptimizationValidator

try:
    import distributed

    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False


class S2OptimizedConverter:
    """Optimized Sentinel-2 to GeoZarr converter."""

    def __init__(
        self,
        enable_sharding: bool = True,
        spatial_chunk: int = 1024,
        compression_level: int = 3,
        max_retries: int = 3,
        enable_streaming: bool = True,
    ):
        self.enable_sharding = enable_sharding
        self.spatial_chunk = spatial_chunk
        self.compression_level = compression_level
        self.max_retries = max_retries
        self.enable_streaming = enable_streaming

        # Initialize components - choose between streaming and traditional
        if enable_streaming:
            self.pyramid_creator = S2StreamingMultiscalePyramid(enable_sharding, spatial_chunk)
        else:
            self.pyramid_creator = S2MultiscalePyramid(enable_sharding, spatial_chunk)
        self.validator = S2OptimizationValidator()

    def convert_s2_optimized(
        self,
        dt_input: xr.DataTree,
        output_path: str,
        create_geometry_group: bool = True,
        create_meteorology_group: bool = True,
        validate_output: bool = True,
        verbose: bool = False,
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
            print("Starting S2 optimized conversion...")
            print(f"Input: {len(dt_input.groups)} groups")
            print(f"Output: {output_path}")

        # Validate input is S2
        if not self._is_sentinel2_dataset(dt_input):
            raise ValueError("Input dataset is not a Sentinel-2 product")

        # Step 1: Consolidate data from scattered structure
        print("Step 1: Consolidating EOPF data structure...")
        consolidator = S2DataConsolidator(dt_input)
        measurements_data, geometry_data, meteorology_data = (
            consolidator.consolidate_all_data()
        )

        if verbose:
            print(
                f"  Measurements data extracted: {sum(len(d['bands']) for d in measurements_data.values())} bands"
            )
            print(f"  Geometry variables: {len(geometry_data)}")
            print(f"  Meteorology variables: {len(meteorology_data)}")

        # Step 2: Create multiscale measurements
        print("Step 2: Creating multiscale measurements pyramid...")
        if self.enable_streaming:
            # Use streaming approach - computation happens during write
            pyramid_datasets = self.pyramid_creator.create_multiscale_measurements_streaming(
                measurements_data, output_path
            )
        else:
            # Use traditional approach
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
        
        # Step 5: Create measurements group and add multiscales metadata
        print("Step 5: Creating measurements group...")
        measurement_path = f"{output_path}/measurements"
        measurement_dt = self._write_measurements_group(measurement_path, pyramid_datasets, verbose)
        
        # Step 6: Simple root-level consolidation
        print("Step 6: Final root-level metadata consolidation...")
        self._simple_root_consolidation(output_path, pyramid_datasets)

        # Step 7: Validation
        if validate_output:
            print("Step 7: Validating optimized dataset...")
            validation_results = self.validator.validate_optimized_dataset(output_path)
            if not validation_results["is_valid"]:
                print("  Warning: Validation issues found:")
                for issue in validation_results["issues"]:
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
        stac_props = dt.attrs.get("stac_discovery", {}).get("properties", {})
        mission = stac_props.get("mission", "")

        if mission.lower().startswith("sentinel-2"):
            return True

        # Check for characteristic S2 groups
        s2_indicators = [
            "/measurements/reflectance",
            "/conditions/geometry",
            "/quality/atmosphere",
        ]

        found_indicators = sum(
            1 for indicator in s2_indicators if indicator in dt.groups
        )
        return found_indicators >= 2

    def _write_auxiliary_group(
        self, dataset: xr.Dataset, group_path: str, group_type: str, verbose: bool
    ) -> None:
        """Write auxiliary group (geometry or meteorology)."""
        # Create simple encoding following geozarr.py pattern
        from zarr.codecs import BloscCodec

        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0)
        encoding = {}
        for var_name in dataset.data_vars:
            encoding[var_name] = {"compressors": [compressor]}
        for coord_name in dataset.coords:
            encoding[coord_name] = {"compressors": None}

        # Write dataset with progress bar
        storage_options = get_storage_options(group_path)

        # Create zarr write job with progress bar
        write_job = dataset.to_zarr(
            group_path,
            mode="w",
            consolidated=True,
            zarr_format=3,
            encoding=encoding,
            storage_options=storage_options,
            compute=False,
        )
        write_job = write_job.persist()

        # Show progress bar if distributed is available
        if DISTRIBUTED_AVAILABLE:
            try:
                # this will return an interactive (non-blocking) widget if in a notebook
                # environment. To force the widget to block, provide notebook=False.
                distributed.progress(write_job, notebook=False)
            except Exception as e:
                print(f"    Warning: Could not display progress bar: {e}")
                write_job.compute()
        else:
            print(f"    Writing {group_type} zarr file...")
            write_job.compute()

        if verbose:
            print(f"  {group_type.title()} group written: {len(dataset.data_vars)} variables")
    
    def _write_measurements_group(
        self,
        group_path: str,
        pyramid_datasets: Dict[int, xr.Dataset],
        verbose: bool
    ) -> None:
        """Write measurements group metadata and consolidate all level metadata."""
        import zarr
        import os

        print("  Creating measurements group with consolidated metadata...")
        
        # Create multiscales metadata
        multiscales_attrs = self._create_multiscales_metadata_with_rio(pyramid_datasets)
        
        # Get storage options
        storage_options = get_storage_options(group_path)
        
        # Open/create the measurements group
        if storage_options:
            store = zarr.storage.FSStore(group_path, **storage_options)
        else:
            store = group_path
            
        # Create or open the measurements group
        group = zarr.open_group(store, mode='a')
        
        # Add multiscales metadata
        if multiscales_attrs:
            group.attrs['multiscales'] = [multiscales_attrs]
            if verbose:
                num_levels = len(multiscales_attrs.get('tile_matrix_set', {}).get('matrices', []))
                print(f"  Multiscales metadata added with {num_levels} levels")
        
        # Consolidate all level metadata into the group
        print("  Consolidating metadata from all pyramid levels...")
        try:
            # Force consolidation of the entire measurements tree
            zarr.consolidate_metadata(store)
            print("  ✅ Measurements group metadata consolidated")
        except Exception as e:
            print(f"  ⚠️ Warning: Metadata consolidation failed: {e}")

        return None
    
    def _create_multiscales_metadata_with_rio(self, pyramid_datasets: Dict[int, xr.Dataset]) -> Dict:
        """Create multiscales metadata using rioxarray .rio accessor, following geozarr.py format."""
        if not pyramid_datasets:
            return {}

        # Get the first available dataset to extract spatial information using .rio
        reference_ds = None
        for level in sorted(pyramid_datasets.keys()):
            if pyramid_datasets[level] is not None:
                reference_ds = pyramid_datasets[level]
                break

        if not reference_ds or not reference_ds.data_vars:
            return {}

        try:
            # Use .rio accessor to get CRS and bounds directly from the dataset
            if not hasattr(reference_ds, "rio") or not reference_ds.rio.crs:
                return {}

            native_crs = reference_ds.rio.crs
            native_bounds = reference_ds.rio.bounds()

            # Create overview levels list following geozarr.py format
            overview_levels = []
            for level in sorted(pyramid_datasets.keys()):
                if pyramid_datasets[level] is not None:
                    level_ds = pyramid_datasets[level]
                    resolution = self.pyramid_creator.pyramid_levels.get(
                        level, level * 10
                    )

                    if hasattr(level_ds, "rio"):
                        width = level_ds.rio.width
                        height = level_ds.rio.height
                        scale_factor = 2**level if level > 0 else 1

                        overview_levels.append(
                            {
                                "level": level,
                                "width": width,
                                "height": height,
                                "scale_factor": scale_factor,
                                "zoom": max(0, level),  # Simple zoom calculation
                            }
                        )

            if not overview_levels:
                return {}

            # Create tile matrix set following geozarr.py exactly
            tile_matrix_set = create_native_crs_tile_matrix_set(
                native_crs,
                native_bounds,
                overview_levels,
                "measurements",  # group prefix
            )

            # Create tile matrix limits following geozarr.py exactly
            tile_matrix_limits = _create_tile_matrix_limits(
                overview_levels, 256
            )  # tile_width=256

            # Create multiscales metadata following geozarr.py format exactly
            multiscales_metadata = {
                "tile_matrix_set": tile_matrix_set,
                "resampling_method": "average",
                "tile_matrix_limits": tile_matrix_limits,
            }

            return multiscales_metadata

        except Exception as e:
            print(
                f"  Warning: Could not create multiscales metadata with .rio accessor: {e}"
            )
            return {}

    def _simple_root_consolidation(
        self, output_path: str, pyramid_datasets: Dict[int, xr.Dataset]
    ) -> None:
        """Simple root-level metadata consolidation with proper zarr group creation."""
        try:
            print("  Performing root consolidation...")
            storage_options = get_storage_options(output_path)

            # First, ensure the root zarr group exists
            import zarr
            import os

            if storage_options:
                store = zarr.storage.FSStore(output_path, **storage_options)
            else:
                store = output_path

            # Create root zarr group if it doesn't exist
            print("  Creating root zarr group...")
            root_group = zarr.open_group(store, mode='a')
            root_group.attrs.update({
                "title": "Optimized Sentinel-2 Dataset",
                "description": "Multiscale pyramid structure for efficient access",
                "zarr_format": 3
            })

            # Ensure subgroups are properly linked
            if self.enable_streaming:
                # In streaming mode, link existing subgroups
                for subgroup in ['measurements', 'geometry', 'meteorology']:
                    subgroup_path = os.path.join(output_path, subgroup)
                    if os.path.exists(subgroup_path):
                        try:
                            if subgroup not in root_group:
                                # Link the subgroup to the root
                                subgroup_obj = zarr.open_group(subgroup_path, mode='r')
                                # Copy attributes to root group reference
                                root_group.attrs[f"{subgroup}_info"] = f"Subgroup: {subgroup}"
                        except Exception as e:
                            print(f"    Warning: Could not link subgroup {subgroup}: {e}")

            # Consolidate metadata
            try:
                zarr.consolidate_metadata(store)
                print("  ✅ Root consolidation completed")
            except Exception as e:
                print(f"  ⚠️ Warning: Metadata consolidation failed: {e}")
             
        except Exception as e:
            print(f"  ⚠️ Warning: Root consolidation failed: {e}")

    def _create_result_datatree(self, output_path: str) -> xr.DataTree:
        """Create result DataTree from written output."""
        try:
            storage_options = get_storage_options(output_path)
            return xr.open_datatree(
                output_path,
                engine="zarr",
                chunks="auto",
                storage_options=storage_options,
            )
        except Exception as e:
            print(f"Warning: Could not open result DataTree: {e}")
            return xr.DataTree()

    def _print_optimization_summary(
        self, dt_input: xr.DataTree, dt_output: xr.DataTree, output_path: str
    ) -> None:
        """Print optimization summary statistics."""
        print("\n" + "=" * 50)
        print("OPTIMIZATION SUMMARY")
        print("=" * 50)

        # Count groups
        input_groups = len(dt_input.groups) if hasattr(dt_input, "groups") else 0
        output_groups = len(dt_output.groups) if hasattr(dt_output, "groups") else 0

        print(
            f"Groups: {input_groups} → {output_groups} ({((output_groups - input_groups) / input_groups * 100):+.1f}%)"
        )

        # Estimate file count reduction
        estimated_input_files = input_groups * 10  # Rough estimate
        estimated_output_files = output_groups * 5  # Fewer files per group
        print(
            f"Estimated files: {estimated_input_files} → {estimated_output_files} ({((estimated_output_files - estimated_input_files) / estimated_input_files * 100):+.1f}%)"
        )

        # Show structure
        print("\nNew structure:")
        print("  /measurements/  (multiscale: levels 0-6)")
        if f"{output_path}/geometry" in str(dt_output):
            print("  /geometry/      (consolidated)")
        if f"{output_path}/meteorology" in str(dt_output):
            print("  /meteorology/   (consolidated)")

        print("=" * 50)


def convert_s2_optimized(
    dt_input: xr.DataTree, output_path: str, **kwargs
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
    # Separate constructor args from method args
    constructor_args = {
        "enable_sharding": kwargs.pop("enable_sharding", True),
        "spatial_chunk": kwargs.pop("spatial_chunk", 1024),
        "compression_level": kwargs.pop("compression_level", 3),
        "max_retries": kwargs.pop("max_retries", 3),
        "enable_streaming": kwargs.pop("enable_streaming", True),
    }

    # Remaining kwargs are for the convert_s2_optimized method
    method_args = kwargs

    converter = S2OptimizedConverter(**constructor_args)
    return converter.convert_s2_optimized(dt_input, output_path, **method_args)
