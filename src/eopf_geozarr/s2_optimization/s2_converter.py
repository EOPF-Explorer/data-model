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
from eopf_geozarr.conversion.fs_utils import get_storage_options, normalize_path

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
            print(f"Starting S2 optimized conversion...")
            print(f"Input: {len(dt_input.groups)} groups")
            print(f"Output: {output_path}")
        
        # Validate input is S2
        if not self._is_sentinel2_dataset(dt_input):
            raise ValueError("Input dataset is not a Sentinel-2 product")
        
        # Step 1: Consolidate data from scattered structure
        print("Step 1: Consolidating EOPF data structure...")
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
        # Create simple encoding following geozarr.py pattern
        from zarr.codecs import BloscCodec
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0)
        encoding = {}
        for var_name in dataset.data_vars:
            encoding[var_name] = {'compressors': [compressor]}
        for coord_name in dataset.coords:
            encoding[coord_name] = {'compressors': None}
        
        # Write dataset with progress bar
        storage_options = get_storage_options(group_path)
        
        # Create zarr write job with progress bar
        write_job = dataset.to_zarr(
            group_path,
            mode='w',
            consolidated=True,
            zarr_format=3,
            encoding=encoding,
            storage_options=storage_options,
            compute=False
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
    
    def _add_root_multiscales_metadata(
        self,
        output_path: str,
        pyramid_datasets: Dict[int, xr.Dataset]
    ) -> None:
        """Add multiscales metadata at root level."""
        from eopf_geozarr.conversion.geozarr import create_native_crs_tile_matrix_set, calculate_overview_levels
        
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
            from eopf_geozarr.conversion.geozarr import consolidate_metadata
            from eopf_geozarr.conversion.fs_utils import open_zarr_group
            
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
    # Separate constructor args from method args
    constructor_args = {
        'enable_sharding': kwargs.pop('enable_sharding', True),
        'spatial_chunk': kwargs.pop('spatial_chunk', 1024),
        'compression_level': kwargs.pop('compression_level', 3),
        'max_retries': kwargs.pop('max_retries', 3)
    }
    
    # Remaining kwargs are for the convert_s2_optimized method
    method_args = kwargs
    
    converter = S2OptimizedConverter(**constructor_args)
    return converter.convert_s2_optimized(dt_input, output_path, **method_args)
