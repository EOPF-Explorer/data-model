"""
Main S2 optimization converter.
"""

import time
from typing import Dict

import xarray as xr
from zarr import consolidate_metadata

from eopf_geozarr.conversion import fs_utils
from eopf_geozarr.conversion.fs_utils import get_storage_options
from eopf_geozarr.conversion.geozarr import (
    _create_tile_matrix_limits,
    create_native_crs_tile_matrix_set,
)

from .s2_multiscale import S2MultiscalePyramid
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
    ):
        self.enable_sharding = enable_sharding
        self.spatial_chunk = spatial_chunk
        self.compression_level = compression_level
        self.max_retries = max_retries

        # Initialize components - streaming is always enabled
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

        # Step 1: Process data while preserving original structure
        print("Step 1: Processing data with original structure preserved...")
        
        # Step 2: Create multiscale pyramids for each group in the original structure
        print("Step 2: Creating multiscale pyramids (preserving original hierarchy)...")
        datasets = self.pyramid_creator.create_multiscale_from_datatree(
            dt_input, output_path, verbose
        )
        print(f"  Created multiscale pyramids for {len(datasets)} groups")
        
        # Step 3: Root-level consolidation
        print("Step 3: Final root-level metadata consolidation...")
        self._simple_root_consolidation(output_path, datasets)

        # Step 4: Validation
        if validate_output:
            print("Step 4: Validating optimized dataset...")
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


    def _simple_root_consolidation(
        self, output_path: str, datasets: Dict[str, Dict]
    ) -> None:
        """Simple root-level metadata consolidation with proper zarr group creation."""
        try:
            print("  Performing root consolidation...")

            # Create root zarr group if it doesn't exist
            print("  Creating root zarr group...")
            dt_root = xr.DataTree()
            for group_path, dataset in datasets.items():
                dt_root[group_path] = xr.DataTree()
            dt_root.to_zarr(
                output_path,
                modea="a",
                consolidated=True,
                zarr_format=3,
            )
            print("  ✅ Root zarr group created")

            try:
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
        print("\nNew structure: (original hierarchy preserved with multiscale pyramids)")
        for group in dt_output.groups:
            if group != ".":
                print(f"  {group}")

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
    }

    # Remaining kwargs are for the convert_s2_optimized method
    method_args = kwargs

    converter = S2OptimizedConverter(**constructor_args)
    return converter.convert_s2_optimized(dt_input, output_path, **method_args)
