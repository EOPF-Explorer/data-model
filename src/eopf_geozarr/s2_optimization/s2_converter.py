"""
Main S2 optimization converter.
"""

import importlib.util
import time
from typing import Any, Dict

import structlog
import xarray as xr

from eopf_geozarr.conversion.fs_utils import get_storage_options

from .s2_multiscale import S2MultiscalePyramid
from .s2_validation import S2OptimizationValidator

log = structlog.get_logger()

DISTRIBUTED_AVAILABLE = importlib.util.find_spec("distributed") is not None


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
            log.info(
                "Starting S2 optimized conversion",
                num_groups=len(dt_input.groups),
                output_path=output_path,
            )

        # Validate input is S2
        if not self._is_sentinel2_dataset(dt_input):
            raise ValueError("Input dataset is not a Sentinel-2 product")

        # Step 1: Process data while preserving original structure
        log.info("Step 1: Processing data with original structure preserved...")
        
        # Step 2: Create multiscale pyramids for each group in the original structure
        log.info("Step 2: Creating multiscale pyramids (preserving original hierarchy)...")
        datasets = self.pyramid_creator.create_multiscale_from_datatree(
            dt_input, output_path, verbose
        )
        log.info(f"  Created multiscale pyramids for {len(datasets)} groups")
        
        # Step 3: Root-level consolidation
        log.info("Step 3: Final root-level metadata consolidation...")
        self._simple_root_consolidation(output_path, datasets)

        # Step 4: Validation
        if validate_output:
            log.info("Step 4: Validating optimized dataset...")
            validation_results = self.validator.validate_optimized_dataset(output_path)
            if not validation_results["is_valid"]:
                log.info("  Warning: Validation issues found:")
                for issue in validation_results["issues"]:
                    log.info(f"    - {issue}")

        # Create result DataTree
        result_dt = self._create_result_datatree(output_path)

        total_time = time.time() - start_time
        log.info("Optimization complete", duration_seconds=round(total_time, 2))

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
            log.info("Performing root consolidation")

            # create missing intermediary groups (/conditions, /quality, etc.)
            # using the keys of the datasets dict
            missing_groups = set()
            for group_path in datasets.keys():
                # extract all the parent paths
                parts = group_path.strip("/").split("/")
                for i in range(1, len(parts)):
                    parent_path = "/" + "/".join(parts[:i])
                    if parent_path not in datasets:
                        missing_groups.add(parent_path)
            for group_path in missing_groups:
                dt_parent = xr.DataTree()
                dt_parent.to_zarr(
                    output_path + group_path,
                    mode="a",
                    zarr_format=3,
                )

            # Create root zarr group if it doesn't exist
            log.info("Creating root zarr group")
            dt_root = xr.DataTree()
            dt_root.to_zarr(
                output_path,
                mode="a",
                consolidated=True,
                zarr_format=3,
            )
            dt_root = xr.DataTree()
            for group_path, dataset in datasets.items():
                dt_root[group_path] = xr.DataTree()
            dt_root.to_zarr(
                output_path,
                mode="r+",
                consolidated=True,
                zarr_format=3,
            )
            log.info("Root zarr group created")

            try:
                log.info("Root consolidation completed")
            except Exception as e:
                log.warning("Metadata consolidation failed", error=str(e))

        except Exception as e:
            log.warning("Root consolidation failed", error=str(e))

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
            log.warning("Could not open result DataTree", error=str(e))
            return xr.DataTree()

    def _print_optimization_summary(
        self, dt_input: xr.DataTree, dt_output: xr.DataTree, output_path: str
    ) -> None:
        """Print optimization summary statistics."""
        # Count groups
        input_groups = len(dt_input.groups) if hasattr(dt_input, "groups") else 0
        output_groups = len(dt_output.groups) if hasattr(dt_output, "groups") else 0

        # Estimate file count reduction
        estimated_input_files = input_groups * 10  # Rough estimate
        estimated_output_files = output_groups * 5  # Fewer files per group
        group_change_pct = (
            ((output_groups - input_groups) / input_groups * 100)
            if input_groups > 0
            else 0
        )
        file_change_pct = (
            (
                (estimated_output_files - estimated_input_files)
                / estimated_input_files
                * 100
            )
            if estimated_input_files > 0
            else 0
        )

        log.info(
            "OPTIMIZATION SUMMARY",
            input_groups=input_groups,
            output_groups=output_groups,
            group_change_pct=f"{group_change_pct:+.1f}%",
            estimated_input_files=estimated_input_files,
            estimated_output_files=estimated_output_files,
            file_change_pct=f"{file_change_pct:+.1f}%",
            output_path=output_path,
            groups=[g for g in dt_output.groups if g != "."],
        )


def convert_s2_optimized(
    dt_input: xr.DataTree, output_path: str, **kwargs: Any
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
