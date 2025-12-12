"""
Main S2 optimization converter.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, TypedDict

import structlog
import xarray as xr
import zarr
from pydantic import TypeAdapter
from pyproj import CRS
from zarr.core.sync import sync
from zarr.storage._common import make_store

from eopf_geozarr.conversion.fs_utils import get_storage_options
from eopf_geozarr.conversion.geozarr import get_zarr_group
from eopf_geozarr.data_api.s1 import Sentinel1Root
from eopf_geozarr.data_api.s2 import Sentinel2Root
from eopf_geozarr.zarrio import ChunkEncodingSpec, reencode_group

from .s2_multiscale import create_multiscale_from_datatree, write_geo_metadata

log = structlog.get_logger()


def initialize_crs_from_dataset(dt_input: xr.DataTree) -> CRS:
    """
    Initialize CRS from dataset by checking data variables.

    Args:
        dt_input: Input DataTree

    Returns:
        CRS object if found, None otherwise
    """
    # For CPM >= 2.6.0, the EPSG code is stored in root attributes
    epsg_cpm_260 = dt_input.attrs.get("other_metadata", {}).get("horizontal_CRS_code", None)
    if epsg_cpm_260 is not None:
        try:
            # Handle both integer (32632) and string ("EPSG:32632" or "32632") formats
            if isinstance(epsg_cpm_260, str):
                # Extract numeric part from string like "EPSG:32632" or "32632"
                epsg_code = int(epsg_cpm_260.split(":")[-1])
            else:
                # Already an integer
                epsg_code = int(epsg_cpm_260)
            crs = CRS.from_epsg(epsg_code)
            log.info("Initialized CRS from CPM 2.6.0+ metadata", epsg=epsg_code)
        except Exception as e:
            log.warning(
                "Failed to initialize CRS from CPM 2.6.0+ metadata",
                epsg=epsg_cpm_260,
                error=str(e),
            )
        else:
            return crs

    for group_path in dt_input.groups:
        if group_path == ".":
            continue
        group_node = dt_input[group_path]
        if not hasattr(group_node, "ds") or group_node.ds is None:
            continue
        dataset = group_node.ds

        # Check if dataset has rio accessor with CRS
        if hasattr(dataset, "rio"):
            try:
                crs = dataset.rio.crs
                if crs is not None:
                    log.info("Initialized CRS from dataset", crs=str(crs))
                    return crs
            except Exception:
                pass

        # Check data variables for CRS information
        for var in dataset.data_vars.values():
            if hasattr(var, "rio"):
                try:
                    crs = var.rio.crs
                    if crs is not None:
                        log.info("Initialized CRS from variable", crs=str(crs))
                        return crs
                except Exception:
                    pass

            # Check for proj:epsg attribute
            if "proj:epsg" in var.attrs:
                try:
                    epsg = var.attrs["proj:epsg"]
                    crs = CRS.from_epsg(epsg)
                    log.info("Initialized CRS from EPSG code", epsg=epsg)
                except Exception:
                    pass
    raise ValueError("No CRS found.")


def convert_s2(
    dt_input: xr.DataTree,
    output_path: str,
    validate_output: bool,
    enable_sharding: bool,
    spatial_chunk: int,
) -> xr.DataTree:
    """
    Convert S2 dataset to optimized structure.

        Args:
            dt_input: Input Sentinel-2 DataTree
            output_path: Output path for optimized dataset
            validate_output: Whether to validate the output
            verbose: Enable verbose logging

        Returns:
            Optimized DataTree
    """
    start_time = time.time()

    log.info(
        "Starting S2 optimized conversion",
        num_groups=len(dt_input.groups),
        output_path=output_path,
    )

    # Validate input is S2
    if not is_sentinel2_dataset(get_zarr_group(dt_input)):
        raise ValueError("Input dataset is not a Sentinel-2 product")

    # Step 1: Process data while preserving original structure
    log.info("Step 1: Processing data with original structure preserved")

    # Step 2: Create multiscale pyramids for each group in the original structure
    log.info("Step 2: Creating multiscale pyramids (preserving original hierarchy)")
    datasets = create_multiscale_from_datatree(
        dt_input,
        output_path,
        spatial_chunk=spatial_chunk,
        enable_sharding=enable_sharding,
    )

    log.info("Created multiscale pyramids", num_groups=len(datasets))

    # Step 3: Root-level consolidation
    log.info("Step 3: Final root-level metadata consolidation")
    simple_root_consolidation(output_path, datasets)

    # Step 4: Validation
    if validate_output:
        log.info("Step 4: Validating optimized dataset")
        validation_results = validate_optimized_dataset(output_path)
        if not validation_results["is_valid"]:
            log.warning("Validation issues found", issues=validation_results["issues"])

    # Create result DataTree
    result_dt = create_result_datatree(output_path)

    total_time = time.time() - start_time
    log.info("Optimization complete", duration_seconds=round(total_time, 2))

    optimization_summary(dt_input, result_dt, output_path)

    return result_dt


class ConvertS2Params(TypedDict):
    enable_sharding: bool
    spatial_chunk: int
    compression_level: int
    max_retries: int


def add_crs_and_grid_mapping(group: zarr.Group, crs: CRS) -> None:
    """
    Add crs and grid mapping elements to a dataset.
    """
    ds = xr.open_dataset(group.store, group=group.path, engine="zarr", consolidated=False)
    write_geo_metadata(ds, crs=crs)

    for var in ds.data_vars:
        new_attrs = ds[var].attrs.copy()
        new_attrs["coordinates"] = "spatial_ref"
        group[var].attrs.update(new_attrs | {"grid_mapping": "spatial_ref"})

    group.create_array(
        "spatial_ref",
        shape=ds["spatial_ref"].shape,
        dtype=ds["spatial_ref"].dtype,
        attributes=ds["spatial_ref"].attrs,
        compressors=None,
        filters=None,
    )

    # Set grid_mapping attribute on the group itself
    group.attrs.update({"grid_mapping": "spatial_ref"})


def convert_s2_optimized(
    dt_input: xr.DataTree,
    *,
    output_path: str,
    enable_sharding: bool,
    spatial_chunk: int,
    compression_level: int,
    validate_output: bool,
    omit_nodes: set[str] | None = None,
    max_retries: int = 3,
) -> xr.DataTree:
    """
    Convenience function for S2 optimization.

    Args:
        dt_input: Input Sentinel-2 DataTree
        output_path: Output path
        enable_sharding: Enable Zarr v3 sharding
        spatial_chunk: Spatial chunk size
        compression_level: Compression level 1-9
        validate_output: Whether to validate the output
        max_retries: Maximum number of retries for network operations

    Returns:
        Optimized DataTree
    """

    if omit_nodes is None:
        omit_nodes = set()

    start_time = time.time()
    zg = get_zarr_group(dt_input)
    s2root_model = Sentinel2Root.from_zarr(zg)
    crs = s2root_model.crs

    log.info(
        "Starting S2 optimized conversion",
        num_groups=len(dt_input.groups),
        output_path=output_path,
    )
    # Validate input is S2
    if not is_sentinel2_dataset(zg):
        raise ValueError("Input dataset is not a Sentinel-2 product")

    out_store = sync(make_store(output_path))

    log.info("Re-encoding source data to Zarr V3")

    # Define a chunk reencoding function based on the requested chunking
    def chunk_reencoder(array: zarr.Array[Any]) -> ChunkEncodingSpec:
        group_name = str(Path(array.path).parent)
        in_measurements_group = (
            group_name.startswith("r")
            and group_name.endswith("m")
            and "/measurements/" in group_name
        )
        if in_measurements_group and array.ndim > 0:
            if array.ndim == 1:
                return {"write_chunks": (spatial_chunk,)}
            if array.ndim == 2:
                return {"write_chunks": (spatial_chunk, spatial_chunk)}
            return {"write_chunks": (1,) * (array.ndim - 2) + (spatial_chunk, spatial_chunk)}
        return {"write_chunks": array.chunks}

    out_group = reencode_group(
        zg,
        out_store,
        path="",
        overwrite=True,
        chunk_reencoder=chunk_reencoder,
        omit_nodes=omit_nodes,
    )

    if "measurements" not in omit_nodes:
        log.info("Adding CRS elements to datasets in measurements")
        for _, subgroup in out_group["measurements"].groups():
            for _, dataset in subgroup.groups():
                add_crs_and_grid_mapping(dataset, crs=crs)

    if "quality" not in omit_nodes:
        log.info("Adding CRS elements to quality datasets")
        for _, subgroup in out_group["quality"].groups():
            for _, dataset in subgroup.groups():
                add_crs_and_grid_mapping(dataset, crs=crs)

    # Step 2: Create multiscale pyramids for each group in the original structure
    log.info("Adding multiscale levels")
    new_dt_input = xr.open_datatree(
        out_store, engine="zarr", chunks="auto", decode_timedelta=True, consolidated=False
    )
    datasets = create_multiscale_from_datatree(
        new_dt_input,
        output_path,
        spatial_chunk=spatial_chunk,
        enable_sharding=enable_sharding,
        crs=crs,
    )

    log.info("Created multiscale pyramids", num_groups=len(datasets))

    # Step 3: Root-level consolidation
    log.info("Step 3: Final root-level metadata consolidation")
    simple_root_consolidation(output_path, datasets)

    # Step 4: Validation
    if validate_output:
        log.info("Step 4: Validating optimized dataset")
        validation_results = validate_optimized_dataset(output_path)
        if not validation_results["is_valid"]:
            log.warning("Validation issues found", issues=validation_results["issues"])

    # Create result DataTree
    result_dt = create_result_datatree(output_path)

    total_time = time.time() - start_time
    log.info("Optimization complete", duration_seconds=round(total_time, 2))

    optimization_summary(dt_input, result_dt, output_path)

    return result_dt


def simple_root_consolidation(output_path: str, datasets: dict[str, dict]) -> None:
    """Simple root-level metadata consolidation with proper zarr group creation."""
    # create missing intermediary groups (/conditions, /quality, etc.)
    # using the keys of the datasets dict
    missing_groups = set()
    for group_path in datasets:
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
            consolidated=False,
        )

    # Create root zarr group if it doesn't exist
    log.info("Creating root zarr group")
    dt_root = xr.DataTree()
    dt_root.to_zarr(
        output_path,
        mode="a",
        consolidated=False,
        zarr_format=3,
    )
    dt_root = xr.DataTree()
    for group_path in datasets:
        dt_root[group_path] = xr.DataTree()

    dt_root.to_zarr(
        output_path,
        mode="r+",
        consolidated=False,
        zarr_format=3,
    )
    log.info("Root zarr group created")

    # consolidate reflectance group metadata
    zarr.consolidate_metadata(output_path + "/measurements/reflectance", zarr_format=3)

    # consolidate root group metadata
    zarr.consolidate_metadata(output_path, zarr_format=3)


def optimization_summary(dt_input: xr.DataTree, dt_output: xr.DataTree, output_path: str) -> None:
    """Print optimization summary statistics."""
    # Count groups
    input_groups = len(dt_input.groups) if hasattr(dt_input, "groups") else 0
    output_groups = len(dt_output.groups) if hasattr(dt_output, "groups") else 0

    # Estimate file count reduction
    estimated_input_files = input_groups * 10  # Rough estimate
    estimated_output_files = output_groups * 5  # Fewer files per group
    group_change_pct = (
        ((output_groups - input_groups) / input_groups * 100) if input_groups > 0 else 0
    )
    file_change_pct = (
        ((estimated_output_files - estimated_input_files) / estimated_input_files * 100)
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


def create_result_datatree(output_path: str) -> xr.DataTree:
    """Create result DataTree from written output."""
    try:
        storage_options = get_storage_options(output_path)
        return xr.open_datatree(
            output_path,
            engine="zarr",
            chunks="auto",
            storage_options=storage_options,
            decode_timedelta=True,
        )
    except Exception as e:
        log.warning("Could not open result DataTree", error=str(e))
        return xr.DataTree()


def is_sentinel2_dataset(group: zarr.Group) -> bool:
    from eopf_geozarr.pyz.v2 import GroupSpec

    adapter = TypeAdapter(Sentinel1Root | Sentinel2Root)  # type: ignore[var-annotated]
    try:
        model = adapter.validate_python(GroupSpec.from_zarr(group).model_dump())
    except ValueError as e:
        log.warning("Could not validate Sentinel-2 dataset", error=str(e))
        return False

    return isinstance(model, Sentinel2Root)


def validate_optimized_dataset(dataset_path: str) -> dict[str, Any]:
    """
    Validate an optimized Sentinel-2 dataset.

    Args:
        dataset_path: Path to the optimized dataset

    Returns:
        Validation results dictionary
    """
    return {"is_valid": True, "issues": [], "warnings": [], "summary": {}}

    # Placeholder for validation logic
