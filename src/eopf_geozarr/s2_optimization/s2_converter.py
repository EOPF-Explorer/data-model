"""
Main S2 optimization converter.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, TypedDict

import structlog
import xarray as xr
import zarr
from pydantic import TypeAdapter
from pyproj import CRS

from eopf_geozarr.conversion.fs_utils import get_storage_options
from eopf_geozarr.conversion.geozarr import get_zarr_group
from eopf_geozarr.data_api.s1 import Sentinel1Root
from eopf_geozarr.data_api.s2 import Sentinel2Root
from eopf_geozarr.types import (
    BoundingBox2D,
    make_bounding_box,
    make_crs_code,
    make_epsg_code,
)

from .s2_multiscale import create_multiscale_from_datatree

if TYPE_CHECKING:
    from collections.abc import Mapping

log = structlog.get_logger()


def initialize_crs_from_dataset(dt_input: xr.DataTree) -> CRS | None:
    """
    Initialize CRS from dataset by checking data variables.

    Args:
        dt_input: Input DataTree

    Returns:
        CRS object if found, None otherwise
    """
    # For CPM >= 2.6.0, the EPSG code is stored in root attributes
    epsg_cpm_260 = dt_input.attrs.get("other_metadata", {}).get(
        "horizontal_CRS_code",
        dt_input.attrs.get("other_metadata", {}).get("horizontal_crs_code", None),
    )
    if epsg_cpm_260 is not None:
        try:
            # Handles both integer (32632) and string ("EPSG:32632" or "32632") formats
            epsg_code = make_epsg_code(epsg_cpm_260)
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

        # Check if dataset has rio accessor with CRS. rioxarray returns a
        # rasterio CRS; convert it to a pyproj CRS (the declared return type),
        # which also validates the value at runtime.
        if hasattr(dataset, "rio"):
            try:
                rio_crs = dataset.rio.crs
                if rio_crs is not None:
                    ds_crs = CRS.from_user_input(rio_crs)
                    log.info("Initialized CRS from dataset", crs=str(ds_crs))
                    return ds_crs
            except Exception:
                log.debug("Failed to get CRS from dataset rio accessor")

        # Check data variables for CRS information
        for var in dataset.data_vars.values():
            if hasattr(var, "rio"):
                try:
                    rio_crs = var.rio.crs
                    if rio_crs is not None:
                        var_crs = CRS.from_user_input(rio_crs)
                        log.info("Initialized CRS from variable", crs=str(var_crs))
                        return var_crs
                except Exception:
                    log.debug("Failed to get CRS from variable rio accessor")

            # Check for proj:epsg attribute
            if "proj:epsg" in var.attrs:
                try:
                    epsg = make_epsg_code(var.attrs["proj:epsg"])
                    crs = CRS.from_epsg(epsg)
                    log.info("Initialized CRS from EPSG code", epsg=epsg)
                except Exception:
                    log.debug("Failed to initialize CRS from proj:epsg attribute")
                else:
                    return crs

    log.warning("Could not initialize CRS from dataset")
    return None


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
        output_group=zarr.open_group(output_path),
        spatial_chunk=spatial_chunk,
        enable_sharding=enable_sharding,
        keep_scale_offset=False,
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


def convert_s2_optimized(
    dt_input: xr.DataTree,
    *,
    output_path: str,
    enable_sharding: bool,
    spatial_chunk: int,
    compression_level: int,
    validate_output: bool,
    keep_scale_offset: bool,
    experimental_scale_offset_codec: bool = False,
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
        keep_scale_offset: Whether to preserve scale-offset encoding of the source data.
        experimental_scale_offset_codec: Push CF scale-offset into zarr codec pipeline.
        max_retries: Maximum number of retries for network operations

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

    # Initialize CRS from dataset
    crs = initialize_crs_from_dataset(dt_input)

    # Step 1: Process data while preserving original structure
    log.info("Step 1: Processing data with original structure preserved")

    # Step 2: Create multiscale pyramids for each group in the original structure
    log.info("Step 2: Creating multiscale pyramids (preserving original hierarchy)")

    output_group = zarr.open_group(output_path)

    datasets = create_multiscale_from_datatree(
        dt_input,
        output_group=output_group,
        spatial_chunk=spatial_chunk,
        enable_sharding=enable_sharding,
        crs=crs,
        keep_scale_offset=keep_scale_offset,
        experimental_scale_offset_codec=experimental_scale_offset_codec,
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


def simple_root_consolidation(output_path: str, datasets: Mapping[str, object]) -> None:
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

    # Write the store-root spatial footprint (geozarr minispec, Store Root section).
    # Aggregates child-group `spatial:bbox` values, reprojects them to EPSG:4326
    # and writes the union on the root `zarr.json`.
    write_store_root_bbox(output_path)

    # consolidate reflectance group metadata
    zarr.consolidate_metadata(output_path + "/measurements/reflectance", zarr_format=3)

    # consolidate root group metadata
    zarr.consolidate_metadata(output_path, zarr_format=3)


def _as_bbox(value: object) -> BoundingBox2D | None:
    """Return *value* as a validated bounding box, or ``None`` if it is not one.

    ``spatial:bbox`` is read from stored metadata, so its type is not known
    statically; unlike a bare ``make_bounding_box`` call, the store walker
    tolerates absent or malformed values by skipping the group.
    """
    try:
        return make_bounding_box(value)
    except (TypeError, ValueError):
        return None


def write_store_root_bbox(output_path: str) -> None:
    """Write `spatial:bbox` and `proj:code` at the store root.

    Walks the zarr store, collects every child-group `spatial:bbox` along with
    its `proj:code`, reprojects each to EPSG:4326 and writes the union plus
    the CRS code on the root group. The CRS is always declared explicitly per
    the Store Root section of the minispec — there is no implicit default.
    """
    from pyproj import Transformer

    root = zarr.open_group(output_path, mode="r+")

    bboxes_4326: list[BoundingBox2D] = []

    def _walk(group: zarr.Group) -> None:
        attrs = dict(group.attrs)
        bbox = attrs.get("spatial:bbox")
        raw_code = attrs.get("proj:code")
        # spatial:bbox comes from stored metadata; verify it is a 4-element
        # numeric sequence before use rather than trusting the type.
        corners = _as_bbox(bbox)
        if corners is not None:
            x0, y0, x1, y1 = corners
            # A group with a bbox but no proj:code is treated as already EPSG:4326;
            # a present-but-malformed code fails loudly rather than being skipped.
            code = make_crs_code(raw_code) if raw_code is not None else None
            if code is not None and code != "EPSG:4326":
                transformer = Transformer.from_crs(code, "EPSG:4326", always_xy=True)
                xmin, ymin = transformer.transform(x0, y0)
                xmax, ymax = transformer.transform(x1, y1)
                bboxes_4326.append(BoundingBox2D((xmin, ymin, xmax, ymax)))
            else:
                bboxes_4326.append(corners)
        for child in group.groups():
            _walk(child[1])

    for _, child_group in root.groups():
        _walk(child_group)

    if not bboxes_4326:
        log.warning("No child-group spatial:bbox found; skipping store-root bbox")
        return

    xmin = min(b[0] for b in bboxes_4326)
    ymin = min(b[1] for b in bboxes_4326)
    xmax = max(b[2] for b in bboxes_4326)
    ymax = max(b[3] for b in bboxes_4326)
    root.attrs["spatial:bbox"] = [xmin, ymin, xmax, ymax]
    root.attrs["proj:code"] = "EPSG:4326"
    log.info("Wrote store-root spatial:bbox", bbox=[xmin, ymin, xmax, ymax])


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
    storage_options = get_storage_options(output_path)
    return xr.open_datatree(
        output_path,
        engine="zarr",
        chunks="auto",
        storage_options=storage_options,
    )


def is_sentinel2_dataset(group: zarr.Group) -> bool:
    from eopf_geozarr.pyz.v2 import GroupSpec

    adapter = TypeAdapter(Sentinel1Root | Sentinel2Root)
    try:
        model = adapter.validate_python(GroupSpec.from_zarr(group).model_dump())
    except ValueError as e:
        log.warning("Could not validate Sentinel-2 dataset", error=str(e))
        return False

    return isinstance(model, Sentinel2Root)


class ValidationResult(TypedDict):
    """Result of validating an optimized Sentinel-2 dataset."""

    is_valid: bool
    issues: list[str]
    warnings: list[str]
    summary: dict[str, object]


def validate_optimized_dataset(dataset_path: str) -> ValidationResult:
    """
    Validate an optimized Sentinel-2 dataset.

    Args:
        dataset_path: Path to the optimized dataset

    Returns:
        Validation results dictionary
    """
    return {"is_valid": True, "issues": [], "warnings": [], "summary": {}}

    # Placeholder for validation logic
