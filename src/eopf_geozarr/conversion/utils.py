"""Utility functions for GeoZarr conversion."""

# mypy: disable-error-code=import-not-found

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio  # noqa: F401  # Import to enable .rio accessor
import xarray as xr

from . import fs_utils

# Verbose logging flag controlled by environment variable
_VERBOSE = os.environ.get("EOPF_GEOZARR_VERBOSE", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def log(message: str) -> None:
    """Print message when verbose mode is enabled via EOPF_GEOZARR_VERBOSE."""
    if _VERBOSE:
        print(message)


def _safe_get_chunk_shape(meta: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    """Extract chunk shape from Zarr v3 or v2 metadata."""
    cg = meta.get("chunk_grid")
    if isinstance(cg, dict):
        chunk_shape_val = cg.get("chunk_shape")
        if isinstance(chunk_shape_val, list):
            return tuple(int(x) for x in chunk_shape_val)
        grid = cg.get("grid")
        if isinstance(grid, dict):
            grid_cs = grid.get("chunk_shape")
            if isinstance(grid_cs, list):
                return tuple(int(x) for x in grid_cs)
    # Zarr v2 style
    chunks_val = meta.get("chunks")
    if isinstance(chunks_val, (list, tuple)):
        return tuple(int(x) for x in chunks_val)
    return None


def estimate_array_completeness(array_dir: str) -> Tuple[bool, Dict[str, Any]]:
    """Estimate if a Zarr array directory appears complete.

    Heuristics:
    - zarr.json exists and is readable
    - shape present in metadata
    - chunk directory exists and non-empty
    - if chunk shape known: count of chunk files approximates expected grid size
    - no zero-byte chunk files among a sample
    """
    info: Dict[str, Any] = {"path": array_dir, "problems": []}
    try:
        meta_path = fs_utils.normalize_path(f"{array_dir}/zarr.json")
        meta = fs_utils.read_json_metadata(meta_path)
    except Exception as e:
        info["problems"].append(f"metadata_unreadable:{e}")
        return False, info

    shape = meta.get("shape") if isinstance(meta, dict) else None
    if not isinstance(shape, (list, tuple)):
        info["problems"].append("missing_shape")
        return False, info
    shape_t = tuple(int(x) for x in shape)
    chunk_shape = _safe_get_chunk_shape(meta)
    info["shape"] = shape_t
    info["chunk_shape"] = chunk_shape

    fs = fs_utils.get_filesystem(array_dir)
    chunks_base = fs_utils.normalize_path(f"{array_dir}/c")
    if not fs.exists(chunks_base):
        info["problems"].append("missing_chunk_dir")
        return False, info

    # Find chunk files under c/
    try:
        files = [p for p in fs.find(chunks_base) if not p.endswith("/")]
    except Exception:
        # Fallback to ls if find not available
        try:
            files = fs.ls(chunks_base, detail=False)
        except Exception as e:
            info["problems"].append(f"list_failed:{e}")
            return False, info

    info["chunk_files_found"] = len(files)

    if len(files) == 0:
        info["problems"].append("no_chunks")
        return False, info

    # Zero-byte sample check
    zero_bytes = 0
    sample = files[:5] + files[-5:] if len(files) > 10 else files
    for p in sample:
        try:
            s = fs.size(p)
            if s == 0:
                zero_bytes += 1
        except Exception:
            continue
    if zero_bytes > 0:
        info["problems"].append("zero_byte_chunks")

    # Expected chunk count estimate if chunk_shape known
    if chunk_shape is not None:
        import math

        dims = len(shape_t)
        # align chunk_shape length with dims (prepend 1s if needed)
        cs = (1,) * (dims - len(chunk_shape)) + tuple(chunk_shape)
        expected = 1
        for i in range(dims):
            expected *= math.ceil(shape_t[i] / max(1, cs[i]))
        info["expected_chunk_count"] = expected
        # allow some leeway; if far fewer files, likely incomplete
        if len(files) < max(1, int(0.9 * expected)):
            info["problems"].append("chunk_count_below_expected")

    is_complete = len(info["problems"]) == 0
    return is_complete, info


def audit_group_completeness(group_path: str, variables: List[str]) -> Tuple[bool, Dict[str, Any]]:
    """Audit a Zarr group for variable completeness.

    Returns (ok, report). ok = False if any variable appears incomplete.
    """
    report: Dict[str, Any] = {"group": group_path, "variables": {}}
    all_ok = True
    for v in variables:
        array_dir = fs_utils.normalize_path(f"{group_path}/{v}")
        ok, details = estimate_array_completeness(array_dir)
        report["variables"][v] = details
        if not ok:
            all_ok = False
    report["ok"] = all_ok
    return all_ok, report


def downsample_2d_array(
    source_data: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """
    Downsample a 2D array using block averaging.

    Parameters
    ----------
    source_data : numpy.ndarray
        Source 2D array
    target_height : int
        Target height
    target_width : int
        Target width

    Returns
    -------
    numpy.ndarray
        Downsampled 2D array
    """
    source_height, source_width = source_data.shape

    # Calculate block sizes
    block_size_y = source_height // target_height
    block_size_x = source_width // target_width

    if block_size_y > 1 and block_size_x > 1:
        # Block averaging
        reshaped = source_data[: target_height * block_size_y, : target_width * block_size_x]
        reshaped = reshaped.reshape(target_height, block_size_y, target_width, block_size_x)
        downsampled = reshaped.mean(axis=(1, 3))
    else:
        # Simple subsampling
        y_indices = np.linspace(0, source_height - 1, target_height, dtype=int)
        x_indices = np.linspace(0, source_width - 1, target_width, dtype=int)
        downsampled = source_data[np.ix_(y_indices, x_indices)]

    return downsampled


def is_grid_mapping_variable(ds: xr.Dataset, var_name: str) -> bool:
    """
    Check if a variable is a grid_mapping variable by looking for references to it.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to check
    var_name : str
        Variable name to check

    Returns
    -------
    bool
        True if this variable is referenced as a grid_mapping
    """
    for data_var in list(ds.data_vars):
        try:
            if data_var != var_name and "grid_mapping" in ds[data_var].attrs:
                if ds[data_var].attrs.get("grid_mapping") == var_name:
                    return True
        except Exception:
            continue
    return False


def calculate_aligned_chunk_size(dimension_size: int, target_chunk_size: int) -> int:
    """
    Calculate a chunk size that divides evenly into the dimension size.

    This ensures that Zarr chunks align properly with the data dimensions,
    preventing chunk overlap issues when writing with Dask.

    Parameters
    ----------
    dimension_size : int
        Size of the dimension to chunk
    target_chunk_size : int
        Desired chunk size

    Returns
    -------
    int
        Aligned chunk size that divides evenly into dimension_size
    """
    if target_chunk_size >= dimension_size:
        return dimension_size

    # Find the largest divisor of dimension_size that is <= target_chunk_size
    for chunk_size in range(target_chunk_size, 0, -1):
        if dimension_size % chunk_size == 0:
            return chunk_size

    # Fallback: return 1 if no good divisor found
    return 1


def validate_existing_band_data(
    existing_group: xr.Dataset, var_name: str, reference_ds: xr.Dataset
) -> bool:
    """
    Validate that a specific band exists and is complete in the dataset.

    Parameters
    ----------
    existing_group : xarray.Dataset
        Existing dataset to validate
    var_name : str
        Name of the variable to validate
    reference_ds : xarray.Dataset
        Reference dataset structure for comparison

    Returns
    -------
    bool
        True if the variable exists and is valid, False otherwise
    """
    try:
        # Check if the variable exists
        if var_name not in existing_group.data_vars and var_name not in existing_group.coords:
            return False

        # Check shape matches
        if var_name in reference_ds.data_vars:
            expected_shape = reference_ds[var_name].shape
            existing_shape = existing_group[var_name].shape

            if expected_shape != existing_shape:
                return False

        # Check required attributes for data variables
        if var_name in reference_ds.data_vars and not is_grid_mapping_variable(
            reference_ds, var_name
        ):
            required_attrs = ["_ARRAY_DIMENSIONS", "standard_name"]
            for attr in required_attrs:
                if attr not in existing_group[var_name].attrs:
                    return False

        # Check rio CRS
        if existing_group.rio.crs != reference_ds.rio.crs:
            return False

        # Basic data integrity check for data variables
        if var_name in existing_group.data_vars and not is_grid_mapping_variable(
            existing_group, var_name
        ):
            try:
                # Just check if we can access the array metadata without reading data
                array_info = existing_group[var_name]
                if array_info.size == 0:
                    return False
                # read a piece of data to ensure it's valid
                test = array_info.isel({dim: 0 for dim in array_info.dims}).values.mean()
                if np.isnan(test):
                    return False
            except Exception as e:
                print(f"Error validating variable {var_name}: {e}")
                return False

        return True

    except Exception:
        return False


def validate_existing_band_metadata(
    existing_group: xr.Dataset, var_name: str, reference_ds: xr.Dataset
) -> bool:
    """Fast validation without reading data values.

    Checks existence, shape, minimal attrs, and CRS equality only.
    """
    try:
        if var_name not in existing_group.data_vars and var_name not in existing_group.coords:
            return False

        if var_name in reference_ds.data_vars:
            expected_shape = reference_ds[var_name].shape
            existing_shape = existing_group[var_name].shape
            if expected_shape != existing_shape:
                return False

        if var_name in reference_ds.data_vars and not is_grid_mapping_variable(
            reference_ds, var_name
        ):
            for attr in ("_ARRAY_DIMENSIONS", "standard_name"):
                if attr not in existing_group[var_name].attrs:
                    return False

        if getattr(existing_group, "rio", None) is not None:
            if existing_group.rio.crs != reference_ds.rio.crs:
                return False

        return True
    except Exception:
        return False
