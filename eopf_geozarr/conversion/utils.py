"""Utility functions for GeoZarr conversion."""

import numpy as np
import rasterio  # noqa: F401  # Import to enable .rio accessor
import xarray as xr


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
    for data_var in ds.data_vars:
        if data_var != var_name and "grid_mapping" in ds[data_var].attrs:
            if ds[data_var].attrs["grid_mapping"] == var_name:
                return True
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
