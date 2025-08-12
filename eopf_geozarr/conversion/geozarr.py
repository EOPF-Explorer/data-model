"""
GeoZarr-spec 0.4 compliant conversion tools for EOPF datasets.

This module provides functions to convert EOPF datasets to GeoZarr-spec 0.4 compliant format
while maintaining native projections and using /2 downsampling logic.

Key compliance features:
- _ARRAY_DIMENSIONS attributes on all arrays
- CF standard names for all variables
- grid_mapping attributes referencing CF grid_mapping variables
- GeoTransform attributes in grid_mapping variables
- Native CRS preservation (no TMS reprojection)
- Proper multiscales metadata structure
"""

import dataclasses
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import zarr
import zarr.api
import zarr.api.asynchronous
import zarr.core
import zarr.core.common
import zarr.core.group
from zarr.core.sync import sync
from zarr.storage import StoreLike
from zarr.storage._common import make_store_path

from . import fs_utils, utils


def create_geozarr_dataset(
    dt_input: xr.DataTree,
    groups: List[str],
    output_path: str,
    spatial_chunk: int = 4096,
    min_dimension: int = 256,
    tile_width: int = 256,
    max_retries: int = 3,
    crs_groups: Optional[List[str]] = None,
) -> xr.DataTree:
    """
    Create a GeoZarr-spec 0.4 compliant dataset from EOPF data.

    Parameters
    ----------
    dt_input : xr.DataTree
        Input EOPF DataTree
    groups : list[str]
        List of group names to process as Geozarr datasets
    output_path : str
        Output path for the Zarr store
    spatial_chunk : int, default 4096
        Spatial chunk size for encoding
    min_dimension : int, default 256
        Minimum dimension for overview levels
    tile_width : int, default 256
        Tile width for TMS compatibility
    max_retries : int, default 3
        Maximum number of retries for network operations
    crs_groups : list[str], optional
        List of group names that need CRS information added on best-effort basis

    Returns
    -------
    xr.DataTree
        DataTree containing the GeoZarr compliant data
    """
    from zarr.codecs import BloscCodec

    dt = dt_input.copy()

    # Set up compression
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0)

    # Get the measurements datasets prepared for GeoZarr compliance
    geozarr_groups = setup_datatree_metadata_geozarr_spec_compliant(dt, groups)

    # Create the GeoZarr compliant store through recursive processing
    # CRS groups will be handled within recursive_copy before writing
    dt_geozarr = recursive_copy(
        dt,
        geozarr_groups,
        output_path,
        "",
        compressor,
        spatial_chunk,
        min_dimension,
        tile_width,
        max_retries,
        crs_groups,
    )

    return dt_geozarr


def prepare_dataset_with_crs_info(
    ds: xr.Dataset, reference_crs: Optional[str] = None
) -> xr.Dataset:
    """
    Prepare a dataset with CRS information without writing it to disk.
    
    This function adds proper coordinate metadata and CRS information where possible.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset to prepare with CRS information
    reference_crs : str, optional
        Reference CRS to use (e.g., "epsg:4326")
    
    Returns
    -------
    xr.Dataset
        Dataset with CRS information added
    """
    ds = ds.copy()
    
    # Set up coordinate variables with proper attributes
    for coord_name in ds.coords:
        if coord_name == "x":
            ds[coord_name].attrs.update({
                "_ARRAY_DIMENSIONS": ["x"],
                "standard_name": "projection_x_coordinate",
                "units": "m",
                "long_name": "x coordinate of projection"
            })
        elif coord_name == "y":
            ds[coord_name].attrs.update({
                "_ARRAY_DIMENSIONS": ["y"],
                "standard_name": "projection_y_coordinate", 
                "units": "m",
                "long_name": "y coordinate of projection"
            })
        elif coord_name == "angle":
            ds[coord_name].attrs.update({
                "_ARRAY_DIMENSIONS": ["angle"],
                "standard_name": "angle",
                "long_name": "angle coordinate"
            })
        elif coord_name == "band":
            ds[coord_name].attrs.update({
                "_ARRAY_DIMENSIONS": ["band"],
                "standard_name": "band",
                "long_name": "spectral band identifier"
            })
        elif coord_name == "detector":
            ds[coord_name].attrs.update({
                "_ARRAY_DIMENSIONS": ["detector"],
                "standard_name": "detector",
                "long_name": "detector identifier"
            })
        else:
            # Generic coordinate
            if "_ARRAY_DIMENSIONS" not in ds[coord_name].attrs:
                ds[coord_name].attrs["_ARRAY_DIMENSIONS"] = [coord_name]
    
    # Set up data variables with proper attributes
    for var_name in ds.data_vars:
        # Add _ARRAY_DIMENSIONS attribute if missing
        if "_ARRAY_DIMENSIONS" not in ds[var_name].attrs and hasattr(ds[var_name], "dims"):
            ds[var_name].attrs["_ARRAY_DIMENSIONS"] = list(ds[var_name].dims)
        
        # Add grid_mapping reference if spatial coordinates are present and we have a reference CRS
        if "x" in ds.coords and "y" in ds.coords and reference_crs:
            ds[var_name].attrs["grid_mapping"] = "spatial_ref"
    
    # Add CRS information if we have spatial coordinates and a reference CRS
    if "x" in ds.coords and "y" in ds.coords and reference_crs:
        print(f"  Adding CRS information: {reference_crs}")
        ds = ds.rio.write_crs(reference_crs)
        
        # Ensure spatial_ref variable has proper attributes
        if "spatial_ref" in ds:
            ds["spatial_ref"].attrs["_ARRAY_DIMENSIONS"] = []
            
            # Add GeoTransform if we can calculate it from coordinates
            if len(ds.coords["x"]) > 1 and len(ds.coords["y"]) > 1:
                x_coords = ds.coords["x"].values
                y_coords = ds.coords["y"].values
                
                # Calculate pixel size
                pixel_size_x = float(x_coords[1] - x_coords[0])
                pixel_size_y = float(y_coords[0] - y_coords[1])  # Usually negative
                
                # Create GeoTransform (GDAL format)
                transform_str = f"{x_coords[0]} {pixel_size_x} 0.0 {y_coords[0]} 0.0 {pixel_size_y}"
                ds["spatial_ref"].attrs["GeoTransform"] = transform_str
    
    return ds




def setup_datatree_metadata_geozarr_spec_compliant(
    dt: xr.DataTree, groups: List[str]
) -> Dict[str, xr.Dataset]:
    """
    Set up GeoZarr-spec compliant CF standard names and CRS information.

    Parameters
    ----------
    dt : xr.DataTree
        The data tree containing the datasets to process
    groups : list[str]
        List of group names to process as Geozarr datasets

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary of datasets with GeoZarr compliance applied
    """
    geozarr_groups = {}

    for key in groups:
        if not dt[key].data_vars:
            continue

        print(f"Processing group for GeoZarr compliance: {key}")
        ds = dt[key].to_dataset().copy()

        # Create a CF grid_mapping variable name
        grid_mapping_var_name = "spatial_ref"

        # Loop over the bands in the group
        for band in ds.data_vars:
            print(f"  Processing band: {band}")

            # Set CF standard name (required by GeoZarr spec)
            ds[band].attrs["standard_name"] = "toa_bidirectional_reflectance"

            # Add _ARRAY_DIMENSIONS attribute (required by GeoZarr spec)
            if hasattr(ds[band], "dims"):
                ds[band].attrs["_ARRAY_DIMENSIONS"] = list(ds[band].dims)

            # Set grid_mapping to reference our CF grid_mapping variable
            ds[band].attrs["grid_mapping"] = grid_mapping_var_name

            # Check if the band has the proj:epsg attribute to get CRS info
            if "proj:epsg" in ds[band].attrs:
                epsg = ds[band].attrs["proj:epsg"]
                print(f"    Setting CRS for {band} to EPSG:{epsg}")
                ds = ds.rio.write_crs(f"epsg:{epsg}")

        # Add _ARRAY_DIMENSIONS to coordinate variables and ensure proper attributes
        for coord_name, coord_var in ds.coords.items():
            ds[coord_name].attrs["_ARRAY_DIMENSIONS"] = [coord_name]

            # Add appropriate standard names for coordinates
            if coord_name == "x":
                ds[coord_name].attrs["standard_name"] = "projection_x_coordinate"
                ds[coord_name].attrs["units"] = "m"
                ds[coord_name].attrs["long_name"] = "x coordinate of projection"
            elif coord_name == "y":
                ds[coord_name].attrs["standard_name"] = "projection_y_coordinate"
                ds[coord_name].attrs["units"] = "m"
                ds[coord_name].attrs["long_name"] = "y coordinate of projection"
            elif coord_name == "time":
                ds[coord_name].attrs["standard_name"] = "time"
            elif coord_name == "angle":
                ds[coord_name].attrs["standard_name"] = "angle"
                ds[coord_name].attrs["long_name"] = "angle coordinate"
            elif coord_name == "band":
                ds[coord_name].attrs["standard_name"] = "band"
                ds[coord_name].attrs["long_name"] = "spectral band identifier"
            elif coord_name == "detector":
                ds[coord_name].attrs["standard_name"] = "detector"
                ds[coord_name].attrs["long_name"] = "detector identifier"

        # Set up spatial_ref variable with GeoZarr required attributes
        if ds.rio.crs and "spatial_ref" in ds:
            ds["spatial_ref"].attrs["_ARRAY_DIMENSIONS"] = []  # Required for auxiliary variables

            # Add GeoTransform if available
            if ds.rio.transform():
                transform_gdal = ds.rio.transform().to_gdal()
                transform_str = " ".join([str(i) for i in transform_gdal])
                ds["spatial_ref"].attrs["GeoTransform"] = transform_str

            # Update all data variables to reference the grid_mapping
            ds.attrs["grid_mapping"] = grid_mapping_var_name
            for band in ds.data_vars:
                if band != "spatial_ref":
                    ds[band].attrs["grid_mapping"] = grid_mapping_var_name

        geozarr_groups[key] = ds

    return geozarr_groups


def recursive_copy(
    dt_node: xr.DataTree,
    geozarr_groups: Dict[str, xr.Dataset],
    output_path: str,
    group_prefix: str,
    compressor: Any,
    spatial_chunk: int = 4096,
    min_dimension: int = 256,
    tile_width: int = 256,
    max_retries: int = 3,
    crs_groups: Optional[List[str]] = None,
) -> xr.DataTree:
    """
    Recursively copy groups from original DataTree to GeoZarr DataTree.

    Parameters
    ----------
    dt_node : xarray.DataTree
        Current node of the DataTree to copy from
    geozarr_groups : dict[str, xr.Dataset]
        Dictionary of GeoZarr groups to process
    output_path : str
        Output path for the Zarr store
    group_prefix : str
        Prefix for group names in the GeoZarr store
    compressor : Any
        Compressor to use for encoding
    spatial_chunk : int, default 4096
        Spatial chunk size for encoding
    min_dimension : int, default 256
        Minimum dimension for overview levels
    tile_width : int, default 256
        Tile width for TMS compatibility
    max_retries : int, default 3
        Maximum number of retries for network operations
    crs_groups : list[str], optional
        List of group names that need CRS information added on best-effort basis

    Returns
    -------
    xarray.DataTree
        Updated GeoZarr DataTree with copied groups and variables including multiscale children
    """
    no_children = True

    if not dt_node.is_leaf:
        for group_name, group in dt_node.items():
            # Create a new group in the GeoZarr DataTree
            new_group_name = f"{group_prefix}/{group_name}"

            if new_group_name in geozarr_groups:
                dt_node = dt_node.drop_nodes(group_name)  # Remove existing group
                dt_node[group_name] = write_geozarr_group(
                    new_group_name,
                    geozarr_groups[new_group_name],
                    output_path,
                    spatial_chunk=spatial_chunk,
                    compressor=compressor,
                    max_retries=max_retries,
                    min_dimension=min_dimension,
                    tile_width=tile_width,
                )
                no_children = False
                continue

            # First go recursively into children groups
            ds = recursive_copy(
                group,
                geozarr_groups,
                output_path,
                new_group_name,
                compressor,
                spatial_chunk,
                min_dimension,
                tile_width,
                max_retries,
                crs_groups,
            )
            dt_node[group_name] = ds
            no_children = False

    print(f"Writing group '{group_prefix}' to GeoZarr DataTree")

    encoding = {}
    is_dataset = False
    ds = dt_node.to_dataset().drop_encoding()

    # Copy the current group to the GeoZarr DataTree
    if dt_node.data_vars:
        # Set up encoding for variables with proper chunk alignment
        for var in ds.data_vars:
            # Get the current chunks from the data array if it's dask-backed
            if hasattr(ds[var].data, "chunks"):
                # Use existing dask chunks to ensure alignment
                current_chunks = ds[var].chunks
                if len(current_chunks) >= 2:
                    # For 2D+ data, use the existing chunk structure
                    chunking = tuple(
                        current_chunks[i][0] if len(current_chunks[i]) > 0 else ds[var].shape[i]
                        for i in range(len(current_chunks))
                    )
                else:
                    # For 1D data
                    chunking = (
                        current_chunks[0][0] if len(current_chunks[0]) > 0 else ds[var].shape[0],
                    )
            else:
                # Fallback for non-dask arrays - use reasonable chunk sizes
                data_shape = ds[var].shape
                if len(data_shape) >= 2:
                    # Use spatial_chunk size but ensure it doesn't exceed data dimensions
                    chunk_y = min(spatial_chunk, data_shape[-2])
                    chunk_x = min(spatial_chunk, data_shape[-1])
                    if len(data_shape) == 3:
                        chunking = (1, chunk_y, chunk_x)
                    else:
                        chunking = (chunk_y, chunk_x)
                else:
                    chunking = (min(spatial_chunk, data_shape[-1]),)

            encoding[var] = {"compressors": [compressor], "chunks": chunking}
        for coord in ds.coords:
            encoding[coord] = {
                "compressors": None,  # No compression for coordinates
            }
        dt_node = ds
        is_dataset = True

        # Fix double slash issue by normalizing the path
        if group_prefix.startswith("/"):
            group_path = f"{output_path}{group_prefix}"
        else:
            group_path = f"{output_path}/{group_prefix}"

        # Normalize path and get storage options
        group_path = fs_utils.normalize_path(group_path)
        storage_options = fs_utils.get_storage_options(group_path)

        ds.to_zarr(
            group_path,
            mode="w" if no_children else "a",  # Write if no children, append otherwise
            consolidated=is_dataset,  # Consolidate metadata if it's a dataset
            zarr_format=3,
            encoding=encoding,
            storage_options=storage_options,
        )
    else:
        # Write manually the group in zarr.json
        print(f"Writing group metadata for '{group_prefix}'")
        group_path = f"{output_path}/{group_prefix}"

        zarr_json_content = {
            "attributes": {},
            "zarr_format": 3,
            "consolidated_metadata": None,
            "node_type": "group",
        }

        # Write JSON metadata using unified function
        zarr_json_path = fs_utils.normalize_path(f"{group_path}/zarr.json")
        fs_utils.write_json_metadata(zarr_json_path, zarr_json_content)

        # Consolidate metadata without removing existing metadata from children
        zarr_group = fs_utils.open_zarr_group(fs_utils.normalize_path(group_path), mode="r+")
        consolidate_metadata(zarr_group.store)

    # Ensure we always return a DataTree
    if isinstance(dt_node, xr.Dataset):
        dt_node = xr.DataTree(dt_node)
    
    return dt_node


def consolidate_metadata(
    store: StoreLike,
    path: Optional[str] = None,
    zarr_format: Optional[zarr.core.common.ZarrFormat] = None,
) -> zarr.Group:
    """
    Consolidate metadata of all nodes in a hierarchy.

    Parameters
    ----------
    store : StoreLike
        The store-like object whose metadata to consolidate
    path : str, optional
        Path to a group in the store to consolidate at
    zarr_format : {2, 3, None}, optional
        The zarr format of the hierarchy

    Returns
    -------
    zarr.Group
        The group with consolidated metadata
    """
    return zarr.Group(sync(async_consolidate_metadata(store, path=path, zarr_format=zarr_format)))


async def async_consolidate_metadata(
    store: StoreLike,
    path: Optional[str] = None,
    zarr_format: Optional[zarr.core.common.ZarrFormat] = None,
) -> zarr.core.group.AsyncGroup:
    """
    Consolidate metadata of all nodes in a hierarchy asynchronously.

    Parameters
    ----------
    store : StoreLike
        The store-like object whose metadata to consolidate
    path : str, optional
        Path to a group in the store to consolidate at
    zarr_format : {2, 3, None}, optional
        The zarr format of the hierarchy

    Returns
    -------
    zarr.core.group.AsyncGroup
        The group with consolidated metadata
    """
    store_path = await make_store_path(store, path=path)

    if not store_path.store.supports_consolidated_metadata:
        store_name = type(store_path.store).__name__
        raise TypeError(
            f"The Zarr Store in use ({store_name}) doesn't support consolidated metadata",
        )

    group = await zarr.core.group.AsyncGroup.open(
        store_path, zarr_format=zarr_format, use_consolidated=False
    )
    group.store_path.store._check_writable()

    members_metadata = {
        k: v.metadata
        async for k, v in group.members(max_depth=None, use_consolidated_for_children=False)
    }

    zarr.core.group.ConsolidatedMetadata._flat_to_nested(members_metadata)

    consolidated_metadata = zarr.core.group.ConsolidatedMetadata(metadata=members_metadata)
    metadata = dataclasses.replace(group.metadata, consolidated_metadata=consolidated_metadata)
    group = dataclasses.replace(
        group,
        metadata=metadata,
    )

    await group._save_metadata()
    return group


def write_geozarr_group(
    group_name: str,
    ds: xr.Dataset,
    output_path: str,
    spatial_chunk: int = 4096,
    compressor: Any = None,
    max_retries: int = 3,
    min_dimension: int = 256,
    tile_width: int = 256,
) -> xr.DataTree:
    """
    Write a group to a GeoZarr dataset with multiscales support.

    Parameters
    ----------
    group_name : str
        Name of the group to write
    ds : xarray.Dataset
        Dataset to write
    output_path : str
        Output path for the GeoZarr dataset (local path or S3 URL)
    spatial_chunk : int, default 4096
        Spatial chunk size
    compressor : Any, optional
        Compressor to use for encoding
    max_retries : int, default 3
        Maximum number of retries for writing
    min_dimension : int, default 256
        Minimum dimension for overview levels
    tile_width : int, default 256
        Tile width for TMS compatibility

    Returns
    -------
    xarray.DataTree
        The written GeoZarr DataTree with multiscale groups as children
    """
    print(f"\n=== Processing {group_name} with GeoZarr-spec compliance ===")

    # Create a new container for the group as we will need
    # to create siblings for the multiscales
    dt = xr.DataTree()
    group_path = fs_utils.normalize_path(f"{output_path}/{group_name}")

    # Copy the attributes from the original dataset to the DataTree
    dt.attrs = ds.attrs.copy()

    # Get storage options and write DataTree
    storage_options = fs_utils.get_storage_options(group_path)
    dt.to_zarr(
        group_path,
        mode="a",  # Append mode to add to the group
        consolidated=False,  # No consolidate metadata
        zarr_format=3,  # Use Zarr format 3
        storage_options=storage_options,
    )

    # Create encoding for all variables in this group
    encoding = {}
    for var in ds.data_vars:
        if utils.is_grid_mapping_variable(ds, var):
            encoding[var] = {"compressors": None}
        else:
            # We must ensure spatial chunking is aligned with the actual data shape
            data_shape = ds[var].shape
            if len(data_shape) >= 2:
                height, width = data_shape[-2:]
                # Calculate aligned chunk size that divides evenly into the data dimensions
                spatial_chunk_aligned = min(
                    spatial_chunk,
                    utils.calculate_aligned_chunk_size(width, spatial_chunk),
                    utils.calculate_aligned_chunk_size(height, spatial_chunk),
                )
            else:
                spatial_chunk_aligned = spatial_chunk

            encoding[var] = {
                "chunks": (spatial_chunk_aligned, spatial_chunk_aligned),
                "compressors": compressor,
            }

    # Add coordinate encoding
    for coord in ds.coords:
        encoding[coord] = {"compressors": None}

    # Write native data in the group 0 (overview level 0)
    native_dataset_path = f"{group_path}/0"

    # Try to open the existing group path
    existing_native_dataset = None
    try:
        if fs_utils.path_exists(native_dataset_path):
            storage_options = fs_utils.get_storage_options(native_dataset_path)
            existing_native_dataset = xr.open_zarr(
                native_dataset_path, zarr_format=3, storage_options=storage_options, chunks="auto"
            )
            print(f"Found existing native dataset at {native_dataset_path}")
    except Exception as e:
        print(f"Warning: Could not open existing native dataset at {native_dataset_path}: {e}")
        existing_native_dataset = None

    # Write native data band by band to avoid losing all work if one band fails
    success, ds = write_dataset_band_by_band_with_validation(
        ds,
        existing_native_dataset,
        native_dataset_path,
        encoding,
        max_retries,
        group_name,
        False,
    )
    if not success:
        raise RuntimeError(f"Failed to write all bands for {group_name}")

    # Create GeoZarr-spec compliant multiscales (overview levels as children groups)
    try:
        print(f"Creating GeoZarr-spec compliant multiscales for {group_name}")
        create_geozarr_compliant_multiscales(
            ds=ds,
            output_path=output_path,
            group_name=group_name,
            min_dimension=min_dimension,
            tile_width=tile_width,
            spatial_chunk=spatial_chunk,
        )
    except Exception as e:
        print(f"Warning: Failed to create GeoZarr-spec compliant multiscales for {group_name}: {e}")
        print("Continuing with next group...")

    print(f"  Consolidating metadata for group {group_name}...")

    ds.close()  # Close the original dataset to release resources

    # Open the zarr group and consolidate
    zarr_group = fs_utils.open_zarr_group(group_path, mode="r+")
    consolidate_metadata(zarr_group.store)

    print("  ✅ Metadata consolidated")

    # Create a DataTree that includes all multiscale groups as children
    # Reopen the main group as a DataTree to include all children
    storage_options = fs_utils.get_storage_options(group_path)
    
    try:
        # Open as DataTree to get all children groups (multiscale levels)
        dt_result = xr.open_datatree(
            group_path,
            engine="zarr",
            zarr_format=3,
            storage_options=storage_options,
            chunks="auto",
        )
        print(f"  ✅ Successfully opened DataTree with {len(dt_result.groups)} groups")
        return dt_result
    except Exception as e:
        print(f"  ⚠️  Could not open as DataTree, falling back to Dataset: {e}")
        # Fallback to Dataset if DataTree opening fails
        ds = xr.open_dataset(
            group_path,
            engine="zarr",
            zarr_format=3,
            decode_coords="all",
            storage_options=storage_options,
            chunks="auto",
        ).compute()
        
        # Convert Dataset to DataTree for consistency
        dt_result = xr.DataTree(ds)
        return dt_result


def create_geozarr_compliant_multiscales(
    ds: xr.Dataset,
    output_path: str,
    group_name: str,
    min_dimension: int = 256,
    tile_width: int = 256,
    spatial_chunk: int = 4096,
) -> Dict[str, Any]:
    """
    Create GeoZarr-spec compliant multiscales following the specification exactly.

    According to GeoZarr spec:
    - Multiscales MUST be encoded in children groups
    - Multiscale group name is the zoom level identifier (e.g. '0')
    - Multiscale group contains all DataArrays generated for this specific zoom level

    Parameters
    ----------
    ds : xarray.Dataset
        Source dataset with all variables
    output_path : str
        Output path for the Zarr store
    group_name : str
        Name of the resolution group
    min_dimension : int, default 256
        Minimum dimension for overview levels
    tile_width : int, default 256
        Tile width for TMS compatibility
    spatial_chunk : int, default 4096
        Spatial chunk size for encoding

    Returns
    -------
    dict
        Dictionary with overview levels information
    """
    from zarr.codecs import BloscCodec

    # Set up compression
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

    # Get spatial information from the first data variable (excluding grid_mapping variables)
    data_vars = [var for var in ds.data_vars if not utils.is_grid_mapping_variable(ds, var)]
    if not data_vars:
        return {}

    first_var = data_vars[0]
    native_height, native_width = ds[first_var].shape[-2:]
    native_crs = ds.rio.crs
    native_bounds = ds.rio.bounds()

    print(f"Creating GeoZarr-compliant multiscales for {group_name}")
    print(f"Native resolution: {native_width} x {native_height}")
    print(f"Native CRS: {native_crs}")

    # Calculate overview levels
    overview_levels = calculate_overview_levels(
        native_width, native_height, min_dimension, tile_width
    )

    print(f"Total overview levels: {len(overview_levels)}")
    for ol in overview_levels:
        print(
            f"Overview level {ol['level']}: {ol['width']} x {ol['height']} (scale factor: {ol['scale_factor']})"
        )

    # Create native CRS tile matrix set
    tile_matrix_set = create_native_crs_tile_matrix_set(
        native_crs, native_bounds, overview_levels, None
    )

    # Create tile matrix limits
    tile_matrix_limits = {}
    for ol in overview_levels:
        level_str = str(ol["level"])
        max_tile_col = int(np.ceil(ol["width"] / tile_width)) - 1
        max_tile_row = int(np.ceil(ol["height"] / tile_width)) - 1

        tile_matrix_limits[level_str] = {
            "tileMatrix": level_str,
            "minTileCol": 0,
            "maxTileCol": max_tile_col,
            "minTileRow": 0,
            "maxTileRow": max_tile_row,
        }

    # Add multiscales metadata to the group
    zarr_json_path = fs_utils.normalize_path(f"{output_path}/{group_name}/zarr.json")

    # Handle JSON metadata using unified functions
    zarr_json = fs_utils.read_json_metadata(zarr_json_path)

    zarr_json["attributes"]["multiscales"] = {
        "tile_matrix_set": tile_matrix_set,
        "resampling_method": "average",
        "tile_matrix_limits": tile_matrix_limits,
    }

    fs_utils.write_json_metadata(zarr_json_path, zarr_json)

    print(f"Added multiscales metadata to {group_name}")

    # Create overview levels as children groups (GeoZarr spec requirement)
    # Skip level 0 as it's the native resolution in the root group
    # Use pyramid approach: create each level from the previous level for efficiency
    timing_data = []
    previous_level_ds = ds  # Start with the native resolution dataset

    overview_datasets = {}

    for overview in overview_levels:
        level = overview["level"]
        width = overview["width"]
        height = overview["height"]
        scale_factor = overview["scale_factor"]

        # Skip level 0 - native resolution is in the root group
        if level == 0:
            print("Skipping level 0 - native resolution is already in group 0")
            continue

        print(f"\nCreating overview level {level} (1:{scale_factor} scale)...")
        print(f"Target dimensions: {width} x {height}")

        # Use pyramid approach: create level n+1 from level n for much better performance
        print(f"  Using pyramid approach: creating level {level} from level {level - 1}")

        # Create overview dataset with all variables using the previous level dataset
        overview_ds = create_overview_dataset_all_vars(
            previous_level_ds,
            level,
            width,
            height,
            native_crs,
            native_bounds,
            data_vars,
        )

        # Create encoding for this overview level
        encoding = {}
        for var in overview_ds.data_vars:
            if utils.is_grid_mapping_variable(overview_ds, var):
                encoding[var] = {"compressors": None}
            else:
                # Use smaller chunks for overview levels
                spatial_chunk_aligned = min(
                    spatial_chunk,
                    utils.calculate_aligned_chunk_size(width, spatial_chunk),
                    utils.calculate_aligned_chunk_size(height, spatial_chunk),
                )
                encoding[var] = {
                    "chunks": (spatial_chunk_aligned, spatial_chunk_aligned),
                    "compressors": compressor,
                }

        # Add coordinate encoding
        for coord in overview_ds.coords:
            encoding[coord] = {"compressors": None}

        # Write overview level as children group (GeoZarr spec requirement)
        overview_path = fs_utils.normalize_path(f"{output_path}/{group_name}/{level}")

        start_time = time.time()

        # Get storage options and write overview dataset
        storage_options = fs_utils.get_storage_options(overview_path)
        print(f"Writing overview level {level} at {overview_path}")

        # Ensure the directory exists for local paths
        if not fs_utils.is_s3_path(overview_path):
            os.makedirs(os.path.dirname(overview_path), exist_ok=True)

        # Write the overview dataset to Zarr
        overview_ds.to_zarr(
            overview_path,
            mode="w",
            consolidated=True,
            zarr_format=3,
            encoding=encoding,
            align_chunks=True,
            storage_options=storage_options,
        )

        overview_datasets[level] = overview_ds
        proc_time = time.time() - start_time

        timing_data.append(
            {
                "level": level,
                "time": proc_time,
                "pixels": width * height,
                "width": width,
                "height": height,
                "scale_factor": scale_factor,
            }
        )

        print(f"Level {level}: Successfully created in {proc_time:.2f}s")

        # Update previous_level_ds for the next iteration (pyramid approach)
        previous_level_ds = overview_ds

    print(
        f"\n✅ Created {len(overview_levels)} GeoZarr-compliant overview levels using pyramid approach"
    )

    return {
        "overview_datasets": overview_datasets,
        "levels": overview_levels,
        "timing": timing_data,
        "tile_matrix_set": tile_matrix_set,
        "tile_matrix_limits": tile_matrix_limits,
    }


def calculate_overview_levels(
    native_width: int, native_height: int, min_dimension: int = 256, tile_width: int = 256
) -> List[Dict[str, Any]]:
    """
    Calculate overview levels following COG /2 downsampling logic.

    Parameters
    ----------
    native_width : int
        Width of the native resolution data
    native_height : int
        Height of the native resolution data
    min_dimension : int, default 256
        Stop creating overviews when dimension is smaller than this
    tile_width : int, default 256
        Tile width for TMS compatibility calculations

    Returns
    -------
    list
        List of overview level dictionaries with level, zoom, width, height, scale_factor
    """
    overview_levels = []
    level = 0
    current_width = native_width
    current_height = native_height

    while min(current_width, current_height) >= min_dimension:
        # For native CRS TMS compatibility, calculate zoom level that can accommodate this resolution
        # This is for serving purposes - the data stays in native CRS
        zoom_for_width = max(0, int(np.ceil(np.log2(current_width / tile_width))))
        zoom_for_height = max(0, int(np.ceil(np.log2(current_height / tile_width))))
        zoom = max(zoom_for_width, zoom_for_height)

        overview_levels.append(
            {
                "level": level,
                "zoom": zoom,  # For TMS serving compatibility
                "width": current_width,
                "height": current_height,
                "scale_factor": 2**level,
            }
        )

        level += 1
        # COG-style /2 downsampling
        current_width = native_width // (2**level)
        current_height = native_height // (2**level)

    return overview_levels


def create_native_crs_tile_matrix_set(
    native_crs: Any,
    native_bounds: Tuple[float, float, float, float],
    overview_levels: List[Dict[str, Any]],
    group_prefix: str = "",
) -> Dict[str, Any]:
    """
    Create a custom Tile Matrix Set for the native CRS following GeoZarr spec.

    Parameters
    ----------
    native_crs : rasterio.crs.CRS
        Native CRS of the data
    native_bounds : tuple
        Native bounds (left, bottom, right, top)
    overview_levels : list
        List of overview level dictionaries
    group_prefix : str, optional
        Group prefix for the tile matrix IDs

    Returns
    -------
    dict
        Tile Matrix Set definition following OGC standard
    """
    left, bottom, right, top = native_bounds

    # Create tile matrices for each overview level
    tile_matrices = []
    for overview in overview_levels:
        level = overview["level"]
        width = overview["width"]
        height = overview["height"]

        # Calculate cell size (pixel size in CRS units)
        cell_size_x = (right - left) / width
        cell_size_y = (top - bottom) / height
        cell_size = max(cell_size_x, cell_size_y)  # Use the larger dimension

        # Calculate scale denominator (for cartographic purposes)
        # Assuming 1 meter = 1 unit in the CRS (adjust if needed)
        scale_denominator = cell_size * 3779.5275  # Approximate conversion factor

        # Calculate matrix dimensions (number of tiles)
        tile_width = 256
        tile_height = 256
        matrix_width = int(np.ceil(width / tile_width))
        matrix_height = int(np.ceil(height / tile_height))

        # The Tile Matrix identifier MUST be the relative path to the Zarr group (GeoZarr spec requirement)
        matrix_id = f"{group_prefix}/{level}" if group_prefix else str(level)

        tile_matrix = {
            "id": matrix_id,
            "scaleDenominator": scale_denominator,
            "cellSize": cell_size,
            "pointOfOrigin": [left, top],  # Top-left corner
            "tileWidth": tile_width,
            "tileHeight": tile_height,
            "matrixWidth": matrix_width,
            "matrixHeight": matrix_height,
        }

        tile_matrices.append(tile_matrix)

    # Create the complete Tile Matrix Set
    crs_uri = (
        f"http://www.opengis.net/def/crs/EPSG/0/{native_crs.to_epsg()}"
        if native_crs.to_epsg()
        else native_crs.to_wkt()
    )

    tile_matrix_set = {
        "id": f"Native_CRS_{native_crs.to_epsg() if native_crs.to_epsg() else 'Custom'}",
        "title": f"Native CRS Tile Matrix Set ({native_crs})",
        "crs": crs_uri,
        "supportedCRS": crs_uri,  # Required by GeoZarr spec
        "orderedAxes": ["X", "Y"],
        "tileMatrices": tile_matrices,
    }

    return tile_matrix_set


def create_overview_dataset_all_vars(
    ds: xr.Dataset,
    level: int,
    width: int,
    height: int,
    native_crs: Any,
    native_bounds: Tuple[float, float, float, float],
    data_vars: List[str],
) -> xr.Dataset:
    """
    Create an overview dataset containing all variables for a specific level.

    Parameters
    ----------
    ds : xarray.Dataset
        Source dataset
    level : int
        Overview level number
    width : int
        Width of this overview level
    height : int
        Height of this overview level
    native_crs : rasterio.crs.CRS
        Native CRS of the data
    native_bounds : tuple
        Native bounds (left, bottom, right, top)
    data_vars : list
        List of data variable names to include

    Returns
    -------
    xarray.Dataset
        Overview dataset with all variables
    """
    import rasterio.transform

    # Calculate the transform for this overview level
    overview_transform = rasterio.transform.from_bounds(*native_bounds, width, height)

    # Create coordinate arrays in native CRS
    left, bottom, right, top = native_bounds
    x_coords = np.linspace(left, right, width, endpoint=False)
    y_coords = np.linspace(top, bottom, height, endpoint=False)

    # Create overview dataset
    overview_data_vars = {}
    overview_coords = {
        "x": (
            ["x"],
            x_coords,
            {
                "units": "m",
                "long_name": "x coordinate of projection",
                "standard_name": "projection_x_coordinate",
                "_ARRAY_DIMENSIONS": ["x"],
            },
        ),
        "y": (
            ["y"],
            y_coords,
            {
                "units": "m",
                "long_name": "y coordinate of projection",
                "standard_name": "projection_y_coordinate",
                "_ARRAY_DIMENSIONS": ["y"],
            },
        ),
    }

    # Find the grid_mapping variable name from the source dataset
    # Check both dataset attributes and individual variable attributes
    grid_mapping_var_name = ds.attrs.get("grid_mapping", None)
    if not grid_mapping_var_name and data_vars:
        # Try to find grid_mapping from the first data variable
        first_var = data_vars[0]
        if first_var in ds and "grid_mapping" in ds[first_var].attrs:
            grid_mapping_var_name = ds[first_var].attrs["grid_mapping"]
    
    # If still not found, use default name
    if not grid_mapping_var_name:
        grid_mapping_var_name = "spatial_ref"

    # Downsample all data variables
    for var in data_vars:
        print(f"  Downsampling {var}...")

        # Get source data
        source_data = ds[var].values

        # Create downsampled data
        if source_data.ndim == 3:
            # Handle 3D data (e.g., time, y, x)
            downsampled_data = np.zeros(
                (source_data.shape[0], height, width), dtype=source_data.dtype
            )
            for i in range(source_data.shape[0]):
                downsampled_data[i] = utils.downsample_2d_array(source_data[i], height, width)
            dims = ["time", "y", "x"] if "time" in ds[var].dims else [ds[var].dims[0], "y", "x"]
        else:
            # Handle 2D data (y, x)
            downsampled_data = utils.downsample_2d_array(source_data, height, width)
            dims = ["y", "x"]

        # Create data array with proper grid_mapping reference
        attrs = {
            "standard_name": ds[var].attrs.get("standard_name", "toa_bidirectional_reflectance"),
            "_ARRAY_DIMENSIONS": dims,
            "grid_mapping": grid_mapping_var_name,  # Ensure all data variables have grid_mapping
        }

        overview_data_vars[var] = (dims, downsampled_data, attrs)

    # Create overview dataset
    overview_ds = xr.Dataset(overview_data_vars, coords=overview_coords)

    # Ensure the grid_mapping variable is properly added to the overview dataset
    # First, try to find it in the source dataset
    if grid_mapping_var_name in ds:
        # Copy the existing grid_mapping variable and update its attributes
        grid_mapping_attrs = ds[grid_mapping_var_name].attrs.copy()
        
        # Update GeoTransform for this overview level
        transform_gdal = overview_transform.to_gdal()
        transform_str = " ".join([str(i) for i in transform_gdal])
        grid_mapping_attrs["GeoTransform"] = transform_str
        grid_mapping_attrs["_ARRAY_DIMENSIONS"] = []  # Required for auxiliary variables

        # Create the grid_mapping variable
        overview_ds[grid_mapping_var_name] = xr.DataArray(
            data=ds[grid_mapping_var_name].values,  # Copy the original data
            attrs=grid_mapping_attrs,
        )
    else:
        # Create a new grid_mapping variable if it doesn't exist in source
        print(f"  Creating new grid_mapping variable '{grid_mapping_var_name}' for overview level {level}")
        
        # Update GeoTransform for this overview level
        transform_gdal = overview_transform.to_gdal()
        transform_str = " ".join([str(i) for i in transform_gdal])
        
        # Create grid_mapping attributes based on the CRS
        grid_mapping_attrs = {
            "_ARRAY_DIMENSIONS": [],  # Required for auxiliary variables
            "GeoTransform": transform_str,
        }
        
        # Add CRS-specific attributes
        if native_crs:
            if native_crs.to_epsg():
                grid_mapping_attrs["spatial_ref"] = native_crs.to_wkt()
                grid_mapping_attrs["crs_wkt"] = native_crs.to_wkt()
            else:
                grid_mapping_attrs["spatial_ref"] = native_crs.to_wkt()
                grid_mapping_attrs["crs_wkt"] = native_crs.to_wkt()

        # Create the grid_mapping variable
        overview_ds[grid_mapping_var_name] = xr.DataArray(
            data=np.array(b"", dtype="S1"),  # Empty scalar
            attrs=grid_mapping_attrs,
        )

    # Set CRS using rioxarray to ensure proper CRS handling
    overview_ds = overview_ds.rio.write_crs(native_crs)
    
    # Set dataset-level grid_mapping attribute after rio.write_crs to ensure it's preserved
    overview_ds.attrs["grid_mapping"] = grid_mapping_var_name

    return overview_ds


def write_dataset_band_by_band_with_validation(
    ds: xr.Dataset,
    existing_group: Optional[xr.Dataset],
    output_path: str,
    encoding: Dict[str, Any],
    max_retries: int,
    group_name: str,
    force_overwrite: bool = False,
) -> Tuple[bool, xr.Dataset]:
    """
    Write dataset band by band with individual band validation to avoid losing all work if one band fails.

    This approach validates each band individually and only writes bands that are missing or invalid.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to write
    existing_group : xarray.Dataset, optional
        Existing group on the target Zarr store
    output_path : str
        Path to the output Zarr store
    encoding : dict
        Encoding configuration for variables
    max_retries : int
        Maximum number of retries for each band
    group_name : str
        Name of the group (for logging)
    force_overwrite : bool, default False
        Force overwrite existing bands even if they're valid

    Returns
    -------
    tuple[bool, xarray.Dataset]
        (True if all bands were written successfully, updated dataset)
    """
    print(
        f"Writing GeoZarr-spec compliant base resolution for {group_name} band by band with validation"
    )

    # Get data variables (excluding grid_mapping variables)
    data_vars = [var for var in ds.data_vars if not utils.is_grid_mapping_variable(ds, var)]
    grid_mapping_vars = [var for var in ds.data_vars if utils.is_grid_mapping_variable(ds, var)]

    successful_vars = []
    failed_vars = []
    skipped_vars = []

    # Check if zarr store exists
    store_exists = existing_group is not None and len(existing_group.data_vars) > 0

    # Write data variables one by one with validation
    for i, var in enumerate(data_vars):
        # Check if this band already exists and is valid
        if not force_overwrite and store_exists:
            if utils.validate_existing_band_data(existing_group, var, ds):
                ds.drop_vars(var)
                ds[var] = existing_group[var]
                print(f"  ✅ Band {var} already exists and is valid, skipping")
                skipped_vars.append(var)
                successful_vars.append(var)
                continue

        print(f"  Writing data variable {var}...")

        # Create a single-variable dataset with its coordinates
        single_var_ds = ds[[var]]

        # Create encoding for this variable only
        var_encoding = {}
        if var in encoding:
            var_encoding[var] = encoding[var]

        # Add coordinate encoding (coordinates will be written automatically)
        for coord in single_var_ds.coords:
            if coord in encoding:
                var_encoding[coord] = encoding[coord]

        # Try to write this variable with retries
        success = False
        for attempt in range(max_retries):
            try:
                # Determine write mode
                if len(successful_vars) == 0 and len(skipped_vars) == 0:
                    # First variable - create new store
                    mode = "w"
                else:
                    # Subsequent variables - append to existing store
                    mode = "a"
                    # remove coordinates from encoding to avoid duplication
                    var_encoding = {
                        k: v for k, v in var_encoding.items() if k not in single_var_ds.coords
                    }

                # Ensure the dataset is properly chunked to align with encoding
                if var in var_encoding and "chunks" in var_encoding[var]:
                    target_chunks = var_encoding[var]["chunks"]
                    # Create chunk dict using the actual dimensions of the variable
                    var_dims = single_var_ds[var].dims
                    chunk_dict = {}
                    for i, dim in enumerate(var_dims):
                        if i < len(target_chunks):
                            chunk_dict[dim] = target_chunks[i]
                    # Rechunk the dataset to match the target chunks
                    single_var_ds = single_var_ds.chunk(chunk_dict)
                else:
                    single_var_ds = single_var_ds.chunk()

                # Get storage options and write variable
                storage_options = fs_utils.get_storage_options(output_path)
                single_var_ds.to_zarr(
                    output_path,
                    mode=mode,
                    consolidated=False,
                    zarr_format=3,
                    encoding=var_encoding,
                    storage_options=storage_options,
                )

                print(f"    ✅ Successfully wrote {var}")
                successful_vars.append(var)
                success = True
                break

            except Exception as e:
                # delete the started data array to avoid conflict on next attempt
                if os.path.exists(os.path.join(output_path, var)):
                    shutil.rmtree(os.path.join(output_path, var))
                if attempt < max_retries - 1:
                    print(
                        f"    ⚠️  Attempt {attempt + 1} failed for {var}, retrying in 2 seconds..."
                    )
                    time.sleep(2)
                else:
                    print(f"    ❌ Failed to write {var} after {max_retries} attempts: {e}")
                    failed_vars.append(var)
                    break

        if not success:
            print(f"  Failed to write data variable {var}")

    # Write grid_mapping variables separately if any
    for var in grid_mapping_vars:
        print(f"  Writing grid_mapping variable {var}...")

        # Create a single-variable dataset
        single_var_ds = ds[[var]]

        # Create encoding for this variable only
        var_encoding = {}
        if var in encoding:
            var_encoding[var] = encoding[var]

        # Try to write this variable with retries
        success = False
        for attempt in range(max_retries):
            try:
                # Get storage options and write grid_mapping variable
                storage_options = fs_utils.get_storage_options(output_path)
                single_var_ds.to_zarr(
                    output_path,
                    mode="a",  # Always append for grid_mapping variables
                    consolidated=False,
                    zarr_format=3,
                    encoding=var_encoding,
                    storage_options=storage_options,
                )

                print(f"    ✅ Successfully wrote {var}")
                successful_vars.append(var)
                success = True
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    print(
                        f"    ⚠️  Attempt {attempt + 1} failed for {var}, retrying in 2 seconds..."
                    )
                    time.sleep(2)
                else:
                    print(f"    ❌ Failed to write {var} after {max_retries} attempts: {e}")
                    failed_vars.append(var)
                    break

        if not success:
            print(f"  Failed to write grid_mapping variable {var}")

    # Consolidate metadata without removing existing metadata from children
    zarr_group = fs_utils.open_zarr_group(output_path, mode="r+")
    consolidate_metadata(zarr_group.store)

    print(f"  ✅ Metadata consolidated for {len(successful_vars)} variables")

    # Close the dataset to release resources
    ds.close()

    # Reopen the dataset
    storage_options = fs_utils.get_storage_options(output_path)
    ds = xr.open_dataset(
        output_path,
        engine="zarr",
        zarr_format=3,
        decode_coords="all",
        storage_options=storage_options,
        chunks="auto",
    ).compute()

    # Report results
    if failed_vars:
        print(f"❌ Failed to write {len(failed_vars)} variables for {group_name}: {failed_vars}")
        print(f"✅ Successfully wrote {len(successful_vars) - len(skipped_vars)} new variables")
        print(f"⏭️  Skipped {len(skipped_vars)} existing valid variables: {skipped_vars}")
        return False, ds
    else:
        print(f"✅ Successfully processed all {len(successful_vars)} variables for {group_name}")
        if skipped_vars:
            print(f"   - Wrote {len(successful_vars) - len(skipped_vars)} new variables")
            print(f"   - Skipped {len(skipped_vars)} existing valid variables")
        return True, ds
