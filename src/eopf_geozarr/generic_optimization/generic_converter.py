"""
GeoZarr compliant conversion tools for EOPF datasets.

This module provides functions to convert EOPF datasets to GeoZarr format
while maintaining native projections and using /2 downsampling logic.

Key compliance features:
- _ARRAY_DIMENSIONS attributes on all arrays
- CF standard names for all variables
- grid_mapping attributes referencing CF grid_mapping variables
- GeoTransform attributes in grid_mapping variables
- Native CRS preservation (no TMS reprojection)
- Proper multiscales metadata structure
"""

import time

import structlog
import xarray as xr
import zarr

from eopf_geozarr.conversion import utils

log = structlog.get_logger()


def create_generic_geozarr_dataset(
    dt_input: xr.DataTree,
    output_path: str,
    spatial_chunk: int,
    enable_sharding: bool,
    compression_level: int = 3,
    keep_scale_offset: bool = True,
) -> xr.DataTree:
    """
    Create a GeoZarr-spec compliant dataset from EOPF data with CPM 3.0.0.
    Possibly backward compatabile but not verified and not required, as this generic converter is (so far) only used by EODC which uses solely CPM 3.0.0

    Parameters
    ----------
    dt_input : xr.DataTree
        Input EOPF DataTree
    output_path : str
        Output path for the Zarr store
    spatial_chunk : int, default 1024 (loaded in the writer.py script)
        Spatial chunk size for encoding
    enable_sharding : bool
        Enable zarr sharding for spatial dimensions of each variable
    compression_level: int, default 3
    keep_scale_offset: bool, default True
        Keep uint16 and scale/offset attributes

    Returns
    -------
    xr.DataTree
        DataTree containing the GeoZarr compliant data
    """
    start_time = time.time()

    ouput_group = zarr.open_group(output_path)
    processed_groups = {}
    # slc_chunks = chunk_info.s1slc_chunks

    # rechunk everything
    for group_path in dt_input.groups:
        if group_path == "/":
            continue

        group_node = dt_input[group_path]

        # Skip parent groups that have children (only process leaf groups)
        if hasattr(group_node, "children") and len(group_node.children) > 0:
            # this silently fails for groups which have data variables at group level (eg.: S3 OLC EFR) and children groups -> if orphans are assigned to measurements!
            # ERR works, as it has no orphans!
            # does this need to be considered? maybe, as generic verison will likely have this issue (ans its a stupid scheem anyway)
            continue

        base_dataset = group_node.to_dataset()

        # Skip empty groups
        if not base_dataset.data_vars:
            log.info("Skipping empty group: {}", group_path=group_path)
            continue

        log.info("Copying original group: {}", group_path=group_path)

        dataset = utils._rechunk_ds(base_dataset, spatial_chunk)

        encoding = utils.create_uniform_encoding(
            dataset,
            spatial_chunk=spatial_chunk,
            enable_sharding=enable_sharding,
            keep_scale_offset=keep_scale_offset,
            compression_level=compression_level,
        )

        # Write dataset -> does NOT add geo metadata
        ds_out = utils.stream_write_dataset(
            dataset,
            path=group_path,
            group=ouput_group,
            encoding=encoding,
            enable_sharding=enable_sharding,
        )
        processed_groups[group_path] = ds_out

    # root/sub-root level consolidation and attribute handling
    utils.updated_root_consolidation(dt_input, output_path, processed_groups)

    # Create result DataTree
    result_dt = utils.create_result_datatree(output_path)

    total_time = time.time() - start_time
    log.info("Optimization complete", duration_seconds=round(total_time, 2))

    utils.optimization_summary(dt_input, result_dt, output_path)

    return result_dt
