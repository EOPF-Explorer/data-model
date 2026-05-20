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
import itertools
import os
import time
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import structlog
import xarray as xr
import zarr
from pyproj import CRS
from rasterio.warp import calculate_default_transform
from zarr.codecs import BloscCodec
from zarr.core.sync import sync
from zarr.storage import StoreLike
from zarr.storage._common import make_store_path
from zarr_cm import geo_proj
from zarr_cm import multiscales as multiscales_cm
from zarr_cm import spatial as spatial_cm

from eopf_geozarr.data_api.geozarr.multiscales import zcm
from eopf_geozarr.data_api.geozarr.multiscales.geozarr import (
    MultiscaleGroupAttrs,
    MultiscaleMeta,
)
from eopf_geozarr.types import (
    OverviewLevelJSON,
    StandardLatCoordAttrsJSON,
    StandardLonCoordAttrsJSON,
    StandardXCoordAttrsJSON,
    StandardYCoordAttrsJSON,
    XarrayEncodingJSON,
)

from . import fs_utils, utils
from .fs_utils import sanitize_dataset_attributes
from .sentinel1_reprojection import reproject_sentinel1_with_gcps

log = structlog.get_logger()


def create_geozarr_dataset(
    dt_input: xr.DataTree,
    groups: Iterable[str],
    output_path: str,
    spatial_chunk: int = 4096,
    min_dimension: int = 256,
    max_retries: int = 3,
    crs_groups: Iterable[str] | None = None,
    gcp_group: str | None = None,
    enable_sharding: bool = False,
) -> xr.DataTree:
    """
    Create a GeoZarr-spec 0.4 compliant dataset from EOPF data.

    Parameters
    ----------
    dt_input : xr.DataTree
        Input EOPF DataTree
    groups : list[str]
        List of group names to process as Geozarr datasets.
    output_path : str
        Output path for the Zarr store
    spatial_chunk : int, default 4096
        Spatial chunk size for encoding
    min_dimension : int, default 256
        Minimum dimension for overview levels
    max_retries : int, default 3
        Maximum number of retries for network operations
    crs_groups : Iterabl[str], optional
        Iterable of group names that need CRS information added on best-effort basis
    gcp_group : str, optional
        Group name where GCPs (Ground Control Points) are located.
    enable_sharding : bool, default False
        Enable zarr sharding for spatial dimensions of each variable

    Returns
    -------
    xr.DataTree
        DataTree containing the GeoZarr compliant data
    """
    dt = dt_input.copy()
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0)

    if enable_sharding:
        log.info("🔧 Zarr sharding enabled for spatial dimensions")

    if _is_sentinel1(dt_input):
        if gcp_group is None:
            raise ValueError("Detected Sentinel-1 GRD product but GCP group not provided")

        # process sentinel-1 VV and VH polarization top-level groups
        vv_vh_group_names = [f"/{name}" for name in list(dt_input.children)]
        assert len(vv_vh_group_names) == 2, str(vv_vh_group_names)

        groups = [
            vv_vh + "/" + grp.lstrip("/")
            for vv_vh, grp in itertools.product(vv_vh_group_names, groups)
        ]
        if crs_groups is not None:
            crs_groups = [
                vv_vh + "/" + grp.lstrip("/")
                for vv_vh, grp in itertools.product(vv_vh_group_names, crs_groups)
            ]

        # pick only one gcp group (both groups from VV and VH should be equal)
        gcp_group = vv_vh_group_names[0] + "/" + gcp_group.lstrip("/")
        if gcp_group not in dt_input.groups:
            raise ValueError(f"GCP group '{gcp_group}' not found in input datatree")

    # Get the measurements datasets prepared for GeoZarr compliance
    geozarr_groups = setup_datatree_metadata_geozarr_spec_compliant(dt, groups, gcp_group)

    log.info("GeoZarr groups prepared", groups_prepared=list(geozarr_groups.keys()))

    # Create the GeoZarr compliant store through iterative processing
    dt_geozarr = iterative_copy(
        dt,
        geozarr_groups,
        output_path,
        compressor,
        spatial_chunk,
        min_dimension,
        max_retries,
        crs_groups,
        gcp_group,
        enable_sharding,
    )

    # Consolidate metadata at the root level AFTER all groups are written
    log.info("Consolidating metadata at root level for consistent zarr access...")
    try:
        zarr_group = fs_utils.open_zarr_group(output_path, mode="r+")
        consolidate_metadata(zarr_group.store)
        log.info("✅ Root level metadata consolidation completed")
    except Exception as e:
        log.warning("Root level consolidation failed", error=str(e))

    return dt_geozarr


def setup_datatree_metadata_geozarr_spec_compliant(
    dt: xr.DataTree, groups: Iterable[str], gcp_group: str | None = None
) -> dict[str, xr.Dataset]:
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
    geozarr_groups: dict[str, xr.Dataset] = {}
    grid_mapping_var_name = "spatial_ref"
    epsg_CPM_260 = dt.attrs.get("other_metadata", {}).get(
        "horizontal_CRS_code", dt.attrs.get("other_metadata", {}).get("horizontal_crs_code", None)
    )
    if epsg_CPM_260 is not None:
        epsg_CPM_260 = epsg_CPM_260.split(":")[-1]

    for key in groups:
        # Check if key exists in DataTree by attempting to access it
        try:
            node = dt[key]
        except KeyError:
            log.info("Skipping group %s - not found in DataTree", key)
            continue

        if not node.data_vars:
            log.info("Skipping group %s - no data variables", key)
            continue

        log.info("Processing group %s for GeoZarr compliance", key)
        ds = node.to_dataset().copy()

        ds_gcp = dt[gcp_group].to_dataset() if gcp_group is not None else None

        # Apply Sentinel-1 reprojection if needed
        if _is_sentinel1(dt) and ds_gcp is not None:
            log.info("Applying Sentinel-1 reprojection for group %s", key)
            ds = reproject_sentinel1_with_gcps(ds, ds_gcp, target_crs="EPSG:4326")
            log.info("✅ Reprojection complete")

        # Process all variables in the group
        for var_name in ds.data_vars:
            log.info("Processing variable / band %s", var_name)

            # Sanitize source-only and misleading attributes
            is_float = np.issubdtype(ds[var_name].dtype, np.floating)
            ds[var_name].attrs = utils.sanitize_array_attrs(
                ds[var_name].attrs,
                is_decoded_float=is_float,
            )
            if is_float:
                ds[var_name].attrs["_FillValue"] = np.nan

            # Set CF standard name and _ARRAY_DIMENSIONS
            if _is_sentinel1(dt):
                ds[var_name].attrs["standard_name"] = (
                    "surface_backwards_scattering_coefficient_of_radar_wave"
                )
                ds[var_name].attrs["units"] = "1"
            else:  # Default to optical data standard name
                ds[var_name].attrs["standard_name"] = "toa_bidirectional_reflectance"

            if hasattr(ds[var_name], "dims"):
                ds[var_name].attrs["_ARRAY_DIMENSIONS"] = list(ds[var_name].dims)
            ds[var_name].attrs["grid_mapping"] = grid_mapping_var_name

            # Set CRS if available
            if "proj:epsg" in ds[var_name].attrs:
                epsg = ds[var_name].attrs["proj:epsg"]
                log.info("Setting CRS for variable %s to EPSG %s", var_name, epsg)
                ds = ds.rio.write_crs(f"epsg:{epsg}")
            elif epsg_CPM_260:
                log.info(
                    "Setting CRS for variable %s to EPSG (CPM 2.6.0 default)",
                    var_name,
                )
                ds = ds.rio.write_crs(f"epsg:{epsg_CPM_260}")

        # Add _ARRAY_DIMENSIONS to coordinate variables
        _add_coordinate_metadata(ds)

        # Set up spatial_ref variable with GeoZarr required attributes
        _setup_grid_mapping(ds, grid_mapping_var_name)

        geozarr_groups[key] = ds
        log.info("Added %s to geozarr_groups", key)

    log.info(
        "Returning geozarr_groups with %s groups: %s",
        len(geozarr_groups),
        list(geozarr_groups.keys()),
    )
    return geozarr_groups


def iterative_copy(
    dt_input: xr.DataTree,
    geozarr_groups: dict[str, xr.Dataset],
    output_path: str,
    compressor: Any,
    spatial_chunk: int = 4096,
    min_dimension: int = 256,
    max_retries: int = 3,
    crs_groups: Iterable[str] | None = None,
    gcp_group: str | None = None,
    enable_sharding: bool = False,
) -> xr.DataTree:
    """
    Iteratively copy groups from original DataTree to GeoZarr DataTree.

    Parameters
    ----------
    dt_input : xarray.DataTree
        Input DataTree to copy from
    geozarr_groups : dict[str, xr.Dataset]
        Dictionary of GeoZarr groups to process
    output_path : str
        Output path for the Zarr store
    compressor : Any
        Compressor to use for encoding
    spatial_chunk : int, default 4096
        Spatial chunk size for encoding
    min_dimension : int, default 256
        Minimum dimension for overview levels
    max_retries : int, default 3
        Maximum number of retries for network operations
    crs_groups : Iterable[str], optional
        Iterable of group names that need CRS information added on best-effort basis
    gcp_group : str, optional
        Group name where GCPs (Ground Control Points) are located

    Returns
    -------
    xarray.DataTree
        Updated GeoZarr DataTree with copied groups and variables including multiscale children
    """
    # Create result DataTree and initialize storage
    dt_result = xr.DataTree()
    storage_options = fs_utils.get_storage_options(output_path)
    dt_result.to_zarr(
        output_path,
        mode="a",
        consolidated=False,
        compute=True,
        storage_options=storage_options,
    )

    written_groups: set[str] = set()
    reference_crs = None

    # Process all groups in the tree using iterative approach
    for relative_path, node in dt_input.subtree_with_keys:
        if relative_path == ".":
            continue
        if relative_path.endswith(("_VH", "_VV")):
            # skip sentinel-1 top-level polarization groups
            continue

        current_group_path = "/" + relative_path
        log.info("Processing group %s in iterative copy", current_group_path)

        if current_group_path in geozarr_groups:
            log.info("Processing %s as GeoZarr group", current_group_path)
            write_geozarr_group(
                dt_input,
                dt_result,
                current_group_path,
                geozarr_groups[current_group_path],
                output_path,
                spatial_chunk=spatial_chunk,
                compressor=compressor,
                max_retries=max_retries,
                min_dimension=min_dimension,
                gcp_group=gcp_group,
                enable_sharding=enable_sharding,
            )
            written_groups.add(current_group_path)
            continue

        # Get dataset from the node
        ds = node.to_dataset().drop_encoding()

        # Add CRS information if needed
        if crs_groups and current_group_path in crs_groups:
            log.info("Adding CRS information for group %s", current_group_path)
            if reference_crs is None:
                reference_crs = _find_reference_crs(geozarr_groups)
            ds = prepare_dataset_with_crs_info(ds, reference_crs=reference_crs)

        # Process groups with data variables
        if node.data_vars:
            log.info("Writing %s with data variables to GeoZarr DataTree", current_group_path)

            # Sanitize NaN values in attributes before writing
            ds = sanitize_dataset_attributes(ds)

            # Set up encoding
            encoding = _create_encoding(ds, compressor, spatial_chunk)

            # Write the dataset
            group_param = current_group_path.lstrip("/") if current_group_path else None
            ds.to_zarr(
                output_path,
                group=group_param,
                mode="w",
                consolidated=False,
                zarr_format=3,
                encoding=encoding,
                storage_options=storage_options,
            )

            dt_result[relative_path] = xr.DataTree(ds)

        written_groups.add(current_group_path)

    return dt_result if isinstance(dt_result, xr.DataTree) else xr.DataTree(dt_result)


def prepare_dataset_with_crs_info(ds: xr.Dataset, reference_crs: str | None = None) -> xr.Dataset:
    """
    Prepare a dataset with CRS information without writing it to disk.

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
    _add_coordinate_metadata(ds)

    # Add CRS information if we have spatial coordinates and a reference CRS
    if "x" in ds.coords and "y" in ds.coords and reference_crs:
        log.info("Adding CRS information with reference CRS %s", reference_crs)
        ds = ds.rio.write_crs(reference_crs)
        ds.attrs["grid_mapping"] = "spatial_ref"

        # Ensure spatial_ref variable has proper attributes
        if "spatial_ref" in ds:
            _add_geotransform(ds, "spatial_ref")

    # Set up data variables with proper attributes
    for var_name in ds.data_vars:
        if "_ARRAY_DIMENSIONS" not in ds[var_name].attrs and hasattr(ds[var_name], "dims"):
            ds[var_name].attrs["_ARRAY_DIMENSIONS"] = list(ds[var_name].dims)

        # Add grid_mapping reference if spatial coordinates are present
        if "x" in ds[var_name].coords and "y" in ds[var_name].coords and reference_crs:
            ds[var_name].attrs["grid_mapping"] = "spatial_ref"
            ds[var_name].attrs["proj:epsg"] = reference_crs.split(":")[-1]
            if "spatial_ref" in ds and "GeoTransform" in ds["spatial_ref"].attrs:
                ds[var_name].attrs["proj:transform"] = ds["spatial_ref"].attrs["GeoTransform"]

    return ds


def write_geozarr_group(
    dt_input: xr.DataTree,
    dt_result: xr.DataTree,
    group_name: str,
    ds: xr.Dataset,
    output_path: str,
    spatial_chunk: int = 4096,
    compressor: Any = None,
    max_retries: int = 3,
    min_dimension: int = 256,
    gcp_group: str | None = None,
    enable_sharding: bool = False,
) -> xr.DataTree:
    """
    Write a group to a GeoZarr dataset with multiscales support.

    Parameters
    ----------
    dt_input : xr.DataTree
        The original DataTree
    dt_result : xr.DataTree
        Result DataTree to update
    group_name : str
        Name of the group to write
    ds : xarray.Dataset
        Dataset to write
    output_path : str
        Output path for the GeoZarr dataset
    spatial_chunk : int, default 4096
        Spatial chunk size
    compressor : Any, optional
        Compressor to use for encoding
    max_retries : int, default 3
        Maximum number of retries for writing
    min_dimension : int, default 256
        Minimum dimension for overview levels
    gcp_group : str, optional
        Group name where GCPs (Ground Control Points) are located
        in the input DataTree (ignored if ``dt_input`` does not
        correspond to a Sentinel-1 product)

    Returns
    -------
    xarray.DataTree
        The written GeoZarr DataTree with multiscale groups as children
    """
    log.info("Processing group %s with GeoZarr-spec compliance", group_name)

    # Create a new container for the group
    dt = xr.DataTree()
    dt_result[group_name.lstrip("/")] = dt
    dt.attrs = ds.attrs.copy()

    # Create encoding for all variables
    encoding = _create_geozarr_encoding(ds, compressor, spatial_chunk, enable_sharding)

    # Write native data directly at the group path (GeoZarr 0.4 layout)
    native_dataset_group_name = group_name
    native_dataset_path = f"{output_path}/{native_dataset_group_name.lstrip('/')}"

    # Check for existing dataset
    existing_native_dataset = _load_existing_dataset(native_dataset_path)

    # Write native data band by band
    success, ds = write_dataset_band_by_band_with_validation(
        ds,
        existing_native_dataset,
        output_path,
        encoding,
        max_retries,
        native_dataset_group_name,
        False,
    )
    if not success:
        raise RuntimeError(f"Failed to write all bands for {group_name}")

    # Create GeoZarr-spec compliant multiscales
    if _is_sentinel1(dt_input):
        assert gcp_group is not None, "GCP group required for processing Sentinel-1"
        ds_gcp = dt_input[gcp_group].to_dataset()
        # For Sentinel-1, ds_gcp is set to None since data is now reprojected and doesn't need GCP handling
        ds_gcp = None
    else:
        ds_gcp = None

    try:
        log.info("Creating GeoZarr-spec compliant multiscales", group_name=group_name)
        create_geozarr_compliant_multiscales(
            ds=ds,
            output_path=output_path,
            group_name=group_name,
            min_dimension=min_dimension,
            spatial_chunk=spatial_chunk,
            ds_gcp=ds_gcp,
            enable_sharding=enable_sharding,
        )
    except Exception as e:
        log.warning(
            "Failed to create GeoZarr-spec compliant multiscales",
            group_name=group_name,
            error=str(e),
        )
        log.info("Continuing with next group...")

    # Consolidate metadata
    log.info("Consolidating metadata for group %s", group_name)
    group_path = fs_utils.normalize_path(f"{output_path}/{group_name.lstrip('/')}")
    zarr_group = fs_utils.open_zarr_group(group_path, mode="r+")
    consolidate_metadata(zarr_group.store)
    log.info("✅ Metadata consolidated")

    return dt


def create_geozarr_compliant_multiscales(
    ds: xr.Dataset,
    output_path: str,
    group_name: str,
    min_dimension: int = 256,
    spatial_chunk: int = 4096,
    ds_gcp: xr.Dataset | None = None,
    enable_sharding: bool = False,
) -> dict[str, Any]:
    """
    Create GeoZarr-spec 0.4 compliant multiscales for a group.

    Native data is expected to already be written at ``group_name``.
    Overview sub-groups are written at ``{group_name}/r{scale}``
    (e.g. ``r2``, ``r4``, ``r8`` …) and the experimental multiscales
    convention metadata is added to ``group_name``.

    Parameters
    ----------
    ds : xarray.Dataset
        Source dataset with all variables (native resolution)
    output_path : str
        Output path for the Zarr store
    group_name : str
        Path of the group that already holds the native data
    min_dimension : int, default 256
        Minimum dimension for overview levels
    spatial_chunk : int, default 4096
        Spatial chunk size for encoding
    ds_gcp : xr.Dataset, optional
        Source dataset with Sentinel-1 ground control points
        at native resolution

    Returns
    -------
    dict
        Dictionary with overview_datasets and overview levels information
    """
    import rasterio.transform

    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

    # Get spatial information from the first data variable
    data_vars = [var for var in ds.data_vars if not utils.is_grid_mapping_variable(ds, var)]
    if not data_vars:
        return {}

    first_var = data_vars[0]
    native_height, native_width = ds[first_var].shape[-2:]
    native_crs = ds.rio.crs

    if ds_gcp is None:
        native_bounds = ds.rio.bounds()
    else:
        if "azimuth_time" in ds.dims and "ground_range" in ds.dims:
            ds.rio.set_spatial_dims(x_dim="ground_range", y_dim="azimuth_time", inplace=True)

        try:
            if ds.rio.get_gcps() is not None:
                transform, width, height = calculate_default_transform(
                    ds.rio.crs,
                    CRS.from_epsg(4326),
                    ds.rio.width,
                    ds.rio.height,
                    gcps=ds.rio.get_gcps(),
                )
                native_bounds = (
                    transform[2],
                    transform[5] + height * transform[4],
                    transform[2] + width * transform[0],
                    transform[5],
                )
            else:
                native_bounds = ds.rio.bounds()

        except Exception as e:
            log.info("Error computing native bounds", error=str(e))
            native_bounds = (
                ds_gcp["longitude"].values.min(),
                ds_gcp["latitude"].values.min(),
                ds_gcp["longitude"].values.max(),
                ds_gcp["latitude"].values.max(),
            )

    log.info("Creating GeoZarr-compliant multiscales in group %s", group_name)
    log.info("Native resolution", width=native_width, height=native_height)
    log.info("Native CRS: %s", native_crs)

    # Calculate overview levels (integer indexed, 0 = native)
    overview_levels = calculate_overview_levels(native_width, native_height, min_dimension)

    log.info("Total overview levels: %s", len(overview_levels))
    for ol in overview_levels:
        log.info(
            "Overview level",
            level=ol["level"],
            width=ol["width"],
            height=ol["height"],
        )

    # Build experimental multiscales layout ----------------------------------
    # Level 0 → asset "." (native data written at the group root)
    # Level N → asset "r{2**N}" (overview sub-group)

    def _spatial_transform_for(
        bounds: tuple[float, float, float, float], w: int, h: int
    ) -> tuple[float, ...]:
        t = rasterio.transform.from_bounds(*bounds, w, h)
        return tuple(float(v) for v in tuple(t)[:6])

    layout: list[zcm.ScaleLevel] = []
    for i, ol in enumerate(overview_levels):
        level_int = int(ol["level"])
        asset = "." if level_int == 0 else f"r{2 ** level_int}"
        scale_level_data: dict[str, Any] = {"asset": asset}
        if i > 0:
            prev_level_int = int(overview_levels[i - 1]["level"])
            prev_asset = "." if prev_level_int == 0 else f"r{2 ** prev_level_int}"
            scale_level_data["derived_from"] = prev_asset
            scale_level_data["transform"] = zcm.Transform(
                scale=(2.0, 2.0),
                translation=(0.0, 0.0),
            )
        w, h = int(ol["width"]), int(ol["height"])
        scale_level_data["spatial:shape"] = (h, w)
        spatial_tf = _spatial_transform_for(native_bounds, w, h)
        if not all(v == 0.0 for v in spatial_tf):
            scale_level_data["spatial:transform"] = spatial_tf
        layout.append(zcm.ScaleLevel(**scale_level_data))

    multiscale_attrs = MultiscaleGroupAttrs(
        zarr_conventions=(multiscales_cm.CMO, spatial_cm.CMO, geo_proj.CMO),
        multiscales=MultiscaleMeta(
            layout=layout,
            resampling_method="average",
        ),
    )
    attrs_to_write = multiscale_attrs.model_dump()
    if native_crs and native_bounds:
        attrs_to_write["spatial:dimensions"] = ["y", "x"]
        attrs_to_write["spatial:bbox"] = list(native_bounds)
        attrs_to_write["spatial:registration"] = "pixel"
        if hasattr(native_crs, "to_epsg") and native_crs.to_epsg():
            attrs_to_write["proj:code"] = f"EPSG:{native_crs.to_epsg()}"
        elif hasattr(native_crs, "to_wkt"):
            attrs_to_write["proj:wkt2"] = native_crs.to_wkt()

    group_path = fs_utils.normalize_path(f"{output_path}/{group_name.lstrip('/')}")
    zarr_group = fs_utils.open_zarr_group(group_path, mode="r+")
    zarr_group.attrs.update(attrs_to_write)

    log.info("Added multiscales metadata to group %s", group_name)

    # Create overview sub-groups as r2, r4, r8, ... -------------------------
    timing_data = []
    previous_level_ds = ds
    overview_datasets: dict[str, xr.Dataset] = {}

    for overview in overview_levels:
        level = int(overview["level"])

        # Skip level 0 — native data already written at the group root
        if level == 0:
            continue

        asset_name = f"r{2 ** level}"
        width = overview["width"]
        height = overview["height"]

        log.info("Creating overview %s (%dx%d)", asset_name, width, height)

        if ds_gcp is not None:
            ds_gcp_overview = utils.compute_overview_gcps(ds_gcp, 2.0, width, height)
        else:
            ds_gcp_overview = None

        # Create overview dataset
        overview_ds = create_overview_dataset_all_vars(
            previous_level_ds,
            level,
            width,
            height,
            native_crs,
            native_bounds,
            data_vars,
            ds_gcp_overview,
            enable_sharding,
        )

        # Create encoding for this overview level
        encoding = _create_geozarr_encoding(overview_ds, compressor, spatial_chunk, enable_sharding)

        overview_group = f"{group_name}/{asset_name}"
        overview_path = fs_utils.normalize_path(f"{output_path}/{overview_group.lstrip('/')}")
        start_time = time.time()

        storage_options = fs_utils.get_storage_options(overview_path)
        log.info("Writing overview level at path %s", overview_path)

        if not fs_utils.is_s3_path(overview_path):
            os.makedirs(os.path.dirname(overview_path), exist_ok=True)

        # Sanitize NaN values in overview dataset attributes before writing
        overview_ds = sanitize_dataset_attributes(overview_ds)

        align_chunks_flag = not enable_sharding
        overview_ds.to_zarr(
            output_path,
            group=overview_group,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding=encoding,
            align_chunks=align_chunks_flag,
            storage_options=storage_options,
        )

        overview_datasets[asset_name] = overview_ds
        proc_time = time.time() - start_time

        timing_data.append(
            {
                "level": asset_name,
                "time": proc_time,
                "pixels": width * height,
                "width": width,
                "height": height,
            }
        )

        log.info("%s created in %s seconds", asset_name, round(proc_time, 2))

        # Consolidate metadata for this overview group
        ov_group_path = fs_utils.normalize_path(f"{output_path}/{overview_group.lstrip('/')}")
        ov_zarr_group = fs_utils.open_zarr_group(ov_group_path, mode="r+")
        consolidate_metadata(ov_zarr_group.store)
        log.info("✅ Metadata consolidated for overview %s", asset_name)

        previous_level_ds = overview_ds

    log.info(
        "✅ Created GeoZarr-compliant overview levels using pyramid approach",
        count=len(overview_levels),
    )

    return {
        "overview_datasets": overview_datasets,
        "levels": overview_levels,
        "timing": timing_data,
    }


def calculate_overview_levels(
    native_width: int,
    native_height: int,
    min_dimension: int = 256,
) -> list[OverviewLevelJSON]:
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

    Returns
    -------
    list
        List of overview level dictionaries (level 0 = native resolution)
    """
    overview_levels: list[OverviewLevelJSON] = []
    level = 0
    current_width = native_width
    current_height = native_height

    while min(current_width, current_height) >= min_dimension:
        overview_level: dict[str, Any] = {
            "level": level,
            "width": current_width,
            "height": current_height,
            "translation_relative": 0.0,
            "scale_relative": 2**level,
        }
        overview_levels.append(overview_level)  # type: ignore[arg-type]

        level += 1
        current_width = native_width // (2**level)
        current_height = native_height // (2**level)

    return overview_levels


def create_overview_dataset_all_vars(
    ds: xr.Dataset,
    level: int,
    width: int,
    height: int,
    native_crs: Any,
    native_bounds: tuple[float, float, float, float],
    data_vars: Sequence[Hashable],
    ds_gcp: xr.Dataset | None = None,
    enable_sharding: bool = False,
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
    data_vars : Sequence[Hashable]
        Sequence of data variable names to include
    ds_gcp : xr.Dataset, optional
        Source dataset with Sentinel-1 ground control points
        at native resolution

    Returns
    -------
    xarray.Dataset
        Overview dataset with all variables
    """
    import rasterio.transform

    # Calculate the transform for this overview level
    overview_transform = rasterio.transform.from_bounds(*native_bounds, width, height)

    # Determine coordinate dtype from parent dataset to maintain alignment
    parent_x_dtype = ds.coords["x"].dtype if "x" in ds.coords else np.float64
    parent_y_dtype = ds.coords["y"].dtype if "y" in ds.coords else np.float64

    # Create coordinate arrays preserving parent dtype
    left, bottom, right, top = native_bounds
    x_coords = np.linspace(left, right, width, endpoint=False, dtype=parent_x_dtype)
    y_coords = np.linspace(top, bottom, height, endpoint=False, dtype=parent_y_dtype)

    # Check if we're dealing with geographic coordinates (EPSG:4326)
    if native_crs and native_crs.to_epsg() == 4326:
        lon_attrs = _get_lon_coord_attrs()
        lat_attrs = _get_lat_coord_attrs()
        overview_coords = {
            "x": (["x"], x_coords, lon_attrs),
            "y": (["y"], y_coords, lat_attrs),
        }

    else:
        x_attrs = _get_x_coord_attrs()
        y_attrs = _get_y_coord_attrs()

        overview_coords = {
            "x": (["x"], x_coords, x_attrs),
            "y": (["y"], y_coords, y_attrs),
        }

    # Determine standard name based on whether this is Sentinel-1 data
    # TODO: use a better way to determine this than just checking for ds_gcp
    if ds_gcp is not None:
        standard_name = "surface_backwards_scattering_coefficient_of_radar_wave"
    else:
        standard_name = "toa_bidirectional_reflectance"

    spatial_dims = ["y", "x"]

    # Find the grid_mapping variable name
    grid_mapping_var_name = _find_grid_mapping_var_name(ds, data_vars)

    # Downsample all data variables
    overview_data_vars = {}
    for var in data_vars:
        log.info("Downsampling variable %s", var)

        source_data = ds[var].values

        # Create downsampled data
        if source_data.ndim == 3:
            downsampled_data = np.zeros(
                (source_data.shape[0], height, width), dtype=source_data.dtype
            )
            for i in range(source_data.shape[0]):
                downsampled_data[i] = utils.downsample_2d_array(source_data[i], height, width)
            dim0 = ["time"] if "time" in ds[var].dims else [ds[var].dims[0]]
            dims = dim0 + spatial_dims
        else:
            downsampled_data = utils.downsample_2d_array(source_data, height, width)
            dims = spatial_dims

        attrs = {
            "standard_name": ds[var].attrs.get("standard_name", standard_name),
            "_ARRAY_DIMENSIONS": dims,
            "grid_mapping": grid_mapping_var_name,
        }
        if np.issubdtype(downsampled_data.dtype, np.floating):
            attrs["_FillValue"] = np.nan

        overview_data_vars[var] = (dims, downsampled_data, attrs)

    # Create overview dataset
    overview_ds = xr.Dataset(overview_data_vars, coords=overview_coords)

    # Set CRS using rioxarray first
    overview_ds.rio.write_crs(native_crs, inplace=True)
    overview_ds.attrs["grid_mapping"] = grid_mapping_var_name

    # Add grid_mapping variable after setting CRS
    # TODO: refactor? grid mapping attributes and variables are handled
    # below and above in different function bodies in a confusing way.
    # ds.rio.write_crs may conflict with manual metadata handling
    # (i.e., rioxarray writes grid_mapping attributes to Xarray encoding, not attrs)
    # --
    _add_grid_mapping_variable(
        overview_ds, ds, grid_mapping_var_name, overview_transform, native_crs
    )

    return overview_ds


def write_dataset_band_by_band_with_validation(
    ds: xr.Dataset,
    existing_dataset: xr.Dataset | None,
    output_path: str,
    encoding: dict[Hashable, XarrayEncodingJSON],
    max_retries: int,
    group_name: str,
    force_overwrite: bool = False,
) -> tuple[bool, xr.Dataset]:
    """
    Write dataset band by band with individual band validation.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to write
    existing_dataset : xarray.Dataset, optional
        Existing dataset on the target Zarr store
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
    log.info(
        "Writing GeoZarr-spec compliant base resolution band by band with validation for group %s",
        group_name,
    )

    # Get data variables
    data_vars = [var for var in ds.data_vars if not utils.is_grid_mapping_variable(ds, var)]

    successful_vars = []
    failed_vars = []
    skipped_vars = []

    store_exists = existing_dataset is not None and len(existing_dataset.data_vars) > 0

    store_storage_options = fs_utils.get_storage_options(output_path)
    fs = fs_utils.get_filesystem(output_path, **(store_storage_options or {}))

    def cleanup_prefix(prefix: str) -> None:
        key = prefix.lstrip("/")
        base_path = output_path.rstrip("/")
        if fs_utils.is_s3_path(base_path):
            target_path = fs_utils.normalize_path(f"{base_path}/{key}")
        else:
            target_path = os.path.join(base_path, key)
        try:
            fs.rm(target_path, recursive=True)
        except FileNotFoundError:
            log.debug("Target not found, skipping removal", target_path=target_path)
        except Exception as cleanup_error:
            log.info(
                "    ⚠️ Failed to remove",
                target_path=target_path,
                error=str(cleanup_error),
            )

    # Write data variables one by one with validation
    for var in data_vars:
        # Check if this variable already exists and is valid
        if not force_overwrite and store_exists:
            if utils.validate_existing_band_data(existing_dataset, var, ds):
                ds.drop_vars(str(var))
                ds[var] = existing_dataset[var]  # type: ignore[index]
                log.info("✅ Band %s already exists and is valid, skipping.", var)
                skipped_vars.append(var)
                successful_vars.append(var)
                continue
            # Remove invalid existing variable using filesystem-agnostic method
            log.info("    🧹 Removing invalid existing variable", var=var)
            cleanup_prefix(f"{group_name.lstrip('/')}/{var}")

        log.info("  Writing data variable", var=var)

        # Create a single-variable dataset with its coordinates
        single_var_ds = ds[[var]]

        # Create encoding for this variable only
        var_encoding = {}
        if var in encoding:
            var_encoding[var] = encoding[var]

        # Add coordinate encoding if not already present
        for coord in single_var_ds.coords:
            if coord in encoding and (
                existing_dataset is None or coord not in existing_dataset.coords
            ):
                var_encoding[coord] = encoding[coord]

        # Try to write this variable with retries
        success = False
        for attempt in range(max_retries):
            try:
                # Ensure the dataset is properly chunked to align with encoding
                if (
                    var in var_encoding
                    and "shards" in var_encoding[var]
                    and var_encoding[var]["shards"] is not None
                ):
                    # For sharded variables, use the shards dimensions
                    shard_dims = var_encoding[var].get("shards", None)
                    if shard_dims is not None:
                        var_dims = single_var_ds[var].dims
                        chunk_dict = {}
                        for i, dim in enumerate(var_dims):
                            if i < len(shard_dims):
                                chunk_dict[dim] = shard_dims[i]
                        single_var_ds[var] = single_var_ds[var].chunk(chunk_dict)
                elif var in var_encoding and "chunks" in var_encoding[var]:
                    target_chunks = var_encoding[var]["chunks"]
                    # Create chunk dict using the actual dimensions of the variable
                    var_dims = single_var_ds[var].dims
                    chunk_dict = {}
                    for i, dim in enumerate(var_dims):
                        if i < len(target_chunks):
                            chunk_dict[dim] = target_chunks[i]
                    # Rechunk the dataset to match the target chunks
                    single_var_ds[var] = single_var_ds[var].chunk(chunk_dict)
                else:
                    single_var_ds[var] = single_var_ds[var].chunk()

                # Sanitize NaN values in single variable dataset attributes
                single_var_ds = sanitize_dataset_attributes(single_var_ds)

                single_var_ds.to_zarr(
                    output_path,
                    group=group_name,
                    mode="a",
                    consolidated=False,
                    zarr_format=3,
                    encoding=var_encoding,
                    storage_options=store_storage_options,
                )

                log.info("    ✅ Successfully wrote", var=var)
                successful_vars.append(var)
                success = True
                if existing_dataset is None:
                    group_path = fs_utils.normalize_path(f"{output_path}/{group_name.lstrip('/')}")
                    existing_dataset = xr.open_dataset(
                        group_path,
                        mode="r",
                        engine="zarr",
                        decode_coords="all",
                        chunks="auto",
                        storage_options=store_storage_options,
                    )
                break

            except Exception as e:
                # Delete the started data array to avoid conflict on next attempt
                for written_var in var_encoding:
                    target_components = [group_name.lstrip("/"), str(written_var)]
                    target_prefix = "/".join(
                        component for component in target_components if component
                    )
                    cleanup_prefix(target_prefix)
                if attempt < max_retries - 1:
                    log.warning(
                        "    ⚠️  Attempt failed, retrying in 2 seconds",
                        attempt=attempt + 1,
                        var=var,
                        error=str(e),
                    )
                    time.sleep(2)
                else:
                    log.exception(
                        "    ❌ Failed to write variable after max retries",
                        var=var,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    failed_vars.append(var)
                    break

        if not success:
            log.info("  Failed to write data variable", var=var)

    # Consolidate metadata
    group_path = fs_utils.normalize_path(f"{output_path}/{group_name.lstrip('/')}")
    zarr_group = fs_utils.open_zarr_group(group_path, mode="r+")
    consolidate_metadata(zarr_group.store)

    log.info("✅ Metadata consolidated for %s variables", len(successful_vars))

    # Report results
    if failed_vars:
        log.error(
            "❌ Failed to write %s variables in %s: %s", len(failed_vars), group_name, failed_vars
        )
        log.info("✅ Successfully wrote %s new variables", len(successful_vars) - len(skipped_vars))
        log.info("Skipped %s existing valid variables: %s", len(skipped_vars), skipped_vars)
        return False, ds
    log.info("✅ Successfully processed %s variables in %s", len(successful_vars), group_name)
    if skipped_vars:
        log.info("Wrote %s new variables", len(successful_vars) - len(skipped_vars))
        log.info("Skipped %s existing valid variables", len(skipped_vars))
    return True, ds


def consolidate_metadata(
    store: StoreLike,
    path: str | None = None,
    zarr_format: zarr.core.common.ZarrFormat | None = None,
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
    path: str | None = None,
    zarr_format: zarr.core.common.ZarrFormat | None = None,
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


# Helper functions
def _add_coordinate_metadata(ds: xr.Dataset) -> None:
    """Add proper metadata to coordinate variables."""
    for coord_name in ds.coords:
        if coord_name == "x":
            # Check if this is geographic coordinates (EPSG:4326)
            if ds.rio.crs and ds.rio.crs.to_epsg() == 4326:
                ds[coord_name].attrs.update(
                    {
                        "_ARRAY_DIMENSIONS": ["x"],
                        "standard_name": "longitude",
                        "units": "degrees_east",
                        "long_name": "longitude",
                    }
                )
            else:
                ds[coord_name].attrs.update(
                    {
                        "_ARRAY_DIMENSIONS": ["x"],
                        "standard_name": "projection_x_coordinate",
                        "units": "m",
                        "long_name": "x coordinate of projection",
                    }
                )
        elif coord_name == "y":
            # Check if this is geographic coordinates (EPSG:4326)
            if ds.rio.crs and ds.rio.crs.to_epsg() == 4326:
                ds[coord_name].attrs.update(
                    {
                        "_ARRAY_DIMENSIONS": ["y"],
                        "standard_name": "latitude",
                        "units": "degrees_north",
                        "long_name": "latitude",
                    }
                )
            else:
                ds[coord_name].attrs.update(
                    {
                        "_ARRAY_DIMENSIONS": ["y"],
                        "standard_name": "projection_y_coordinate",
                        "units": "m",
                        "long_name": "y coordinate of projection",
                    }
                )
        elif coord_name == "time":
            ds[coord_name].attrs.update({"_ARRAY_DIMENSIONS": ["time"], "standard_name": "time"})
        elif coord_name == "angle":
            ds[coord_name].attrs.update(
                {
                    "_ARRAY_DIMENSIONS": ["angle"],
                    "standard_name": "angle",
                    "long_name": "angle coordinate",
                }
            )
        elif coord_name == "band":
            ds[coord_name].attrs.update(
                {
                    "_ARRAY_DIMENSIONS": ["band"],
                    "standard_name": "band",
                    "long_name": "spectral band identifier",
                }
            )
        elif coord_name == "detector":
            ds[coord_name].attrs.update(
                {
                    "_ARRAY_DIMENSIONS": ["detector"],
                    "standard_name": "detector",
                    "long_name": "detector identifier",
                }
            )
        else:
            # Generic coordinate
            if "_ARRAY_DIMENSIONS" not in ds[coord_name].attrs:
                ds[coord_name].attrs["_ARRAY_DIMENSIONS"] = [coord_name]


def _setup_grid_mapping(ds: xr.Dataset, grid_mapping_var_name: str) -> None:
    """Set up spatial_ref variable with GeoZarr required attributes."""

    # Use standard CRS and transform if available
    if ds.rio.crs and "spatial_ref" in ds:
        ds["spatial_ref"].attrs["_ARRAY_DIMENSIONS"] = []
        if ds.rio.transform():
            transform_gdal = ds.rio.transform().to_gdal()
            transform_str = " ".join([str(i) for i in transform_gdal])
            ds["spatial_ref"].attrs["GeoTransform"] = transform_str

    # Update all data variables to reference the grid_mapping
    ds.attrs["grid_mapping"] = grid_mapping_var_name
    for band in ds.data_vars:
        if band != "spatial_ref":
            ds[band].attrs["grid_mapping"] = grid_mapping_var_name


def _add_geotransform(ds: xr.Dataset, grid_mapping_var: str) -> None:
    """Add GeoTransform to grid_mapping variable."""
    ds[grid_mapping_var].attrs["_ARRAY_DIMENSIONS"] = []

    if len(ds.coords["x"]) > 1 and len(ds.coords["y"]) > 1:
        x_coords = ds.coords["x"].values
        y_coords = ds.coords["y"].values

        pixel_size_x = float(x_coords[1] - x_coords[0])
        pixel_size_y = float(y_coords[0] - y_coords[1])

        transform_str = f"{x_coords[0]} {pixel_size_x} 0.0 {y_coords[0]} 0.0 {pixel_size_y}"
        ds[grid_mapping_var].attrs["GeoTransform"] = transform_str


def _find_reference_crs(geozarr_groups: Mapping[str, xr.Dataset]) -> str | None:
    """Find the reference CRS in the geozarr groups."""
    for group in geozarr_groups.values():
        if group.rio.crs:
            crs_string: str = group.rio.crs.to_string()
            return crs_string
    return None


def _create_encoding(
    ds: xr.Dataset, compressor: Any, spatial_chunk: int
) -> dict[Hashable, XarrayEncodingJSON]:
    """Create encoding for dataset variables."""
    encoding: dict[Hashable, XarrayEncodingJSON] = {}
    chunking: tuple[int, ...]
    for var in ds.data_vars:
        if hasattr(ds[var].data, "chunks"):
            current_chunks = ds[var].chunks
            if len(current_chunks) >= 2:
                chunking = tuple(
                    current_chunks[i][0] if len(current_chunks[i]) > 0 else ds[var].shape[i]
                    for i in range(len(current_chunks))
                )
            else:
                chunking = (
                    current_chunks[0][0] if len(current_chunks[0]) > 0 else ds[var].shape[0],
                )
        else:
            data_shape = ds[var].shape
            if len(data_shape) >= 2:
                chunk_y = min(spatial_chunk, data_shape[-2])
                chunk_x = min(spatial_chunk, data_shape[-1])
                chunking = (1, chunk_y, chunk_x) if len(data_shape) == 3 else (chunk_y, chunk_x)
            else:
                chunking = (min(spatial_chunk, data_shape[-1]),)

        var_encoding: XarrayEncodingJSON = {
            "compressors": [compressor],
            "chunks": chunking,
        }
        fv = utils.explicit_fill_value(ds[var])
        if fv is not utils.UNSET:
            var_encoding["fill_value"] = fv
        encoding[var] = var_encoding

    # Add coordinate encoding
    for coord in ds.coords:
        encoding[coord] = {"compressors": None}

    return encoding


def _create_geozarr_encoding(
    ds: xr.Dataset, compressor: Any, spatial_chunk: int, enable_sharding: bool = False
) -> dict[Hashable, XarrayEncodingJSON]:
    """Create encoding for GeoZarr dataset variables."""
    encoding: dict[Hashable, XarrayEncodingJSON] = {}
    chunks: tuple[int, ...]
    for var in ds.data_vars:
        if utils.is_grid_mapping_variable(ds, var):
            encoding[var] = {"compressors": None}
        else:
            data_shape = ds[var].shape
            if len(data_shape) >= 2:
                height, width = data_shape[-2:]
                spatial_chunk_aligned = min(
                    spatial_chunk,
                    utils.calculate_aligned_chunk_size(width, spatial_chunk),
                    utils.calculate_aligned_chunk_size(height, spatial_chunk),
                )

                if len(data_shape) == 3:
                    chunks = (1, spatial_chunk_aligned, spatial_chunk_aligned)
                else:
                    chunks = (spatial_chunk_aligned, spatial_chunk_aligned)
            else:
                spatial_chunk_aligned = spatial_chunk
                chunks = (spatial_chunk_aligned,)

            shards: tuple[int, ...] | None = None

            if enable_sharding:
                # Calculate shard dimensions that are divisible by chunk dimensions
                if len(data_shape) == 3:
                    # For 3D data (time, y, x), ensure shard dimensions are divisible by chunks
                    shard_time = data_shape[0]  # Keep full time dimension
                    shard_y = _calculate_shard_dimension(data_shape[1], chunks[1])  # type: ignore[misc]
                    shard_x = _calculate_shard_dimension(data_shape[2], chunks[2])  # type: ignore[misc]
                    shards = (shard_time, shard_y, shard_x)
                    log.info(
                        "Sharding config for variable %s: ",
                        var,
                        data_shape=data_shape,
                        chunks=chunks,
                        shards=shards,
                    )
                elif len(data_shape) == 2:
                    # For 2D data (y, x), ensure shard dimensions are divisible by chunks
                    shard_y = _calculate_shard_dimension(data_shape[0], chunks[0])
                    shard_x = _calculate_shard_dimension(data_shape[1], chunks[1])  # type: ignore[misc]
                    shards = (shard_y, shard_x)
                    log.info(
                        "  🔧 Sharding config",
                        var=var,
                        data_shape=data_shape,
                        chunks=chunks,
                        shards=shards,
                    )
                else:
                    # For 1D data, use the full dimension
                    shards = (data_shape[0],)
                    log.info(
                        "  🔧 Sharding config",
                        var=var,
                        data_shape=data_shape,
                        chunks=chunks,
                        shards=shards,
                    )

                # Validate that shards are evenly divisible by chunks
                for i, (shard_dim, chunk_dim) in enumerate(zip(shards, chunks, strict=False)):
                    if shard_dim % chunk_dim != 0:
                        log.warning(
                            "  ⚠️  Warning: Shard dimension not evenly divisible by chunk dimension",
                            shard_dim=shard_dim,
                            chunk_dim=chunk_dim,
                            axis=i,
                        )

            var_encoding: XarrayEncodingJSON = {
                "chunks": chunks,
                "compressors": compressor,
                "shards": shards,
            }
            if np.issubdtype(ds[var].dtype, np.floating):
                var_encoding["fill_value"] = "NaN"
            else:
                fv = utils.explicit_fill_value(ds[var])
                if fv is not utils.UNSET:
                    var_encoding["fill_value"] = fv
            encoding[var] = var_encoding

    # Add coordinate encoding
    for coord in ds.coords:
        encoding[coord] = {"compressors": None}

    return encoding


def _load_existing_dataset(path: str) -> xr.Dataset | None:
    """Load existing dataset if it exists."""
    try:
        if fs_utils.path_exists(path):
            storage_options = fs_utils.get_storage_options(path)
            return xr.open_dataset(
                path,
                zarr_format=3,
                storage_options=storage_options,
                engine="zarr",
                chunks="auto",
                decode_coords="all",
            )
    except Exception as e:
        log.warning("Could not open existing dataset at path", path=path, error=str(e))
    return None


def _get_x_coord_attrs() -> StandardXCoordAttrsJSON:
    """Get standard attributes for x coordinate."""
    return {
        "units": "m",
        "long_name": "x coordinate of projection",
        "standard_name": "projection_x_coordinate",
        "_ARRAY_DIMENSIONS": ["x"],
    }


def _get_y_coord_attrs() -> StandardYCoordAttrsJSON:
    """Get standard attributes for y coordinate."""
    return {
        "units": "m",
        "long_name": "y coordinate of projection",
        "standard_name": "projection_y_coordinate",
        "_ARRAY_DIMENSIONS": ["y"],
    }


def _get_lon_coord_attrs() -> StandardLonCoordAttrsJSON:
    """Get standard attributes for longitude coordinate."""
    return {
        "units": "degrees_east",
        "long_name": "longitude",
        "standard_name": "longitude",
        "_ARRAY_DIMENSIONS": ["x"],
    }


def _get_lat_coord_attrs() -> StandardLatCoordAttrsJSON:
    """Get standard attributes for latitude coordinate."""
    return {
        "units": "degrees_north",
        "long_name": "latitude",
        "standard_name": "latitude",
        "_ARRAY_DIMENSIONS": ["y"],
    }


def _find_grid_mapping_var_name(ds: xr.Dataset, data_vars: Sequence[Hashable]) -> str:
    """Find the grid_mapping variable name from the dataset."""
    grid_mapping_var_name = ds.attrs.get("grid_mapping", None)
    if not grid_mapping_var_name and data_vars:
        first_var = data_vars[0]
        if first_var in ds and "grid_mapping" in ds[first_var].attrs:
            grid_mapping_var_name = ds[first_var].attrs["grid_mapping"]

    if not grid_mapping_var_name:
        grid_mapping_var_name = "spatial_ref"

    return str(grid_mapping_var_name)


def _add_grid_mapping_variable(
    overview_ds: xr.Dataset,
    ds: xr.Dataset,
    grid_mapping_var_name: str,
    overview_transform: Any,
    native_crs: Any,
) -> None:
    """Add grid_mapping variable to overview dataset."""

    base_attrs: dict[str, Any] = {
        "_ARRAY_DIMENSIONS": [],
    }

    if overview_transform is not None:
        transform_gdal = overview_transform.to_gdal()
        transform_str = " ".join([str(i) for i in transform_gdal])
        base_attrs["GeoTransform"] = transform_str

    if grid_mapping_var_name in ds:
        grid_mapping_attrs = ds[grid_mapping_var_name].attrs.copy()
        grid_mapping_attrs.update(base_attrs)

        overview_ds.coords[grid_mapping_var_name] = xr.DataArray(
            data=ds[grid_mapping_var_name].values,
            attrs=grid_mapping_attrs,
        )
    else:
        log.info(
            "  Creating new grid_mapping variable",
            grid_mapping_var_name=grid_mapping_var_name,
        )

        grid_mapping_attrs = base_attrs.copy()

        if native_crs:
            if native_crs.to_epsg():
                grid_mapping_attrs["spatial_ref"] = native_crs.to_wkt()
                grid_mapping_attrs["crs_wkt"] = native_crs.to_wkt()
            else:
                grid_mapping_attrs["spatial_ref"] = native_crs.to_wkt()
                grid_mapping_attrs["crs_wkt"] = native_crs.to_wkt()

        overview_ds.coords[grid_mapping_var_name] = xr.DataArray(
            data=np.array(b"", dtype="S1"),
            attrs=grid_mapping_attrs,
        )

    # Ensure all data variables have the grid_mapping attribute
    for var_name in overview_ds.data_vars:
        if (
            not utils.is_grid_mapping_variable(overview_ds, var_name)
            and "grid_mapping" not in overview_ds[var_name].attrs
        ):
            overview_ds[var_name].attrs["grid_mapping"] = grid_mapping_var_name
            log.info("Added grid_mapping attribute to %s", var_name)


def _calculate_shard_dimension(data_dim: int, chunk_dim: int) -> int:
    """
    Calculate shard dimension that is evenly divisible by chunk dimension.

    For Zarr v3 sharding with Dask, the shard dimension must be evenly
    divisible by the chunk dimension to avoid checksum mismatches.

    Parameters
    ----------
    data_dim : int
        Size of the data dimension
    chunk_dim : int
        Size of the chunk dimension

    Returns
    -------
    int
        Shard dimension that is evenly divisible by chunk_dim
    """
    # If chunk is larger than data dimension, the effective chunk will be data_dim
    # In this case, shard should also be data_dim to maintain divisibility
    if chunk_dim >= data_dim:
        return data_dim

    # Calculate how many complete chunks fit in the data dimension
    num_complete_chunks = data_dim // chunk_dim

    # If we have at least 2 complete chunks, use a multiple of chunk_dim
    if num_complete_chunks >= 2:
        # Use a shard size that's a multiple of chunk_dim
        for multiplier in range(num_complete_chunks + 1, 2, -1):
            shard_size = multiplier * chunk_dim
            if shard_size <= data_dim:
                return shard_size

    # Fallback: use the largest multiple of chunk_dim that fits
    # If no complete chunks fit, use data_dim (this handles edge cases)
    return num_complete_chunks * chunk_dim if num_complete_chunks > 0 else data_dim


def _is_sentinel1(dt: xr.DataTree) -> bool:
    """Return True if the input DataTree represents a Sentinel-1 product."""
    stac_props = dt.attrs.get("stac_discovery", {}).get("properties", {})
    return bool(stac_props.get("product:type", "not-a-product").startswith("S01"))


def get_zarr_group(data: xr.DataTree) -> zarr.Group:
    return data._close.__self__.zarr_group
