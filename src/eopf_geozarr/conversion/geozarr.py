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

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import xarray as xr

from . import fs_utils, utils
from .helpers import build_var_encodings, inject_crs_if_requested, maybe_step, normalize_group_path
from .hierarchy import ensure_group_hierarchy as _ensure_group_hierarchy
from .hierarchy import safe_rmtree as _safe_rmtree
from .io import load_existing_dataset

# encoding creation for level-0 is now minimal and done inline
from .metadata import consolidate_metadata
from .multiscales import create_geozarr_compliant_multiscales
from .prepare import (
    _find_reference_crs,
    prepare_dataset_with_crs_info,
    setup_datatree_metadata_geozarr_spec_compliant,
)
from .zarr_write import write_dataset_zarr

log = logging.getLogger(__name__)


def _normalize_spatial_dim_order(ds: xr.Dataset) -> xr.Dataset:
    """Ensure spatial dims appear in the order ('y','x') for all data vars.

    Keeps any leading non-spatial dims (e.g., 'band', 'time') unchanged and appends
    'y','x' at the end when both are present. Grid-mapping variables are ignored.
    """
    try:
        for var in list(ds.data_vars):
            if utils.is_grid_mapping_variable(ds, var):
                continue
            dims = tuple(ds[var].dims)
            if "y" in dims and "x" in dims:
                leading = [d for d in dims if d not in ("y", "x")]
                new_order = tuple(leading + ["y", "x"])
                if new_order != dims:
                    ds[var] = ds[var].transpose(*new_order)
        return ds
    except Exception:
        return ds


def create_geozarr_dataset(
    dt_input: xr.DataTree,
    groups: List[str],
    output_path: str,
    spatial_chunk: int = 4096,
    min_dimension: int = 256,
    tile_width: int = 256,
    max_retries: int = 3,
    crs_groups: Optional[List[str]] = None,
    overwrite: str = "fail",
    max_overview_levels: Optional[int] = None,
    skip_overviews: bool = False,
    base_write_mode: str = "single",
    copy_non_selected_groups: bool = False,
    crs_injection_groups: Optional[List[str]] = None,
) -> xr.DataTree:
    """
    Create a GeoZarr-spec 0.4 compliant dataset from EOPF data.

    Parameters
    ----------
    dt_input : xr.DataTree
        Input EOPF DataTree
    groups : list[str]
        List of group names to process as GeoZarr datasets
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
        List of group names that need CRS information added on best-effort basis (no-op here)
    overwrite : str, default "fail"
        Overwrite policy: one of {"fail", "skip", "merge", "replace"}

    Returns
    -------
    xr.DataTree
        DataTree containing the GeoZarr compliant data
    """
    # Initialize an empty result and create the root store
    dt_result = xr.DataTree()
    storage_options = fs_utils.get_storage_options(output_path)
    write_dataset_zarr(
        dt_result.to_dataset() if hasattr(dt_result, "to_dataset") else xr.Dataset(),
        output_path,
        mode="a",
        consolidated=True,
        safe_chunks=False,
        storage_options=storage_options,
    )

    # Prepare GeoZarr-compliant datasets per ADR (CF attrs, _ARRAY_DIMENSIONS, grid_mapping)
    with maybe_step(None, "prepare_metadata"):
        geozarr_groups = setup_datatree_metadata_geozarr_spec_compliant(dt_input, groups)

    # ADR compression: Blosc zstd level 3 with shuffle
    try:
        from zarr.codecs import BloscCodec

        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0)
    except Exception:
        compressor = None

    # Orchestrator uses ADR compressor
    writer = GeoZarrWriter(
        output_path=output_path,
        compressor=compressor,
        spatial_chunk=spatial_chunk,
        min_dimension=min_dimension,
        tile_width=tile_width,
        max_retries=max_retries,
        overwrite=overwrite,
        max_overview_levels=max_overview_levels,
        skip_overviews=skip_overviews,
        base_write_mode=base_write_mode,
    )

    requested = {normalize_group_path(g) for g in groups}

    # Write only the requested target groups
    crs_requested = {normalize_group_path(g) for g in (crs_groups or [])} if crs_groups else None

    def ref_crs_func() -> Any:
        return _find_reference_crs(geozarr_groups)

    for key, ds in geozarr_groups.items():
        current_group = normalize_group_path(key)
        ds = inject_crs_if_requested(
            current_group,
            ds,
            crs_requested,
            ref_crs_func,
            prepare_dataset_with_crs_info,
            verbose=True if crs_requested else False,
        )
        with maybe_step(None, "write_group"):
            dt_result = writer.write_group(dt_result, current_group, ds)

    # Report any explicitly requested crs_groups that were not found
    if crs_groups:
        existing = {normalize_group_path(g) for g in geozarr_groups.keys()}
        for g in crs_groups:
            g_norm = normalize_group_path(g)
            if g_norm not in existing:
                print(f"CRS group {g_norm} not found in DataTree (best-effort skip)")

    # Optionally copy non-selected groups verbatim (no overviews) for parity
    if copy_non_selected_groups:
        storage_options = fs_utils.get_storage_options(output_path)
        for rel_path, node in dt_input.subtree_with_keys:
            if rel_path == ".":
                continue
            grp = "/" + rel_path
            if grp in requested:
                continue
            try:
                ds = node.to_dataset().drop_encoding()
            except Exception:
                continue
            if len(ds.data_vars) == 0 and len(ds.coords) == 0:
                continue

            # Optional CRS injection for specified groups
            if crs_injection_groups and (
                grp in crs_injection_groups or rel_path in crs_injection_groups
            ):
                try:
                    ref_crs = _find_reference_crs(geozarr_groups)
                    ds = prepare_dataset_with_crs_info(ds, reference_crs=ref_crs)
                except Exception as e:
                    log.debug(f"CRS injection skipped for {grp}: {e}")

            # Minimal encoding: reuse helper
            encoding: Dict[str, Any] = build_var_encodings(ds, compressor)

            # Clear legacy encodings to avoid zarr v2 artifacts
            try:
                for name in list(ds.variables):
                    enc = getattr(ds[name], "encoding", None)
                    if isinstance(enc, dict):
                        enc.clear()
            except Exception:
                pass

            # Ensure hierarchy exists and write dataset directly (no multiscales)
            _ensure_group_hierarchy(output_path, grp)
            mode = "a"
            target_path = fs_utils.normalize_path(f"{output_path}/{rel_path}")
            try:
                exists = fs_utils.path_exists(target_path)
            except Exception:
                exists = False
            if overwrite == "replace" and exists:
                try:
                    _safe_rmtree(target_path)
                    exists = False
                except Exception as e:
                    log.warning(f"Failed to clean existing group at {target_path}: {e}")
            if not exists:
                mode = "w"

            write_dataset_zarr(
                ds,
                output_path,
                group=rel_path,
                mode=mode,
                consolidated=True,
                zarr_format=3,
                encoding=encoding if encoding else None,
                align_chunks=True,
                safe_chunks=False,
                storage_options=storage_options,
            )

    # Final root-level metadata consolidation per ADR
    try:
        with maybe_step(None, "root_consolidate"):
            zgroup = fs_utils.open_zarr_group(fs_utils.normalize_path(output_path), mode="r+")
            consolidate_metadata(zgroup.store)
    except Exception as e:
        log.warning(f"Root metadata consolidation skipped: {e}")

    return dt_result


def write_geozarr_group(
    dt_result: xr.DataTree,
    group_name: str,
    ds: xr.Dataset,
    output_path: str,
    spatial_chunk: int = 4096,
    compressor: Any = None,
    max_retries: int = 3,
    min_dimension: int = 256,
    tile_width: int = 256,
    overwrite: str = "fail",
) -> xr.DataTree:
    """Thin wrapper delegating to GeoZarrWriter for backward compatibility."""
    writer = GeoZarrWriter(
        output_path=output_path,
        compressor=compressor,
        spatial_chunk=spatial_chunk,
        min_dimension=min_dimension,
        tile_width=tile_width,
        max_retries=max_retries,
        overwrite=overwrite,
    )
    return writer.write_group(dt_result, group_name, ds)


def write_dataset_band_by_band_with_validation(
    ds: xr.Dataset,
    existing_dataset: Optional[xr.Dataset],
    output_path: str,
    encoding: Dict[str, Any],
    max_retries: int,
    group_name: str,
    force_overwrite: bool = False,
) -> Tuple[bool, xr.Dataset]:
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
    log.info(f"Writing base resolution for {group_name} band-by-band with validation")

    # Get data variables
    data_vars: List[str] = [
        str(var) for var in ds.data_vars if not utils.is_grid_mapping_variable(ds, str(var))
    ]

    successful_vars = []
    failed_vars = []
    skipped_vars = []

    store_exists = existing_dataset is not None and len(existing_dataset.data_vars) > 0

    # Write data variables one by one with validation
    for var in data_vars:
        # Check if this band already exists and is valid
        if not force_overwrite and store_exists:
            if existing_dataset is not None and utils.validate_existing_band_data(
                existing_dataset, var, ds
            ):
                # Ensure the on-disk array looks complete before skipping
                try:
                    array_dir = fs_utils.normalize_path(
                        f"{output_path}/{group_name.lstrip('/')}" + f"/{var}"
                    )
                    ok, _rep = utils.audit_group_completeness(array_dir, [var])
                except Exception as e:
                    log.warning(f"Audit failed for {var} ({e}); will rewrite")
                    ok = False
                if ok:
                    ds.drop_vars(var)
                    ds[var] = existing_dataset[var]
                    log.info(f"Band {var} already exists and is valid; skipping")
                    skipped_vars.append(var)
                    successful_vars.append(var)
                    continue
                else:
                    # Remove incomplete array directory and proceed to write
                    try:
                        if os.path.exists(array_dir):
                            _safe_rmtree(array_dir)
                            log.info(f"Removed incomplete array for {var}; will rewrite")
                    except Exception as e:
                        log.warning(f"Failed to clean incomplete array for {var}: {e}")
            try:
                _arr_dir = fs_utils.normalize_path(f"{output_path}/{group_name.lstrip('/')}/{var}")
                if os.path.exists(_arr_dir):
                    _safe_rmtree(_arr_dir)
            except Exception:
                pass

        log.info(f"Writing data variable {var}...")

        # Create a single-variable dataset with its coordinates
        single_var_ds = ds[[var]]

        # Create encoding for this variable only
        var_encoding = {}
        if var in encoding:
            var_encoding[var] = encoding[var]
            # Avoid grid_mapping duplication in encoding dict
            if isinstance(var_encoding[var], dict):
                var_encoding[var].pop("grid_mapping", None)

        # Add coordinate encoding if not already present
        for coord in single_var_ds.coords:
            if coord in encoding and (
                existing_dataset is None or coord not in existing_dataset.coords
            ):
                var_encoding[coord] = encoding[coord]
                if isinstance(var_encoding[coord], dict):
                    var_encoding[coord].pop("grid_mapping", None)

        # Try to write this variable with retries
        success = False
        for attempt in range(max_retries):
            try:
                # Do not supply encoding for grid-mapping variable even if present
                if "spatial_ref" in var_encoding:
                    var_encoding.pop("spatial_ref", None)
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
                    group=group_name.lstrip("/"),
                    mode="a",
                    consolidated=False,
                    zarr_format=3,
                    encoding=var_encoding,
                    align_chunks=True,
                    safe_chunks=False,
                    storage_options=storage_options,
                )

                log.info(f"Successfully wrote {var}")
                successful_vars.append(var)
                success = True
                if existing_dataset is None:
                    group_path = fs_utils.normalize_path(f"{output_path}/{group_name.lstrip('/')}")
                    storage_options = fs_utils.get_storage_options(output_path)
                    existing_dataset = xr.open_dataset(
                        group_path,
                        mode="r",
                        engine="zarr",
                        decode_coords="all",
                        chunks="auto",
                        storage_options=storage_options,
                    )
                break

            except Exception as e:
                # Delete the started data array to avoid conflict on next attempt
                for written_var in var_encoding.keys():
                    if os.path.exists(
                        os.path.join(output_path, group_name.lstrip("/"), written_var)
                    ):
                        _safe_rmtree(os.path.join(output_path, group_name.lstrip("/"), written_var))
                if attempt < max_retries - 1:
                    log.warning(f"Attempt {attempt + 1} failed for {var}: {e}; retrying in 2s")
                    time.sleep(2)
                else:
                    log.error(f"Failed to write {var} after {max_retries} attempts: {e}")
                    failed_vars.append(var)
                    break

        if not success:
            log.error(f"Failed to write data variable {var}")

    # Consolidate metadata
    group_path = fs_utils.normalize_path(f"{output_path}/{group_name.lstrip('/')}")
    zarr_group = fs_utils.open_zarr_group(group_path, mode="r+")
    consolidate_metadata(zarr_group.store)

    log.info(f"Metadata consolidated for {len(successful_vars)} variables")

    # Report results
    if failed_vars:
        log.error(f"Failed to write {len(failed_vars)} variables for {group_name}: {failed_vars}")
        log.info(f"Successfully wrote {len(successful_vars) - len(skipped_vars)} new variables")
        log.info(f"Skipped {len(skipped_vars)} existing valid variables: {skipped_vars}")
        return False, ds
    else:
        log.info(f"Successfully processed all {len(successful_vars)} variables for {group_name}")
        if skipped_vars:
            log.info(f"- Wrote {len(successful_vars) - len(skipped_vars)} new variables")
            log.info(f"- Skipped {len(skipped_vars)} existing valid variables")
        return True, ds


class GeoZarrWriter:
    """Orchestrates GeoZarr group writes, including level-0, overviews, and metadata."""

    def __init__(
        self,
        *,
        output_path: str,
        compressor: Any,
        spatial_chunk: int,
        min_dimension: int,
        tile_width: int,
        max_retries: int,
        overwrite: str,
        _load_existing_dataset: Any = load_existing_dataset,
        max_overview_levels: Optional[int] = None,
        skip_overviews: bool = False,
        base_write_mode: str = "single",
    ) -> None:
        self.output_path = output_path
        self.compressor = compressor
        self.spatial_chunk = spatial_chunk
        self.min_dimension = min_dimension
        self.tile_width = tile_width
        self.max_retries = max_retries
        self.overwrite = overwrite
        self._load_existing_dataset = _load_existing_dataset
        self.max_overview_levels = max_overview_levels
        self.skip_overviews = skip_overviews
        self.base_write_mode = (
            base_write_mode if base_write_mode in {"single", "band-validated"} else "single"
        )

    def write_group(self, dt_result: xr.DataTree, group_name: str, ds: xr.Dataset) -> xr.DataTree:
        from .helpers import (  # local import to avoid cycles
            build_var_encodings,
            normalize_group_path,
        )

        norm_group = normalize_group_path(group_name)
        _ensure_group_hierarchy(self.output_path, norm_group)

        # Prepare minimal encoding to preserve source layout and avoid coord conflicts
        # Only set compressors and dtype for data variables; do not force chunks or encode coords
        encoding: Dict[str, Any] = build_var_encodings(ds, self.compressor)

        level0_group = f"{norm_group}/0"
        level0_path = fs_utils.normalize_path(f"{self.output_path}/{level0_group.lstrip('/')}")

        # Handle overwrite policy for base level
        mode = "a"
        try:
            exists = fs_utils.path_exists(level0_path)
        except Exception:
            exists = False
        if self.overwrite == "replace" and exists:
            try:
                _safe_rmtree(level0_path)
                exists = False
            except Exception as e:
                log.warning(f"Failed to clean existing level-0 group at {level0_path}: {e}")
        if not exists:
            mode = "w"

        # Clear any legacy encodings (e.g., numcodecs 'compressor') from source variables
        try:
            for name in list(ds.variables):
                enc = getattr(ds[name], "encoding", None)
                if isinstance(enc, dict):
                    enc.clear()
        except Exception:
            pass

        storage_options = fs_utils.get_storage_options(self.output_path)

        # Decide write strategy for level-0
        if self.base_write_mode == "band-validated":
            # Load existing dataset if present to enable skip/merge
            existing = None
            try:
                if fs_utils.path_exists(level0_path):
                    existing = self._load_existing_dataset(level0_path)
            except Exception:
                existing = None

            # If overwrite=replace and exists, clean and reset
            if self.overwrite == "replace" and fs_utils.path_exists(level0_path):
                try:
                    _safe_rmtree(level0_path)
                    existing = None
                except Exception as e:
                    log.warning(f"Failed to clean existing level-0 group at {level0_path}: {e}")

            # Use band-by-band validated writing
            ok, _ = write_dataset_band_by_band_with_validation(
                ds,
                existing,
                self.output_path,
                encoding,
                self.max_retries,
                level0_group,
                force_overwrite=(self.overwrite == "replace"),
            )
            if not ok and self.overwrite == "fail":
                raise RuntimeError(f"Level-0 write failed for {level0_group}")
        else:
            # Single-shot write of the full dataset at level 0 (preserve values as-is)
            write_dataset_zarr(
                ds,
                self.output_path,
                group=level0_group.lstrip("/"),
                mode=mode,
                consolidated=True,
                zarr_format=3,
                encoding=encoding if encoding else None,
                align_chunks=True,
                safe_chunks=False,
                storage_options=storage_options,
            )

        # Consolidate metadata at level 0
        try:
            zarr_group = fs_utils.open_zarr_group(level0_path, mode="r+")
            consolidate_metadata(zarr_group.store)
        except Exception as e:
            log.warning(f"Level-0 metadata consolidation skipped for {level0_group}: {e}")

        ds_updated = ds

        # Generate GeoZarr multiscales and write overviews
        if not self.skip_overviews:
            try:
                create_geozarr_compliant_multiscales(
                    ds_updated,
                    self.output_path,
                    norm_group,
                    min_dimension=self.min_dimension,
                    tile_width=self.tile_width,
                    spatial_chunk=self.spatial_chunk,
                    overwrite=self.overwrite,
                    max_retries=self.max_retries,
                    max_overview_levels=self.max_overview_levels,
                )
            except Exception as e:
                print(f"Warning: skipping overviews for {norm_group} due to error: {e}")
                log.warning(f"Skipping overviews for {norm_group}: {e}")

        # Update result tree
        if norm_group == "/":
            # Assign dataset to root of the DataTree
            dt_result = xr.DataTree(ds_updated)
        else:
            dt_result[norm_group.lstrip("/")] = xr.DataTree(ds_updated)
        return dt_result
