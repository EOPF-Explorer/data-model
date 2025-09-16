"""Multiscales overview generation and metadata helpers.

Downsampling relies on numpy which may lack full type stubs in minimal
environments; suppress missing-import noise locally.
"""

# mypy: disable-error-code=import-not-found

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from . import fs_utils, utils
from .encoding import create_geozarr_encoding
from .metadata import consolidate_metadata
from .zarr_write import write_dataset_zarr

log = logging.getLogger(__name__)


def calculate_overview_levels(
    native_width: int,
    native_height: int,
    min_dimension: int = 256,
    tile_width: int = 256,
) -> List[Dict[str, Any]]:
    levels: List[Dict[str, Any]] = []
    level = 0
    while True:
        width = native_width // (2**level)
        height = native_height // (2**level)
        if min(width, height) < min_dimension:
            break
        zoom_w = max(0, int(np.ceil(np.log2(width / tile_width))))
        zoom_h = max(0, int(np.ceil(np.log2(height / tile_width))))
        levels.append(
            {
                "level": level,
                "zoom": max(zoom_w, zoom_h),
                "width": width,
                "height": height,
                "scale_factor": 2**level,
            }
        )
        level += 1
    return levels


def create_native_crs_tile_matrix_set(
    native_crs: Any,
    native_bounds: Tuple[float, float, float, float],
    overview_levels: List[Dict[str, Any]],
    group_prefix: Optional[str] = "",
) -> Dict[str, Any]:
    left, bottom, right, top = native_bounds

    def matrix_entry(ol: Dict[str, Any]) -> Dict[str, Any]:
        width, height = ol["width"], ol["height"]
        cell_size = max((right - left) / width, (top - bottom) / height)
        scale_denominator = cell_size * 3779.5275
        matrix_width = int(np.ceil(width / 256))
        matrix_height = int(np.ceil(height / 256))
        mid = f"{group_prefix}/{ol['level']}" if group_prefix else str(ol["level"])
        return {
            "id": mid,
            "scaleDenominator": scale_denominator,
            "cellSize": cell_size,
            "pointOfOrigin": [left, top],
            "tileWidth": 256,
            "tileHeight": 256,
            "matrixWidth": matrix_width,
            "matrixHeight": matrix_height,
        }

    epsg_code = native_crs.to_epsg() if native_crs else None
    crs_uri = (
        f"http://www.opengis.net/def/crs/EPSG/0/{epsg_code}"
        if epsg_code
        else (native_crs.to_wkt() if native_crs else "")
    )
    return {
        "id": f"Native_CRS_{epsg_code if epsg_code else 'Custom'}",
        "title": f"Native CRS Tile Matrix Set ({native_crs})",
        "crs": crs_uri,
        "supportedCRS": crs_uri,
        "orderedAxes": ["X", "Y"],
        "tileMatrices": [matrix_entry(ol) for ol in overview_levels],
    }


def _get_x_coord_attrs() -> Dict[str, Any]:
    return {
        "units": "m",
        "long_name": "x coordinate of projection",
        "standard_name": "projection_x_coordinate",
        "_ARRAY_DIMENSIONS": ["x"],
    }


def _get_y_coord_attrs() -> Dict[str, Any]:
    return {
        "units": "m",
        "long_name": "y coordinate of projection",
        "standard_name": "projection_y_coordinate",
        "_ARRAY_DIMENSIONS": ["y"],
    }


def _find_grid_mapping_var_name(ds: xr.Dataset, data_vars: List[str]) -> str:
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
    if grid_mapping_var_name in ds:
        grid_mapping_attrs = ds[grid_mapping_var_name].attrs.copy()
        transform_gdal = overview_transform.to_gdal()
        grid_mapping_attrs["GeoTransform"] = " ".join([str(i) for i in transform_gdal])
        grid_mapping_attrs["_ARRAY_DIMENSIONS"] = []
        overview_ds[grid_mapping_var_name] = xr.DataArray(
            data=ds[grid_mapping_var_name].values,
            attrs=grid_mapping_attrs,
        )
    else:
        log.debug(f"Creating new grid_mapping variable '{grid_mapping_var_name}'")
        transform_gdal = overview_transform.to_gdal()
        transform_str = " ".join([str(i) for i in transform_gdal])
        grid_mapping_new: Dict[str, Any] = {
            "_ARRAY_DIMENSIONS": [],
            "GeoTransform": transform_str,
        }
        if native_crs:
            grid_mapping_new["spatial_ref"] = native_crs.to_wkt()
            grid_mapping_new["crs_wkt"] = native_crs.to_wkt()
        overview_ds[grid_mapping_var_name] = xr.DataArray(
            data=np.array(b"", dtype="S1"), attrs=grid_mapping_new
        )


def create_overview_dataset_all_vars(
    ds: xr.Dataset,
    level: int,
    width: int,
    height: int,
    native_crs: Any,
    native_bounds: Tuple[float, float, float, float],
    data_vars: List[str],
) -> xr.Dataset:
    import rasterio.transform

    overview_transform = rasterio.transform.from_bounds(*native_bounds, width, height)
    grid_mapping_var_name = _find_grid_mapping_var_name(ds, data_vars)

    overview_vars: Dict[str, xr.DataArray] = {}
    for var in data_vars:
        log.debug(f"Downsampling {var} with lazy coarsen")
        arr = ds[var]
        coarsen_kwargs: Dict[str, int] = {}
        if "y" in arr.dims:
            coarsen_kwargs["y"] = 2
        if "x" in arr.dims:
            coarsen_kwargs["x"] = 2
        if not coarsen_kwargs:
            overview_vars[var] = arr
            continue
        coarsener = arr.coarsen(boundary="trim", **coarsen_kwargs)
        down = coarsener.mean()
        attrs = dict(arr.attrs)
        attrs.setdefault("standard_name", "toa_bidirectional_reflectance")
        attrs["_ARRAY_DIMENSIONS"] = list(down.dims)
        attrs["grid_mapping"] = grid_mapping_var_name
        overview_vars[var] = down.assign_attrs(attrs)

    overview_ds: xr.Dataset = xr.Dataset(overview_vars)
    left, bottom, right, top = native_bounds
    x_coords = np.linspace(left, right, width, endpoint=False)
    y_coords = np.linspace(top, bottom, height, endpoint=False)
    overview_ds = overview_ds.assign_coords(x=("x", x_coords), y=("y", y_coords))
    if "x" in overview_ds.coords:
        overview_ds["x"].attrs = _get_x_coord_attrs()
    if "y" in overview_ds.coords:
        overview_ds["y"].attrs = _get_y_coord_attrs()

    _add_grid_mapping_variable(
        overview_ds, ds, grid_mapping_var_name, overview_transform, native_crs
    )

    # rioxarray can set dataset-level CRS/grid mapping and sometimes drop
    # variable-level 'grid_mapping' attributes. Write CRS first, then
    # re-inject per-variable attribute to guarantee persistence.
    overview_ds = overview_ds.rio.write_crs(native_crs)
    for var in list(overview_ds.data_vars):
        if utils.is_grid_mapping_variable(overview_ds, var):
            continue
        try:
            overview_ds[var].attrs["grid_mapping"] = grid_mapping_var_name
        except Exception:
            pass
    overview_ds.attrs["grid_mapping"] = grid_mapping_var_name
    return overview_ds


def _create_tile_matrix_limits(
    overview_levels: List[Dict[str, Any]], tile_width: int
) -> Dict[str, Any]:
    return {
        str(ol["level"]): {
            "tileMatrix": str(ol["level"]),
            "minTileCol": 0,
            "maxTileCol": int(np.ceil(ol["width"] / tile_width)) - 1,
            "minTileRow": 0,
            "maxTileRow": int(np.ceil(ol["height"] / tile_width)) - 1,
        }
        for ol in overview_levels
    }


def create_geozarr_compliant_multiscales(
    ds: xr.Dataset,
    output_path: str,
    group_name: str,
    min_dimension: int = 256,
    tile_width: int = 256,
    spatial_chunk: int = 4096,
    overwrite: str = "fail",
    max_retries: int = 3,
    max_overview_levels: Optional[int] = None,
) -> Dict[str, Any]:
    from zarr.codecs import BloscCodec

    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

    data_vars: List[str] = [
        str(var) for var in ds.data_vars if not utils.is_grid_mapping_variable(ds, str(var))
    ]
    if not data_vars:
        return {}

    first_var: str = data_vars[0]
    native_height, native_width = ds[first_var].shape[-2:]
    native_crs = ds.rio.crs
    native_bounds = ds.rio.bounds()

    log.info(f"Creating GeoZarr-compliant multiscales for {group_name}")
    log.debug(f"Native resolution: {native_width} x {native_height}")
    log.debug(f"Native CRS: {native_crs}")

    overview_levels: List[Dict[str, Any]] = calculate_overview_levels(
        native_width, native_height, min_dimension, tile_width
    )
    if max_overview_levels is not None:
        try:
            # levels include 0; keep levels <= max_overview_levels
            overview_levels = [
                ol for ol in overview_levels if int(ol.get("level", 0)) <= int(max_overview_levels)
            ]
        except Exception:
            pass
    log.info(f"Total overview levels: {len(overview_levels)}")
    for ol in overview_levels:
        log.debug(
            "Overview level %s: %s x %s (scale: %s)",
            ol["level"],
            ol["width"],
            ol["height"],
            ol["scale_factor"],
        )

    tile_matrix_set: Dict[str, Any] = create_native_crs_tile_matrix_set(
        native_crs, native_bounds, overview_levels, None
    )
    tile_matrix_limits: Dict[str, Any] = _create_tile_matrix_limits(overview_levels, tile_width)

    zarr_json_path = fs_utils.normalize_path(f"{output_path}/{group_name.lstrip('/')}/zarr.json")
    zarr_json = fs_utils.read_json_metadata(zarr_json_path)
    zarr_json_attributes = zarr_json.get("attributes", {})
    zarr_json_attributes["multiscales"] = {
        "tile_matrix_set": tile_matrix_set,
        "resampling_method": "average",
        "tile_matrix_limits": tile_matrix_limits,
    }
    fs_utils.write_json_metadata(zarr_json_path, zarr_json)
    log.debug(f"Added multiscales metadata to {group_name}")

    timing_data: List[Dict[str, Any]] = []
    previous_level_ds = ds
    overview_datasets: Dict[int, xr.Dataset] = {}

    for overview in overview_levels:
        level = overview["level"]
        if level == 0:
            log.debug("Skipping level 0 - native resolution is already in group 0")
            continue

        width = overview["width"]
        height = overview["height"]
        scale_factor = overview["scale_factor"]
        log.info(f"Creating overview level {level} (1:{scale_factor})")
        log.debug(f"Target dimensions: {width} x {height}")
        log.debug(f"Creating level {level} from level {level - 1}")

        overview_ds = create_overview_dataset_all_vars(
            previous_level_ds,
            level,
            width,
            height,
            native_crs,
            native_bounds,
            data_vars,
        )
        encoding = create_geozarr_encoding(overview_ds, compressor, spatial_chunk)
        overview_path = fs_utils.normalize_path(f"{output_path}/{group_name.lstrip('/')}/{level}")
        start_time = time.time()
        storage_options = fs_utils.get_storage_options(overview_path)
        log.info(f"Writing overview level {level} at {overview_path}")

        if fs_utils.path_exists(overview_path) and overwrite in {"skip", "merge"}:
            try:
                ok, _rep = utils.audit_group_completeness(
                    overview_path, [v for v in data_vars if v in overview_ds.data_vars]
                )
            except Exception as e:
                log.warning(f"Overview audit failed ({e}); will rewrite level {level}")
                ok = False
            if ok:
                log.info(f"Skipping overview level {level} (already exists and complete)")
                overview_datasets[level] = overview_ds
                previous_level_ds = overview_ds
                continue

        if not fs_utils.is_s3_path(overview_path):
            os.makedirs(os.path.dirname(overview_path), exist_ok=True)

        overview_group = f"{group_name}/{level}"
        # Write with retries to mitigate transient network disconnects
        last_err: Optional[Exception] = None
        for attempt in range(int(max_retries) if max_retries else 1):
            try:
                write_dataset_zarr(
                    overview_ds,
                    output_path,
                    group=overview_group.lstrip("/"),
                    mode="w",
                    consolidated=True,
                    zarr_format=3,
                    encoding=encoding,
                    align_chunks=True,
                    safe_chunks=False,
                    storage_options=storage_options,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < int(max_retries) - 1:
                    log.warning(
                        f"Overview level {level}: attempt {attempt + 1} failed ({e}); retrying in 2s"
                    )
                    time.sleep(2)
                else:
                    log.error(f"Overview level {level}: failed after {max_retries} attempts: {e}")
        if last_err is not None:
            raise last_err

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
        log.info(f"Level {level}: Successfully created in {proc_time:.2f}s")

        group_path = fs_utils.normalize_path(f"{output_path}/{overview_group.lstrip('/')}")
        zarr_group = fs_utils.open_zarr_group(group_path, mode="r+")
        consolidate_metadata(zarr_group.store)
        log.debug(f"Metadata consolidated for overview level {level}")

        previous_level_ds = overview_ds

    log.info(
        f"Created {len(overview_levels)} GeoZarr-compliant overview levels using pyramid approach"
    )
    return {
        "overview_datasets": overview_datasets,
        "levels": overview_levels,
        "timing": timing_data,
        "tile_matrix_set": tile_matrix_set,
        "tile_matrix_limits": tile_matrix_limits,
    }
