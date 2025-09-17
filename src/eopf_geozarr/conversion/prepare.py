from __future__ import annotations

import logging
from typing import Dict, Optional

import rioxarray  # noqa: F401  # Registers the .rio accessor
import xarray as xr


def setup_datatree_metadata_geozarr_spec_compliant(
    dt: xr.DataTree, groups: list[str]
) -> Dict[str, xr.Dataset]:
    geozarr_groups: Dict[str, xr.Dataset] = {}
    grid_mapping_var_name = "spatial_ref"

    for key in groups:
        logging.getLogger(__name__).debug(f"Processing group for GeoZarr compliance: {key}")
        ds = dt[key].to_dataset().copy()

        for band in ds.data_vars:
            logging.getLogger(__name__).debug(f"  Processing band: {band}")
            ds[band].attrs["standard_name"] = "toa_bidirectional_reflectance"
            if hasattr(ds[band], "dims"):
                ds[band].attrs["_ARRAY_DIMENSIONS"] = list(ds[band].dims)
            ds[band].attrs["grid_mapping"] = grid_mapping_var_name

            if "proj:epsg" in ds[band].attrs:
                epsg = ds[band].attrs["proj:epsg"]
                logging.getLogger(__name__).debug(f"    Setting CRS for {band} to EPSG:{epsg}")
                crs_str = f"epsg:{epsg}"
                ds = ds.rio.write_crs(crs_str)
                if "spatial_ref" not in ds:
                    ds["spatial_ref"] = xr.DataArray(0)

        _add_coordinate_metadata(ds)
        _setup_grid_mapping(ds, grid_mapping_var_name)
        geozarr_groups[key] = ds

    return geozarr_groups


def prepare_dataset_with_crs_info(
    ds: xr.Dataset, reference_crs: Optional[str] = None
) -> xr.Dataset:
    ds = ds.copy()
    _add_coordinate_metadata(ds)
    if "x" in ds.coords and "y" in ds.coords and reference_crs:
        logging.getLogger(__name__).debug(f"  Adding CRS information: {reference_crs}")
        ds = ds.rio.write_crs(reference_crs)
        if "spatial_ref" not in ds:
            ds["spatial_ref"] = xr.DataArray(0)
        _add_geotransform(ds, "spatial_ref")

    for var_name in ds.data_vars:
        if "_ARRAY_DIMENSIONS" not in ds[var_name].attrs and hasattr(ds[var_name], "dims"):
            ds[var_name].attrs["_ARRAY_DIMENSIONS"] = list(ds[var_name].dims)
        if "x" in ds[var_name].coords and "y" in ds[var_name].coords and reference_crs:
            ds[var_name].attrs["grid_mapping"] = "spatial_ref"
            ds[var_name].attrs["proj:epsg"] = reference_crs.split(":")[-1]
            if "spatial_ref" in ds and "GeoTransform" in ds["spatial_ref"].attrs:
                ds[var_name].attrs["proj:transform"] = ds["spatial_ref"].attrs["GeoTransform"]

    return ds


def _add_coordinate_metadata(ds: xr.Dataset) -> None:
    for coord_name in ds.coords:
        if coord_name == "x":
            ds[coord_name].attrs.update(
                {
                    "_ARRAY_DIMENSIONS": ["x"],
                    "standard_name": "projection_x_coordinate",
                    "units": "m",
                    "long_name": "x coordinate of projection",
                }
            )
        elif coord_name == "y":
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
            if "_ARRAY_DIMENSIONS" not in ds[coord_name].attrs:
                ds[coord_name].attrs["_ARRAY_DIMENSIONS"] = [coord_name]


def _setup_grid_mapping(ds: xr.Dataset, grid_mapping_var_name: str) -> None:
    if ds.rio.crs and "spatial_ref" in ds:
        ds[grid_mapping_var_name].attrs["_ARRAY_DIMENSIONS"] = []
        if ds.rio.transform():
            transform_gdal = ds.rio.transform().to_gdal()
            transform_str = " ".join([str(i) for i in transform_gdal])
            ds[grid_mapping_var_name].attrs["GeoTransform"] = transform_str
        for band in ds.data_vars:
            if band != grid_mapping_var_name:
                ds[band].attrs["grid_mapping"] = grid_mapping_var_name


def _add_geotransform(ds: xr.Dataset, grid_mapping_var: str) -> None:
    ds[grid_mapping_var].attrs["_ARRAY_DIMENSIONS"] = []
    if len(ds.coords["x"]) > 1 and len(ds.coords["y"]) > 1:
        x_coords = ds.coords["x"].values
        y_coords = ds.coords["y"].values
        pixel_size_x = float(x_coords[1] - x_coords[0])
        pixel_size_y = float(y_coords[0] - y_coords[1])
        transform_str = f"{x_coords[0]} {pixel_size_x} 0.0 {y_coords[0]} 0.0 {pixel_size_y}"
        ds[grid_mapping_var].attrs["GeoTransform"] = transform_str


def _find_reference_crs(geozarr_groups: Dict[str, xr.Dataset]) -> Optional[str]:
    for _, group in geozarr_groups.items():
        if group.rio.crs:
            return group.rio.crs.to_string()  # type: ignore[no-any-return]
    return None


# No manual CRS writers required; rioxarray is a hard dependency in this project.
