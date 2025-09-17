"""
Sentinel-1 reprojection utilities for GeoZarr conversion.

This module provides functions to reproject Sentinel-1 GRD data from radar geometry
to geographic coordinates (lat/lon) using Ground Control Points (GCPs).
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rioxarray  # noqa: F401  # Import to enable .rio accessor
import xarray as xr
from pyproj import CRS
from typing import Tuple, Optional


def reproject_sentinel1_with_gcps(
    ds: xr.Dataset,
    ds_gcp: xr.Dataset,
    target_crs: str = "EPSG:4326",
    resampling: Resampling = Resampling.bilinear,
) -> xr.Dataset:
    """
    Reproject Sentinel-1 dataset from radar geometry to geographic coordinates using GCPs.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input Sentinel-1 dataset with azimuth_time/ground_range dimensions
    ds_gcp : xr.Dataset
        Dataset containing Ground Control Points with line, pixel, latitude, longitude, height
    target_crs : str, default "EPSG:4326"
        Target coordinate reference system
    resampling : rasterio.warp.Resampling, default Resampling.bilinear
        Resampling method for reprojection
        
    Returns
    -------
    xr.Dataset
        Reprojected dataset with x/y coordinates in target CRS
    """
    print(f"Reprojecting Sentinel-1 data to {target_crs} using GCPs...")
    
    # Set up GCPs from the GCP dataset
    gcps = _create_gcps_from_dataset(ds_gcp)
    
    # Get the first data variable to determine dimensions and calculate transform
    data_vars = [var for var in ds.data_vars if var != 'spatial_ref']
    if not data_vars:
        raise ValueError("No data variables found in dataset")
    
    first_var = data_vars[0]
    src_height, src_width = ds[first_var].shape[-2:]
    
    # Calculate the target transform and dimensions
    transform, width, height = calculate_default_transform(
        src_crs="EPSG:4326",  # GCPs are in lat/lon
        dst_crs=target_crs,
        width=src_width,
        height=src_height,
        gcps=gcps,
    )
    
    print(f"Calculated target dimensions: {width} x {height}")
    print(f"Transform: {transform}")
    
    # Create target coordinate arrays
    target_coords = _create_target_coordinates(transform, width, height, target_crs)
    
    # Reproject all data variables
    reprojected_data_vars = {}
    for var_name in data_vars:
        print(f"  Reprojecting variable: {var_name}")
        reprojected_var = _reproject_data_variable(
            ds[var_name], gcps, transform, width, height, target_crs, resampling
        )
        reprojected_data_vars[var_name] = reprojected_var
    
    # Create the reprojected dataset
    reprojected_ds = xr.Dataset(
        data_vars=reprojected_data_vars,
        coords=target_coords,
        attrs=ds.attrs.copy()
    )
    
    # Set CRS information
    reprojected_ds = reprojected_ds.rio.write_crs(target_crs)
    
    print(f"✅ Successfully reprojected Sentinel-1 data to {target_crs}")
    return reprojected_ds


def _create_gcps_from_dataset(ds_gcp: xr.Dataset) -> list[rasterio.control.GroundControlPoint]:
    """Create rasterio GCPs from GCP dataset."""
    # Flatten the GCP dataset to get all points
    ds_gcp_flat = ds_gcp.stack(points=list(ds_gcp.dims))
    
    rows = ds_gcp_flat["line"].values
    cols = ds_gcp_flat["pixel"].values
    x = ds_gcp_flat["longitude"].values
    y = ds_gcp_flat["latitude"].values
    z = ds_gcp_flat["height"].values
    
    gcps = []
    for i in range(ds_gcp_flat.sizes["points"]):
        gcp = rasterio.control.GroundControlPoint(
            row=float(rows[i]),
            col=float(cols[i]),
            x=float(x[i]),
            y=float(y[i]),
            z=float(z[i]),
            id=str(i),
            info="",
        )
        gcps.append(gcp)
    
    print(f"Created {len(gcps)} Ground Control Points")
    return gcps


def _create_target_coordinates(
    transform: rasterio.Affine, 
    width: int, 
    height: int, 
    target_crs: str
) -> dict:
    """Create target coordinate arrays for reprojected dataset."""
    # Calculate coordinate arrays from transform
    x_coords = np.array([transform * (i + 0.5, 0) for i in range(width)])[:, 0]
    y_coords = np.array([transform * (0, j + 0.5) for j in range(height)])[:, 1]
    
    coords = {
        "x": (
            ["x"], 
            x_coords, 
            {
                "_ARRAY_DIMENSIONS": ["x"],
                "standard_name": "projection_x_coordinate" if "EPSG:4326" in target_crs else "longitude",
                "units": "degrees_east" if "EPSG:4326" in target_crs else "m",
                "long_name": "longitude" if "EPSG:4326" in target_crs else "x coordinate of projection",
            }
        ),
        "y": (
            ["y"], 
            y_coords, 
            {
                "_ARRAY_DIMENSIONS": ["y"],
                "standard_name": "projection_y_coordinate" if "EPSG:4326" in target_crs else "latitude", 
                "units": "degrees_north" if "EPSG:4326" in target_crs else "m",
                "long_name": "latitude" if "EPSG:4326" in target_crs else "y coordinate of projection",
            }
        ),
    }
    
    return coords


def _reproject_data_variable(
    data_var: xr.DataArray,
    gcps: list[rasterio.control.GroundControlPoint],
    transform: rasterio.Affine,
    width: int,
    height: int,
    target_crs: str,
    resampling: Resampling,
) -> xr.DataArray:
    """Reproject a single data variable using GCPs."""
    # Handle different dimensionalities
    if data_var.ndim == 2:
        # 2D array (azimuth_time, ground_range)
        reprojected_data = _reproject_2d_array(
            data_var.values, gcps, transform, width, height, resampling
        )
        dims = ["y", "x"]
        
    elif data_var.ndim == 3:
        # 3D array (time, azimuth_time, ground_range)
        time_size = data_var.shape[0]
        reprojected_data = np.zeros((time_size, height, width), dtype=data_var.dtype)
        
        for t in range(time_size):
            reprojected_data[t] = _reproject_2d_array(
                data_var.values[t], gcps, transform, width, height, resampling
            )
        
        dims = ["time", "y", "x"]
        
    else:
        raise ValueError(f"Unsupported data variable dimensionality: {data_var.ndim}")
    
    # Create attributes for reprojected variable
    attrs = data_var.attrs.copy()
    attrs["_ARRAY_DIMENSIONS"] = dims
    attrs["grid_mapping"] = "spatial_ref"
    
    return xr.DataArray(
        data=reprojected_data,
        dims=dims,
        attrs=attrs
    )


def _reproject_2d_array(
    src_array: np.ndarray,
    gcps: list[rasterio.control.GroundControlPoint],
    dst_transform: rasterio.Affine,
    dst_width: int,
    dst_height: int,
    resampling: Resampling,
) -> np.ndarray:
    """Reproject a 2D array using GCPs."""
    src_height, src_width = src_array.shape
    
    # Create destination array
    dst_array = np.zeros((dst_height, dst_width), dtype=src_array.dtype)
    
    # Create a temporary in-memory rasterio dataset with GCPs
    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=src_height,
            width=src_width,
            count=1,
            dtype=src_array.dtype,
            crs='EPSG:4326',  # GCPs are in lat/lon
        ) as src_dataset:
            # Write source data
            src_dataset.write(src_array, 1)
            
            # Set GCPs
            src_dataset.gcps = (gcps, 'EPSG:4326')
            
            # Perform reprojection
            reproject(
                source=rasterio.band(src_dataset, 1),
                destination=dst_array,
                src_transform=src_dataset.transform,
                src_crs=src_dataset.crs,
                dst_transform=dst_transform,
                dst_crs='EPSG:4326',
                resampling=resampling,
                src_nodata=None,
                dst_nodata=None,
            )
    
    return dst_array


def calculate_reprojected_bounds(
    ds_gcp: xr.Dataset,
    target_crs: str = "EPSG:4326"
) -> Tuple[float, float, float, float]:
    """
    Calculate bounds of reprojected data based on GCPs.
    
    Parameters
    ----------
    ds_gcp : xr.Dataset
        Dataset containing Ground Control Points
    target_crs : str, default "EPSG:4326"
        Target coordinate reference system
        
    Returns
    -------
    tuple
        Bounds as (left, bottom, right, top)
    """
    # Get GCP coordinates
    lons = ds_gcp["longitude"].values
    lats = ds_gcp["latitude"].values
    
    if target_crs == "EPSG:4326":
        # Direct bounds from lat/lon
        left = float(np.min(lons))
        right = float(np.max(lons))
        bottom = float(np.min(lats))
        top = float(np.max(lats))
    else:
        # Transform bounds to target CRS
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        
        # Transform corner points
        x_coords, y_coords = transformer.transform(lons.flatten(), lats.flatten())
        
        left = float(np.min(x_coords))
        right = float(np.max(x_coords))
        bottom = float(np.min(y_coords))
        top = float(np.max(y_coords))
    
    return (left, bottom, right, top)
