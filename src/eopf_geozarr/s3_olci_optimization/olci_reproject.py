"""Reprojection of OLCI swath measurements to a regular grid.

OLCI L1 EFR is a curvilinear swath geolocated by dense per-pixel 2-D
latitude/longitude arrays.  Generic GeoZarr readers (titiler, rioxarray)
require a regular grid with an affine transform and a declared CRS, so the
converter warps the swath once at native resolution using rasterio's
geolocation-array support (``src_geoloc_array``, rasterio >= 1.4).
"""

from __future__ import annotations

import numpy as np
import rioxarray  # noqa: F401  # enables the .rio accessor
import structlog
import xarray as xr
from pyproj import CRS as ProjCRS
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject

log = structlog.get_logger()

#: CRS of the OLCI geolocation arrays (per-pixel lat/lon in degrees).
GEOLOC_CRS = "EPSG:4326"

#: Dimension names of the reprojected regular grid.
GRID_DIMS: tuple[str, str] = ("y", "x")

_SWATH_DIMS = ("rows", "columns")


def _nodata_for(var: xr.DataArray) -> float:
    """Warp nodata for *var*: its ``_FillValue`` if present, else a dtype default.

    Integer variables without a fill value get the dtype maximum (matching the
    OLCI convention of 65535 for uint16 radiances); floats get NaN.
    """
    fill = var.attrs.get("_FillValue")
    if fill is None:
        fill = var.encoding.get("_FillValue")
    if fill is not None:
        return float(fill)
    if np.issubdtype(var.dtype, np.integer):
        return float(np.iinfo(var.dtype).max)
    return float("nan")


def _unpacked_degrees(var: xr.DataArray) -> np.ndarray:
    """Geolocation values as float64 degrees, with fill pixels as NaN.

    The pipeline contract is ``mask_and_scale=False`` (raw values), and real
    OLCI stores latitude/longitude as int32 microdegrees with
    ``scale_factor = 1e-06`` — feeding raw values to the warp as if they were
    degrees would georeference the output a factor of 10^6 off.  Mirrors the
    unpacking the native pipeline's geodesic centroid path performs.
    """
    vals = np.asarray(var.values, dtype="float64")
    fill = var.attrs.get("_FillValue")
    if fill is None:
        fill = var.encoding.get("_FillValue")
    if fill is not None and not np.isnan(float(fill)):
        vals = np.where(vals == float(fill), np.nan, vals)
    scale = var.attrs.get("scale_factor")
    if scale is None:
        scale = var.encoding.get("scale_factor", 1.0)
    offset = var.attrs.get("add_offset")
    if offset is None:
        offset = var.encoding.get("add_offset", 0.0)
    return vals * float(scale) + float(offset)


def _grid_coord_attrs(target_crs: str) -> tuple[dict[str, str], dict[str, str]]:
    """CF attrs for the 1-D (y, x) dimension coordinates in *target_crs*."""
    if ProjCRS.from_user_input(target_crs).is_geographic:
        y_attrs = {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"}
        x_attrs = {"standard_name": "longitude", "units": "degrees_east", "axis": "X"}
    else:
        y_attrs = {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"}
        x_attrs = {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"}
    return y_attrs, x_attrs


def reproject_olci(
    ds: xr.Dataset,
    *,
    target_crs: str = "EPSG:4326",
    resampling: Resampling = Resampling.bilinear,
) -> xr.Dataset:
    """Warp an OLCI swath dataset onto a regular *target_crs* grid.

    Every numeric variable spanning exactly ``(rows, columns)`` — radiance
    bands and the 2-D ``altitude`` coordinate alike — is warped onto a common
    grid sized by :func:`rasterio.warp.calculate_default_transform` from the
    dense per-pixel geolocation (~native resolution preserved).  The
    ``latitude``/``longitude`` geolocation arrays are consumed by the warp and
    replaced by 1-D ``y``/``x`` dimension coordinates.  Variables that carry a
    swath dim but cannot live on the grid (e.g. per-scan-line ``time_stamp``)
    are dropped; variables without swath dims pass through unchanged.

    Off-swath cells are set to each warped variable's ``_FillValue`` (dtype
    max for integer variables without one, NaN for float variables without
    one), and ``_FillValue`` is recorded in the output attrs of every warped
    variable — including NaN for float variables — so downstream fill-aware
    averaging keeps working.  Passthrough variables that carry no swath dim
    are not warped and carry no such guarantee.

    Raises
    ------
    ValueError
        If the 2-D ``latitude``/``longitude`` coordinates are missing, or if
        the geolocation spans a zero-area (degenerate) extent.
    """
    for required in ("latitude", "longitude"):
        if required not in ds.coords:
            raise ValueError(f"cannot reproject OLCI swath: missing 2-D coordinate {required!r}")
    lat = ds.coords["latitude"]
    lon = ds.coords["longitude"]
    if tuple(str(d) for d in lat.dims) != _SWATH_DIMS or lat.dims != lon.dims:
        raise ValueError("latitude/longitude must be 2-D over (rows, columns)")
    lat_vals = _unpacked_degrees(lat)
    lon_vals = _unpacked_degrees(lon)
    if np.nanmax(lat_vals) == np.nanmin(lat_vals) or np.nanmax(lon_vals) == np.nanmin(lon_vals):
        raise ValueError("degenerate geolocation extent: latitude/longitude span zero area")

    src_height, src_width = lat_vals.shape
    geoloc = np.stack([lon_vals, lat_vals])  # (2, H, W): x first, then y
    transform, width, height = calculate_default_transform(
        src_crs=CRS.from_string(GEOLOC_CRS),
        dst_crs=CRS.from_string(target_crs),
        width=src_width,
        height=src_height,
        src_geoloc_array=geoloc,
    )
    assert width is not None
    assert height is not None
    log.info(
        "Reprojecting OLCI swath",
        target_crs=target_crs,
        src_shape=(src_height, src_width),
        dst_shape=(height, width),
    )

    result_vars: dict[str, xr.DataArray] = {}
    passthrough_coords: dict[str, xr.DataArray] = {}
    all_names = [str(k) for k in ds.data_vars] + [
        str(k) for k in ds.coords if str(k) not in ("latitude", "longitude")
    ]
    for name in all_names:
        var = ds[name] if name in ds.data_vars else ds.coords[name]
        var_dims = tuple(str(d) for d in var.dims)
        # Order-insensitive swath detection: a variant product storing a band
        # transposed as (columns, rows) must still be warped, not silently
        # dropped through the swath-dim fallthrough (matches the native
        # pipeline's normalization in reduce_swath).
        if (
            len(var_dims) == 2
            and set(var_dims) == set(_SWATH_DIMS)
            and np.issubdtype(var.dtype, np.number)
        ):
            if var_dims != _SWATH_DIMS:
                log.info("Transposing band to canonical swath dim order", band=name)
                var = var.transpose(*_SWATH_DIMS)
            nodata = _nodata_for(var)
            src_nodata = (
                nodata
                if (
                    var.attrs.get("_FillValue") is not None
                    or var.encoding.get("_FillValue") is not None
                )
                else None
            )
            dest = np.full((height, width), nodata, dtype=var.dtype)
            reproject(
                source=np.ascontiguousarray(var.values),
                destination=dest,
                src_crs=CRS.from_string(GEOLOC_CRS),
                src_geoloc_array=geoloc,
                src_nodata=src_nodata,
                dst_crs=CRS.from_string(target_crs),
                dst_transform=transform,
                dst_nodata=nodata,
                resampling=resampling,
            )
            out_attrs = dict(var.attrs)
            if np.issubdtype(var.dtype, np.integer):
                out_attrs["_FillValue"] = int(nodata)
            else:
                out_attrs["_FillValue"] = float(nodata)  # NaN included
            out_attrs.pop("coordinates", None)  # swath geolocation is gone
            result_vars[name] = xr.DataArray(dest, dims=GRID_DIMS, attrs=out_attrs)
        elif any(d in var_dims for d in _SWATH_DIMS):
            log.info("Dropping swath-bound variable with no grid home", variable=name)
        elif name in ds.coords:
            passthrough_coords[name] = var
        else:
            result_vars[name] = var

    xs = transform.c + transform.a * (np.arange(width) + 0.5)
    ys = transform.f + transform.e * (np.arange(height) + 0.5)
    y_attrs, x_attrs = _grid_coord_attrs(target_crs)
    out = xr.Dataset(
        result_vars,
        coords={
            "y": ("y", ys, y_attrs),
            "x": ("x", xs, x_attrs),
            **passthrough_coords,
        },
        attrs=dict(ds.attrs),
    )
    out = out.rio.write_crs(target_crs)
    if not isinstance(out, xr.Dataset):
        raise TypeError(f"expected an xarray.Dataset after write_crs, got {type(out).__name__}")
    # rioxarray records grid_mapping in encoding; pin it in attrs so it is
    # guaranteed to reach the zarr store.
    for name in out.data_vars:
        out[name].attrs["grid_mapping"] = "spatial_ref"
    return out
