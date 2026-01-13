"""
Streaming multiscale pyramid creation for optimized S2 structure.
Uses lazy evaluation to minimize memory usage during dataset preparation.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import structlog
import xarray as xr
from dask import delayed
from dask.array import from_delayed
from pydantic.experimental.missing_sentinel import MISSING
from zarr.codecs import BloscCodec

from eopf_geozarr.conversion.geozarr import (
    _create_tile_matrix_limits,
    create_native_crs_tile_matrix_set,
)
from eopf_geozarr.data_api.geozarr.geoproj import ProjConventionMetadata
from eopf_geozarr.data_api.geozarr.multiscales import tms, zcm
from eopf_geozarr.data_api.geozarr.multiscales.geozarr import (
    MultiscaleGroupAttrs,
    MultiscaleMeta,
)
from eopf_geozarr.data_api.geozarr.spatial import SpatialConventionMetadata
from eopf_geozarr.data_api.geozarr.types import (
    CF_SCALE_OFFSET_KEYS,
    XARRAY_ENCODING_KEYS,
    XarrayDataArrayEncoding,
)
from eopf_geozarr.s2_optimization.common import DISTRIBUTED_AVAILABLE

from .s2_resampling import determine_variable_type, downsample_variable

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    import zarr
    from pyproj import CRS

    from eopf_geozarr.types import OverviewLevelJSON

log = structlog.get_logger()

MultiscalesFlavor = Literal["ogc_tms", "experimental_multiscales_convention"]

pyramid_levels = {
    0: 10,  # Level 0: 10m (native for b02,b03,b04,b08)
    1: 20,  # Level 1: 20m (native for b05,b06,b07,b11,b12,b8a + all quality)
    2: 60,  # Level 2: 60m (native for b01,b09,b10)
    3: 120,  # Level 3: 120m (2x downsampling from 60m)
    4: 360,  # Level 4: 360m (3x downsampling from 120m)
    5: 720,  # Level 5: 720m (2x downsampling from 360m)
}


def get_grid_spacing(ds: xr.DataArray, coords: tuple[Hashable, ...]) -> tuple[float | int, ...]:
    """
    Get the grid spacing of a regularly-gridded DataArray along the specified coordinates.
    """
    return tuple(np.abs(ds.coords[coord][0].data - ds.coords[coord][1].data) for coord in coords)


def create_multiscale_levels(group: zarr.Group, path: str) -> None:
    """
    Add additional multiscale levels to an existing Zarr group
    """
    ds_levels = (2, 6, 12, 36, 72)
    # Construct the full path to the group containing resolution levels
    full_path = f"{group.path}/{path}" if group.path else path

    for cur_factor, next_factor in pairwise((1, *ds_levels)):
        cur_group_name = f"r{10 * cur_factor}m"
        next_group_name = f"r{10 * next_factor}m"
        # Open the current resolution level as a dataset
        cur_group_path = f"{full_path}/{cur_group_name}"
        cur_ds = xr.open_dataset(group.store, group=cur_group_path, engine="zarr")

        scale = next_factor // cur_factor
        to_downsample: dict[str, xr.DataArray] = {}
        to_copy: dict[str, xr.DataArray] = {}

        # Iterate over all variables (data_vars and coords)
        for var_name in list(cur_ds.data_vars) + list(cur_ds.coords):
            # Convert to string for type safety
            var_name_str = str(var_name)
            var = cur_ds[var_name]
            next_level_path = f"{path}/{next_group_name}"

            # Skip if already exists in next level
            if f"{next_level_path}/{var_name_str}" in group:
                continue

            # Decide whether to downsample or copy
            # Spatial data variables (2D+) should be downsampled
            if var_name in cur_ds.data_vars and var.ndim >= 2:
                to_downsample[var_name_str] = var
            # Everything else (coordinates, 0D/1D variables like spatial_ref) should be copied
            else:
                to_copy[var_name_str] = var

        log.info("downsampling %s into %s", tuple(sorted(to_downsample.keys())), next_group_name)
        if to_copy:
            log.info("copying %s into %s", tuple(sorted(to_copy.keys())), next_group_name)

        # Only process if there's something to downsample or copy
        if to_downsample or to_copy:
            # Create downsampled dataset with data variables to downsample
            downsampled_ds = create_downsampled_resolution_group(
                xr.Dataset(data_vars=to_downsample), factor=scale
            )

            # Add variables to copy (like spatial_ref)
            for var_name, var in to_copy.items():
                if var_name not in downsampled_ds.coords:
                    downsampled_ds = downsampled_ds.assign_coords({var_name: var})
            next_group_path = f"{full_path}/{next_group_name}"
            downsampled_ds.to_zarr(
                group.store, group=next_group_path, consolidated=False, mode="a", compute=True
            )


def update_encoding(
    array: xr.DataArray, encoding: XarrayDataArrayEncoding
) -> XarrayDataArrayEncoding:
    """
    Update an xarray encoding of a variable against a dataarray. Used when ensuring that a downsampled
    dataarray has an encoding consistent with both the source array and also its newly reduced shape.
    Shape-related quantities like chunks and shards need to be adjusted to match the shape of the array.
    All other elements of the encoding are preserved
    """
    new_encoding: XarrayDataArrayEncoding = {**encoding}
    if "chunks" in new_encoding:
        new_encoding["chunks"] = tuple(
            min(s, c) for s, c in zip(array.shape, new_encoding["chunks"], strict=True)
        )
        if new_encoding.get("shards") is not None:
            # new shards are the largest multiple of the new chunks that fits inside the new shape
            new_shards = tuple(
                (shape // chunk) * chunk
                for chunk, shape in zip(new_encoding["chunks"], array.shape, strict=True)
            )
            # calculate the number of inner chunks within the shard
            num_subchunks = np.prod(
                tuple(
                    shard // chunk
                    for shard, chunk in zip(new_shards, new_encoding["chunks"], strict=True)
                )
            )
            # If we would generate shards with a single chunk, there is no longer value in sharding, and we should
            # use regular chunking
            if num_subchunks == 1:
                new_encoding["shards"] = None
            else:
                new_encoding["shards"] = new_shards
    if "preferred_chunks" in new_encoding:
        new_encoding["preferred_chunks"] = {
            k: min(array.shape[array.dims.index(k)], v)
            for k, v in new_encoding["preferred_chunks"].items()
        }
    return new_encoding


def auto_chunks(shape: tuple[int, ...], target_chunk_size: int) -> tuple[int, ...]:
    """
    Compute a chunk size from a shape and a target chunk size. This logic is application-specific.
    For 0D data, the empty tuple is returned.
    For 1D data, the minimum of the length of the data and the target chunk size is returned.
    For 2D, 3D, etc, a chunk edge length is computed based on each of the last two dimensions, and
    all other dimensions have chunks set to 1.
    """
    ndim = len(shape)

    if ndim == 0:
        return ()

    if ndim == 1:
        return (min(target_chunk_size, shape[0]),)

    height, width = shape[-2:]

    spatial_chunk_aligned = min(
        target_chunk_size,
        calculate_aligned_chunk_size(width, target_chunk_size),
        calculate_aligned_chunk_size(height, target_chunk_size),
    )
    return ((1,) * (ndim - 2)) + (spatial_chunk_aligned, spatial_chunk_aligned)


def create_measurements_encoding(
    dataset: xr.Dataset,
    *,
    spatial_chunk: int,
    enable_sharding: bool = True,
    keep_scale_offset: bool = True,
) -> dict[str, XarrayDataArrayEncoding]:
    """
    Create optimized encoding for a pyramid level with advanced chunking and sharding.
    """
    encoding: dict[str, XarrayDataArrayEncoding] = {}

    for var_name, var_data in dataset.data_vars.items():
        # start with the original encoding
        var_encoding: XarrayDataArrayEncoding = {}

        chunks: tuple[int, ...] = ()
        if var_data.ndim >= 2:
            height, width = var_data.shape[-2:]

            # Use advanced aligned chunk calculation
            spatial_chunk_aligned = min(
                spatial_chunk,
                calculate_aligned_chunk_size(width, spatial_chunk),
                calculate_aligned_chunk_size(height, spatial_chunk),
            )

            if var_data.ndim == 3:
                # Single file per variable per time: chunk time dimension to 1
                chunks = (1, spatial_chunk_aligned, spatial_chunk_aligned)
            else:
                chunks = (spatial_chunk_aligned, spatial_chunk_aligned)
        else:
            chunks = (min(spatial_chunk, var_data.shape[0]),)

        # Configure encoding - use proper compressor following geozarr.py pattern
        from zarr.codecs import BloscCodec

        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0)

        var_encoding["chunks"] = chunks
        var_encoding["compressors"] = (compressor,)

        # Add advanced sharding if enabled - shards match x/y dimensions exactly
        if enable_sharding and var_data.ndim >= 2:
            shard_dims = calculate_simple_shard_dimensions(var_data.shape, chunks)
            var_encoding["shards"] = shard_dims
        else:
            var_encoding["shards"] = None

        # Forward-propagate the existing encoding, minus keys that should be omitted
        keep_keys = XARRAY_ENCODING_KEYS - {"compressors", "shards", "chunks"}

        if not keep_scale_offset:
            keep_keys = keep_keys - CF_SCALE_OFFSET_KEYS

        for key in keep_keys:
            if key in var_data.encoding:
                var_encoding[key] = var_data.encoding[key]  # type: ignore[literal-required]

        if len(set(var_data.encoding.keys()) - XARRAY_ENCODING_KEYS) > 0:
            log.warning(
                "Unknown encoding keys in %s: %s",
                var_name,
                set(var_data.encoding.keys()) - XARRAY_ENCODING_KEYS,
            )
        encoding[var_name] = var_encoding

    # Add coordinate encoding
    for coord_name in dataset.coords:
        encoding[coord_name] = {"compressors": []}  # type: ignore[typeddict-item]

    return encoding


def calculate_aligned_chunk_size(dimension_size: int, target_chunk: int) -> int:
    """
    Calculate aligned chunk size following geozarr.py logic.

    This ensures good chunk alignment without complex calculations.
    """
    if target_chunk >= dimension_size:
        return dimension_size

    # Find the largest divisor of dimension_size that's close to target_chunk
    best_chunk = target_chunk
    for chunk_candidate in range(target_chunk, max(target_chunk // 2, 1), -1):
        if dimension_size % chunk_candidate == 0:
            best_chunk = chunk_candidate
            break

    return best_chunk


def calculate_simple_shard_dimensions(
    data_shape: tuple[int, ...], chunks: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Calculate shard dimensions that are compatible with chunk dimensions.

    Shard dimensions must be evenly divisible by chunk dimensions for Zarr v3.
    When possible, shards should match x/y dimensions exactly as required.
    """
    shard_dims = []

    for i, (dim_size, chunk_size) in enumerate(zip(data_shape, chunks, strict=False)):
        if i == 0 and len(data_shape) == 3:
            # First dimension in 3D data (time) - use single time slice per shard
            shard_dims.append(1)
        else:
            # For x/y dimensions, try to use full dimension size
            # But ensure it's divisible by chunk size
            if dim_size % chunk_size == 0:
                # Perfect: full dimension is divisible by chunk
                shard_dims.append(dim_size)
            else:
                # Find the largest multiple of chunk_size that fits
                num_chunks = dim_size // chunk_size
                if num_chunks > 0:
                    shard_size = num_chunks * chunk_size
                    shard_dims.append(shard_size)
                else:
                    # Fallback: use chunk size itself
                    shard_dims.append(chunk_size)

    return tuple(shard_dims)


def create_multiscales_metadata(
    out_group: zarr.Group,
    multiscales_flavor: set[MultiscalesFlavor] | None = None,
) -> None:
    """Add GeoZarr-compliant multiscales metadata to parent group."""
    # Sort by resolution (finest to coarsest)
    if multiscales_flavor is None:
        multiscales_flavor = {"ogc_tms", "experimental_multiscales_convention"}
    res_order = {
        "r10m": 10,
        "r20m": 20,
        "r60m": 60,
        "r120m": 120,
        "r360m": 360,
        "r720m": 720,
    }
    res_groups = tuple(out_group[k] for k in res_order)
    first_dataset = xr.open_zarr(out_group.store, group=next(iter(res_groups)).path)

    # Get CRS and bounds
    native_crs = first_dataset.rio.crs if hasattr(first_dataset, "rio") else None
    if native_crs is None:
        raise ValueError("Cannot determine native CRS for multiscales metadata")

    try:
        native_bounds = first_dataset.rio.bounds()
    except (AttributeError, TypeError):
        # Try alternative method or construct from coordinates
        x_coords = first_dataset.x.values
        y_coords = first_dataset.y.values
        native_bounds = (x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max())

    # Create overview_levels structure following the multiscales v1.0 specification
    overview_levels: list[OverviewLevelJSON] = []
    for res_group in res_groups:
        res_name = res_group.basename
        # Use resolution order for consistent scale calculations
        res_meters = res_order[res_name]

        dataset = xr.open_zarr(res_group.store, group=res_group.path)

        # Get first data variable to extract dimensions
        first_var = next(iter(dataset.data_vars.values()))
        height, width = first_var.shape[-2:]

        # Calculate spatial transform (affine transformation)
        transform = None
        if hasattr(dataset, "rio") and hasattr(dataset.rio, "transform"):
            try:
                # Try to get transform as property first
                rio_transform = dataset.rio.transform
                if callable(rio_transform):
                    rio_transform = rio_transform()
                transform = tuple(rio_transform)[:6]  # Get 6 coefficients
                log.info("Got transform from rio accessor", transform=transform, level=res_name)
            except (AttributeError, TypeError) as e:
                log.warning(
                    "Could not get transform from rio accessor", error=str(e), level=res_name
                )

        if transform is None or all(t == 0 for t in transform):
            # Fallback: construct from grid spacing and bounds
            if "x" in dataset.coords and "y" in dataset.coords:
                # Use coordinate arrays to calculate spacing
                x_coords = dataset.coords["x"].values
                y_coords = dataset.coords["y"].values

                if len(x_coords) > 1 and len(y_coords) > 1:
                    pixel_size_x = float(np.abs(x_coords[1] - x_coords[0]))
                    pixel_size_y = float(np.abs(y_coords[1] - y_coords[0]))
                    x_min = float(x_coords.min())
                    y_max = float(y_coords.max())
                    transform = (pixel_size_x, 0.0, x_min, 0.0, -pixel_size_y, y_max)
                    log.info(
                        "Calculated transform from coordinates",
                        transform=transform,
                        pixel_size_x=pixel_size_x,
                        pixel_size_y=pixel_size_y,
                        level=res_name,
                    )
                else:
                    log.warning(
                        "Insufficient coordinate points for transform calculation",
                        x_len=len(x_coords),
                        y_len=len(y_coords),
                        level=res_name,
                    )
            else:
                log.warning(
                    "Missing x/y coordinates for transform calculation",
                    coords=list(dataset.coords.keys()),
                    level=res_name,
                )

        # Calculate zoom level (higher resolution = higher zoom)
        tile_width = 256
        zoom_for_width = max(0, int(np.ceil(np.log2(width / tile_width))))
        zoom_for_height = max(0, int(np.ceil(np.log2(height / tile_width))))
        zoom = max(zoom_for_width, zoom_for_height)

        # Calculate relative scale and translation vs first resolution
        finest_res_meters = res_order[res_groups[0].basename]
        relative_scale = res_meters / finest_res_meters
        relative_translation = (res_meters - finest_res_meters) / 2

        # Get chunks in the correct format
        var_chunks = dataset.data_vars[first_var.name].chunks
        if var_chunks is not None:
            chunks = tuple(tuple(int(c) for c in chunk_dim) for chunk_dim in var_chunks)
        else:
            chunks = None
            log.warning(
                "Could not determine chunking information for overview level; 'chunks' will be set to None",
                level=res_name,
                variable=str(first_var.name),
            )

        layout_entry: OverviewLevelJSON = {
            "level": res_name,  # Use string-based level name
            "zoom": zoom,
            "width": width,
            "height": height,
            "translation_relative": relative_translation,
            "scale_absolute": res_meters,
            "scale_relative": relative_scale,
            "spatial_transform": None,
            "chunks": chunks,
            "spatial_shape": (height, width),
        }

        # Only add spatial_transform if we have valid transform data
        if transform is not None and not all(t == 0 for t in transform):
            layout_entry["spatial_transform"] = transform

        overview_levels.append(layout_entry)

    multiscales: dict[str, Any] = {"multiscales": {}}
    layout: list[zcm.ScaleLevel] | MISSING = MISSING
    tile_matrix_set: tms.TileMatrixSet | MISSING = MISSING
    tile_matrix_limits: dict[str, tms.TileMatrixLimit] | MISSING = MISSING

    if "ogc_tms" in multiscales_flavor:
        # Create tile matrix set using geozarr function
        tile_matrix_set = create_native_crs_tile_matrix_set(
            native_crs,
            native_bounds,
            overview_levels,
            group_prefix=None,
        )

        # Create tile matrix limits
        tile_matrix_limits = _create_tile_matrix_limits(
            overview_levels,
            tile_width=256,
        )
        multiscales["multiscales"].update(
            {
                "tile_matrix_set": tile_matrix_set,
                "resampling_method": "average",
                "tile_matrix_limits": tile_matrix_limits,
            }
        )
    if "experimental_multiscales_convention" in multiscales_flavor:
        layout = []

        # Define the correct derivation chain
        derivation_chain = {
            "r10m": None,  # base resolution
            "r20m": "r10m",
            "r60m": "r10m",
            "r120m": "r60m",
            "r360m": "r120m",
            "r720m": "r360m",
        }

        for i, overview_level in enumerate(overview_levels):
            # Create scale level with required fields
            asset = str(overview_level["level"])

            # Build complete dict for ScaleLevel initialization
            scale_level_data: dict[str, Any] = {"asset": asset}

            if i > 0:  # Not the first (base) resolution
                derived_from = derivation_chain.get(asset, str(res_groups[0].basename))
                multiscale_transform = zcm.Transform(
                    scale=(overview_level["scale_relative"],) * 2,
                    translation=(overview_level["translation_relative"],) * 2,
                )
                scale_level_data["derived_from"] = derived_from
                scale_level_data["transform"] = multiscale_transform

            # Add spatial properties
            scale_level_data["spatial:shape"] = overview_level["spatial_shape"]
            if "spatial_transform" in overview_level:
                spatial_transform = overview_level["spatial_transform"]
                # Only add spatial_transform if we have valid transform data (not all zeros)
                if spatial_transform is not None and not all(t == 0 for t in spatial_transform):
                    scale_level_data["spatial:transform"] = spatial_transform

            scale_level = zcm.ScaleLevel(**scale_level_data)
            layout.append(scale_level)

    # Create convention metadata for all three conventions
    multiscale_attrs = MultiscaleGroupAttrs(
        zarr_conventions=(
            zcm.MultiscaleConventionMetadata(),
            SpatialConventionMetadata(),
            ProjConventionMetadata(),
        ),
        multiscales=MultiscaleMeta(
            layout=layout,
            resampling_method="average",
            tile_matrix_set=tile_matrix_set,
            tile_matrix_limits=tile_matrix_limits,
        ),
    )

    # Add multiscale attributes
    out_group.attrs.update(multiscale_attrs.model_dump())
    # Add spatial and proj attributes at group level following specifications
    if native_crs and native_bounds:
        # Add spatial convention attributes
        spatial_attrs: dict[str, object] = {
            "spatial:dimensions": ("y", "x"),
            "spatial:bbox": tuple(native_bounds),
            "spatial:registration": "pixel",
        }

        # Add proj convention attributes
        if hasattr(native_crs, "to_epsg") and native_crs.to_epsg():
            spatial_attrs["proj:code"] = f"EPSG:{native_crs.to_epsg()}"
        elif hasattr(native_crs, "to_wkt"):
            spatial_attrs["proj:wkt2"] = native_crs.to_wkt()

        out_group.attrs.update(spatial_attrs)


def create_original_encoding(dataset: xr.Dataset) -> dict[str, XarrayDataArrayEncoding]:
    """Write a group preserving its original chunking and encoding."""

    # Simple encoding that preserves original structure
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0)
    encoding = {}

    for var_name in dataset.data_vars:
        # start with the original encoding
        var_data = dataset.data_vars[var_name]
        var_encoding: XarrayDataArrayEncoding = {}
        var_encoding["compressors"] = (compressor,)
        for key in XARRAY_ENCODING_KEYS - {"compressors"}:
            if key in var_data.encoding:
                var_encoding[key] = var_data.encoding[key]  # type: ignore[literal-required]
        if len(set(var_data.encoding.keys()) - XARRAY_ENCODING_KEYS) > 0:
            log.warning(
                "Unknown encoding keys in %s: %s",
                var_name,
                set(var_data.encoding.keys()) - XARRAY_ENCODING_KEYS,
            )
        encoding[var_name] = var_encoding

    for coord_name in dataset.coords:
        encoding[coord_name] = {"compressors": None}

    return encoding


def create_downsampled_resolution_group(source_dataset: xr.Dataset, factor: int) -> xr.Dataset:
    """Create a downsampled version of a dataset by given factor."""

    # Downsample all variables
    lazy_vars: dict[str, xr.DataArray] = {}
    for data_var_name, data_var in source_dataset.data_vars.items():
        if data_var.ndim >= 2:
            var_typ = determine_variable_type(data_var_name, data_var)
            if var_typ == "quality_mask":
                lazy_downsampled = data_var.coarsen(
                    {"x": factor, "y": factor}, boundary="trim"
                ).max()
            elif var_typ == "reflectance":
                lazy_downsampled = data_var.coarsen(
                    {"x": factor, "y": factor}, boundary="trim"
                ).mean()
            elif var_typ == "classification":
                lazy_downsampled = data_var.coarsen(
                    {"x": factor, "y": factor}, boundary="trim"
                ).reduce(subsample_2)
            elif var_typ == "probability":
                lazy_downsampled = data_var.coarsen(
                    {"x": factor, "y": factor}, boundary="trim"
                ).mean()
            else:
                raise ValueError(f"Unknown variable type {var_typ}")

            lazy_downsampled = lazy_downsampled.astype(data_var.dtype)
            lazy_downsampled.encoding = update_encoding(lazy_downsampled, data_var.encoding)
            for coord_name, coord in lazy_downsampled.coords.items():
                lazy_downsampled.coords[coord_name].encoding = update_encoding(
                    coord, data_var.coords[coord_name].encoding
                )
            lazy_vars[data_var_name] = lazy_downsampled

    # Create dataset with lazy variables and coordinates
    return xr.Dataset(lazy_vars, attrs=source_dataset.attrs)


def subsample_2(a: xr.DataArray, axis: tuple[int, ...] | None = None) -> xr.DataArray:
    if axis is None:
        return a[((slice(None, None, 2),) * a.ndim)]
    indexer = [slice(None, None, 2) if i in axis else slice(None) for i in range(a.ndim)]
    return a[tuple(indexer)]


def create_downsampled_coordinates(
    level_2_dataset: xr.Dataset,
    target_height: int,
    target_width: int,
    downsample_factor: int,
) -> dict[str, Any]:
    """Create downsampled coordinates for higher pyramid levels."""

    # Get original coordinates from level 2
    if "x" not in level_2_dataset.coords or "y" not in level_2_dataset.coords:
        return {}

    x_coords_orig = level_2_dataset.coords["x"].values
    y_coords_orig = level_2_dataset.coords["y"].values

    # Calculate downsampled coordinates by taking every nth point
    # where n is the downsample_factor
    x_coords_downsampled = x_coords_orig[::downsample_factor][:target_width]
    y_coords_downsampled = y_coords_orig[::downsample_factor][:target_height]

    # Create coordinate dictionary with proper attributes
    coords = {}

    # Copy x coordinate with attributes
    x_attrs = level_2_dataset.coords["x"].attrs.copy()
    coords["x"] = (["x"], x_coords_downsampled, x_attrs)

    # Copy y coordinate with attributes
    y_attrs = level_2_dataset.coords["y"].attrs.copy()
    coords["y"] = (["y"], y_coords_downsampled, y_attrs)

    # Copy any other coordinates that might exist
    coords.update(
        {
            coord_name: coord_data
            for coord_name, coord_data in level_2_dataset.coords.items()
            if coord_name not in ["x", "y"]
        }
    )

    return coords


def create_lazy_downsample_operation_from_existing(
    source_data: xr.DataArray, target_height: int, target_width: int
) -> xr.DataArray:
    """Create lazy downsampling operation from existing data."""

    @delayed  # type: ignore[misc]
    def downsample_operation() -> Any:
        var_type = determine_variable_type(source_data.name, source_data)
        return downsample_variable(source_data, target_height, target_width, var_type)

    # Create delayed operation
    lazy_result = downsample_operation()

    # Estimate output shape and chunks
    output_shape: tuple[int, ...]
    chunks: tuple[int, ...]
    if source_data.ndim == 3:
        output_shape = (source_data.shape[0], target_height, target_width)
        chunks = (1, min(256, target_height), min(256, target_width))
    else:
        output_shape = (target_height, target_width)
        chunks = (min(256, target_height), min(256, target_width))

    # Create Dask array from delayed operation
    dask_array = from_delayed(lazy_result, shape=output_shape, dtype=source_data.dtype).rechunk(
        chunks
    )

    # Return as xarray DataArray with lazy data - no coords to avoid alignment issues
    # Coordinates will be set when the lazy operation is computed
    return xr.DataArray(
        dask_array,
        dims=source_data.dims,
        attrs=source_data.attrs.copy(),
        name=source_data.name,
    )


def stream_write_dataset(
    dataset: xr.Dataset,
    *,
    path: str,
    group: zarr.Group,
    encoding: dict[str, XarrayDataArrayEncoding],
    enable_sharding: bool,
    crs: CRS | None = None,
) -> xr.Dataset:
    """
    Stream write a lazy dataset with advanced chunking and sharding.

    This is where the magic happens: all the lazy downsampling operations
    are executed as the data is streamed to storage with optimal performance.

    Args:
        dataset: Dataset to write
        dataset_path: Output path for dataset
        encoding: Encoding dictionary for variables
        enable_sharding: Enable Zarr v3 sharding
        crs: Coordinate Reference System for geographic metadata

    Returns:
        Written dataset
    """
    # Check if level already exists
    if path in group:
        log.info(
            "Level path {} already exists. Skipping write.",
            dataset_path=path,
        )
        return xr.open_dataset(
            dataset_path,
            engine="zarr",
            chunks={},
            decode_coords="all",
            consolidated=False,
        )

    log.info("Streaming computation and write to {}", dataset_path=path)
    log.info("Variables", variables=list(dataset.data_vars.keys()))

    # Rechunk dataset to align with encoding
    dataset = rechunk_dataset_for_encoding(dataset, encoding)

    # Add the geo metadata before writing for
    # - /measurements/ groups
    # - /quality/ groups
    if "/measurements/" in path or "/quality/" in path:
        write_geo_metadata(dataset, crs=crs)

    # Write with streaming computation and progress tracking
    # The to_zarr operation will trigger all lazy computations
    write_job = dataset.to_zarr(
        group.store,
        mode="w",
        consolidated=False,
        zarr_format=3,
        encoding=encoding,
        group=path,
        compute=False,  # Create job first for progress tracking
    )

    if DISTRIBUTED_AVAILABLE:
        try:
            import distributed

            # Try to get current client for better status monitoring
            try:
                client = distributed.Client.current()
                # Use client.compute to get a proper Future with status
                future = client.compute(write_job)
                log.info("Using distributed client for write job monitoring")

                try:
                    distributed.progress(future, notebook=False)
                except Exception as progress_error:
                    log.warning("Could not display progress bar: {}", e=progress_error)

                # Get result and raise if computation failed
                future.result()
            except ValueError:
                # No current client, fall back to regular distributed.progress
                log.info("No distributed client available, using regular progress")
                distributed.progress(write_job, notebook=False)
                write_job.compute()

        except Exception as e:
            log.warning("Could not use distributed features: {}", e=e)
            write_job.compute()
    else:
        log.info("Writing zarr file...")
        write_job.compute()

    log.info("✅ Streaming write complete for dataset {}", dataset_path=path)
    return dataset


def write_geo_metadata(
    dataset: xr.Dataset,
    crs: CRS,
    *,
    grid_mapping_var_name: str = "spatial_ref",
) -> None:
    """
    Write geographic metadata to the dataset.

    Args:
        dataset: Dataset to write metadata to
        grid_mapping_var_name: Name for grid mapping variable
        crs: Coordinate Reference System to use (if None, attempts to detect from dataset)
    """
    dataset.rio.write_crs(crs, grid_mapping_name=grid_mapping_var_name, inplace=True)
    dataset.rio.write_grid_mapping(grid_mapping_var_name, inplace=True)

    if crs is not None:
        # Write CRS using rioxarray
        # NOTE: for now rioxarray only supports writing grid mapping using CF conventions
        dataset.rio.write_crs(crs, grid_mapping_name=grid_mapping_var_name, inplace=True)
        dataset.rio.write_grid_mapping(grid_mapping_var_name, inplace=True)
        dataset.attrs["grid_mapping"] = grid_mapping_var_name

        for var in dataset.data_vars.values():
            var.rio.write_grid_mapping(grid_mapping_var_name, inplace=True)
            var.attrs["grid_mapping"] = grid_mapping_var_name

        # Also add proj: and spatial: zarr conventions at dataset level
        # TODO : Remove once rioxarray supports writing these conventions directly
        # https://github.com/corteva/rioxarray/pull/883

        # Add spatial convention attributes
        dataset.attrs["spatial:dimensions"] = ["y", "x"]  # Required field
        dataset.attrs["spatial:registration"] = "pixel"  # Default registration type

        # Calculate and add spatial bbox if coordinates are available
        if "x" in dataset.coords and "y" in dataset.coords:
            x_coords = dataset.coords["x"].values
            y_coords = dataset.coords["y"].values
            x_min, x_max = float(x_coords.min()), float(x_coords.max())
            y_min, y_max = float(y_coords.min()), float(y_coords.max())
            dataset.attrs["spatial:bbox"] = [x_min, y_min, x_max, y_max]

            # Calculate spatial transform (affine transformation)
            spatial_transform = None
            if hasattr(dataset, "rio") and hasattr(dataset.rio, "transform"):
                try:
                    rio_transform = dataset.rio.transform
                    if callable(rio_transform):
                        rio_transform = rio_transform()
                    spatial_transform = list(rio_transform)[:6]
                except (AttributeError, TypeError):
                    # Fallback: construct from coordinate spacing
                    pixel_size_x = float(get_grid_spacing(dataset, ("x",))[0])
                    pixel_size_y = float(get_grid_spacing(dataset, ("y",))[0])
                    spatial_transform = [
                        pixel_size_x,
                        0.0,
                        x_min,
                        0.0,
                        -pixel_size_y,
                        y_max,
                    ]

            # Only add spatial:transform if we have valid transform data (not all zeros)
            if spatial_transform is not None and not all(t == 0 for t in spatial_transform):
                dataset.attrs["spatial:transform"] = spatial_transform

            # Add spatial shape if data variables exist
            if dataset.data_vars:
                first_var = next(iter(dataset.data_vars.values()))
                if first_var.ndim >= 2:
                    height, width = first_var.shape[-2:]
                    dataset.attrs["spatial:shape"] = [height, width]

        # Add proj convention attributes
        if hasattr(crs, "to_epsg") and crs.to_epsg():
            dataset.attrs["proj:code"] = f"EPSG:{crs.to_epsg()}"
        elif hasattr(crs, "to_wkt"):
            dataset.attrs["proj:wkt2"] = crs.to_wkt()


def rechunk_dataset_for_encoding(
    dataset: xr.Dataset, encoding: dict[str, XarrayDataArrayEncoding]
) -> xr.Dataset:
    """
    Rechunk dataset variables to align with sharding dimensions when sharding is enabled.

    When using Zarr v3 sharding, Dask chunks must align with shard dimensions to avoid
    checksum validation errors.
    """
    rechunked_vars = {}

    for var_name, var_data in dataset.data_vars.items():
        if var_name in encoding:
            var_encoding = encoding[var_name]

            # If sharding is enabled, rechunk based on shard dimensions
            if "shards" in var_encoding and var_encoding["shards"] is not None:
                target_chunks = var_encoding["shards"]  # Use shard dimensions for rechunking
            elif "chunks" in var_encoding:
                target_chunks = var_encoding["chunks"]  # Fallback to chunk dimensions
            else:
                # No specific chunking needed, use original variable
                rechunked_vars[var_name] = var_data
                continue

            # Create chunk dict using the actual dimensions of the variable
            var_dims = var_data.dims
            chunk_dict = {}
            for i, dim in enumerate(var_dims):
                if i < len(target_chunks):
                    chunk_dict[dim] = target_chunks[i]

            # Rechunk the variable to match the target dimensions
            rechunked_vars[var_name] = var_data.chunk(chunk_dict)
        else:
            # No specific chunking needed, use original variable
            rechunked_vars[var_name] = var_data

    # Create new dataset with rechunked variables, preserving coordinates
    return xr.Dataset(rechunked_vars, coords=dataset.coords, attrs=dataset.attrs)


def extract_scale_offset_encoding(
    attrs: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """
    extract the scale / offset encoding from _eopf_attrs
    """
    encoding = {}
    encoding["add_offset"] = attrs["_eopf_attrs"]["add_offset"]
    encoding["scale_factor"] = attrs["_eopf_attrs"]["scale_factor"]
    encoding["dtype"] = attrs["_eopf_attrs"]["dtype"]
    return encoding
