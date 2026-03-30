"""
Streaming multiscale pyramid creation for optimized S2 structure.
Uses lazy evaluation to minimize memory usage during dataset preparation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import xarray as xr
from dask import delayed
from dask.array import from_delayed
from zarr.codecs import BloscCodec
from zarr_cm import geo_proj
from zarr_cm import spatial as spatial_cm

from eopf_geozarr.conversion.fs_utils import sanitize_dataset_attributes
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


log = structlog.get_logger()

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
            # When stripping scale/offset, also strip _FillValue since the original
            # _FillValue is in raw integer units and meaningless for decoded float data.
            keep_keys = keep_keys - CF_SCALE_OFFSET_KEYS - {"_FillValue"}
            # Set zarr fill_value to NaN so nodata regions are correctly identified
            # as transparent by zarr-aware viewers (e.g. OpenLayers GeoZarr source).
            var_encoding["fill_value"] = float("nan")

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
            group.store,
            engine="zarr",
            chunks={},
            decode_coords="all",
            consolidated=False,
            group=f"{group.path}/{path}",
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

    # Sanitize NaN values in dataset attributes before writing
    dataset = sanitize_dataset_attributes(dataset)

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

    # Add zarr convention declarations
    conventions = [
        spatial_cm.CMO,
        geo_proj.CMO,
    ]
    dataset.attrs["zarr_conventions"] = conventions


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
