"""Utility functions for GeoZarr conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, cast, runtime_checkable

import numpy as np
import structlog
import xarray as xr
import zarr
import zarr_cm
from pyproj import CRS
from pyproj.exceptions import CRSError
from zarr_cm import GeoProjAttrs, MultiConventionAttrs, MultiscalesAttrs, SpatialAttrs
from zarr_cm import geo_proj as geo_proj_cm
from zarr_cm import spatial as spatial_cm

from eopf_geozarr.conversion import constants, encoding_utils, fs_utils
from eopf_geozarr.data_api.geozarr.types import (
    XARRAY_ENCODING_KEYS,
    XarrayDataArrayEncoding,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence


from importlib.util import find_spec

DISTRIBUTED_AVAILABLE = find_spec("distributed") is not None

# Sentinel: distinguish "no explicit fill_value" from a legitimate `None`.
UNSET: Any = object()

# Dimension names that represent a "band-like" axis (polarization) to allow a per-"band" sharding if they extend beyond ram
# purposedly doeStn inlcude the 'band' option to not impleemnt on small enOugh arrays
BAND_LIKE_DIM_NAMES = frozenset({"polarization"})

log = structlog.get_logger()


GeoTransform = tuple[float, float, float, float, float, float]
"""Affine transform as GDAL-style 6-tuple `(a, b, c, d, e, f)`."""
Bbox = tuple[float, float, float, float]
"""Bounding box `(xmin, ymin, xmax, ymax)`."""


def _has_regular_xy_grid(dataset: xr.Dataset) -> bool:
    """Whether `dataset` has x/y coordinates with at least two pixels along each axis.

    Then the pixel size is known and `_transform_from_coordinates` can be called.
    """
    return (
        "x" in dataset.coords
        and "y" in dataset.coords
        and len(dataset.coords["x"].values) >= 2
        and len(dataset.coords["y"].values) >= 2
    )


def _transform_from_coordinates(dataset: xr.Dataset) -> GeoTransform:
    """Affine transform of the dataset's x/y grid. Requires `_has_regular_xy_grid(dataset)`."""
    x_coords = dataset.coords["x"].values
    y_coords = dataset.coords["y"].values
    pixel_size_x = float(np.abs(x_coords[1] - x_coords[0]))
    pixel_size_y = float(np.abs(y_coords[1] - y_coords[0]))
    # Coordinates are pixel centres; the transform origin is the outer pixel edge.
    x_min = float(x_coords.min()) - pixel_size_x / 2
    y_max = float(y_coords.max()) + pixel_size_y / 2
    return (pixel_size_x, 0.0, x_min, 0.0, -pixel_size_y, y_max)


def _bbox_from_coordinates(dataset: xr.Dataset) -> list[float]:
    """Outer pixel edges ``[xmin, ymin, xmax, ymax]`` of the dataset's x/y grid.

    Falls back to the coordinate extent when the grid has fewer than two pixels
    along an axis, because the pixel size is then unknown.
    """
    if not _has_regular_xy_grid(dataset):
        x_coords = dataset.coords["x"].values
        y_coords = dataset.coords["y"].values
        return [
            float(x_coords.min()),
            float(y_coords.min()),
            float(x_coords.max()),
            float(y_coords.max()),
        ]

    pixel_size_x, _, x_min, _, neg_pixel_size_y, y_max = _transform_from_coordinates(dataset)
    x_max = x_min + dataset.sizes["x"] * pixel_size_x
    y_min = y_max + dataset.sizes["y"] * neg_pixel_size_y
    return [x_min, y_min, x_max, y_max]


def _rio_transform(dataset: xr.Dataset) -> GeoTransform:
    """Affine transform from the rio accessor.

    Raises AttributeError, TypeError or ValueError when the dataset has no usable
    one; check `_has_rio_transform(dataset)` first.
    """
    rio_value = dataset.rio.transform
    if callable(rio_value):
        rio_value = rio_value()
    # rio transform value is dynamically typed; it is iterable at runtime.
    rio_iter = cast("tuple[float, ...]", tuple(rio_value))  # pyright: ignore[reportArgumentType]
    a, b, c, d, e, f = (float(value) for value in rio_iter[:6])
    return (a, b, c, d, e, f)


def _has_rio_transform(dataset: xr.Dataset) -> bool:
    """Whether `_rio_transform(dataset)` can be called."""
    try:
        _rio_transform(dataset)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _has_spatial_transform(dataset: xr.Dataset) -> bool:
    """Whether `_preferred_spatial_transform(dataset)` can be called."""
    return _has_regular_xy_grid(dataset) or _has_rio_transform(dataset)


def _rio_transform_matches_coordinates(
    transform: GeoTransform, coordinate_transform: GeoTransform
) -> bool:
    """Check whether rio-derived metadata matches the current x/y grid."""
    return all(np.isclose(a, b) for a, b in zip(transform, coordinate_transform, strict=False))


def _preferred_spatial_transform(dataset: xr.Dataset) -> GeoTransform:
    """Prefer rio metadata only when it matches the current coordinate grid.

    Requires `_has_spatial_transform(dataset)`.
    """
    if not _has_rio_transform(dataset):
        return _transform_from_coordinates(dataset)
    rio_transform = _rio_transform(dataset)
    if not _has_regular_xy_grid(dataset):
        return rio_transform

    coordinate_transform = _transform_from_coordinates(dataset)
    if not all(value == 0 for value in rio_transform) and _rio_transform_matches_coordinates(
        rio_transform, coordinate_transform
    ):
        return rio_transform
    return coordinate_transform


def _band_like_dim_index(var_data: xr.DataArray) -> int | None:
    """Index of the band-like dimension (see `BAND_LIKE_DIM_NAMES`) of `var_data`.

    Only arrays with more than two dimensions can have one; returns None otherwise
    or when no dimension is band-like.
    """
    if len(var_data.dims) <= 2:
        log.info(
            "Band-like dimension index is only looked up for arrays with more than two dimensions",
            dims=var_data.dims,
            band_like_dim=BAND_LIKE_DIM_NAMES,
        )
    else:
        for i, dim in enumerate(var_data.dims):
            if dim in BAND_LIKE_DIM_NAMES:
                return i
    return None


def _rechunk_ds(ds: xr.Dataset, spatial_chunk: int) -> xr.Dataset:
    chunks = {dim: (min(spatial_chunk, size)) for dim, size in ds.sizes.items()}
    return ds.chunk(chunks)


def _rechunk_iterable(
    iterable: Iterable, encoding: dict[str, XarrayDataArrayEncoding]
) -> dict[Hashable, xr.DataArray]:
    collector: dict[Hashable, xr.DataArray] = {}
    for var_name, var_data in iterable:
        if str(var_name) in encoding:
            var_encoding = encoding[str(var_name)]

            # If sharding is enabled, rechunk based on shard dimensions
            shards_encoding = var_encoding.get("shards")
            chunks_encoding = var_encoding.get("chunks")
            if shards_encoding is not None:
                target_chunks = shards_encoding  # Use shard dimensions for rechunking
            elif chunks_encoding is not None:
                target_chunks = chunks_encoding  # Fallback to chunk dimensions
            else:
                # No specific chunking needed, use original variable
                collector[var_name] = var_data
                continue

            # Create chunk dict using the actual dimensions of the variable
            if len(var_data.dims) == len(target_chunks):
                chunk_dict = dict(zip(var_data.dims, target_chunks, strict=True))

                # Rechunk the variable to match the target dimensions
                collector[var_name] = var_data.chunk(chunk_dict)
            else:
                raise ValueError(
                    f"Given dataarray '{var_name}' with dimension '{var_data.dims}' has different shape than chunks '{target_chunks}'"
                )
        else:
            # No specific chunking needed, use original variable
            collector[var_name] = var_data
    return collector


def _is_bbox(value: object) -> TypeGuard[Sequence[float]]:
    """Whether *value* is a ``spatial:bbox``: a list or tuple of four numbers.

    ``spatial:bbox`` is read from stored metadata, so its type is not known
    statically; this verifies the shape at runtime rather than asserting it.
    """
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(v, (int, float)) for v in value)
    )


def _as_bbox(value: Sequence[float]) -> Bbox:
    """Return *value* as a 4-tuple of floats. Requires `_is_bbox(value)`."""
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _has_proj_crs(attrs: Mapping[str, Any]) -> bool:
    """Whether a group's attributes declare a CRS with ``proj:code``, ``proj:wkt2`` or ``proj:projjson``."""
    return (
        isinstance(attrs.get("proj:code"), str)
        or isinstance(attrs.get("proj:wkt2"), str)
        or isinstance(attrs.get("proj:projjson"), dict)
    )


def _crs_from_attrs(attrs: Mapping[str, Any]) -> CRS:
    """Resolve a pyproj CRS from a group's ``proj:*`` attributes. Requires `_has_proj_crs(attrs)`.

    Uses ``proj:code``, else ``proj:wkt2``, else ``proj:projjson``. Raises
    `pyproj.exceptions.CRSError` for a malformed value, since the attributes come
    from stored metadata.
    """
    code = attrs.get("proj:code")
    if isinstance(code, str):
        return CRS.from_user_input(code)
    wkt2 = attrs.get("proj:wkt2")
    if isinstance(wkt2, str):
        return CRS.from_wkt(wkt2)
    return CRS.from_json_dict(attrs["proj:projjson"])


def optimization_summary(dt_input: xr.DataTree, dt_output: xr.DataTree, output_path: str) -> None:
    """Print optimization summary statistics."""
    # Count groups
    input_groups = len(dt_input.groups) if hasattr(dt_input, "groups") else 0
    output_groups = len(dt_output.groups) if hasattr(dt_output, "groups") else 0

    log.info(
        "OPTIMIZATION SUMMARY",
        input_groups=input_groups,
        output_groups=output_groups,
        output_path=output_path,
        groups=[g for g in dt_output.groups if g != "."],
    )


def create_result_datatree(output_path: str) -> xr.DataTree:
    """Create result DataTree from written output."""
    storage_options = fs_utils.get_storage_options(output_path)
    return xr.open_datatree(
        output_path,
        engine="zarr",
        chunks="auto",
        storage_options=storage_options,
    )


def updated_root_consolidation(
    dt_input: xr.DataTree,
    output_path: str,
    datasets: Mapping[str, object],
    additional_consolidate_paths: list[str] | None = None,
) -> None:
    """
    Simple root-level and sub-root level metadata consolidation with proper zarr group creation.
    Consolidation has to happen from child first to prevent possible overwriting issues if root is consolidated first.
    The root (and subroot) metadata is written from `dt_input.attrs`, so product-specific STAC
    additions (e.g. the S2 assets) have to be applied to them before calling this function.
    Three steps have to be done to achieve a uniform processing of all modes:
        1.) Create consolidated 'measurement' groups for S1, S2 (in reflectance), and S3
        2.) Consolidate any additional groups which are added (this is reserved for multiscale groups)
        3.) Write metadata for subroots (first) and roots, and consolidate subroots OR root
    Note: The addition of multiscale induced assets to STAC has to be implemented before this function call with add_multiscale_pyramids_to_stac_metadata(...)
    Note: Nothing may write group attributes after this call, else the root's consolidated metadata becomes stale
    """
    # this might be a potential issue for s3, as many paths added to output_path are '/*' which will lead to '*//*'
    output_path = output_path.rstrip("/")

    def _group_url(group_path: str) -> str:
        return f"{output_path}/{group_path.strip('/')}"

    # step 1: consolidate all /measurements (inlcuding subroots and S2 reflectance)
    for group_path in datasets:
        _, _, name = group_path.rstrip("/").rpartition("/")
        if name == "measurements":
            zarr.consolidate_metadata(_group_url(group_path), zarr_format=constants.ZARR_FORMAT)

    # step 2: consolidate any additional groups (multiscales, ...) which are passed to this function
    if additional_consolidate_paths:
        for additional_path in additional_consolidate_paths:
            zarr.consolidate_metadata(
                _group_url(additional_path), zarr_format=constants.ZARR_FORMAT
            )

    # step 3: add metadata to subroots and roots, consolidated either one
    # subroot groups are only present for s1, and we need to consolidate its measurements with /subgroup/measurements
    # its guaranteed, that this is the deepest consolidation for the dtree, above it is only the subroot
    subroot_groups = get_subroots(dt_input.groups)
    if subroot_groups:
        for subroot in subroot_groups:
            # add spatial and stac related metadata
            write_store_root_metadata(
                _group_url(subroot),
                attrs=cast("dict[str, dict[str, Any]]", dt_input[subroot].attrs),
            )

            # consolidate the subroot itself
            zarr.consolidate_metadata(_group_url(subroot), zarr_format=constants.ZARR_FORMAT)

    # also add the geo and stac metadata to the root
    write_store_root_metadata(
        output_path,
        attrs=cast("dict[str, dict[str, Any]]", dt_input.attrs),
    )

    # consolidate metadata in root OR in each subroot
    if not subroot_groups:
        zarr.consolidate_metadata(output_path, zarr_format=constants.ZARR_FORMAT)


def write_store_root_metadata(output_path: str, attrs: dict[str, dict[str, Any]]) -> None:
    """Function wrapper for calling two metadata writing functions to add geo and stac metadata to zarr groups."""
    write_store_geo_metadata(output_path, input_root_attrs=attrs)
    write_store_stac_metadata(output_path, input_root_attrs=attrs)


def get_subroots(groups: tuple[str, ...]) -> set[str] | None:
    subroots = set()
    for path in groups:
        parent, _, name = path.rstrip("/").rpartition("/")
        if name in constants.SUBROOT_MARKERS and parent and parent not in subroots:
            subroots.add(parent)
    return subroots or None


def write_dataset_geo_metadata(
    dataset: xr.Dataset,
    grid_mapping_var_name: str = "spatial_ref",
    crs: CRS | None = None,
) -> xr.Dataset:
    """
    Return a copy of the dataset with geographic metadata added; `dataset` is not modified.
    Requires a crs or a dataset from which a CRS can be derived.

    Args:
        dataset: Dataset to add metadata to
        grid_mapping_var_name: Name for grid mapping variable
        crs: Coordinate Reference System to use (if None, attempts to detect from dataset)

    Returns:
        The dataset with the metadata, or an unchanged copy if no CRS is found.
    """
    # Shallow copy: new variables with their own attrs and encoding, data is shared.
    dataset = dataset.copy(deep=False)

    # Use provided CRS or try to detect from dataset
    if crs is None:
        for var in dataset.data_vars.values():
            if hasattr(var, "rio") and var.rio.crs:
                crs = var.rio.crs
                break
            if "proj:epsg" in var.attrs:
                epsg = var.attrs["proj:epsg"]
                crs = CRS.from_epsg(epsg)
                break

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

        # Assemble spatial convention data
        spatial_data: spatial_cm.SpatialAttrs = {
            "spatial:dimensions": ["y", "x"],  # Required field
            "spatial:registration": "pixel",  # Default registration type
        }

        # Calculate and add spatial bbox if coordinates are available
        if "x" in dataset.coords and "y" in dataset.coords:
            spatial_data["spatial:bbox"] = _bbox_from_coordinates(dataset)

            if _has_spatial_transform(dataset):
                spatial_transform = _preferred_spatial_transform(dataset)
                # Only add spatial:transform if we have valid transform data (not all zeros)
                if not all(t == 0 for t in spatial_transform):
                    spatial_data["spatial:transform"] = list(spatial_transform)

            # Add spatial shape if data variables exist
            if dataset.data_vars:
                first_var = next(iter(dataset.data_vars.values()))
                if first_var.ndim >= 2:
                    height, width = first_var.shape[-2:]
                    spatial_data["spatial:shape"] = [height, width]

        # Build validated spatial + proj convention attrs (data + CMOs) via zarr-cm
        dataset.attrs.update(build_convention_attrs(spatial=spatial_data, crs=crs))

    return dataset


def stream_write_dataset(
    dataset: xr.Dataset,
    *,
    path: str,
    group: zarr.Group,
    encoding: dict[str, XarrayDataArrayEncoding],
    add_geo_metadata: bool = False,
    chunk_and_shard_coords: bool = True,
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

    Returns:
        Written dataset
    """
    # Check if level already exists
    if path in group:
        log.info(
            "Level path {} already exists. Skipping write.",
            dataset_path=path,
        )
        # The zarr backend accepts a zarr `Store` here at runtime, but xarray's
        # `open_dataset` stub only types the first arg as path/buffer/datastore.
        return xr.open_dataset(
            group.store,  # type: ignore[arg-type]
            engine="zarr",
            chunks={},
            decode_coords="all",
            group=path,
            mask_and_scale=False,
        )

    log.info("Streaming computation and write to {}", dataset_path=path)
    log.info("Variables", variables=list(dataset.data_vars.keys()))

    # Rechunk dataset to align with encoding
    dataset = rechunk_dataset_for_encoding(
        dataset, encoding, chunk_and_shard_coords=chunk_and_shard_coords
    )

    if add_geo_metadata:
        dataset = write_dataset_geo_metadata(dataset, crs=crs)

    # Sanitize NaN values in dataset attributes before writing
    dataset = fs_utils.sanitize_dataset_attributes(dataset)

    # Write with streaming computation and progress tracking
    # The to_zarr operation will trigger all lazy computations

    write_job = dataset.to_zarr(
        store=group.store,
        mode="w",
        consolidated=False,
        zarr_format=constants.ZARR_FORMAT,
        encoding=encoding,
        group=path,
        compute=False,  # Create job first for progress tracking
    )
    write_job = write_job.persist()

    if DISTRIBUTED_AVAILABLE:
        try:
            import distributed

            # Try to get current client for better status monitoring
            try:
                client = distributed.Client.current()
                # client.compute is untyped (returns Any); verify we got a
                # Future rather than asserting it with a cast.
                future = client.compute(write_job)
                if not isinstance(future, distributed.Future):
                    raise TypeError(f"expected a distributed.Future, got {type(future).__name__}")
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

    log.info("Streaming write complete for dataset {}", dataset_path=path)
    return dataset


def rechunk_dataset_for_encoding(
    dataset: xr.Dataset,
    encoding: dict[str, XarrayDataArrayEncoding],
    chunk_and_shard_coords: bool = False,
) -> xr.Dataset:
    """
    Rechunk dataset variables to align with sharding dimensions when sharding is enabled.

    When using Zarr v3 sharding, Dask chunks must align with shard dimensions to avoid
    checksum validation errors.

    chunk_and_shard_coords: bool
        allows (primarily) the sharding of coordinate arrays, usually not too relevant but
        takes effect for large 2dim coordinate arrays eg. lat/lon S3A OLCI LFR
    """
    if chunk_and_shard_coords:
        # Create new dataset with rechunked variables, also sharding coordinates
        rechunked_vars = _rechunk_iterable(iterable=dataset.data_vars.items(), encoding=encoding)
        rechunked_cords = _rechunk_iterable(iterable=dataset.coords.items(), encoding=encoding)

        return xr.Dataset(rechunked_vars, coords=rechunked_cords, attrs=dataset.attrs)
    # Create new dataset with rechunked variables, preserving coordinates
    rechunked_vars = _rechunk_iterable(iterable=dataset.data_vars.items(), encoding=encoding)

    return xr.Dataset(rechunked_vars, coords=dataset.coords, attrs=dataset.attrs)


def get_chunking_for_encoding(
    var_data: xr.DataArray, shard_along_smallest_dimension: bool = False
) -> tuple[int, ...]:
    """
    requires a prior rechunking of the dataset by calling _rechunk_ds() to rechunk non-metadata arrays to spatial_chunk
    get a tuple of maximal chunksize for the dataarray
    -> (spatial_chukn, spatial_chukn) for spatial arrays
    -> (x, y, z, ..) for multidimensional metadata arrays (just to allow sharding later on)

    Args:
        var_data: DataArray to get the chunks from

    """
    if var_data.chunks:
        # get the maximal chunk shape for zarr encoding -> theoretically it wouldnt be necessary to take the max, as non-uniform chukning (1024, 806)
        # has irregular chunksizes trailing, but the syntax and goal of the code is much clearer this way
        max_chunksizes = [max(c) for c in var_data.chunks]

        if shard_along_smallest_dimension:
            band_dim = _band_like_dim_index(var_data)
            if band_dim is not None:
                max_chunksizes[band_dim] = 1

        return tuple(max_chunksizes)
    raise ValueError(
        f"Datavariable {var_data.name!r} is not chunked already, cannot derive Zarr encoding chunks -> will lead to unchunked array"
    )


def _forward_unpacked_encoding(
    var_name: Hashable, var_data: xr.DataArray, var_encoding: XarrayDataArrayEncoding
) -> XarrayDataArrayEncoding:
    """Update the source encoding of an unpacked variable, with an explicit Zarr fill value."""
    # Pin the Zarr fill_value rather than letting xarray versions infer different ones.
    fv = explicit_fill_value(var_data)
    if fv is not UNSET:
        var_encoding["fill_value"] = fv
    elif "fill_value" in var_data.attrs:
        # On Zarr v3 CPM products the EOPF `fill_value` attribute is the only nodata value.
        var_encoding["_FillValue"] = var_data.attrs["fill_value"]

    # Integer variables without CF packing (e.g. SCL) also need a CF `_FillValue`,
    # as packed variables have, since some clients require it. Only a declared
    # nodata value is used: the Zarr fill value (often 0) can be valid data, e.g.
    # for probabilities and bitmasks.
    if np.issubdtype(var_data.dtype, np.integer):
        nodata = var_encoding.get("_FillValue", encoding_utils.find_fill_value(var_data))
        if nodata is not None:
            var_encoding["_FillValue"] = int(nodata)
            var_encoding.setdefault("fill_value", int(nodata))
            # Keep it on the variable too: the attributes are sanitized afterwards,
            # and the coarser pyramid levels are built from this variable.
            var_data.encoding["_FillValue"] = int(nodata)

    for key in XARRAY_ENCODING_KEYS - {"compressors", "shards", "chunks", "fill_value"}:
        if key in var_data.encoding:
            var_encoding[key] = var_data.encoding[key]

    unknown = set(var_data.encoding) - XARRAY_ENCODING_KEYS
    if unknown:
        log.warning("Unknown encoding keys in %s: %s", var_name, unknown)

    return var_encoding


def build_dataarray_encoding(
    var: xr.DataArray,
    enable_sharding: bool = True,
    compression_level: int = 3,
    shard_along_smallest_dimension: bool = False,
) -> XarrayDataArrayEncoding:
    import math

    from zarr.codecs import BloscCodec

    _encoding: XarrayDataArrayEncoding = {}
    compressor = BloscCodec(cname="zstd", clevel=compression_level, shuffle="shuffle", blocksize=0)

    encoding_chunks = get_chunking_for_encoding(var, shard_along_smallest_dimension)

    _encoding["chunks"] = encoding_chunks
    _encoding["compressors"] = (compressor,)

    # --- Shards: cover the whole array, one shard per array -----------
    if enable_sharding:
        # select next largest mutliple of chunksize to fit full array
        shards_ = [
            math.ceil(shape / chunk) * chunk
            for shape, chunk in zip(var.shape, encoding_chunks, strict=True)
        ]
        if shard_along_smallest_dimension:
            band_dim = _band_like_dim_index(var)
            if band_dim is None:
                log.warning(
                    "shard_along_smallest_dimension=True but %s has no "
                    "recognized band-like dimension (%s); falling back to "
                    "whole-array sharding",
                    var.name,
                    list(var.dims),
                )
            else:
                shards_[band_dim] = encoding_chunks[band_dim]

        _encoding["shards"] = tuple(shards_)
    else:
        _encoding["shards"] = None

    return _encoding


def create_uniform_encoding(
    dataset: xr.Dataset,
    *,
    enable_sharding: bool = True,
    compression_level: int = 3,
    scale_offset_codec: bool = False,
    chunk_and_shard_coords: bool = False,
    shard_along_smallest_dimension: bool = False,
) -> dict[str, XarrayDataArrayEncoding]:
    """
    Create encoding (compression, chunking, sharding) for a dataset.

    Chunking is taken from the input dataset's existing chunks which HAS to be present, which has to be enforced before calling this fucntion.
    Tests inside this fucntion from `get_chunking_for-encoding` will raise a ValueError if no chunks are present. Sharding always covers the *entire* array along every
    dimension, sized as the smallest multiple of that dimension's chunk size
    that is >= the array's shape — so a shard always contains a whole number
    of chunks and there is exactly one shard per array. This avoids partial
    edge chunks ending up in their own oddly-sized shard when e.g. shape=1830
    and chunk=1024 (shard becomes 2048, i.e. 2 chunks, not some 1830-based
    value that would clip/overlap the second chunk).

    Packed variables (see `packing_of`) keep their source integer dtype on disk.
    By default they are written as in the ESA product: packed integers with CF
    `scale_factor`, `add_offset` and `_FillValue`, and they stay packed integers
    in `dataset` (no cast to float). With `scale_offset_codec`, they are
    replaced in `dataset` by their decoded float32 values, and the Zarr
    `scale_offset` + `cast_value` codecs pack them, with no CF scale
    attributes. Other variables keep their dtype and fill value.

    """
    encoding: dict[str, XarrayDataArrayEncoding] = {}

    for var_name in list(dataset.data_vars):
        packing = encoding_utils.packing_of(dataset[var_name])
        if packing is not None:
            dataset[var_name] = encoding_utils.normalize_packed(
                dataset[var_name], packing, scale_offset_codec=scale_offset_codec
            )
        else:
            dataset[var_name] = encoding_utils._drop_trivial_scaling(dataset[var_name])

        var_data = dataset[var_name]
        var_encoding = build_dataarray_encoding(
            var_data,
            enable_sharding=enable_sharding,
            compression_level=compression_level,
            shard_along_smallest_dimension=shard_along_smallest_dimension,
        )

        # `to_zarr(encoding=...)` replaces each variable's `.encoding`, so the
        # variables keep their packing for the pyramid levels while only the
        # keys below are written.
        if packing is not None and not scale_offset_codec:
            # The data are already the packed integers and the CF `scale_factor`
            # and `add_offset` are attributes, so xarray writes them unchanged.
            if packing.fill_value is not None:
                var_encoding["_FillValue"] = packing.fill_value
        elif packing is not None:
            var_encoding["filters"] = encoding_utils._scale_offset_filters(packing)
            var_encoding["fill_value"] = "NaN"
            # CF _FillValue attribute for xarray issue #11345.
            var_data.attrs["_FillValue"] = np.nan
        else:
            var_encoding = _forward_unpacked_encoding(var_name, var_data, var_encoding)
            # Sanitize source-only attributes (replace dict — ``.update`` cannot
            # remove keys, so stale ``_eopf_attrs`` / ``dtype`` / ``valid_*`` would
            # otherwise leak into the output).
            is_float = np.issubdtype(var_data.dtype, np.floating)
            var_data.attrs = sanitize_array_attrs(var_data.attrs, is_decoded_float=is_float)

        encoding[str(var_name)] = var_encoding

    for coord_name, coord_data in dataset.coords.items():
        # coord_encoding: XarrayDataArrayEncoding = {}

        if chunk_and_shard_coords:
            if (
                coord_name in dataset.xindexes
            ):  # skip indexed coords which are likely not chunked -> check for chunked
                continue
            coord_encoding = build_dataarray_encoding(
                coord_data,
                enable_sharding=enable_sharding,
                compression_level=compression_level,
                shard_along_smallest_dimension=shard_along_smallest_dimension,
            )
        else:
            coord_encoding: XarrayDataArrayEncoding = {"compressors": ()}

        coord_data.attrs = sanitize_array_attrs(coord_data.attrs)
        encoding[str(coord_name)] = coord_encoding

    return encoding


@runtime_checkable
class CRSLike(Protocol):
    """A coordinate reference system that can serialize to EPSG/WKT2.

    Both ``pyproj.CRS`` and ``rasterio.crs.CRS`` satisfy this; the conversion
    code accepts either, so we depend on the shared interface rather than a
    concrete class.
    """

    def to_epsg(self) -> int | None: ...

    def to_wkt(self) -> str: ...


def proj_attrs_for_crs(crs: CRSLike | None) -> GeoProjAttrs:
    """Build the ``proj`` convention data keys for a CRS.

    Prefers an EPSG code (``proj:code``) and falls back to WKT2
    (``proj:wkt2``). Returns an empty mapping when *crs* is ``None`` or exposes
    no EPSG code.
    """
    if crs is None:
        return GeoProjAttrs()
    epsg = crs.to_epsg()
    if epsg:
        return GeoProjAttrs({"proj:code": f"EPSG:{epsg}"})
    return GeoProjAttrs({"proj:wkt2": crs.to_wkt()})


def build_convention_attrs(
    *,
    spatial: SpatialAttrs | None,
    crs: CRSLike | None,
    multiscales: MultiscalesAttrs | None = None,
) -> MultiConventionAttrs:
    """Build validated multiscales + ``spatial`` + ``proj`` convention attributes.

    Delegates to :func:`zarr_cm.create_many`, which validates each convention's
    data and emits the matching convention-metadata objects into a combined
    ``zarr_conventions`` array. The CMOs are ordered multiscales (if present),
    then spatial, then proj. *spatial* holds the ``spatial:*`` keys; the proj
    keys are derived from *crs* via :func:`proj_attrs_for_crs`.

    The proj convention is only included when *crs* yields a usable CRS
    representation; otherwise only the other conventions are emitted (a proj
    convention with no CRS field is invalid).
    """
    conventions: dict[zarr_cm.ConventionName, MultiscalesAttrs | SpatialAttrs | GeoProjAttrs] = {}
    if multiscales is not None:
        conventions["multiscales"] = multiscales

    if spatial is not None:
        conventions["spatial"] = spatial

    proj = proj_attrs_for_crs(crs)
    if proj:
        conventions["geo-proj"] = proj

    # create_many validates each convention and emits its CMO. It returns a
    # generic JSON dict; narrow to the combined convention TypedDict.
    result = zarr_cm.create_many(conventions)
    return cast("MultiConventionAttrs", result)


def add_multiscale_pyramids_to_stac_metadata(
    datasets: Mapping[str, object], dt_attributes: dict[Hashable, Any]
) -> dict[Hashable, Any]:
    stac_attrs = dt_attributes.get("stac_discovery", {}).get("assets")
    if not stac_attrs:
        return dt_attributes

    # a bit messy but effective split to get group parent from stac attrs
    existing_group_paths = {"/".join(v["href"].split("/")[:-1]) for v in stac_attrs.values()}

    # gEt mismatched ones -> we need pyramids not present
    missing_group_paths = [
        path for path, ds in datasets.items() if path not in existing_group_paths and ds is not None
    ]

    for group_path in missing_group_paths:
        ds = datasets[group_path]

        # catch object != datAset for typing
        if isinstance(ds, xr.Dataset):
            resolution = group_path.rsplit("/", 1)[-1]  # "r120m"
            for var_name in ds.data_vars:
                if var_name == "spatial_ref":
                    continue
                asset_key = f"{var_name}_{resolution}"
                stac_attrs[asset_key] = {"href": f"{group_path}/{var_name}", "title": asset_key}
        else:
            log.warning("Found non-dataset object in datasets!", dataset=ds)

    return dt_attributes


def explicit_fill_value(var: xr.DataArray) -> Any:
    """Pick a zarr-level `fill_value` for `var` based on its source `_FillValue`.

    Different xarray versions infer different on-disk fill values when the
    encoding dict doesn't pin it: older xarray defaults floats to 0.0; newer
    xarray honours the source `_FillValue`. Setting `fill_value` explicitly
    via this helper removes that degree of freedom so the on-disk metadata is
    stable across xarray versions.

    Returns
    -------
    object
        The value to assign to `encoding["fill_value"]`. The sentinel `UNSET`
        is returned when the source has no `_FillValue` (caller should leave
        the encoding entry alone). For non-finite floats, returns the
        JSON-canonical string form (`"NaN"` / `"Infinity"` / `"-Infinity"`)
        that zarr-python serialises.
    """
    source_fill = var.encoding.get("_FillValue")
    if source_fill is None:
        return UNSET
    fill_arr = np.asarray(source_fill)
    if np.issubdtype(fill_arr.dtype, np.floating) and not np.isfinite(fill_arr):
        if np.isnan(fill_arr):
            return "NaN"
        return "Infinity" if fill_arr > 0 else "-Infinity"
    return source_fill


def sanitize_array_attrs(
    attrs: dict[str, Any],
    *,
    is_decoded_float: bool = False,
) -> dict[str, Any]:
    """Return a copy of *attrs* with source-only and misleading keys removed.

    - ``_eopf_attrs`` and ``_FillValue`` are always removed. ``_FillValue``
      belongs in the variable's *encoding* (where the zarr-level fill value is
      carried), not in its attributes; callers that need a CF ``_FillValue``
      attribute (e.g. the NaN workaround for xarray issue #11345) must re-add
      it after sanitizing.

      .. warning:: The Sentinel-3 OLCI converter has its own deliberately
         divergent sanitizer
         (``s3_olci_optimization.olci_converter._sanitize_olci_array_attrs_keep_fill``)
         that **preserves** ``_FillValue`` for raw (non-mask-scaled) input.
         Edits to the strip-list here do not apply there, and vice versa.
    - For decoded float measurement arrays (*is_decoded_float=True*), also
      removes raw-encoding leftovers ``dtype``, ``fill_value``,
      ``valid_min``, ``valid_max`` and rewrites
      ``units: "digital_counts"`` → ``"1"``.

    - Geo-proj *convention* keys (``proj:code``, ``proj:wkt2``,
      ``proj:projjson``) are always removed: per the minispec they belong on
      (or are inherited from) the enclosing group, and source products carry
      them on arrays without the required ``zarr_conventions`` declaration.
      Legacy external keys such as ``proj:epsg`` are left alone.

    CF keys ``scale_factor`` and ``add_offset`` are always preserved.
    """
    dropped = {"_eopf_attrs", "_FillValue", *geo_proj_cm.CONVENTION_KEYS}
    out = {k: v for k, v in attrs.items() if k not in dropped}

    if is_decoded_float:
        for key in ("dtype", "fill_value"):
            out.pop(key, None)
        if out.get("units") == "digital_counts":
            out["units"] = "1"
    return out


def downsample_2d_array(
    source_data: np.ndarray,
    target_height: int,
    target_width: int,
    nodata_value: float | None = None,
) -> np.ndarray:
    """
    Downsample a 2D array using block averaging with proper nodata handling.

    Parameters
    ----------
    source_data : numpy.ndarray
        Source 2D array
    target_height : int
        Target height
    target_width : int
        Target width
    nodata_value : float, optional
        Value representing nodata/fill areas. If provided, these areas will be
        excluded from averaging and preserved in the output.

    Returns
    -------
    numpy.ndarray
        Downsampled 2D array with nodata values preserved
    """
    source_height, source_width = source_data.shape

    # Calculate block sizes
    block_size_y = source_height // target_height
    block_size_x = source_width // target_width

    if block_size_y > 1 and block_size_x > 1:
        # Block averaging with nodata handling
        reshaped = source_data[: target_height * block_size_y, : target_width * block_size_x]
        reshaped = reshaped.reshape(target_height, block_size_y, target_width, block_size_x)

        if nodata_value is not None and not np.isnan(nodata_value):
            # Create mask for valid data (not nodata)
            valid_mask = reshaped != nodata_value

            # Calculate mean only for valid data
            with np.errstate(invalid="ignore", divide="ignore"):
                # Sum valid values and count valid pixels
                valid_sum = np.where(valid_mask, reshaped, 0).sum(axis=(1, 3))
                valid_count = valid_mask.sum(axis=(1, 3))

                # Calculate mean, preserving nodata where no valid data exists
                downsampled = np.where(valid_count > 0, valid_sum / valid_count, nodata_value)
        elif nodata_value is not None and np.isnan(nodata_value):
            # Handle NaN nodata values
            with np.errstate(invalid="ignore"):
                downsampled = np.nanmean(reshaped, axis=(1, 3))
        else:
            # No nodata handling needed
            downsampled = reshaped.mean(axis=(1, 3))
    else:
        # Simple subsampling
        y_indices = np.linspace(0, source_height - 1, target_height, dtype=int)
        x_indices = np.linspace(0, source_width - 1, target_width, dtype=int)
        downsampled = source_data[np.ix_(y_indices, x_indices)]

    return downsampled


def is_grid_mapping_variable(ds: xr.Dataset, var_name: str) -> bool:
    """
    Check if a variable is a grid_mapping variable by looking for references to it.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to check
    var_name : str
        Variable name to check

    Returns
    -------
    bool
        True if this variable is referenced as a grid_mapping
    """
    for data_var in ds.data_vars:
        if (
            data_var != var_name
            and "grid_mapping" in ds[data_var].attrs
            and ds[data_var].attrs["grid_mapping"] == var_name
        ):
            return True
    return False


def calculate_aligned_chunk_size(dimension_size: int, target_chunk_size: int) -> int:
    """
    Calculate a chunk size that divides evenly into the dimension size.

    This ensures that Zarr chunks align properly with the data dimensions,
    preventing chunk overlap issues when writing with Dask.

    Parameters
    ----------
    dimension_size : int
        Size of the dimension to chunk
    target_chunk_size : int
        Desired chunk size

    Returns
    -------
    int
        Aligned chunk size that divides evenly into dimension_size
    """
    if target_chunk_size >= dimension_size:
        return dimension_size

    # Find the largest divisor of dimension_size that is <= target_chunk_size
    for chunk_size in range(target_chunk_size, int(target_chunk_size * 0.51), -1):
        if dimension_size % chunk_size == 0:
            return chunk_size

    # If no divisor is found, return the closest value to target_chunk_size
    return min(target_chunk_size, dimension_size)


def validate_existing_band_data(
    existing_group: xr.Dataset, var_name: str, reference_ds: xr.Dataset
) -> bool:
    """
    Validate that a specific band exists and is complete in the dataset.

    Parameters
    ----------
    existing_group : xarray.Dataset
        Existing dataset to validate
    var_name : str
        Name of the variable to validate
    reference_ds : xarray.Dataset
        Reference dataset structure for comparison

    Returns
    -------
    bool
        True if the variable exists and is valid, False otherwise
    """
    try:
        # Check if the variable exists
        if var_name not in existing_group.data_vars and var_name not in existing_group.coords:
            return False

        # Check shape matches
        if var_name in reference_ds.data_vars:
            expected_shape = reference_ds[var_name].shape
            existing_shape = existing_group[var_name].shape

            if expected_shape != existing_shape:
                return False

        # Check required attributes for data variables
        if var_name in reference_ds.data_vars and not is_grid_mapping_variable(
            reference_ds, var_name
        ):
            required_attrs = ["_ARRAY_DIMENSIONS", "standard_name"]
            for attr in required_attrs:
                if attr not in existing_group[var_name].attrs:
                    return False

        # Check rio CRS
        if existing_group.rio.crs != reference_ds.rio.crs:
            return False

        # Basic data integrity check for data variables
        if var_name in existing_group.data_vars and not is_grid_mapping_variable(
            existing_group, var_name
        ):
            try:
                # Just check if we can access the array metadata without reading data
                array_info = existing_group[var_name]
                if array_info.size == 0:
                    return False
                # read a piece of data to ensure it's valid
                test = array_info.isel(dict.fromkeys(array_info.dims, 0)).values.mean()
                if np.isnan(test):
                    return False
            except Exception as e:
                log.info("Error validating variable", var_name=var_name, error=str(e))
                return False

    except Exception:
        return False
    else:
        return True


def compute_overview_gcps(
    ds_gcp: xr.Dataset, scale_factor: float, width: int, height: int
) -> xr.Dataset:
    """Compute new GCPs for a given overview from the original GCPs.

    Parameters
    ----------
    ds_gcp : xr.Dataset
        the original GCPs
    scale_factor : float
        Overview's scale factor
    width : int
        Overview's width
    height : int
        Overview's height

    Returns
    -------
    ds_gcp_overview : xr.Dataset
        A new dataset where GCPs line and pixel coordinates are updated
        for the overview, and where duplicate line/pixel GCPs are
        merged together by averaging their latitude, longitude and height.

    """
    return (
        # compute the new decimated line/pixel coordinates
        # TODO: trim line values with height and pixel values with width?
        ds_gcp.assign_coords(
            line=np.round(ds_gcp.line / scale_factor).astype(np.int64),
            pixel=np.round(ds_gcp.pixel / scale_factor).astype(np.int64),
        )
        # find duplicate line/pixel GCPs
        # and compute average for latitude, longitude and height
        .pipe(lambda ds: ds.groupby(["line", "pixel"]))
        .mean()
        # re-assign original dimensions
        .rename_dims(line="azimuth_time", pixel="ground_range")
    )


def write_store_geo_metadata(
    output_path: str,
    input_root_attrs: dict[str, dict[str, Any]] | None = None,
    storage_options: dict[str, Any] | None = None,
) -> None:
    """Write the minispec store-root metadata on the root group.

    Walks the zarr store, collects every child-group `spatial:bbox` along with
    its CRS (resolved from ``proj:code`` / ``proj:wkt2`` / ``proj:projjson``),
    reprojects each to EPSG:4326 with edge densification and writes the union
    plus the CRS code and the matching ``zarr_conventions`` declaration on the
    root group. Groups whose CRS cannot be resolved are skipped with a warning
    rather than assumed to be in degrees. The CRS is always declared explicitly
    per the Store Root section of the minispec — there is no implicit default.

    When *storage_options* is ``None``, the store's options are derived from
    *output_path* via :func:`eopf_geozarr.conversion.fs_utils.get_storage_options`
    so remote (e.g. S3) stores honour the configured endpoint and credentials.
    """
    from pyproj import Transformer

    from eopf_geozarr.conversion import fs_utils

    if storage_options is None:
        storage_options = cast("dict[str, Any] | None", fs_utils.get_storage_options(output_path))

    root = zarr.open_group(output_path, mode="r+", storage_options=storage_options)

    bboxes_4326: list[Bbox] = []

    def _collect_bbox_4326(group: zarr.Group, attrs: Mapping[str, Any], corners: Bbox) -> None:
        """Append the group's bbox in EPSG:4326, or log why it is skipped."""
        if not _has_proj_crs(attrs):
            log.warning(
                "Group has spatial:bbox but no proj:* CRS; skipping it for the store-root footprint",
                group=group.path,
            )
            return
        try:
            crs = _crs_from_attrs(attrs)
        except CRSError as e:  # malformed stored metadata
            log.warning(
                "Unresolvable proj:* CRS; skipping group bbox", group=group.path, error=str(e)
            )
            return
        if crs.to_epsg() == 4326:
            bboxes_4326.append(corners)
            return
        try:
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            # transform_bounds densifies the edges, which corner-wise
            # transformation misses (projected edges curve in lon/lat).
            bboxes_4326.append(transformer.transform_bounds(*corners, densify_pts=21))
        except Exception as e:  # never abort the write for one group
            log.warning(
                "Failed to reproject group bbox; skipping it", group=group.path, error=str(e)
            )

    def _walk(group: zarr.Group) -> None:
        attrs = dict(group.attrs)
        bbox = attrs.get("spatial:bbox")
        if _is_bbox(bbox):
            _collect_bbox_4326(group, attrs, _as_bbox(bbox))
        for child in group.groups():
            _walk(child[1])

    for _, child_group in root.groups():
        _walk(child_group)

    if input_root_attrs and not bboxes_4326:
        log.warning(
            "deriving bounding box for minimal spatial geozarr spec from stac metadata geometry -> more stable than bbox attribute"
        )
        try:
            stac_attrs = input_root_attrs.get("stac_discovery")

            if stac_attrs is None or not isinstance(stac_attrs, dict):
                log.warning("No usable stac_discovery block found; skipping store-root metadata")
            elif "geometry" not in stac_attrs:
                log.warning(
                    "stac_discovery present but no geometry found; skipping store-root metadata"
                )
            else:

                def _positions(c: Any) -> Any:
                    # flatten nested GeoJSON coordinates down to [x, y] positions
                    if isinstance(c[0], (int, float)):
                        yield c
                    else:
                        for sub in c:
                            yield from _positions(sub)

                pts = list(_positions(stac_attrs["geometry"]["coordinates"]))
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bboxes_4326.append((min(xs), min(ys), max(xs), max(ys)))
        except KeyError:
            log.warning("No stac_discovery block found at all; skipping store-root metadata")
            return

    if not bboxes_4326:
        log.warning(
            "No spatial:bbox found anywhere in the store; skipping store-root spatial metadata"
        )
        return

    if any(b[0] > b[2] for b in bboxes_4326):
        # At least one footprint crosses the antimeridian; a single
        # [xmin, ymin, xmax, ymax] box cannot represent the union faithfully,
        # so fall back to the full longitude range.
        log.warning(
            "A child bbox crosses the antimeridian; store-root bbox uses the full longitude range"
        )
        xmin, xmax = -180.0, 180.0
    else:
        xmin = min(b[0] for b in bboxes_4326)
        xmax = max(b[2] for b in bboxes_4326)
    ymin = min(b[1] for b in bboxes_4326)
    ymax = max(b[3] for b in bboxes_4326)

    root_attrs: dict[str, Any] = {
        "zarr_conventions": [dict(spatial_cm.CMO), dict(geo_proj_cm.CMO)],
        "spatial:bbox": [xmin, ymin, xmax, ymax],
        "proj:code": "EPSG:4326",
    }
    root.attrs.update(root_attrs)
    log.info("Wrote store-root spatial metadata", bbox=[xmin, ymin, xmax, ymax])


def write_store_stac_metadata(
    output_path: str,
    input_root_attrs: dict[str, dict[str, Any]],
    storage_options: dict[str, Any] | None = None,
    overwrite_root_attrs: bool = False,
) -> None:
    """
    Adds root metadata passed in root_attrs to the new zarr store. This usually includes the stac metadata sstore in the root of the input zarr store.
    Also check if metadta is already present in root -> ioverruled and overwritten by specifing rhe overwrite_root_attrs=True
    """
    from eopf_geozarr.conversion import fs_utils

    if storage_options is None:
        storage_options = cast("dict[str, Any] | None", fs_utils.get_storage_options(output_path))

    root = zarr.open_group(output_path, mode="r+", storage_options=storage_options)

    # prevent the overwriting of attributes in the root node if they are present in the stac metadata.. this is a failsafe for future changes of cpm if the y include zarr metadata
    if not overwrite_root_attrs:
        original_attrs = set(dict(root.attrs).keys())
        new_attrs = set(input_root_attrs.keys())
        for int_attr in original_attrs.intersection(new_attrs):
            input_root_attrs.pop(int_attr)

    root.attrs.update(input_root_attrs)
    log.info(
        "Updated root metadata attributes for STAC ingestion",
        root_attrs=list(input_root_attrs.keys()),
    )

    # use the merged root attrs, since `stac_discovery` may have been popped from `input_root_attrs` above
    write_store_geozarr_version_metadata(
        output_path, input_root_attrs=dict(root.attrs), storage_options=storage_options
    )


def write_store_geozarr_version_metadata(
    output_path: str,
    input_root_attrs: Mapping[str, Any],
    storage_options: dict[str, Any] | None = None,
) -> None:
    """Adds the current git tag to the processing history of stac_discovery."""
    import copy

    from eopf_geozarr.conversion import fs_utils

    def git_version() -> str | None:
        from pathlib import Path

        try:
            import pygit2
        except ImportError:
            return None

        try:
            repo_path = pygit2.discover_repository(str(Path(__file__).resolve().parent))
            if repo_path is None:
                return None
            repo = pygit2.Repository(repo_path)
            return repo.describe(
                describe_strategy=pygit2.enums.DescribeStrategy.TAGS,
                show_commit_oid_as_fallback=True,
                always_use_long_format=True,
                dirty_suffix="-dirty",
            )
        except (LookupError, pygit2.GitError):
            return None

    if storage_options is None:
        storage_options = cast("dict[str, Any] | None", fs_utils.get_storage_options(output_path))

    root = zarr.open_group(output_path, mode="r+", storage_options=storage_options)

    stac = input_root_attrs.get("stac_discovery")
    if stac is None:
        log.info(
            "Could not find `stac_discovery` in passed down attributes, not applying geozarr verisoning."
        )
        return

    def package_version() -> str | None:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("eopf-geozarr")
        except PackageNotFoundError:
            return None

    # fall back to the installed package version if there is no git history (e.g. installed from a wheel)
    eopf_geozarr_version = git_version() or package_version()
    if not eopf_geozarr_version:
        raise ValueError(
            "Could not derive EOPF_GEOZARR version from git version history or the installed package."
        )

    stac = copy.deepcopy(stac)
    software = stac.setdefault("properties", {}).setdefault("processing:software", {})
    software["EOPF_GEOZARR"] = eopf_geozarr_version

    root.attrs["stac_discovery"] = stac

    return
