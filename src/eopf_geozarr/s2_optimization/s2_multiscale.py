"""
Streaming multiscale pyramid creation for optimized S2 structure.
Uses lazy evaluation to minimize memory usage during dataset preparation.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import structlog
import xarray as xr
import zarr
from dask.array import from_delayed
from dask.delayed import delayed
from pydantic.experimental.missing_sentinel import MISSING
from pyproj import CRS
from zarr.codecs import CastValue

from eopf_geozarr.conversion import utils
from eopf_geozarr.conversion.fs_utils import sanitize_dataset_attributes
from eopf_geozarr.data_api.geozarr.multiscales import zcm
from eopf_geozarr.data_api.geozarr.multiscales.geozarr import (
    MultiscaleMeta,
)
from eopf_geozarr.data_api.geozarr.types import (
    CF_SCALE_OFFSET_KEYS,
    XARRAY_ENCODING_KEYS,
    XarrayDataArrayEncoding,
)
from eopf_geozarr.s2_optimization.common import DISTRIBUTED_AVAILABLE
from eopf_geozarr.s2_optimization.s2_band_mapping import BAND_INFO

from .s2_resampling import determine_variable_type, downsample_variable

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from zarr_cm import MultiscalesAttrs
    from zarr_cm import spatial as spatial_cm

    from eopf_geozarr.types import OverviewLevelJSON

log = structlog.get_logger()

MultiscalesFlavor = Literal["experimental_multiscales_convention"]

# Default upper bound on chunk dims propagated from the source when copying
# groups as-is (see the create_original_encoding call site): whole-array
# source chunks (e.g. cpm_v270 aot/wvp, 10980x10980;
# EOPF-Explorer/data-pipeline#339) are unusable for tiled serving. The
# aligned clamp lands on 1830 for the S2 grid sizes — the chunk size the
# EOPF product spec documents as native S2 tiling (eopf-cpm PSFD 6.7.3,
# "6x6 chunks").
MAX_ORIGINAL_CHUNK = 2048

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


def _transform_from_coordinates(
    dataset: xr.Dataset,
) -> tuple[float, float, float, float, float, float] | None:
    """Construct an affine transform from dataset coordinates when possible."""
    if "x" not in dataset.coords or "y" not in dataset.coords:
        return None

    x_coords = dataset.coords["x"].values
    y_coords = dataset.coords["y"].values
    if len(x_coords) < 2 or len(y_coords) < 2:
        return None

    pixel_size_x = float(np.abs(x_coords[1] - x_coords[0]))
    pixel_size_y = float(np.abs(y_coords[1] - y_coords[0]))
    x_min = float(x_coords.min())
    y_max = float(y_coords.max())
    return (pixel_size_x, 0.0, x_min, 0.0, -pixel_size_y, y_max)


def _rio_transform_matches_coordinates(
    transform: tuple[float, float, float, float, float, float] | None,
    coordinate_transform: tuple[float, float, float, float, float, float] | None,
) -> bool:
    """Check whether rio-derived metadata matches the current x/y grid."""
    if transform is None or coordinate_transform is None:
        return False

    return all(np.isclose(a, b) for a, b in zip(transform, coordinate_transform, strict=False))


def _preferred_spatial_transform(
    dataset: xr.Dataset,
) -> tuple[float, float, float, float, float, float] | None:
    """Prefer rio metadata only when it matches the current coordinate grid."""
    coordinate_transform = _transform_from_coordinates(dataset)
    rio_transform: tuple[float, float, float, float, float, float] | None = None

    if hasattr(dataset, "rio") and hasattr(dataset.rio, "transform"):
        try:
            rio_value = dataset.rio.transform
            if callable(rio_value):
                rio_value = rio_value()
            # rio transform value is dynamically typed; it is iterable at runtime.
            rio_iter = cast("tuple[float, ...]", tuple(rio_value))  # pyright: ignore[reportArgumentType]
            rio_values = tuple(float(value) for value in rio_iter[:6])
            if len(rio_values) == 6:
                rio_transform = (
                    rio_values[0],
                    rio_values[1],
                    rio_values[2],
                    rio_values[3],
                    rio_values[4],
                    rio_values[5],
                )
        except (AttributeError, TypeError, ValueError):
            rio_transform = None

    if (
        rio_transform is not None
        and not all(value == 0 for value in rio_transform)
        and _rio_transform_matches_coordinates(rio_transform, coordinate_transform)
    ):
        return rio_transform

    return coordinate_transform or rio_transform


def _coarsen_variable(var_name: str, var_data: xr.DataArray, factor: int) -> xr.DataArray:
    """Coarsen a single variable using type-aware resampling.

    Dispatches to the appropriate coarsen reduction (mean, max, subsample)
    based on `determine_variable_type`.  Preserves encoding and dtype.
    """
    var_type = determine_variable_type(var_name, var_data)
    coarsened = var_data.coarsen({"x": factor, "y": factor}, boundary="trim")
    if var_type in ("reflectance", "probability"):
        # xarray stubs omit reduction methods on DataArrayCoarsen.
        result = coarsened.mean()  # type: ignore[attr-defined]
    elif var_type == "classification":
        result = coarsened.reduce(subsample_2)
    elif var_type == "quality_mask":
        # xarray stubs omit reduction methods on DataArrayCoarsen.
        result = coarsened.max()  # type: ignore[attr-defined]
    else:
        raise ValueError(f"Unknown variable type {var_type}")

    # `xr.DataArray.astype` clears `.encoding`, so we capture it first and
    # restore it on the cast result. Without this, downstream code that
    # inspects encoding (e.g. to push CF scale-offset into a codec pipeline)
    # would see an empty encoding on every coarsened level.
    encoding = var_data.encoding
    cast_result: xr.DataArray = result.astype(var_data.dtype)
    cast_result.encoding = encoding
    return cast_result


def inject_missing_bands(
    dataset: xr.Dataset,
    dt_input: xr.DataTree,
    target_resolution: int,
    *,
    bands: set[str] | None = None,
) -> xr.Dataset:
    """Inject bands whose native resolution is finer than `target_resolution`.

    For each spectral band defined in `BAND_INFO` whose native resolution is
    finer than `target_resolution`, this function checks whether the band is
    already present in `dataset`.  If not, it looks for the band in the
    appropriate source group (e.g. `/measurements/reflectance/r10m`),
    downsamples it to the target grid using the type-aware resampling from
    `determine_variable_type`, and merges it into `dataset`.

    Args:
        dataset: The target-resolution dataset (e.g. the r20m or r60m
            reflectance group).
        dt_input: The full input DataTree (used to locate finer-resolution
            source bands).
        target_resolution: Target resolution in metres (e.g. 20 or 60).
        bands: If provided, only inject these band names.  If `None`
            (default), inject every eligible band from `BAND_INFO`.

    Returns:
        `dataset` with any missing finer-resolution bands added.
    """
    for band_name, info in BAND_INFO.items():
        if bands is not None and band_name not in bands:
            continue
        native_res = info.native_resolution  # type: ignore[attr-defined]
        if native_res >= target_resolution:
            continue
        if band_name in dataset.data_vars:
            continue

        source_path = f"/measurements/reflectance/r{native_res}m"
        if source_path not in dt_input.groups:
            continue

        source_ds = dt_input[source_path].to_dataset()
        if band_name not in source_ds.data_vars:
            continue

        band_src = source_ds[band_name]
        factor = target_resolution // native_res
        band_ds = _coarsen_variable(band_name, band_src, factor)

        # Replace coordinates with the target dataset's coordinates so that
        # xarray.Dataset.assign does not try to align on mismatched values.
        band_ds = xr.DataArray(
            band_ds.values,
            dims=band_ds.dims,
            coords={d: dataset.coords[d] for d in band_ds.dims if d in dataset.coords},
            attrs=band_ds.attrs,
            name=band_name,
        )

        # Preserve source encoding so downstream encoding logic can inspect it
        band_ds.encoding = band_src.encoding.copy()

        dataset = dataset.assign({band_name: band_ds})
        log.info(
            "Injected downsampled band from finer resolution",
            band=band_name,
            source=f"r{native_res}m",
            target=f"r{target_resolution}m",
            shape=band_ds.shape,
        )

    return dataset


def create_multiscale_from_datatree(
    dt_input: xr.DataTree,
    *,
    output_group: zarr.Group,
    enable_sharding: bool,
    spatial_chunk: int,
    crs: CRS | None = None,
    keep_scale_offset: bool,
    experimental_scale_offset_codec: bool = False,
) -> dict[str, dict]:
    """
    Create multiscale versions preserving original structure.
    Keeps all original groups, adds r120m, r360m, r720m downsampled versions.

    Args:
        dt_input: Input DataTree with original structure
        output_path: Base output path
        enable_sharding: Enable Zarr v3 sharding
        spatial_chunk: Spatial chunk size
        crs: Coordinate Reference System for datasets

    Returns:
        Dictionary of processed groups
    """
    processed_groups: dict[str, Any] = {}
    # The scale levels in the output data. 10, 20, 60 already exist in the source data.

    # Step 1: Copy all original groups as-is
    for group_path in dt_input.groups:
        if group_path == ".":
            continue

        # Skip the quicklook groups (`/quality/l1c_quicklook`, `/quality/l2a_quicklook`).
        # They duplicate RGB data that downstream consumers can derive from the
        # reflectance bands, so dropping them shrinks the output store and
        # speeds up conversion. See EOPF-Explorer/data-model#81.
        if group_path.startswith(("/quality/l1c_quicklook", "/quality/l2a_quicklook")):
            log.info("Skipping quicklook group", group_path=group_path)
            continue

        group_node = dt_input[group_path]

        # Skip parent groups that have children (only process leaf groups)
        if hasattr(group_node, "children") and len(group_node.children) > 0:
            continue

        dataset = group_node.to_dataset()

        # Skip empty groups
        if not dataset.data_vars:
            log.info("Skipping empty group: {}", group_path=group_path)
            continue

        log.info("Copying original group: {}", group_path=group_path)

        # Determine if this is a measurement-related resolution group
        group_name = group_path.split("/")[-1]
        is_measurement_group = (
            group_name.startswith("r")
            and group_name.endswith("m")
            and "/measurements/" in group_path
        )

        if is_measurement_group:
            # Inject bands whose native resolution is finer than this group's
            # (e.g. b08 native at 10m into r20m/r60m) so they propagate through
            # the full overview chain (r120m … r720m).
            if group_path.startswith("/measurements/reflectance/"):
                try:
                    group_resolution = int(group_name[1:-1])
                except ValueError:
                    group_resolution = 0
                if group_resolution > 10:
                    dataset = inject_missing_bands(
                        dataset,
                        dt_input,
                        group_resolution,
                        bands={"b08"},
                    )

            # Measurement groups: apply custom encoding
            encoding = create_measurements_encoding(
                dataset,
                spatial_chunk=spatial_chunk,
                enable_sharding=enable_sharding,
                keep_scale_offset=keep_scale_offset,
                experimental_scale_offset_codec=experimental_scale_offset_codec,
            )
            # convert float64 arrays to float32. `xr.DataArray.astype` clears
            # encoding, so we capture and restore it — downstream pyramid
            # levels are coarsened from this dataset and rely on the encoding
            # to drive CF packing / codec filter generation.
            for data_var in dataset.data_vars:
                if dataset[data_var].dtype in (np.dtype("<f8"), np.dtype(">f8")):
                    var_encoding = dataset[data_var].encoding
                    dataset[data_var] = dataset[data_var].astype("float32")
                    dataset[data_var].encoding = var_encoding
            # Clear _FillValue from the DataArray's own encoding to prevent
            # xarray from raising "Zarr does not support _FillValue in encoding".
            if not keep_scale_offset:
                for data_var in dataset.data_vars:
                    dataset[data_var].encoding.pop("_FillValue", None)
        else:
            # Non-measurement groups: preserve original encoding
            encoding = create_original_encoding(
                dataset,
                max_chunk=MAX_ORIGINAL_CHUNK,
                enable_sharding=enable_sharding,
            )

        ds_out = stream_write_dataset(
            dataset,
            path=group_path,
            group=output_group,
            encoding=encoding,
            enable_sharding=enable_sharding,
            crs=crs,
        )
        processed_groups[group_path] = ds_out
    # Step 2: Create downsampled resolution groups ONLY for measurements
    # Find all resolution-based groups under /measurements/ and organize by base path
    resolution_groups: dict[str, xr.Dataset] = {}
    base_path = "/measurements/reflectance"
    for group_path in processed_groups:
        # Only process groups under /measurements/reflectance
        if not group_path.startswith(base_path):
            continue

        group_name = group_path.split("/")[-1]
        if group_name in ["r10m", "r20m", "r60m"]:
            resolution_groups[group_name] = processed_groups[group_path]

    scale_levels = tuple(pyramid_levels.values())

    # iterate over source, dest pairs: (60, 120), (120, 360), ...
    for source_level, dest_level in pairwise(scale_levels[2:]):
        dest_level_name = f"r{dest_level}m"
        dest_level_path = f"{base_path}/{dest_level_name}"

        source_ds = resolution_groups[f"r{source_level}m"]

        downsample_factor = dest_level // source_level
        log.info("Creating level with resolution", level=dest_level_name, resolution=dest_level)

        # Create downsampled dataset
        downsampled_dataset = create_downsampled_resolution_group(
            source_ds, factor=downsample_factor
        )

        log.info("Writing level to path", level=dest_level_name, output_path=dest_level_path)

        # Create encoding
        encoding = create_measurements_encoding(
            downsampled_dataset,
            spatial_chunk=spatial_chunk,
            enable_sharding=enable_sharding,
            keep_scale_offset=keep_scale_offset,
            experimental_scale_offset_codec=experimental_scale_offset_codec,
        )

        # Strip _FillValue from DataArray encoding for downsampled levels too
        if not keep_scale_offset:
            for data_var in downsampled_dataset.data_vars:
                downsampled_dataset[data_var].encoding.pop("_FillValue", None)

        # Write dataset
        ds_out = stream_write_dataset(
            downsampled_dataset,
            path=dest_level_path,
            group=output_group,
            encoding=encoding,
            enable_sharding=enable_sharding,
            crs=crs,
        )

        # Store results
        processed_groups[dest_level_path] = ds_out
        resolution_groups[dest_level_name] = ds_out

    # Step 3: Add multiscales metadata to parent groups
    log.info("Adding multiscales metadata to parent groups")

    # Get the parent group (it was created when writing the resolution groups).
    # `output_group[base_path]` is typed `Array | Group`; `base_path` always
    # addresses a group (the reflectance parent), so verify that at runtime.
    parent_group = output_group[base_path]
    if not isinstance(parent_group, zarr.Group):
        raise TypeError(
            f"expected a zarr.Group at {base_path!r}, got {type(parent_group).__name__}"
        )

    add_multiscales_metadata_to_parent(
        parent_group,
        resolution_groups,
    )
    processed_groups[base_path] = None

    return processed_groups


def create_measurements_encoding(
    dataset: xr.Dataset,
    *,
    spatial_chunk: int,
    enable_sharding: bool = True,
    keep_scale_offset: bool = True,
    experimental_scale_offset_codec: bool = False,
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

        # Whether to inject a CF _FillValue attribute for xarray issue #11345.
        # The injection itself happens after sanitize_array_attrs below, which
        # would otherwise strip it.
        inject_nan_fillvalue = False

        if experimental_scale_offset_codec and not keep_scale_offset:
            # Push CF scale-offset into the zarr codec pipeline instead of
            # decoding to float. The data stays as packed integers on disk,
            # but zarr transparently decodes on read.
            scale_factor = var_data.encoding.get("scale_factor")
            add_offset = var_data.encoding.get("add_offset")
            packed_dtype = var_data.encoding.get("dtype")

            if scale_factor is not None and add_offset is not None and packed_dtype is not None:
                from eopf_geozarr.codecs.scale_offset import scale_offset_from_cf

                so_codec = scale_offset_from_cf(
                    scale_factor=float(scale_factor), add_offset=float(add_offset)
                )
                # CastValue refuses to cast NaN to integer without an explicit
                # mapping, so we need a packed-dtype sentinel for NaN. Prefer
                # the source's existing `_FillValue` (it already encodes the
                # "no data" semantic via xarray's CF mask_and_scale loop), and
                # fall back to the dtype's lowest representable integer.
                packed_np_dtype = np.dtype(packed_dtype)
                source_fill = var_data.encoding.get("_FillValue")
                if source_fill is not None:
                    nan_sentinel = int(source_fill)
                else:
                    nan_sentinel = int(np.iinfo(packed_np_dtype).min)
                cv_codec = CastValue(
                    data_type=packed_np_dtype.name,
                    rounding="nearest-even",
                    scalar_map={
                        "encode": [("NaN", nan_sentinel)],
                        "decode": [(nan_sentinel, "NaN")],
                    },
                )
                var_encoding["filters"] = (so_codec, cv_codec)

            # Strip CF keys and `filters` from `keep_keys` — the codecs handle
            # encoding/decoding now, and we don't want the forward-propagation
            # loop below to overwrite our freshly-set filters with whatever was
            # on the source variable.
            keep_keys = keep_keys - CF_SCALE_OFFSET_KEYS - {"_FillValue", "filters"}
            var_encoding["fill_value"] = "NaN"
            inject_nan_fillvalue = True
        elif not keep_scale_offset:
            # When stripping scale/offset, also strip _FillValue since the original
            # _FillValue is in raw integer units and meaningless for decoded float data.
            keep_keys = keep_keys - CF_SCALE_OFFSET_KEYS - {"_FillValue"}
            # Set zarr fill_value to NaN so nodata regions are correctly identified
            # as transparent by zarr-aware viewers (e.g. OpenLayers GeoZarr source).
            # xarray's zarr backend uses "fill_value" (no underscore) in encoding
            # to set the zarr-level fill value, distinct from "_FillValue" which
            # controls CF-convention attribute masking.
            var_encoding["fill_value"] = "NaN"
            inject_nan_fillvalue = True

        for key in keep_keys:
            if key in var_data.encoding:
                var_encoding[key] = var_data.encoding[key]

        if len(set(var_data.encoding.keys()) - XARRAY_ENCODING_KEYS) > 0:
            log.warning(
                "Unknown encoding keys in %s: %s",
                var_name,
                set(var_data.encoding.keys()) - XARRAY_ENCODING_KEYS,
            )
        # Sanitize source-only attributes (replace dict — ``.update`` cannot
        # remove keys, so stale ``_eopf_attrs`` / ``dtype`` / ``valid_*`` would
        # otherwise leak into the output).
        is_float = np.issubdtype(var_data.dtype, np.floating)
        var_data.attrs = utils.sanitize_array_attrs(var_data.attrs, is_decoded_float=is_float)
        if inject_nan_fillvalue:
            # Inject CF _FillValue attribute for xarray issue #11345
            var_data.attrs["_FillValue"] = np.nan
        encoding[str(var_name)] = var_encoding

    # Add coordinate encoding and sanitize coord attrs (e.g. drop
    # ``_eopf_attrs`` from datetime coords carried in from the source).
    for coord_name, coord_data in dataset.coords.items():
        coord_data.attrs = utils.sanitize_array_attrs(coord_data.attrs)
        encoding[str(coord_name)] = {"compressors": []}  # type: ignore[typeddict-item]

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


def add_multiscales_metadata_to_parent(
    group: zarr.Group,
    res_groups: Mapping[str, xr.Dataset],
) -> None:
    """Add GeoZarr-compliant multiscales metadata to parent group.

    Returns ``None`` in all cases: metadata is written directly to ``group``
    via ``group.attrs.update`` rather than returned as a DataTree.
    """
    # Sort by resolution (finest to coarsest)
    res_order = {
        "r10m": 10,
        "r20m": 20,
        "r60m": 60,
        "r120m": 120,
        "r360m": 360,
        "r720m": 720,
    }

    all_resolutions = sorted(set(res_groups.keys()), key=lambda x: res_order.get(x, 999))

    if len(all_resolutions) < 2:
        log.info(
            "Skipping {} - only one resolution available",
            base_path=group.path,
        )
        return

    # Get CRS and bounds from first available dataset (load from output path)
    first_res = all_resolutions[0]
    first_dataset = res_groups[first_res]

    # Get CRS and bounds
    native_crs = first_dataset.rio.crs if hasattr(first_dataset, "rio") else None
    if native_crs is None:
        log.info("No CRS found, skipping multiscales metadata", base_path=group.path)
        return

    # Calculate bounds directly from coordinates for consistency with the data arrays
    if "x" not in first_dataset.coords or "y" not in first_dataset.coords:
        log.error(
            "Missing x/y coordinates in dataset, cannot determine bounds", base_path=group.path
        )
        return

    x_coords = first_dataset.x.values
    y_coords = first_dataset.y.values
    native_bounds = (
        float(x_coords.min()),
        float(y_coords.min()),
        float(x_coords.max()),
        float(y_coords.max()),
    )

    # Create overview_levels structure following the multiscales v1.0 specification
    overview_levels: list[OverviewLevelJSON] = []
    for res_name in all_resolutions:
        # Use resolution order for consistent scale calculations
        res_meters = res_order[res_name]

        dataset = res_groups[res_name]

        # Defensive guard retained for runtime safety even though the typed
        # contract (Mapping[str, xr.Dataset]) means mypy proves it unreachable.
        if dataset is None:
            continue

        # Get first data variable to extract dimensions
        first_var = next(iter(dataset.data_vars.values()))
        height, width = first_var.shape[-2:]

        transform = _preferred_spatial_transform(dataset)

        # Calculate zoom level (higher resolution = higher zoom)
        tile_width = 256
        zoom_for_width = max(0, int(np.ceil(np.log2(width / tile_width))))
        zoom_for_height = max(0, int(np.ceil(np.log2(height / tile_width))))
        zoom = max(zoom_for_width, zoom_for_height)

        # Calculate relative scale and translation vs parent resolution
        finest_res_meters = res_order[all_resolutions[0]]

        # Fix for issue #114: Translation values should be 0
        relative_translation = 0.0

        # Calculate proper relative scale based on actual parent-child dimension ratios
        if res_name == all_resolutions[0]:  # Base resolution
            relative_scale = 1.0
        else:
            # Define derivation chain to find parent resolution
            derivation_chain = {
                "r10m": None,
                "r20m": "r10m",
                "r60m": "r10m",
                "r120m": "r60m",
                "r360m": "r120m",
                "r720m": "r360m",
            }

            parent_res = derivation_chain.get(res_name)
            if parent_res and parent_res in res_groups:
                # Get actual dimensions of parent and child
                parent_dataset = res_groups[parent_res]
                parent_var = next(iter(parent_dataset.data_vars.values()))
                parent_height, parent_width = parent_var.shape[-2:]

                # Current (child) dimensions
                child_height, child_width = height, width

                # Calculate actual scale ratio based on dimensions
                # Use the larger of the two ratios to be conservative
                scale_x = parent_width / child_width if child_width > 0 else 1.0
                scale_y = parent_height / child_height if child_height > 0 else 1.0
                relative_scale = max(scale_x, scale_y)

                log.info(
                    "Calculated dynamic scale ratio",
                    level=res_name,
                    parent=parent_res,
                    parent_dims=(parent_height, parent_width),
                    child_dims=(child_height, child_width),
                    scale_x=scale_x,
                    scale_y=scale_y,
                    relative_scale=relative_scale,
                )
            else:
                # Fallback to absolute resolution ratio
                relative_scale = res_meters / finest_res_meters
                log.warning(
                    "Using fallback scale calculation",
                    level=res_name,
                    relative_scale=relative_scale,
                )

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

    if len(overview_levels) < 2:
        log.info("    Could not create overview levels for {}", base_path=group.path)
        return

    layout: list[zcm.ScaleLevel] | MISSING = MISSING

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
            derived_from = derivation_chain.get(asset, str(all_resolutions[0]))
            multiscale_transform = zcm.Transform(
                scale=(overview_level["scale_relative"],) * 2,
                translation=(overview_level["translation_relative"],) * 2,
            )
            scale_level_data["derived_from"] = derived_from
            scale_level_data["transform"] = multiscale_transform

        # Add spatial properties
        assert "spatial_shape" in overview_level  # always populated by the producer above
        scale_level_data["spatial:shape"] = overview_level["spatial_shape"]
        if "spatial_transform" in overview_level:
            spatial_transform = overview_level["spatial_transform"]
            # Only add spatial_transform if we have valid transform data (not all zeros)
            if spatial_transform is not None and not all(t == 0 for t in spatial_transform):
                scale_level_data["spatial:transform"] = spatial_transform

        scale_level = zcm.ScaleLevel(**scale_level_data)
        layout.append(scale_level)

    # Validate + serialize the multiscales block via the project model (which
    # also covers the ZCM/TMS duality), then hand all three conventions to
    # zarr-cm, which validates each and emits the matching CMOs in order
    # (multiscales, spatial, proj).
    multiscales_data = cast(
        "MultiscalesAttrs",
        MultiscaleMeta(layout=tuple(layout), resampling_method="average").model_dump(),
    )

    attrs_to_write: dict[str, Any] = {}
    if native_crs and native_bounds:
        attrs_to_write.update(
            utils.build_convention_attrs(
                multiscales=multiscales_data,
                spatial={
                    "spatial:dimensions": ["y", "x"],
                    "spatial:bbox": list(native_bounds),  # [xmin, ymin, xmax, ymax]
                    "spatial:registration": "pixel",
                },
                crs=native_crs,
            )
        )

    # Write attributes directly to the zarr group
    group.attrs.update(attrs_to_write)

    log.info("Added %s multiscale levels to %s", len(overview_levels), group.path)


def create_original_encoding(
    dataset: xr.Dataset,
    *,
    max_chunk: int,
    enable_sharding: bool,
) -> dict[str, XarrayDataArrayEncoding]:
    """Write a group preserving its original chunking and encoding.

    Source chunk dims larger than ``max_chunk`` are clamped to an aligned
    size; when ``enable_sharding`` is true, clamped variables get a
    whole-array shard so each array remains a single storage object.
    """
    from zarr.codecs import BloscCodec

    # Simple encoding that preserves original structure
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0)
    encoding: dict[str, XarrayDataArrayEncoding] = {}

    for var_name in dataset.data_vars:
        # start with the original encoding
        var_data = dataset.data_vars[var_name]
        var_encoding: XarrayDataArrayEncoding = {}
        var_encoding["compressors"] = (compressor,)
        for key in XARRAY_ENCODING_KEYS - {"compressors", "fill_value"}:
            if key in var_data.encoding:
                var_encoding[key] = var_data.encoding[key]
        # Clamp oversized source chunks (e.g. cpm_v270 whole-array aot/wvp)
        # instead of propagating them into the output store.
        chunks = var_encoding.get("chunks")
        if chunks:
            clamped = tuple(
                calculate_aligned_chunk_size(dim, max_chunk) if chunk > max_chunk else chunk
                for dim, chunk in zip(var_data.shape, chunks, strict=True)
            )
            if clamped != tuple(chunks):
                log.info(
                    "Clamping oversized source chunks for %s: %s -> %s",
                    var_name,
                    tuple(chunks),
                    clamped,
                )
                var_encoding["chunks"] = clamped
                # preferred_chunks hints at the old grid — drop it
                var_encoding.pop("preferred_chunks", None)
                if enable_sharding:
                    # keep one storage object per array (same policy as
                    # create_measurements_encoding): whole-array shard with
                    # the clamped chunks as inner tiles
                    var_encoding["shards"] = calculate_simple_shard_dimensions(
                        var_data.shape, clamped
                    )
                else:
                    # a source shard shape may no longer be a multiple of the
                    # clamped chunks
                    var_encoding.pop("shards", None)
        # Set the zarr-level `fill_value` explicitly rather than letting xarray
        # decide — different xarray versions infer different defaults from the
        # variable's `_FillValue`. See `explicit_fill_value` for the rationale.
        fv = utils.explicit_fill_value(var_data)
        if fv is not utils.UNSET:
            var_encoding["fill_value"] = fv
        if len(set(var_data.encoding.keys()) - XARRAY_ENCODING_KEYS) > 0:
            log.warning(
                "Unknown encoding keys in %s: %s",
                var_name,
                set(var_data.encoding.keys()) - XARRAY_ENCODING_KEYS,
            )
        # Sanitize source-only attributes (replace dict — ``.update`` cannot
        # remove keys, so stale ``_eopf_attrs`` would otherwise leak through).
        var_data.attrs = utils.sanitize_array_attrs(var_data.attrs)
        encoding[str(var_name)] = var_encoding

    for coord_name, coord_data in dataset.coords.items():
        coord_data.attrs = utils.sanitize_array_attrs(coord_data.attrs)
        encoding[str(coord_name)] = {"compressors": None}

    return encoding


def create_downsampled_resolution_group(source_dataset: xr.Dataset, factor: int) -> xr.Dataset:
    """Create a downsampled version of a dataset by given factor."""
    if not source_dataset or len(source_dataset.data_vars) == 0:
        return xr.Dataset()

    # Get reference dimensions
    ref_var = next(iter(source_dataset.data_vars.values()))
    if ref_var.ndim < 2:
        return xr.Dataset()

    current_height, current_width = ref_var.shape[-2:]
    target_height = current_height // factor
    target_width = current_width // factor

    if target_height < 1 or target_width < 1:
        return xr.Dataset()

    # Downsample all variables using existing lazy operations
    lazy_vars = {}
    for var_name, var_data in source_dataset.data_vars.items():
        if var_data.ndim < 2:
            continue
        lazy_vars[var_name] = _coarsen_variable(str(var_name), var_data, factor)

    if not lazy_vars:
        return xr.Dataset()

    # Create dataset with lazy variables and coordinates
    return xr.Dataset(lazy_vars, attrs=source_dataset.attrs)


def subsample_2(a: xr.DataArray, axis: tuple[int, ...] | None = None) -> xr.DataArray:
    if axis is None:
        return a[((0,) * a.ndim)]
    indexer = [0 if i in axis else slice(None) for i in range(a.ndim)]
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
            str(coord_name): coord_data
            for coord_name, coord_data in level_2_dataset.coords.items()
            if coord_name not in ["x", "y"]
        }
    )

    return coords


def create_lazy_downsample_operation_from_existing(
    source_data: xr.DataArray, target_height: int, target_width: int
) -> xr.DataArray:
    """Create lazy downsampling operation from existing data."""

    @delayed
    def downsample_operation() -> Any:
        var_type = determine_variable_type(str(source_data.name), source_data)
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
        # The zarr backend accepts a zarr `Store` here at runtime, but xarray's
        # `open_dataset` stub only types the first arg as path/buffer/datastore.
        return xr.open_dataset(
            group.store,  # type: ignore[arg-type]
            engine="zarr",
            chunks={},
            decode_coords="all",
            group=path,
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

    log.info("✅ Streaming write complete for dataset {}", dataset_path=path)
    return dataset


def write_geo_metadata(
    dataset: xr.Dataset,
    grid_mapping_var_name: str = "spatial_ref",
    crs: CRS | None = None,
) -> None:
    """
    Write geographic metadata to the dataset.

    Args:
        dataset: Dataset to write metadata to
        grid_mapping_var_name: Name for grid mapping variable
        crs: Coordinate Reference System to use (if None, attempts to detect from dataset)
    """
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
            x_coords = dataset.coords["x"].values
            y_coords = dataset.coords["y"].values
            x_min, x_max = float(x_coords.min()), float(x_coords.max())
            y_min, y_max = float(y_coords.min()), float(y_coords.max())
            spatial_data["spatial:bbox"] = [x_min, y_min, x_max, y_max]

            spatial_transform = _preferred_spatial_transform(dataset)

            # Only add spatial:transform if we have valid transform data (not all zeros)
            if spatial_transform is not None and not all(t == 0 for t in spatial_transform):
                spatial_data["spatial:transform"] = list(spatial_transform)

            # Add spatial shape if data variables exist
            if dataset.data_vars:
                first_var = next(iter(dataset.data_vars.values()))
                if first_var.ndim >= 2:
                    height, width = first_var.shape[-2:]
                    spatial_data["spatial:shape"] = [height, width]

        # Build validated spatial + proj convention attrs (data + CMOs) via zarr-cm
        dataset.attrs.update(utils.build_convention_attrs(spatial=spatial_data, crs=crs))


def rechunk_dataset_for_encoding(
    dataset: xr.Dataset, encoding: dict[str, XarrayDataArrayEncoding]
) -> xr.Dataset:
    """
    Rechunk dataset variables to align with sharding dimensions when sharding is enabled.

    When using Zarr v3 sharding, Dask chunks must align with shard dimensions to avoid
    checksum validation errors.
    """
    rechunked_vars: dict[Hashable, xr.DataArray] = {}

    for var_name, var_data in dataset.data_vars.items():
        if str(var_name) in encoding:
            var_encoding = encoding[str(var_name)]

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
