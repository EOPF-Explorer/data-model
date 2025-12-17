"""
Streaming multiscale pyramid creation for optimized S2 structure.
Uses lazy evaluation to minimize memory usage during dataset preparation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import structlog
import xarray as xr
from dask import delayed
from dask.array import from_delayed
from pydantic.experimental.missing_sentinel import MISSING
from pyproj import CRS

from eopf_geozarr.conversion import fs_utils
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
    XARRAY_ENCODING_KEYS,
    XarrayDataArrayEncoding,
)
from eopf_geozarr.s2_optimization.common import DISTRIBUTED_AVAILABLE

from .s2_resampling import determine_variable_type, downsample_variable

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

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


def create_multiscale_from_datatree(
    dt_input: xr.DataTree,
    output_path: str,
    enable_sharding: bool,
    spatial_chunk: int,
    crs: CRS | None = None,
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
    processed_groups = {}

    # Step 1: Copy all original groups as-is
    for group_path in dt_input.groups:
        if group_path == ".":
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

        output_group_path = f"{output_path}{group_path}"

        # Determine if this is a measurement-related resolution group
        group_name = group_path.split("/")[-1]
        is_measurement_group = (
            group_name.startswith("r")
            and group_name.endswith("m")
            and "/measurements/" in group_path
        )

        if is_measurement_group:
            # Measurement groups: apply custom encoding
            encoding = create_measurements_encoding(
                dataset, spatial_chunk=spatial_chunk, enable_sharding=enable_sharding
            )
        else:
            # Non-measurement groups: preserve original chunking
            encoding = create_original_encoding(dataset)
        ds_out = stream_write_dataset(
            dataset,
            output_group_path,
            encoding,
            enable_sharding=enable_sharding,
            crs=crs,
        )
        processed_groups[group_path] = ds_out

    # Step 2: Create downsampled resolution groups ONLY for measurements
    # Find all resolution-based groups under /measurements/ and organize by base path
    resolution_groups = {}
    base_path = "/measurements/reflectance"
    for group_path in processed_groups:
        # Only process groups under /measurements/reflectance
        if not group_path.startswith(base_path):
            continue

        group_name = group_path.split("/")[-1]
        if group_name in ["r10m", "r20m", "r60m"]:
            resolution_groups[group_name] = processed_groups[group_path]

    # Find the coarsest resolution (r60m > r20m > r10m)
    source_dataset = None
    source_resolution = None

    for res in ["r60m", "r20m", "r10m"]:
        if res in resolution_groups:
            source_dataset = resolution_groups[res]
            source_resolution = int(res[1:-1])  # Extract number
            break

    if not source_dataset or source_resolution is None:
        log.info("No source resolution found for downsampling, skipping downsampled levels")
        return processed_groups  # Stop processing if no valid source dataset is found

    log.info(
        "Creating downsampled versions",
        source_dataset=source_dataset,
        source_resolution=source_resolution,
    )

    # Create r120m
    try:
        r120m_path = f"{base_path}/r120m"
        factor = 120 // source_resolution
        log.info("Creating r120m with factor {}", factor=factor)

        r120m_dataset = create_downsampled_resolution_group(source_dataset, factor=factor)
        if r120m_dataset and len(r120m_dataset.data_vars) > 0:
            output_path_120 = f"{output_path}{r120m_path}"
            log.info("Writing r120m to {}", output_path_120=output_path_120)
            encoding_120 = create_measurements_encoding(r120m_dataset, spatial_chunk=spatial_chunk)
            ds_120 = stream_write_dataset(
                r120m_dataset,
                output_path_120,
                encoding_120,
                enable_sharding=enable_sharding,
                crs=crs,
            )
            processed_groups[r120m_path] = ds_120
            resolution_groups["r120m"] = ds_120

            # Create r360m from r120m
            try:
                r360m_path = f"{base_path}/r360m"
                log.info("Creating r360m with factor 3")

                r360m_dataset = create_downsampled_resolution_group(r120m_dataset, factor=3)

                if r360m_dataset and len(r360m_dataset.data_vars) > 0:
                    output_path_360 = f"{output_path}{r360m_path}"
                    log.info("Writing r360m to {}", output_path_360=output_path_360)
                    encoding_360 = create_measurements_encoding(
                        r360m_dataset, spatial_chunk=spatial_chunk
                    )
                    ds_360 = stream_write_dataset(
                        r360m_dataset,
                        output_path_360,
                        encoding_360,
                        enable_sharding=enable_sharding,
                        crs=crs,
                    )
                    processed_groups[r360m_path] = ds_360
                    resolution_groups["r360m"] = ds_360

                    # Create r720m from r360m
                    try:
                        r720m_path = f"{base_path}/r720m"
                        log.info("    Creating r720m with factor 2")

                        r720m_dataset = create_downsampled_resolution_group(r360m_dataset, factor=2)

                        if r720m_dataset and len(r720m_dataset.data_vars) > 0:
                            output_path_720 = f"{output_path}{r720m_path}"

                            log.info(
                                "    Writing r720m to {}",
                                output_path_720=output_path_720,
                            )
                            encoding_720 = create_measurements_encoding(
                                r720m_dataset,
                                spatial_chunk=spatial_chunk,
                                enable_sharding=enable_sharding,
                            )
                            ds_720 = stream_write_dataset(
                                r720m_dataset,
                                output_path_720,
                                encoding_720,
                                enable_sharding=enable_sharding,
                                crs=crs,
                            )
                            processed_groups[r720m_path] = ds_720
                            resolution_groups["r720m"] = ds_720
                        else:
                            log.info("    r720m dataset is empty, skipping")
                    except Exception as e:
                        log.warning(
                            "Could not create r720m",
                            base_path=base_path,
                            error=str(e),
                        )
                else:
                    log.info("    r360m dataset is empty, skipping")
            except Exception as e:
                log.warning("Could not create r360m for {}: {}", base_path=base_path, e=e)
            # Track r120m for multiscales if created

            log.info("Tracking r120m for multiscales metadata")
        else:
            log.info("r120m dataset is empty, skipping")
    except Exception as e:
        log.warning("Could not create r120m for {}: {}", base_path=base_path, e=e)

    # Step 3: Add multiscales metadata to parent groups
    log.info("Adding multiscales metadata to parent groups")

    dt_multiscale = add_multiscales_metadata_to_parent(
        output_path,
        base_path,
        resolution_groups,
        multiscales_flavor={"ogc_tms", "experimental_multiscales_convention"},
    )
    processed_groups[base_path] = dt_multiscale

    return processed_groups


def create_measurements_encoding(
    dataset: xr.Dataset, *, spatial_chunk: int, enable_sharding: bool = True
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

        # Forward-propagate the existing encoding
        for key in XARRAY_ENCODING_KEYS - {"compressors", "shards", "chunks"}:
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


def add_multiscales_metadata_to_parent(
    output_path: str,
    base_path: str,
    res_groups: Mapping[str, xr.Dataset],
    multiscales_flavor: set[MultiscalesFlavor] | None = None,
) -> xr.DataTree:
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

    all_resolutions = sorted(set(res_groups.keys()), key=lambda x: res_order.get(x, 999))

    if len(all_resolutions) < 2:
        log.info(
            "Skipping {} - only one resolution available",
            base_path=base_path,
        )
        return None

    # Get CRS and bounds from first available dataset (load from output path)
    first_res = all_resolutions[0]
    first_dataset = res_groups[first_res]

    # Get CRS and bounds
    native_crs = first_dataset.rio.crs if hasattr(first_dataset, "rio") else None
    if native_crs is None:
        log.info("No CRS found, skipping multiscales metadata", base_path=base_path)
        return None

    native_bounds = None
    if hasattr(first_dataset, "rio"):
        try:
            native_bounds = first_dataset.rio.bounds()
        except (AttributeError, TypeError):
            # Try alternative method or construct from coordinates
            try:
                x_coords = first_dataset.x.values
                y_coords = first_dataset.y.values
                native_bounds = (x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max())
            except Exception:
                pass

    if native_bounds is None:
        log.info(
            "No bounds found, skipping multiscales metadata",
            base_path=base_path,
        )
        return None

    # Create overview_levels structure following the multiscales v1.0 specification
    overview_levels: list[OverviewLevelJSON] = []
    for res_name in all_resolutions:
        # Use resolution order for consistent scale calculations
        res_meters = res_order[res_name]

        dataset = res_groups[res_name]

        if dataset is None:
            continue

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
                transform = list(rio_transform)[:6]  # Get 6 coefficients
                log.info("Got transform from rio accessor", transform=transform, level=res_name)
            except (AttributeError, TypeError) as e:
                log.warning(
                    "Could not get transform from rio accessor", error=str(e), level=res_name
                )

        if transform is None or all(t == 0 for t in transform):
            # Fallback: construct from grid spacing and bounds
            try:
                if "x" in dataset.coords and "y" in dataset.coords:
                    # Use coordinate arrays to calculate spacing
                    x_coords = dataset.coords["x"].values
                    y_coords = dataset.coords["y"].values

                    if len(x_coords) > 1 and len(y_coords) > 1:
                        pixel_size_x = float(np.abs(x_coords[1] - x_coords[0]))
                        pixel_size_y = float(np.abs(y_coords[1] - y_coords[0]))
                        x_min = float(x_coords.min())
                        y_max = float(y_coords.max())
                        transform = [pixel_size_x, 0.0, x_min, 0.0, -pixel_size_y, y_max]
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
            except Exception as e:
                log.warning(
                    "Error calculating transform from coordinates", error=str(e), level=res_name
                )

        # Calculate zoom level (higher resolution = higher zoom)
        tile_width = 256
        zoom_for_width = max(0, int(np.ceil(np.log2(width / tile_width))))
        zoom_for_height = max(0, int(np.ceil(np.log2(height / tile_width))))
        zoom = max(zoom_for_width, zoom_for_height)

        # Calculate relative scale and translation vs first resolution
        finest_res_meters = res_order[all_resolutions[0]]
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

        layout_entry = {
            "level": res_name,  # Use string-based level name
            "zoom": zoom,
            "width": width,
            "height": height,
            "translation_relative": relative_translation,
            "scale_absolute": res_meters,
            "scale_relative": relative_scale,
            "chunks": chunks,
            "spatial_shape": [height, width],
        }

        # Only add spatial_transform if we have valid transform data
        if transform is not None and not all(t == 0 for t in transform):
            layout_entry["spatial_transform"] = transform

        overview_levels.append(cast("OverviewLevelJSON", layout_entry))

    if len(overview_levels) < 2:
        log.info("    Could not create overview levels for {}", base_path=base_path)
        return None

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

            if i > 0:  # Not the first (base) resolution
                derived_from = derivation_chain.get(asset, str(all_resolutions[0]))
                multiscale_transform = zcm.Transform(
                    scale=(overview_level["scale_relative"],) * 2,
                    translation=(overview_level["translation_relative"],) * 2,
                )
                scale_level = zcm.ScaleLevel(
                    asset=asset, derived_from=derived_from, transform=multiscale_transform
                )
            else:
                scale_level = zcm.ScaleLevel(asset=asset)

            # Add spatial properties manually after creation (using extra fields)
            setattr(scale_level, "spatial:shape", overview_level["spatial_shape"])
            if "spatial_transform" in overview_level:
                setattr(scale_level, "spatial:transform", overview_level["spatial_transform"])

            layout.append(scale_level)
    # Create convention metadata for all three conventions
    multiscale_attrs = MultiscaleGroupAttrs(
        zarr_conventions=(
            zcm.MultiscaleConventionMetadata(),
            SpatialConventionMetadata(),
            ProjConventionMetadata(),
        ),
        multiscales=MultiscaleMeta(
            version="1.0.0",  # Add required version field
            layout=layout,
            resampling_method="average",
            tile_matrix_set=tile_matrix_set,
            tile_matrix_limits=tile_matrix_limits,
        ),
    )

    # Create parent group path
    parent_group_path = f"{output_path}{base_path}"
    dt_multiscale = xr.DataTree()
    for res in all_resolutions:
        dt_multiscale[res] = xr.DataTree()

    # Add multiscale attributes
    dt_multiscale.attrs.update(multiscale_attrs.model_dump())

    # Add spatial and proj attributes at group level following specifications
    if native_crs and native_bounds:
        # Add spatial convention attributes
        dt_multiscale.attrs["spatial:dimensions"] = ["y", "x"]  # Required field
        dt_multiscale.attrs["spatial:bbox"] = list(native_bounds)  # [xmin, ymin, xmax, ymax]
        dt_multiscale.attrs["spatial:registration"] = "pixel"  # Default registration type

        # Add proj convention attributes
        if hasattr(native_crs, "to_epsg") and native_crs.to_epsg():
            dt_multiscale.attrs["proj:code"] = f"EPSG:{native_crs.to_epsg()}"
        elif hasattr(native_crs, "to_wkt"):
            dt_multiscale.attrs["proj:wkt2"] = native_crs.to_wkt()

    dt_multiscale.to_zarr(
        parent_group_path,
        mode="a",
        consolidated=False,
        zarr_format=3,
    )

    log.info("Added %s multiscale levels to %s", len(overview_levels), base_path)

    return dt_multiscale


def create_original_encoding(dataset: xr.Dataset) -> dict[str, XarrayDataArrayEncoding]:
    """Write a group preserving its original chunking and encoding."""
    from zarr.codecs import BloscCodec

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
        var_typ = determine_variable_type(var_name, var_data)
        if var_typ == "quality_mask":
            lazy_downsampled = var_data.coarsen({"x": factor, "y": factor}, boundary="trim").max()
        elif var_typ == "reflectance":
            lazy_downsampled = var_data.coarsen({"x": factor, "y": factor}, boundary="trim").mean()
        elif var_typ == "classification":
            lazy_downsampled = var_data.coarsen({"x": factor, "y": factor}, boundary="trim").reduce(
                subsample_2
            )
        elif var_typ == "probability":
            lazy_downsampled = var_data.coarsen({"x": factor, "y": factor}, boundary="trim").mean()
        else:
            raise ValueError(f"Unknown variable type {var_typ}")

        # preserve encoding
        lazy_downsampled.encoding = var_data.encoding
        # Ensure that dtype is preserved
        lazy_vars[var_name] = lazy_downsampled.astype(var_data.dtype)

    if not lazy_vars:
        return xr.Dataset()

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
    dataset_path: str,
    encoding: dict[str, XarrayDataArrayEncoding],
    *,
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
    if fs_utils.path_exists(dataset_path):
        log.info(
            "Level path {} already exists. Skipping write.",
            dataset_path=dataset_path,
        )
        return xr.open_dataset(
            dataset_path,
            engine="zarr",
            chunks={},
            decode_coords="all",
        )

    log.info("Streaming computation and write to {}", dataset_path=dataset_path)
    log.info("Variables", variables=list(dataset.data_vars.keys()))

    # Rechunk dataset to align with encoding
    dataset = rechunk_dataset_for_encoding(dataset, encoding)

    # Add the geo metadata before writing for
    # - /measurements/ groups
    # - /quality/ groups
    if "/measurements/" in dataset_path or "/quality/" in dataset_path:
        write_geo_metadata(dataset, crs=crs)

    # Write with streaming computation and progress tracking
    # The to_zarr operation will trigger all lazy computations
    write_job = dataset.to_zarr(
        dataset_path,
        mode="w",
        consolidated=False,
        zarr_format=3,
        encoding=encoding,
        compute=False,  # Create job first for progress tracking
    )
    write_job = write_job.persist()

    if DISTRIBUTED_AVAILABLE:
        try:
            import distributed

            distributed.progress(write_job, notebook=False)
        except Exception as e:
            log.warning("Could not display progress bar: {}", e=e)
            write_job.compute()
    else:
        log.info("Writing zarr file...")
        write_job.compute()

    log.info("✅ Streaming write complete for dataset {}", dataset_path=dataset_path)
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

        # Add spatial convention attributes
        dataset.attrs["spatial:dimensions"] = ["y", "x"]  # Required field
        dataset.attrs["spatial:registration"] = "pixel"  # Default registration type

        # Calculate and add spatial bbox if coordinates are available
        try:
            if "x" in dataset.coords and "y" in dataset.coords:
                x_coords = dataset.coords["x"].values
                y_coords = dataset.coords["y"].values
                x_min, x_max = float(x_coords.min()), float(x_coords.max())
                y_min, y_max = float(y_coords.min()), float(y_coords.max())
                dataset.attrs["spatial:bbox"] = [x_min, y_min, x_max, y_max]

                # Calculate spatial transform (affine transformation)
                if hasattr(dataset, "rio") and hasattr(dataset.rio, "transform"):
                    try:
                        rio_transform = dataset.rio.transform
                        if callable(rio_transform):
                            rio_transform = rio_transform()
                        dataset.attrs["spatial:transform"] = list(rio_transform)[:6]
                    except (AttributeError, TypeError):
                        # Fallback: construct from coordinate spacing
                        pixel_size_x = float(get_grid_spacing(dataset, ("x",))[0])
                        pixel_size_y = float(get_grid_spacing(dataset, ("y",))[0])
                        dataset.attrs["spatial:transform"] = [
                            pixel_size_x,
                            0.0,
                            x_min,
                            0.0,
                            -pixel_size_y,
                            y_max,
                        ]

                # Add spatial shape if data variables exist
                if dataset.data_vars:
                    first_var = next(iter(dataset.data_vars.values()))
                    if first_var.ndim >= 2:
                        height, width = first_var.shape[-2:]
                        dataset.attrs["spatial:shape"] = [height, width]
        except Exception as e:
            log.warning("Could not calculate spatial metadata: {}", e=e)

        # Add proj convention attributes
        try:
            if hasattr(crs, "to_epsg") and crs.to_epsg():
                dataset.attrs["proj:code"] = f"EPSG:{crs.to_epsg()}"
            elif hasattr(crs, "to_wkt"):
                dataset.attrs["proj:wkt2"] = crs.to_wkt()
        except Exception as e:
            log.warning("Could not add proj metadata: {}", e=e)


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
