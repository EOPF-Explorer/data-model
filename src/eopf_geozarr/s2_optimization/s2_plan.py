"""
Declarative planning for S2 zarr hierarchy creation.

This module contains pure functions that build a complete
dict[str, ArrayV3Metadata | GroupMetadata] describing every node in the
output zarr hierarchy. No zarr writes, no xarray opens.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import zarr
from zarr.codecs import ShardingCodec
from zarr.core.chunk_grids import RegularChunkGrid
from zarr.core.group import GroupMetadata
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.dtype import Int64
from zarr_cm import geo_proj
from zarr_cm import multiscales as multiscales_cm
from zarr_cm import spatial as spatial_cm

from eopf_geozarr.conversion.geozarr import (
    _create_tile_matrix_limits,
    create_native_crs_tile_matrix_set,
)
from eopf_geozarr.data_api.geozarr.multiscales import zcm
from eopf_geozarr.data_api.geozarr.multiscales.geozarr import (
    MultiscaleGroupAttrs,
    MultiscaleMeta,
)
from eopf_geozarr.zarrio import ArrayReencoder, replace_json_invalid_floats

if TYPE_CHECKING:
    from pyproj import CRS

    from eopf_geozarr.types import OverviewLevelJSON

log = structlog.get_logger()

# Resolution derivation chain: each level and the level it's derived from
DERIVATION_CHAIN: dict[str, str | None] = {
    "r10m": None,
    "r20m": "r10m",
    "r60m": "r10m",
    "r120m": "r60m",
    "r360m": "r120m",
    "r720m": "r360m",
}

# Ordered list of downsampling steps: (current_meters, next_meters)
DOWNSAMPLE_STEPS: tuple[tuple[int, int], ...] = (
    (10, 20),
    (20, 60),
    (60, 120),
    (120, 360),
    (360, 720),
)

RES_ORDER: dict[str, int] = {
    "r10m": 10,
    "r20m": 20,
    "r60m": 60,
    "r120m": 120,
    "r360m": 360,
    "r720m": 720,
}


@dataclass
class FillSpec:
    """Describes how to fill an array during the execute phase."""

    source_path: str | None = None
    downsample_from: str | None = None
    scale_factor: int = 1
    is_spatial_ref: bool = False
    computed_values: np.ndarray | None = None


def compute_transform_from_coords(
    x_values: np.ndarray, y_values: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """Compute a 6-element affine transform from x and y coordinate arrays."""
    pixel_size_x = float(abs(x_values[1] - x_values[0]))
    pixel_size_y = float(abs(y_values[1] - y_values[0]))
    return (pixel_size_x, 0.0, float(x_values.min()), 0.0, -pixel_size_y, float(y_values.max()))


def build_spatial_ref_metadata(crs: CRS) -> ArrayV3Metadata:
    """
    Build ArrayV3Metadata for a scalar spatial_ref array from a CRS.

    Uses pyproj.CRS.to_cf() to get CF grid mapping attributes.
    """
    cf_attrs = crs.to_cf()
    cf_attrs["spatial_ref"] = crs.to_wkt()

    return ArrayV3Metadata(
        shape=(),
        data_type=Int64(),
        chunk_grid={"name": "regular", "configuration": {"chunk_shape": ()}},
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        fill_value=0,
        dimension_names=None,
        codecs=({"name": "bytes"},),
        attributes=cf_attrs,
    )


def build_geo_group_attrs(
    crs: CRS,
    x_values: np.ndarray,
    y_values: np.ndarray,
    data_shape: tuple[int, ...],
) -> dict[str, object]:
    """
    Compute group-level geo/spatial/proj attributes from CRS and coordinate arrays.
    """
    attrs: dict[str, object] = {
        "grid_mapping": "spatial_ref",
        "spatial:dimensions": ["y", "x"],
        "spatial:registration": "pixel",
    }

    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    attrs["spatial:bbox"] = [x_min, y_min, x_max, y_max]

    if len(x_values) > 1 and len(y_values) > 1:
        transform = compute_transform_from_coords(x_values, y_values)
        if not all(t == 0 for t in transform):
            attrs["spatial:transform"] = list(transform)

    if len(data_shape) >= 2:
        height, width = data_shape[-2:]
        attrs["spatial:shape"] = [height, width]

    if hasattr(crs, "to_epsg") and crs.to_epsg():
        attrs["proj:code"] = f"EPSG:{crs.to_epsg()}"
    elif hasattr(crs, "to_wkt"):
        attrs["proj:wkt2"] = crs.to_wkt()

    conventions = [
        spatial_cm.CMO,
        geo_proj.CMO,
    ]
    attrs["zarr_conventions"] = conventions

    return attrs


def _derive_array_metadata(
    meta: ArrayV3Metadata,
    scale: int,
) -> ArrayV3Metadata:
    """
    Derive ArrayV3Metadata for a downsampled level from a parent array.

    Spatial dimensions (x, y) are divided by scale. Chunk sizes are clamped
    to the new shape. Sharding codecs are adjusted to fit the new shape
    (shard size = largest multiple of inner chunks that fits).
    """
    dim_names = meta.dimension_names or ()
    new_shape = tuple(
        s // scale if (dim_names and dim_names[i] in ("x", "y")) else s
        for i, s in enumerate(meta.shape)
    )

    # Check if parent uses sharding
    has_sharding = any(isinstance(c, ShardingCodec) for c in meta.codecs)

    if has_sharding:
        # Extract the inner chunk shape and inner codecs from the shard
        shard_codec = next(c for c in meta.codecs if isinstance(c, ShardingCodec))
        inner_chunks = shard_codec.chunk_shape
        inner_codecs = shard_codec.codecs

        # Clamp inner chunks to new shape
        new_inner_chunks = tuple(min(c, s) for c, s in zip(inner_chunks, new_shape, strict=True))

        # New shard size = largest multiple of inner chunks that fits in shape
        new_shard_shape = tuple(
            (s // c) * c for s, c in zip(new_shape, new_inner_chunks, strict=True)
        )

        # If shards would contain only 1 chunk, drop sharding
        num_subchunks = 1
        for shard_dim, chunk_dim in zip(new_shard_shape, new_inner_chunks, strict=True):
            if chunk_dim > 0:
                num_subchunks *= shard_dim // chunk_dim

        if num_subchunks <= 1:
            new_chunks = new_inner_chunks
            new_codecs = tuple(inner_codecs)
        else:
            new_chunks = new_shard_shape
            new_codecs = (
                ShardingCodec(
                    chunk_shape=new_inner_chunks,
                    codecs=inner_codecs,
                    index_codecs=shard_codec.index_codecs,
                    index_location=shard_codec.index_location,
                ),
            )
    else:
        old_chunks = meta.chunk_grid.chunk_shape
        new_chunks = tuple(min(c, s) for c, s in zip(old_chunks, new_shape, strict=True))
        new_codecs = tuple(meta.codecs)

    return replace(
        meta,
        shape=new_shape,
        chunk_grid=RegularChunkGrid(chunk_shape=new_chunks),
        codecs=new_codecs,
    )


def _is_geo_leaf_group(path: str) -> bool:
    """
    Check if a group path is a resolution-level leaf group that should get CRS treatment.

    Matches patterns like measurements/reflectance/r10m, quality/mask/r20m, etc.
    """
    parts = path.strip("/").split("/")
    if len(parts) < 3:
        return False
    top = parts[0]
    leaf = parts[-1]
    return top in ("measurements", "quality") and leaf.startswith("r") and leaf.endswith("m")


def _get_first_2d_shape(
    nodes: dict[str, ArrayV3Metadata | GroupMetadata],
    group_prefix: str,
) -> tuple[int, ...] | None:
    """Get the shape of the first 2D+ array under a group prefix."""
    for key, meta in nodes.items():
        if (
            key.startswith(group_prefix + "/")
            and isinstance(meta, ArrayV3Metadata)
            and len(meta.shape) >= 2
        ):
            return meta.shape  # type: ignore[no-any-return]
    return None


def plan_source_nodes(
    source_group: zarr.Group,
    *,
    array_reencoder: ArrayReencoder,
    omit_nodes: set[str] | None = None,
    allow_json_nan: bool = False,
) -> tuple[dict[str, ArrayV3Metadata | GroupMetadata], dict[str, FillSpec]]:
    """
    Plan nodes from the source zarr group (V2 -> V3 re-encoding).

    This replicates the metadata-building logic from reencode_group.
    """
    if omit_nodes is None:
        omit_nodes = set()

    nodes: dict[str, ArrayV3Metadata | GroupMetadata] = {
        "": GroupMetadata(zarr_format=3, attributes=source_group.attrs.asdict())
    }
    fill_specs: dict[str, FillSpec] = {}

    members = dict(source_group.members(max_depth=None))

    for name in omit_nodes:
        if not any(k.startswith(name) for k in members):
            log.warning("The name %s was provided in omit_nodes but no such node exists.", name)

    for name, member in members.items():
        if any(name.startswith(v) for v in omit_nodes):
            continue

        source_meta = member.metadata

        if not allow_json_nan:
            new_attrs = replace_json_invalid_floats(source_meta.attributes)
            source_meta = replace(source_meta, attributes=new_attrs)

        if isinstance(member, zarr.Array):
            new_meta = array_reencoder(member.path, source_meta)
            nodes[name] = new_meta
            fill_specs[name] = FillSpec(source_path=name)
        else:
            nodes[name] = GroupMetadata(
                zarr_format=3,
                attributes=source_meta.attributes,
            )

    return nodes, fill_specs


def enrich_with_geo_metadata(
    nodes: dict[str, ArrayV3Metadata | GroupMetadata],
    fill_specs: dict[str, FillSpec],
    source_group: zarr.Group,
    crs: CRS,
) -> None:
    """
    Enrich leaf groups with CRS/geo metadata: add spatial_ref array,
    set geo attributes on groups, and add grid_mapping to data variables.

    Modifies nodes and fill_specs in place.
    """
    import zarr as zarr_mod

    spatial_ref_meta = build_spatial_ref_metadata(crs)

    # Find all geo leaf groups in the nodes dict
    group_paths = [
        path
        for path, meta in nodes.items()
        if isinstance(meta, GroupMetadata) and _is_geo_leaf_group(path)
    ]

    for group_path in group_paths:
        # Read coordinate values from the source group
        source_path = group_path  # source and output paths match
        try:
            source_subgroup = source_group[source_path]
        except KeyError:
            # This group might be a derived level, not in source
            continue

        x_values = None
        y_values = None
        if isinstance(source_subgroup, zarr_mod.Group):
            if "x" in source_subgroup:
                x_values = source_subgroup["x"][...]
            if "y" in source_subgroup:
                y_values = source_subgroup["y"][...]

        if x_values is None or y_values is None:
            log.warning(
                "No x/y coordinates found in source group %s, skipping geo enrichment", group_path
            )
            continue

        data_shape = _get_first_2d_shape(nodes, group_path)
        if data_shape is None:
            continue

        # Enrich group metadata with geo attrs
        group_meta = nodes[group_path]
        assert isinstance(group_meta, GroupMetadata)
        geo_attrs = build_geo_group_attrs(crs, x_values, y_values, data_shape)
        merged_attrs = {**group_meta.attributes, **geo_attrs}
        nodes[group_path] = GroupMetadata(zarr_format=3, attributes=merged_attrs)

        # Add spatial_ref array
        sr_path = f"{group_path}/spatial_ref"
        nodes[sr_path] = spatial_ref_meta
        fill_specs[sr_path] = FillSpec(is_spatial_ref=True)

        # Add grid_mapping + coordinates to data variable arrays
        prefix = group_path + "/"
        for key in list(nodes.keys()):
            if key.startswith(prefix):
                meta = nodes[key]
                if isinstance(meta, ArrayV3Metadata) and len(meta.shape) >= 2:
                    enriched_attrs = {
                        **meta.attributes,
                        "grid_mapping": "spatial_ref",
                        "coordinates": "spatial_ref",
                    }
                    nodes[key] = replace(meta, attributes=enriched_attrs)


def plan_derived_levels(
    nodes: dict[str, ArrayV3Metadata | GroupMetadata],
    fill_specs: dict[str, FillSpec],
    source_group: zarr.Group,
    crs: CRS,
    reflectance_prefix: str,
) -> dict[str, np.ndarray]:
    """
    Plan derived multiscale levels under the reflectance prefix.

    For each downsampling step (10→20, 20→60, 60→120, etc.), derive array
    metadata from the parent level. Skips arrays that already exist in the
    target level (cross-resolution propagation for native levels).

    Returns a dict mapping level paths to their x/y coordinate values,
    needed for multiscale metadata computation.
    """
    coord_values: dict[str, np.ndarray] = {}

    # Read coordinate values for native levels from source
    for res_name in ("r10m", "r20m", "r60m"):
        group_path = f"{reflectance_prefix}/{res_name}"
        try:
            src = source_group[group_path]
        except KeyError:
            continue
        if "x" in src:
            coord_values[f"{group_path}/x"] = src["x"][...]
        if "y" in src:
            coord_values[f"{group_path}/y"] = src["y"][...]

    for cur_meters, next_meters in DOWNSAMPLE_STEPS:
        cur_name = f"r{cur_meters}m"
        next_name = f"r{next_meters}m"
        cur_prefix = f"{reflectance_prefix}/{cur_name}"
        next_prefix = f"{reflectance_prefix}/{next_name}"
        scale = next_meters // cur_meters

        # Collect arrays from the current level (from nodes dict)
        cur_arrays: dict[str, ArrayV3Metadata] = {}
        for key, meta in nodes.items():
            if key.startswith(cur_prefix + "/") and isinstance(meta, ArrayV3Metadata):
                array_name = key[len(cur_prefix) + 1 :]
                cur_arrays[array_name] = meta

        if not cur_arrays:
            log.warning("No arrays found in %s, skipping derivation", cur_prefix)
            continue

        # Ensure the target group exists in nodes
        if next_prefix not in nodes:
            nodes[next_prefix] = GroupMetadata(zarr_format=3, attributes={})

        # Compute derived coordinate values
        cur_x_key = f"{cur_prefix}/x"
        cur_y_key = f"{cur_prefix}/y"
        parent_x = coord_values.get(cur_x_key)
        parent_y = coord_values.get(cur_y_key)

        if parent_x is not None:
            derived_x = parent_x[::scale][: len(parent_x) // scale]
            coord_values[f"{next_prefix}/x"] = derived_x
        if parent_y is not None:
            derived_y = parent_y[::scale][: len(parent_y) // scale]
            coord_values[f"{next_prefix}/y"] = derived_y

        for array_name, parent_meta in cur_arrays.items():
            target_key = f"{next_prefix}/{array_name}"

            # Skip if already exists (native level or already planned)
            if target_key in nodes:
                continue

            # Skip coordinate arrays and spatial_ref — handle separately
            if array_name in ("x", "y", "spatial_ref"):
                continue

            if len(parent_meta.shape) >= 2:
                # 2D+ data variable: derive downsampled metadata
                derived_meta = _derive_array_metadata(parent_meta, scale)
                nodes[target_key] = derived_meta
                fill_specs[target_key] = FillSpec(
                    downsample_from=f"{cur_prefix}/{array_name}",
                    scale_factor=scale,
                )
            else:
                # 0D/1D variable: copy as-is
                nodes[target_key] = parent_meta
                fill_specs[target_key] = FillSpec(
                    source_path=f"{cur_prefix}/{array_name}",
                )

        # Plan coordinate arrays for derived levels
        for coord_name in ("x", "y"):
            target_key = f"{next_prefix}/{coord_name}"
            if target_key in nodes:
                continue
            parent_coord_key = f"{cur_prefix}/{coord_name}"
            if parent_coord_key not in nodes:
                continue
            parent_coord_meta = nodes[parent_coord_key]
            assert isinstance(parent_coord_meta, ArrayV3Metadata)
            new_length = parent_coord_meta.shape[0] // scale
            parent_chunk = parent_coord_meta.chunk_grid.chunk_shape[0]
            new_chunk = min(parent_chunk, new_length)
            derived_coord_meta = replace(
                parent_coord_meta,
                shape=(new_length,),
                chunk_grid=RegularChunkGrid(chunk_shape=(new_chunk,)),
            )
            nodes[target_key] = derived_coord_meta
            coord_val = coord_values.get(f"{next_prefix}/{coord_name}")
            fill_specs[target_key] = FillSpec(computed_values=coord_val)

        # Plan spatial_ref for derived levels
        sr_key = f"{next_prefix}/spatial_ref"
        if sr_key not in nodes:
            nodes[sr_key] = build_spatial_ref_metadata(crs)
            fill_specs[sr_key] = FillSpec(is_spatial_ref=True)

        # Now enrich the derived level group with geo attrs
        derived_x = coord_values.get(f"{next_prefix}/x")
        derived_y = coord_values.get(f"{next_prefix}/y")
        if derived_x is not None and derived_y is not None:
            data_shape = _get_first_2d_shape(nodes, next_prefix)
            if data_shape is not None:
                geo_attrs = build_geo_group_attrs(crs, derived_x, derived_y, data_shape)
                group_meta = nodes[next_prefix]
                assert isinstance(group_meta, GroupMetadata)
                merged = {**group_meta.attributes, **geo_attrs}
                nodes[next_prefix] = GroupMetadata(zarr_format=3, attributes=merged)

                # Add grid_mapping + coordinates to derived data vars
                for key in list(nodes.keys()):
                    if key.startswith(next_prefix + "/"):
                        meta = nodes[key]
                        if isinstance(meta, ArrayV3Metadata) and len(meta.shape) >= 2:
                            enriched = {
                                **meta.attributes,
                                "grid_mapping": "spatial_ref",
                                "coordinates": "spatial_ref",
                            }
                            nodes[key] = replace(meta, attributes=enriched)

    return coord_values


def compute_multiscales_group_attrs(
    nodes: dict[str, ArrayV3Metadata | GroupMetadata],
    coord_values: dict[str, np.ndarray],
    reflectance_prefix: str,
    crs: CRS,
) -> dict[str, object]:
    """
    Compute multiscale metadata attributes for the reflectance parent group
    from the plan dict. Replaces create_multiscales_metadata.
    """

    res_names = ["r10m", "r20m", "r60m", "r120m", "r360m", "r720m"]

    # Get bounds from finest level
    finest_x = coord_values.get(f"{reflectance_prefix}/r10m/x")
    finest_y = coord_values.get(f"{reflectance_prefix}/r10m/y")
    if finest_x is None or finest_y is None:
        raise ValueError("Cannot compute multiscale metadata: missing r10m coordinates")

    native_bounds = (
        float(finest_x.min()),
        float(finest_y.min()),
        float(finest_x.max()),
        float(finest_y.max()),
    )

    overview_levels: list[OverviewLevelJSON] = []

    for res_name in res_names:
        level_prefix = f"{reflectance_prefix}/{res_name}"
        res_meters = RES_ORDER[res_name]

        # Find first 2D data variable to get shape
        data_shape = _get_first_2d_shape(nodes, level_prefix)
        if data_shape is None:
            continue
        height, width = data_shape[-2:]

        # Get coordinate values for transform
        x_vals = coord_values.get(f"{level_prefix}/x")
        y_vals = coord_values.get(f"{level_prefix}/y")

        transform: tuple[float, ...] | None = None
        if x_vals is not None and y_vals is not None and len(x_vals) > 1 and len(y_vals) > 1:
            transform = compute_transform_from_coords(x_vals, y_vals)
            if all(t == 0 for t in transform):
                transform = None

        # Zoom level
        tile_width = 256
        zoom_w = max(0, int(np.ceil(np.log2(width / tile_width))))
        zoom_h = max(0, int(np.ceil(np.log2(height / tile_width))))
        zoom = max(zoom_w, zoom_h)

        # Scale ratio
        if res_name == "r10m":
            relative_scale: int | float = 1.0
        else:
            parent_res = DERIVATION_CHAIN.get(res_name)
            if parent_res:
                parent_shape = _get_first_2d_shape(nodes, f"{reflectance_prefix}/{parent_res}")
                if parent_shape is not None:
                    parent_h, parent_w = parent_shape[-2:]
                    scale_x = parent_w / width if width > 0 else 1.0
                    scale_y = parent_h / height if height > 0 else 1.0
                    relative_scale = max(scale_x, scale_y)
                else:
                    relative_scale = res_meters / 10
            else:
                relative_scale = res_meters / 10

        # Get chunks from first data var
        chunks: tuple[tuple[int, ...], ...] | None = None
        for key, meta in nodes.items():
            if (
                key.startswith(level_prefix + "/")
                and isinstance(meta, ArrayV3Metadata)
                and len(meta.shape) >= 2
            ):
                chunks = tuple((c,) for c in meta.chunk_grid.chunk_shape)
                break

        level_entry: OverviewLevelJSON = {
            "level": res_name,
            "zoom": zoom,
            "width": width,
            "height": height,
            "translation_relative": 0.0,
            "scale_absolute": res_meters,
            "scale_relative": relative_scale,
            "spatial_transform": None,
            "chunks": chunks,
            "spatial_shape": (height, width),
        }
        if transform is not None:
            level_entry["spatial_transform"] = transform

        overview_levels.append(level_entry)

    # Build TMS
    tile_matrix_set = create_native_crs_tile_matrix_set(
        crs, native_bounds, overview_levels, group_prefix=None
    )
    tile_matrix_limits = _create_tile_matrix_limits(overview_levels, tile_width=256)

    # Build ZCM layout
    layout: list[zcm.ScaleLevel] = []
    for i, ol in enumerate(overview_levels):
        asset = str(ol["level"])
        scale_level_data: dict[str, Any] = {"asset": asset}

        if i > 0:
            derived_from = DERIVATION_CHAIN.get(asset, "r10m")
            multiscale_transform = zcm.Transform(
                scale=(ol["scale_relative"],) * 2,
                translation=(ol["translation_relative"],) * 2,
            )
            scale_level_data["derived_from"] = derived_from
            scale_level_data["transform"] = multiscale_transform

        scale_level_data["spatial:shape"] = ol["spatial_shape"]
        spatial_transform = ol.get("spatial_transform")
        if spatial_transform is not None and not all(t == 0 for t in spatial_transform):
            scale_level_data["spatial:transform"] = spatial_transform

        layout.append(zcm.ScaleLevel(**scale_level_data))

    multiscale_attrs = MultiscaleGroupAttrs(
        zarr_conventions=(
            multiscales_cm.CMO,
            spatial_cm.CMO,
            geo_proj.CMO,
        ),
        multiscales=MultiscaleMeta(
            layout=tuple(layout),
            resampling_method="average",
            tile_matrix_set=tile_matrix_set,
            tile_matrix_limits=tile_matrix_limits,
        ),
    )

    # Combine multiscale attrs with spatial/proj attrs
    result = multiscale_attrs.model_dump()

    spatial_attrs: dict[str, object] = {
        "spatial:dimensions": ("y", "x"),
        "spatial:bbox": tuple(native_bounds),
        "spatial:registration": "pixel",
    }

    if hasattr(crs, "to_epsg") and crs.to_epsg():
        spatial_attrs["proj:code"] = f"EPSG:{crs.to_epsg()}"
    elif hasattr(crs, "to_wkt"):
        spatial_attrs["proj:wkt2"] = crs.to_wkt()

    result.update(spatial_attrs)
    return result


def plan_s2_hierarchy(
    source_group: zarr.Group,
    crs: CRS,
    *,
    array_reencoder: ArrayReencoder,
    omit_nodes: set[str] | None = None,
    allow_json_nan: bool = False,
) -> tuple[dict[str, ArrayV3Metadata | GroupMetadata], dict[str, FillSpec]]:
    """
    Build the complete output hierarchy plan for an S2 conversion.

    Returns
    -------
    nodes
        Complete dict for zarr.create_hierarchy — every group and array
        in the output, with all attributes pre-set.
    fill_specs
        Per-array instructions for the data-filling phase.
    """
    log.info("Planning S2 hierarchy")

    # Step 1: Plan source nodes (V2 -> V3 re-encoding)
    nodes, fill_specs = plan_source_nodes(
        source_group,
        array_reencoder=array_reencoder,
        omit_nodes=omit_nodes,
        allow_json_nan=allow_json_nan,
    )

    # Step 2: Enrich leaf groups with CRS/geo metadata
    enrich_with_geo_metadata(nodes, fill_specs, source_group, crs)

    # Step 3: Plan derived multiscale levels
    reflectance_prefix = "measurements/reflectance"
    coord_values = plan_derived_levels(nodes, fill_specs, source_group, crs, reflectance_prefix)

    # Step 4: Compute and set multiscale metadata on the reflectance parent group
    if reflectance_prefix in nodes:
        ms_attrs = compute_multiscales_group_attrs(nodes, coord_values, reflectance_prefix, crs)
        group_meta = nodes[reflectance_prefix]
        assert isinstance(group_meta, GroupMetadata)
        merged = {**group_meta.attributes, **ms_attrs}
        nodes[reflectance_prefix] = GroupMetadata(zarr_format=3, attributes=merged)

    log.info("Planned %d nodes (%d fill specs)", len(nodes), len(fill_specs))
    return nodes, fill_specs
