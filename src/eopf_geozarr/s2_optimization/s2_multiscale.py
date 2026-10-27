"""
Streaming multiscale pyramid creation for optimized S2 structure.
Uses lazy evaluation to minimize memory usage during dataset preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import rasterio.transform
import rioxarray  # noqa: F401  # enables the .rio accessor
import structlog
import xarray as xr
import zarr
from affine import Affine
from pyproj import CRS
from zarr_cm import LayoutObject, MultiscalesAttrs, SpatialAttrs, Transform

from eopf_geozarr.conversion import encoding_utils, utils
from eopf_geozarr.conversion.constants import ZARR_FORMAT
from eopf_geozarr.conversion.fs_utils import sanitize_dataset_attributes
from eopf_geozarr.cpm.routing import product_type_of
from eopf_geozarr.s2_optimization import s2_resampling
from eopf_geozarr.s2_optimization.common import DISTRIBUTED_AVAILABLE
from eopf_geozarr.s2_optimization.s2_band_mapping import BAND_INFO

from .s2_resampling import determine_variable_type

if TYPE_CHECKING:
    from zarr.core.common import JSON
    from zarr_cm import spatial as spatial_cm

    from eopf_geozarr.data_api.geozarr.types import (
        XarrayDataArrayEncoding,
    )


SCL_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
SCL_LABELS = [
    "NO_DATA",
    "SATURATED_OR_DEFECTIVE",
    "CAST_SHADOWS",
    "CLOUD_SHADOWS",
    "VEGETATION",
    "NOT_VEGETATED",
    "WATER",
    "UNCLASSIFIED",
    "CLOUD_MEDIUM_PROBABILITY",
    "CLOUD_HIGH_PROBABILITY",
    "THIN_CIRRUS",
    "SNOW_ICE",
]

SCL_ATTRS = {
    "flag_values": SCL_VALUES,
    "flag_meanings": SCL_LABELS,
}
SCL_NODATA = SCL_VALUES[SCL_LABELS.index("NO_DATA")]


def _declare_scl_nodata(dataset: xr.Dataset) -> xr.Dataset:
    """Set the SCL NO_DATA class as `_FillValue` when the source declares no nodata value.

    Not all products declare one on SCL (e.g. Zarr v2 products and the r60m SCL of
    Zarr v3 products), but some clients require a CF `_FillValue`.
    """
    if "scl" in dataset.data_vars and encoding_utils.find_fill_value(dataset["scl"]) is None:
        dataset["scl"].encoding["_FillValue"] = SCL_NODATA
    return dataset


class S2Type(StrEnum):
    L1C = "L1C"
    L2A = "L2A"

    @classmethod
    def from_filename(cls, filename: str | None) -> S2Type | None:
        if not filename:
            return None
        for member in cls:
            if member.value in filename:
                return member
        return None

    @classmethod
    def from_datatree(cls, dt: xr.DataTree) -> S2Type | None:
        """Product level from `stac_discovery` `product:type`, else from the tree name.

        A tree opened from a Zarr store has no name, so the name alone is not enough.
        """
        return cls.from_filename(product_type_of(dt)) or cls.from_filename(dt.name)


log = structlog.get_logger()


@dataclass(frozen=True)
class MultiscaleSpec:
    """A group that gets a multiscale pyramid, as input to `generic_multiscales`."""

    base_path: str
    """Group holding the pyramid levels, e.g. `/measurements/reflectance`."""
    scale_levels: tuple[int, ...]
    """Level resolutions in metres, finest first.

    The finest level must exist in the input. Coarser levels present in the input
    are reused; the others are coarsened from the level before them.
    """
    additional_attributes: dict[str, list[Any]] | None = None
    """Attributes written on every variable of every level (e.g. SCL flag meanings)."""
    product_levels: frozenset[S2Type] | None = None
    """Products the group exists in; `None` means every product."""

    @property
    def finest_key(self) -> str:
        return f"r{self.scale_levels[0]}m"

    def applies_to(self, s2_type: S2Type | None) -> bool:
        return self.product_levels is None or s2_type in self.product_levels


REFLECTANCE_MULTISCALE = MultiscaleSpec(
    base_path="/measurements/reflectance",
    scale_levels=(10, 20, 60, 120, 360, 720),
)

# Auxiliary (non-reflectance) groups. Groups missing from the input are skipped with a warning.
_AUX_MULTISCALES: tuple[MultiscaleSpec, ...] = (
    MultiscaleSpec(
        base_path="/conditions/mask/l2a_classification",
        scale_levels=(20, 60, 120, 360, 720),
        additional_attributes=SCL_ATTRS,
        product_levels=frozenset({S2Type.L2A}),
    ),
    MultiscaleSpec(
        base_path="/quality/probability",
        scale_levels=(20, 60, 120, 360, 720),
        product_levels=frozenset({S2Type.L2A}),
    ),
    MultiscaleSpec(
        base_path="/conditions/mask/l1c_classification",
        scale_levels=(60, 120, 360, 720),
    ),
)

MultiscalesFlavor = Literal["experimental_multiscales_convention"]


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
    # Coordinates are pixel centres; the transform origin is the outer pixel edge.
    x_min = float(x_coords.min()) - pixel_size_x / 2
    y_max = float(y_coords.max()) + pixel_size_y / 2
    return (pixel_size_x, 0.0, x_min, 0.0, -pixel_size_y, y_max)


def _bbox_from_coordinates(dataset: xr.Dataset) -> list[float]:
    """Outer pixel edges ``[xmin, ymin, xmax, ymax]`` of the dataset's x/y grid.

    Falls back to the coordinate extent when the grid has fewer than two pixels
    along an axis, because the pixel size is then unknown.
    """
    transform = _transform_from_coordinates(dataset)
    if transform is None:
        x_coords = dataset.coords["x"].values
        y_coords = dataset.coords["y"].values
        return [
            float(x_coords.min()),
            float(y_coords.min()),
            float(x_coords.max()),
            float(y_coords.max()),
        ]

    pixel_size_x, _, x_min, _, neg_pixel_size_y, y_max = transform
    x_max = x_min + dataset.sizes["x"] * pixel_size_x
    y_min = y_max + dataset.sizes["y"] * neg_pixel_size_y
    return [x_min, y_min, x_max, y_max]


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


def _spatial_level_attrs(dataset: xr.Dataset) -> dict[str, list[float] | list[int]]:
    """`spatial:shape` and `spatial:transform` for a multiscales layout entry.

    `write_geo_metadata` leaves out an all-zero transform, but the minispec requires
    `spatial:transform` on every layout entry, so fall back to the coordinate grid.
    """
    transform = dataset.attrs.get("spatial:transform")
    if transform is None:
        transform = list(_preferred_spatial_transform(dataset) or ())
    return {"spatial:shape": dataset.attrs["spatial:shape"], "spatial:transform": transform}


def _coarsen_variable(var_name: str, var_data: xr.DataArray, factor: int) -> xr.DataArray:
    """Coarsen a single variable using type-aware resampling.

    Dispatches to the appropriate coarsen reduction (mean, max, subsample)
    based on `determine_variable_type`.  Preserves encoding and dtype.
    """
    var_type = determine_variable_type(var_name)

    coarsened = var_data.coarsen({"x": factor, "y": factor}, boundary="trim")
    if var_type in ("reflectance", "probability"):
        if np.issubdtype(var_data.dtype, np.floating):
            # Decoded values: nodata is NaN, so skipping NaN excludes it from the mean.
            result = coarsened.mean(skipna=True)  # type: ignore[attr-defined]
        else:
            # Raw integers: mask nodata, average, and round before the integer cast
            # (a plain cast truncates).
            fill_value = encoding_utils.find_fill_value(var_data)
            if fill_value is None:
                result = coarsened.mean().round()  # type: ignore[attr-defined]
            else:
                result = (
                    var_data.where(var_data != fill_value)
                    .coarsen({"x": factor, "y": factor}, boundary="trim")
                    .mean(skipna=True)  # type: ignore[attr-defined]
                    .round()
                    .fillna(fill_value)
                )
    elif var_type == "classification":
        result = coarsened.reduce(subsample_2)
    else:
        raise ValueError(f"Unknown/Unapplicable variable type {var_type}")

    # `xr.DataArray.astype` clears `.encoding`, so we capture it first and
    # restore it on the cast result. Without this, downstream code that
    # inspects encoding (e.g. to push CF scale-offset into a codec pipeline)
    # would see an empty encoding on every coarsened level.
    # The attributes can carry the CF packing (ESA layout), so keep them too:
    # `where` and the arithmetic above do not always do so.
    encoding = var_data.encoding
    cast_result: xr.DataArray = result.astype(var_data.dtype)
    cast_result.encoding = encoding
    cast_result.attrs = dict(var_data.attrs)
    return cast_result


def inject_missing_bands(
    dataset: xr.Dataset,
    dt_input: xr.DataTree,
    target_resolution: int,
    spatial_chunk: int,
    *,
    bands: set[str] | None = None,
    scale_offset_codec: bool = False,
) -> xr.Dataset:
    """Inject bands whose native resolution is finer than `target_resolution`.

    For each spectral band defined in `BAND_INFO` whose native resolution is
    finer than `target_resolution`, this function checks whether the band is
    already present in `dataset`.  If not, it looks for the band in the
    appropriate source group (e.g. `/measurements/reflectance/r10m`),
    downsamples it by the coarsening operation aligned to ESA's S2-PDGS-MPC-ATBD-L2A
    (block-mean), and merges it into `dataset`.

    Args:
        dataset: The target-resolution dataset (e.g. the r20m or r60m
            reflectance group).
        dt_input: The full input DataTree (used to locate finer-resolution
            source bands).
        target_resolution: Target resolution in metres (e.g. 20 or 60).
        spatial_chunk: Spatial chunk size
        bands: If provided, only inject these band names.  If `None`
            (default), inject every eligible band from `BAND_INFO`.
        scale_offset_codec: Encoding mode (see `normalize_packed`). It selects
            the form of the injected packed bands: decoded float32 for the
            Zarr codecs, packed integers for the ESA layout.

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
        packing = encoding_utils.packing_of(band_src)
        if packing is not None:
            # Use the form of the encoding mode, so the injected band has the same
            # dtype as the other bands and the mean skips nodata in both input forms.
            band_src = encoding_utils.normalize_packed(
                band_src, packing, scale_offset_codec=scale_offset_codec
            )
        factor = target_resolution // native_res
        band_ds = _coarsen_variable(band_name, band_src, factor)

        # Replace coordinates with the target dataset's coordinates so that
        # xarray.Dataset.assign does not try to align on mismatched values.
        band_ds = xr.DataArray(
            band_ds.values,
            dims=band_ds.dims,
            coords={d: dataset.coords[d] for d in band_ds.dims if d in dataset.coords},
            attrs=band_src.attrs,
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

    return utils._rechunk_ds(dataset, spatial_chunk)


def select_unique_variable_type(ds: xr.Dataset) -> str:
    variable_types = {
        s2_resampling.determine_variable_type(str(data_var)) for data_var in ds.data_vars
    }
    if len(variable_types) > 1:
        err_msg = f"The given dataset contains multiple data variable types: {variable_types}. It is not possible to map them to a single variable type."
        raise ValueError(err_msg)

    return next(iter(variable_types))


def generic_multiscales(
    spec: MultiscaleSpec,
    src_processed_groups: dict[str, Any],
    output_group: zarr.Group,
    enable_sharding: bool = True,
    crs: CRS | None = None,
    scale_offset_codec: bool = False,
) -> dict[str, Any]:
    base_path, scale_levels = spec.base_path, spec.scale_levels
    finest_dataset_key = spec.finest_key
    additional_attributes = spec.additional_attributes
    # Create downsampled resolution groups
    if finest_dataset_key not in src_processed_groups:
        raise KeyError(
            f"The given `finest_dataset_key` {finest_dataset_key} is not present in `src_processed_groups` {list(src_processed_groups.keys())} which is required for the correct calculation of multiscales."
        )

    # iterate over pre-defined pyramid-dict (or smth) and generate layout data -> use LayoutObject/...
    current_level_name: str = finest_dataset_key
    fine_base: xr.Dataset = src_processed_groups[finest_dataset_key]

    # determine variable type and subsequent resampling method for MS layout
    variable_type = select_unique_variable_type(fine_base)
    resampling_method = (
        "average"
        if variable_type in ("reflectance", "probability")
        else "nearest"
        if variable_type == "classification"
        else None
    )

    # only accept ('reflectance', 'probability', or 'classification') which arr covered by determine_variable_type(..) loudly fail for other input types, as they are not safely implemented yet
    if resampling_method is None:
        raise ValueError(
            f"resampling method for MS generation is derived from variable_type {variable_type} and didnt match any from ('reflectance', 'probability', or 'classification')"
        )

    spatial_levels: dict[str, dict[str, list[float] | list[int]]] = {
        finest_dataset_key: _spatial_level_attrs(fine_base)
    }

    # predefined layout asset
    layout_: list[dict[str, Any]] = [
        {
            "asset": finest_dataset_key,
            **spatial_levels[finest_dataset_key],
        }
    ]

    dst_processed_groups: dict[str, Any] = {f"{base_path}/{finest_dataset_key}": fine_base}

    # iterate over source, dest pairs: (60, 120), (120, 360), ...
    for source_level, dest_level in pairwise(scale_levels):
        src_level_name = f"r{source_level}m"
        src_level_path = f"{base_path}/{src_level_name}"

        dest_level_name = f"r{dest_level}m"
        dest_level_path = f"{base_path}/{dest_level_name}"

        # each step must be an exact integer factor (20->60 = 3, 60->120 = 2, ...)
        if dest_level % source_level:
            raise ValueError(f"{dest_level} m is not an integer multiple of {source_level} m")
        downsample_factor = dest_level // source_level

        if dest_level_name in src_processed_groups:
            # reuse the native group (e.g. reflectance r20m/r60m), no coarsening needed.
            # Fails loudly afterwards if its geo metadata is missing.
            ds_out = src_processed_groups[dest_level_name]

            # Store results
            dst_processed_groups[dest_level_path] = ds_out
        else:
            source_ds = (
                src_processed_groups[src_level_name]
                if src_level_name in src_processed_groups
                else dst_processed_groups[src_level_path]
            )

            log.info("Creating level with resolution", level=dest_level_name, resolution=dest_level)

            # Create downsampled dataset by coarsening
            downsampled_dataset = create_downsampled_resolution_group(
                source_ds, factor=downsample_factor
            )

            log.info("Writing level to path", level=dest_level_name, output_path=dest_level_path)

            # Create encoding
            encoding = utils.create_uniform_encoding(
                downsampled_dataset,
                enable_sharding=enable_sharding,
                scale_offset_codec=scale_offset_codec,
            )

            # add geo metadata
            write_geo_metadata(downsampled_dataset, crs=crs)

            # Write dataset
            ds_out = stream_write_s2dataset(
                downsampled_dataset,
                path=dest_level_path,
                group=output_group,
                encoding=encoding,
                enable_sharding=enable_sharding,
                crs=crs,
            )

            # Store results
            dst_processed_groups[dest_level_path] = ds_out

        # determine multiscale metadata for all datasets
        spatial_levels[dest_level_path] = _spatial_level_attrs(ds_out)

        # Nominal factor (ratio of pixel spacings), not the ratio of shapes: boundary="trim"
        # gives e.g. 305 -> 152 px at r720m. Intentional, see #313.
        transform: Transform = {
            "scale": [downsample_factor, downsample_factor],
            "translation": [0.0, 0.0],
        }
        lo = {
            "asset": dest_level_name,
            "derived_from": current_level_name,
            "transform": transform,
            **spatial_levels[dest_level_path],
        }

        layout_.append(lo)

        current_level_name = dest_level_name

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

    # LayoutObject is a TypedDict (no validation); zarr_cm validates in build_convention_attrs
    layout: list[LayoutObject] = [LayoutObject(**lo) for lo in layout_]
    ms: MultiscalesAttrs = {"layout": layout, "resampling_method": resampling_method}

    # add geozarr attrs to base of /multiscales
    fine_base_spatial = grid_spatial_attrs(
        transform=Affine(*spatial_levels[finest_dataset_key]["spatial:transform"]),
        shape=(fine_base.sizes["y"], fine_base.sizes["x"]),
    )

    conv = utils.build_convention_attrs(multiscales=ms, spatial=fine_base_spatial, crs=crs)
    parent_group.attrs.update(cast("dict[str, JSON]", conv))

    # Add `additional_attributes` to every level in one place. The finest level and
    # levels reused from the source were written before this function ran, so the
    # arrays on disk are updated as well as the in-memory datasets.
    if additional_attributes is not None:
        for level_path, level_ds in dst_processed_groups.items():
            for data_var in level_ds.data_vars:
                level_ds[data_var].attrs.update(additional_attributes)
                level_array = output_group[f"{level_path.lstrip('/')}/{data_var}"]
                level_array.attrs.update(cast("dict[str, JSON]", additional_attributes))

    # add it as none here to be recognized later and can be created as a zarr root with necessary metadata
    dst_processed_groups[base_path] = None

    return dst_processed_groups


def grid_spatial_attrs(transform: Affine, shape: tuple[int, int]) -> SpatialAttrs:
    """Spatial-convention data for a regular grid with an affine *transform*.

    *shape* is ``(height, width)``.  Emits ``spatial:dimensions`` ``["y","x"]``,
    pixel registration, the bounding box, and the 6-element row-major affine
    transform.
    """
    height, width = shape
    left, bottom, right, top = rasterio.transform.array_bounds(height, width, transform)
    return {
        "spatial:dimensions": ["y", "x"],
        "spatial:registration": "pixel",
        "spatial:bbox": [float(left), float(bottom), float(right), float(top)],
        "spatial:transform": [
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        ],
    }


def create_multiscale_from_datatree(
    dt_input: xr.DataTree,
    *,
    output_group: zarr.Group,
    enable_sharding: bool,
    spatial_chunk: int,
    crs: CRS | None = None,
    scale_offset_codec: bool = False,
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
        scale_offset_codec: Pack packed variables with the Zarr scale-offset
            codecs. By default they are written with CF attributes, as in the
            ESA product.

    Returns:
        Dictionary of processed groups
    """
    processed_groups: dict[str, Any] = {}
    # The scale levels in the output data. 10, 20, 60 already exist in the source data.

    s2_type = S2Type.from_datatree(dt_input)
    if s2_type is None:
        log.warning(
            "Could not determine S2 product level (L1C/L2A); no bands are injected",
            product_type=product_type_of(dt_input),
            name=dt_input.name,
        )

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

        base_dataset = group_node.to_dataset()

        # Skip empty groups
        if not base_dataset.data_vars:
            log.info("Skipping empty group", group_path=group_path)
            continue

        log.info("Copying original group", group_path=group_path)

        dataset = _declare_scl_nodata(utils._rechunk_ds(base_dataset, spatial_chunk))

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

                # just add the b08 band
                if s2_type == "L2A":
                    if group_resolution > 10:
                        dataset = inject_missing_bands(
                            dataset,
                            dt_input,
                            group_resolution,
                            spatial_chunk,
                            scale_offset_codec=scale_offset_codec,
                            bands={"b08"},
                        )

                # add all lower level bands here!
                elif s2_type == "L1C":
                    if group_resolution == 20:
                        dataset = inject_missing_bands(
                            dataset,
                            dt_input,
                            group_resolution,
                            spatial_chunk,
                            scale_offset_codec=scale_offset_codec,
                            bands={
                                "b02",
                                "b03",
                                "b04",
                                "b08",
                            },
                        )
                    elif group_resolution == 60:
                        dataset = inject_missing_bands(
                            dataset,
                            dt_input,
                            group_resolution,
                            spatial_chunk,
                            scale_offset_codec=scale_offset_codec,
                            bands={
                                "b02",
                                "b03",
                                "b04",
                                "b08",
                                "b05",
                                "b06",
                                "b07",
                                "b8a",
                                "b11",
                                "b12",
                            },
                        )

            # Measurement groups: apply custom encoding
            encoding = utils.create_uniform_encoding(
                dataset,
                enable_sharding=enable_sharding,
                scale_offset_codec=scale_offset_codec,
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
        else:
            # Non-measurement groups: preserve original encoding
            encoding = utils.create_uniform_encoding(
                dataset,
                enable_sharding=enable_sharding,
                scale_offset_codec=scale_offset_codec,
            )

        # Drop scalar (0-D) coordinates such as a source `band` label: the
        # minispec's DataArray rules forbid scalar arrays in a GeoZarr dataset.
        scalar_coords = [name for name, coord in dataset.coords.items() if coord.ndim == 0]
        if scalar_coords:
            dataset = dataset.drop_vars(scalar_coords)
            for name in scalar_coords:
                encoding.pop(str(name), None)

        ds_out = stream_write_s2dataset(
            dataset,
            path=group_path,
            group=output_group,
            encoding=encoding,
            enable_sharding=enable_sharding,
            crs=crs,
        )
        processed_groups[group_path] = ds_out

    # iterate over auxiliary multisacles to generate non-measurement MS
    for aux in _AUX_MULTISCALES:
        if not aux.applies_to(s2_type):
            continue
        base_path, finest_key = aux.base_path, aux.finest_key
        src_path = f"{base_path}/{finest_key}"
        # Not every processing baseline carries every auxiliary group: skip, don't fail.
        if src_path not in processed_groups:
            log.warning(
                "Skipping S2 auxiliary multiscales: source group missing from input",
                base_path=base_path,
                src_path=src_path,
                product_level=s2_type,
            )
            continue

        log.info("Adding multiscales for S2 auxiliary group", base_path=base_path)
        processed_groups.update(
            generic_multiscales(
                aux,
                src_processed_groups={finest_key: processed_groups[src_path]},
                output_group=output_group,
                enable_sharding=enable_sharding,
                crs=crs,
                scale_offset_codec=scale_offset_codec,
            )
        )

    # generate multiscales for "/measurements/reflectance" from its native r10m/r20m/r60m levels
    reflectance_path = REFLECTANCE_MULTISCALE.base_path
    log.info("Adding multiscales for S2 reflectance", base_path=reflectance_path)
    measurement_processed_groups: dict[str, Any] = {
        key: processed_groups[f"{reflectance_path}/{key}"] for key in ("r10m", "r20m", "r60m")
    }
    ms_measurement_processed_groups = generic_multiscales(
        REFLECTANCE_MULTISCALE,
        src_processed_groups=measurement_processed_groups,
        output_group=output_group,
        enable_sharding=enable_sharding,
        crs=crs,
        scale_offset_codec=scale_offset_codec,
    )

    # update processed groups with multiscale included groups, old instances of dataset are replaced,
    # which is no issue as they were not touched
    processed_groups.update(ms_measurement_processed_groups)

    return processed_groups


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
    shards = []

    for i, (dim_size, chunk_size) in enumerate(zip(data_shape, chunks, strict=False)):
        if i == 0 and len(data_shape) == 3:
            # First dimension in 3D data (time) - use single time slice per shard
            shards.append(1)
        else:
            # For x/y dimensions, try to use full dimension size
            # But ensure it's divisible by chunk size
            if dim_size % chunk_size == 0:
                # Perfect: full dimension is divisible by chunk
                shards.append(dim_size)
            else:
                # Find the largest multiple of chunk_size that fits
                num_chunks = dim_size // chunk_size
                if num_chunks > 0:
                    shard_size = num_chunks * chunk_size
                    shards.append(shard_size)
                else:
                    # Fallback: use chunk size itself
                    shards.append(chunk_size)

    return tuple(shards)


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
    """Pick the centre pixel of each coarsening window (nearest-neighbour).

    `coarsen` places the output coordinates at the window centres, so the
    centre pixel keeps classification aligned with the averaged bands. For an
    even window there is no single centre pixel and the lower-right one of the
    middle four is taken.
    """
    if axis is None:
        axis = tuple(range(a.ndim))
    indexer = [a.shape[i] // 2 if i in axis else slice(None) for i in range(a.ndim)]
    return a[tuple(indexer)]


def stream_write_s2dataset(
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
    dataset = utils.rechunk_dataset_for_encoding(dataset, encoding)

    # Add the geo metadata before writing for
    # - /measurements/ groups
    # - /quality/ groups
    # - /consitions/mask groups
    if "/measurements/" in path or "/quality/" in path or "/conditions/mask" in path:
        write_geo_metadata(dataset, crs=crs)

    # Sanitize NaN values in dataset attributes before writing
    dataset = sanitize_dataset_attributes(dataset)

    # Write with streaming computation and progress tracking
    # The to_zarr operation will trigger all lazy computations
    write_job = dataset.to_zarr(
        group.store,
        mode="w",
        consolidated=False,
        zarr_format=ZARR_FORMAT,
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
            spatial_data["spatial:bbox"] = _bbox_from_coordinates(dataset)

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
