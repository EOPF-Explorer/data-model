"""Top-level Sentinel-3 OLCI L1 EFR -> GeoZarr conversion."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast

import rioxarray  # noqa: F401
import structlog
import xarray as xr
import zarr
from zarr_cm import LayoutObject, MultiscalesAttrs, Transform

from eopf_geozarr.conversion import utils
from eopf_geozarr.conversion.utils import build_convention_attrs
from eopf_geozarr.cpm.routing import product_type_of
from eopf_geozarr.data_api.s3_olci import Sentinel3OlciRoot
from eopf_geozarr.s3_olci_optimization.olci_band_mapping import OLCI_BANDS
from eopf_geozarr.s3_olci_optimization.olci_multiscale import (
    SWATH_DIMS,
    reduce_swath,
)

if TYPE_CHECKING:
    from zarr.core.common import JSON

log = structlog.get_logger()


class S3Type(StrEnum):
    OLC = "OLC"
    SLS = "SLS"

    @classmethod
    def from_filename(cls, filename: str | None) -> S3Type | None:
        if not filename:
            return None
        for member in cls:
            if member.value in filename:
                return member
        return None

    @classmethod
    def from_datatree(cls, dt: xr.DataTree) -> S3Type | None:
        """Product level from `stac_discovery` `product:type`, else from the tree name.

        A tree opened from a Zarr store has no name, so the name alone is not enough.
        """
        return cls.from_filename(product_type_of(dt)) or cls.from_filename(dt.name)


class ResolutionLevel(StrEnum):
    FR = "FR"
    RR = "RR"

    @classmethod
    def from_filename(cls, filename: str | None) -> ResolutionLevel | None:
        if not filename:
            return None
        # cheap solution -> ONLY for S3 OLC as this is this scripts purpose (no other sensors)
        # to avoid confusion with other modes and restrict the impact of characters included after the product definition,
        # this check limits the considered section of filenames to the product indicators
        # also checks out for pre-defined `product_type_of(dt)`
        cropped_filename = filename[:12]
        for member in cls:
            if member.value in cropped_filename:
                return member
        return None

    @classmethod
    def from_datatree(cls, dt: xr.DataTree) -> ResolutionLevel | None:
        """Product level from `stac_discovery` `product:type`, else from the tree name.

        A tree opened from a Zarr store has no name, so the name alone is not enough.
        """
        return cls.from_filename(product_type_of(dt)) or cls.from_filename(dt.name)


@dataclass(frozen=True)
class S3MultiscaleSpec:
    """A group that gets a multiscale pyramid, as input to `generic_multiscales`."""

    base_path: str
    """Group holding the pyramid levels, e.g. `/measurements`."""
    scale_levels: tuple[int, ...]
    """Levels used for the generation of multiscales. The avaiable data is saved as r0, with the others are coarsened from the level before them."""

    product_levels: frozenset[S3Type] | None = None
    """Products the group exists in; `None` means every product."""

    @property
    def finest_key(self) -> str:
        return f"r{self.scale_levels[0]}"

    def applies_to(self, s3_type: S3Type | None) -> bool:
        return self.product_levels is None or s3_type in self.product_levels


FR_REFLECTANCE_MULTISCALE = S3MultiscaleSpec(
    base_path="/measurements",
    scale_levels=(0, 1, 2, 3),
)

RR_REFLECTANCE_MULTISCALE = S3MultiscaleSpec(
    base_path="/measurements",
    scale_levels=(0, 1, 2),
)


def is_sentinel3_olci_dataset(group: zarr.Group) -> bool:
    """Return True if *group* is a Sentinel-3 OLCI L1 EFR product.

    Detection is structural: the group must validate against
    ``Sentinel3OlciRoot`` and its ``measurements`` group must contain the
    first OLCI radiance band.

    Detection is intentionally conservative: ``Sentinel3OlciRoot`` uses a
    closed member set and requires the standard EOPF root attrs, so a
    product with extra top-level groups or missing root metadata fails
    validation and is treated as "not OLCI" (falling through to the generic
    converter in the CLI). Users with such near-miss products should use the
    explicit ``convert-s3-olci-optimized`` subcommand, which skips detection.
    """
    from eopf_geozarr.pyz.v2 import GroupSpec

    try:
        model = Sentinel3OlciRoot.model_validate(GroupSpec.from_zarr(group).model_dump())
    except Exception as e:
        # Classify, never raise: from_zarr/model_dump can fail with types
        # other than ValidationError on malformed or unexpected stores, and
        # any failure simply means "not a recognised OLCI product".
        log.debug("Not an OLCI dataset", error=str(e))
        return False
    try:
        return OLCI_BANDS[0] in model.measurements.members
    except KeyError:
        return False


def update_s3_stac_assets(dt_input: xr.DataTree, datasets: Mapping[str, object]) -> None:
    """Add the S3 pyramid levels (if added) and remove/replace the not existing assets as they are moved to /r0
    Updates `dt_input.attrs["stac_discovery"]` in place, so call it before
    `updated_root_consolidation` writes the root metadata from those attributes.
    """
    if not dt_input.attrs:
        return
    dt_attributes = dt_input.attrs
    stac_attrs = dt_attributes.get("stac_discovery", {}).get("assets")

    if not stac_attrs:
        return

    existing_hrefs = {
        asset["href"].strip("/")
        for asset in stac_attrs.values()
        if isinstance(asset, Mapping) and utils._is_store_href(asset.get("href"))
    }

    # collect all touched hrefs to comapre against existing ones and filter out replaced assets
    touched_hrefs = []

    for group_path, ds in datasets.items():
        if ds is None:
            continue
        # catch object != dataset for typing
        if not isinstance(ds, xr.Dataset):
            log.warning("Found non-dataset object in datasets!", dataset=ds)
            continue
        resolution = group_path.rsplit("/", 1)[-1]  # "r0, r1, r2, ..."
        for var_name in ds.data_vars:
            href = f"{group_path}/{var_name}"
            if href.strip("/") in existing_hrefs:
                touched_hrefs.append(href.strip("/"))
                continue
            asset_key = f"{var_name}_{resolution}"
            if asset_key in stac_attrs:
                log.warning(
                    "STAC asset key already in use, not adding an asset", asset=asset_key, href=href
                )
                continue
            stac_attrs[asset_key] = {"href": href, "title": asset_key}

    # in the end remove the now redundant /measurement/oa18_radiance -> now in
    redundant_measurement_hrefs = existing_hrefs.difference(touched_hrefs)
    redundant_measurement_hrefs.discard("null")

    if len(redundant_measurement_hrefs) > 0:
        for redundant_href in redundant_measurement_hrefs:
            asset_key = redundant_href.split("/")[
                1
            ]  # eg.: 'measurements/oa08_radiance' -> 'oa08_radiance'
            if asset_key in stac_attrs:
                stac_attrs.pop(asset_key)
            else:
                raise KeyError(
                    f"Redundant href {redundant_href} as asset_key {asset_key} SHOULD be available in `stac_attrs`."
                )

    return


def olci_native_multiscales(
    spec: S3MultiscaleSpec,
    base_resolution_dataset: xr.Dataset,
    output_group: zarr.Group,
    spatial_chunk: int,
    enable_sharding: bool = True,
    compression_level: int = 3,
) -> dict[str, Any]:
    """OLCI multiscales calculated on the native sensor grid."""

    base_path, scale_levels = spec.base_path, spec.scale_levels
    pyramid_dims = SWATH_DIMS

    # rename and save to measurements/r300m
    log.info("Writing native-resolution measurements", shape=dict(base_resolution_dataset.sizes))

    base_resolution_dest_level_name = f"r{scale_levels[0]}"
    base_resolution_dest_level_path = f"{base_path}/{base_resolution_dest_level_name}"

    # Write base resolution dataset
    encoding = utils.create_uniform_encoding(
        base_resolution_dataset,
        enable_sharding=enable_sharding,
        compression_level=compression_level,
        chunk_and_shard_coords=True,
    )

    ds_out = utils.stream_write_dataset(
        base_resolution_dataset,
        path=base_resolution_dest_level_path,
        group=output_group,
        encoding=encoding,
    )

    # creating metadata base fir later ingestion
    dst_processed_groups: dict[str, Any] = {base_resolution_dest_level_path: ds_out}

    resampling_method = "average"

    # predefined layout asset
    layout_: list[dict[str, Any]] = [{"asset": base_resolution_dest_level_name}]

    # calculate MS levels with native grid
    log.info("Generating overview levels", n_levels=len(scale_levels))
    current = base_resolution_dataset

    for source_level, dest_level in pairwise(scale_levels):
        src_level_name = f"r{source_level}"

        dest_level_name = f"r{dest_level}"
        dest_level_path = f"{base_path}/{dest_level_name}"

        downsampled_dataset = utils._rechunk_ds(
            reduce_swath(current, factor=2, dims=pyramid_dims), spatial_chunk=spatial_chunk
        )

        log.info(
            "Writing overview",
            group=f"measurements/{dest_level_name}",
            shape=dict(downsampled_dataset.sizes),
        )
        encoding = utils.create_uniform_encoding(
            downsampled_dataset,
            enable_sharding=enable_sharding,
            compression_level=compression_level,
            chunk_and_shard_coords=True,
        )
        ds_out = utils.stream_write_dataset(
            downsampled_dataset,
            path=dest_level_path,
            group=output_group,
            encoding=encoding,
        )

        current = downsampled_dataset

        dst_processed_groups[dest_level_path] = ds_out
        # Nominal factor (ratio of pixel spacings), not the ratio of shapes: boundary="trim"
        # gives e.g. 305 -> 152 px at r720m. Intentional, see #313.
        transform: Transform = {
            "scale": [2, 2],
            "translation": [0.0, 0.0],
        }
        lo = {
            "asset": dest_level_name,
            "derived_from": src_level_name,
            "transform": transform,
        }

        layout_.append(lo)

    # deal with encodings/layout/MS-metadata
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

    conv = build_convention_attrs(multiscales=ms, spatial=None, crs=None)
    parent_group.attrs.update(cast("dict[str, JSON]", conv))

    # add it as none here to be recognized later and can be created as a zarr root with necessary metadata
    dst_processed_groups[base_path] = None

    return dst_processed_groups


def convert_olci_optimized(
    dt_input: xr.DataTree,
    *,
    output_path: str,
    enable_sharding: bool = False,
    spatial_chunk: int = 1024,
    compression_level: int = 3,
    output_grid: str = "native",
) -> xr.DataTree:
    """Convert a Sentinel-3 OLCI DataTree to an optimized GeoZarr store.

    Unlike `convert_s2_optimized` and `create_generic_geozarr_dataset`, there is
    no ``scale_offset_codec`` option: the packed int32/uint32 variables (lat/lon,
    geometry angles) do not round-trip through the float32 decoding of the
    scale_offset codecs, so OLCI is always written in the CF layout. The CPM
    writer (`eopf_geozarr.cpm.writer`) rejects ``scale_offset_codec=True`` for
    this pipeline.
    """
    start_time = time.time()

    processed_groups: dict[str, xr.Dataset | None] = {}
    output_group: zarr.Group = zarr.open_group(output_path)

    s3_type = S3Type.from_datatree(dt_input)
    if s3_type is None:
        log.warning(
            "Could not determine S3 product level (OLCI/SLSTR);",
            product_type=product_type_of(dt_input),
            name=dt_input.name,
        )

    s3_resolution = ResolutionLevel.from_datatree(dt_input)
    if s3_resolution is None:
        log.warning(
            "Could not determine S3 resolution level (FR - full resolution (300m), RR - reduced resolution (1200m));",
            product_type=product_type_of(dt_input),
            name=dt_input.name,
        )

    # rechunk everything
    for group_path in dt_input.groups:
        if group_path == "/":
            continue

        group_node = dt_input[group_path]

        # Skip parent groups that have children (only process leaf groups)
        if (
            hasattr(group_node, "children")
            and len(group_node.children) > 0
            and len(group_node.data_vars) == 0
        ):
            # this catches empty groups which have children but no data variables themselves. This allows the (theoretical)
            # application of the gener_rechunker to S3 OLC data products as they tend to have data variables on
            # /measurements or other groups, while also having an orphan as a child.
            # This extended check does not limit other products, as verified by the test suite.
            # As in the S2 multiscales, the parent is recorded as `None`, so that
            # `updated_root_consolidation` handles it (e.g. consolidates `/measurements` of
            # S3 SLSTR). Only parents with a written descendant exist in the output store.
            if any(len(node.data_vars) > 0 for node in group_node.descendants):
                processed_groups[group_path] = None
            continue

        base_dataset = group_node.to_dataset()

        # Skip empty groups
        if not base_dataset.data_vars:
            log.info("Skipping empty group", group_path=group_path)
            continue

        log.info("Copying original group", group_path=group_path)

        dataset = utils._rechunk_ds(base_dataset, spatial_chunk)
        is_measurement_group = group_path == "/measurements"

        if is_measurement_group:
            # add function call
            if s3_type == S3Type.OLC.value:
                multiscale_spec = (
                    FR_REFLECTANCE_MULTISCALE
                    if s3_resolution == "FR"
                    else RR_REFLECTANCE_MULTISCALE
                )
                ms_measurement_processed_groups = olci_native_multiscales(
                    multiscale_spec,
                    base_resolution_dataset=dataset,
                    output_group=output_group,
                    spatial_chunk=spatial_chunk,
                    enable_sharding=enable_sharding,
                    compression_level=compression_level,
                )

                # add to processed_groups
                processed_groups.update(ms_measurement_processed_groups)
            elif s3_type == S3Type.SLS.value:
                raise ValueError("Processing here for SLSTR is not implemented yet.")
            else:
                raise ValueError("An `S3Type` needs to be assigned to this datatree.")
        else:
            encoding = utils.create_uniform_encoding(
                dataset,
                enable_sharding=enable_sharding,
                compression_level=compression_level,
                chunk_and_shard_coords=True,
            )

            # Write dataset -> does NOT add geo metadata
            ds_out = utils.stream_write_dataset(
                dataset,
                path=group_path,
                group=output_group,
                encoding=encoding,
            )
            processed_groups[group_path] = ds_out

    update_s3_stac_assets(dt_input, processed_groups)
    # root/sub-root level consolidation and attribute handling
    utils.updated_root_consolidation(dt_input, output_path, processed_groups)

    # Create result DataTree
    result_dt = utils.create_result_datatree(output_path)

    total_time = time.time() - start_time
    log.info("Optimization complete", duration_seconds=round(total_time, 2))

    utils.optimization_summary(dt_input, result_dt, output_path)

    return result_dt
