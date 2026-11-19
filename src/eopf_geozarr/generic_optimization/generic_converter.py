"""
GeoZarr compliant conversion tools for EOPF datasets.

This module provides functions to convert EOPF datasets to GeoZarr format
without any application of multiscales. Dataset are rechunked and sharded,
with geozarr attributes only applied where feasible.

"""

import os
import time
from collections.abc import Mapping

import structlog
import xarray as xr
import zarr

from eopf_geozarr.conversion import utils

log = structlog.get_logger()


def update_generic_stac_assets(dt_input: xr.DataTree, datasets: Mapping[str, object]) -> None:
    """Make the `stac_discovery` assets match the written arrays (hotfix for incomplete CPM metadata).

    CPM keys the root assets by variable name, so for products with several subroots
    (e.g. the bursts of S1 SLC) the assets of all but one subroot overwrite each other,
    and arrays such as `conditions/gcp/*` get no asset at all. This function

    1. drops assets whose href points neither at a written array nor at a group above one,
    2. re-keys assets with a bare variable-name key if there are several subroots, so the
       key names its subroot (e.g. `slc` -> `slc_IW3_258176`),
    3. adds an asset for every written array that none references yet,
    4. gives each subroot the root assets that lie inside it.

    Asset keys for subroot arrays are `<var>[_<group>]_<subroot id>`, where the subroot id is
    the subroot name without the prefix all subroots share, and `_<group>` is added only for
    variable names that occur in more than one group of the subroot (e.g. `roll` in
    `conditions/attitude` and `conditions/antenna_pattern`). With a single subroot the id is
    left out. Keys that are not a bare variable name (e.g. `osw_hs`) are kept as they are.

    Updates the `stac_discovery` attributes of `dt_input` in place, so call it before
    `updated_root_consolidation` writes the root and subroot metadata from them.
    """
    stac = dt_input.attrs.get("stac_discovery")
    assets = stac.get("assets") if isinstance(stac, Mapping) else None
    if not isinstance(assets, dict):
        return

    # every written array, as store path without leading "/"
    written: list[str] = []
    for group_path, ds in datasets.items():
        if ds is None:
            continue
        if not isinstance(ds, xr.Dataset):
            log.warning("Found non-dataset object in datasets!", dataset=ds)
            continue
        written.extend(
            f"{group_path.strip('/')}/{var}" for var in ds.data_vars if var != "spatial_ref"
        )
    if not written:
        return
    written_set = set(written)
    written_groups = {
        "/".join(path.split("/")[:i]) for path in written for i in range(1, path.count("/") + 1)
    }

    subroots = sorted(s.strip("/") for s in utils.get_subroots(dt_input.groups) or ())
    # short subroot ids: drop the name prefix all subroots share (cut at a "_")
    if len(subroots) > 1:
        names = [s.rpartition("/")[2] for s in subroots]
        common = os.path.commonprefix(names)
        common = common[: common.rfind("_") + 1]
        subroot_ids = {s: n[len(common) :] or n for s, n in zip(subroots, names, strict=True)}
    else:
        subroot_ids = dict.fromkeys(subroots, "")

    def subroot_of(path: str) -> str | None:
        return next((s for s in subroots if path.startswith(f"{s}/")), None)

    # variable names that occur in more than one group of the same subroot (or of the root)
    groups_by_name: dict[tuple[str | None, str], set[str]] = {}
    for path in written:
        group, _, var = path.rpartition("/")
        groups_by_name.setdefault((subroot_of(path), var), set()).add(group)

    def asset_key(path: str, with_group: bool = False) -> str:
        group, _, var = path.rpartition("/")
        subroot = subroot_of(path)
        parts = [var]
        if with_group or len(groups_by_name[(subroot, var)]) > 1:
            parts.append(group.rpartition("/")[2])
        if subroot is not None and subroot_ids[subroot]:
            parts.append(subroot_ids[subroot])
        return "_".join(parts)

    updated: dict[str, object] = {}
    referenced: set[str] = set()
    for key, asset in assets.items():
        href = asset.get("href") if isinstance(asset, Mapping) else None
        if not utils._is_store_href(href):
            updated[key] = asset  # URLs and assets without href are left alone
            continue
        path = href.strip("/")
        if path not in written_set and path not in written_groups:
            log.info("Removing STAC asset of a not written array", asset=key, href=href)
            continue
        if path in written_set and len(subroots) > 1 and key == path.rpartition("/")[2]:
            new_key = asset_key(path)
            asset = {**asset, "title": new_key} if asset.get("title") == key else asset
            key = new_key
        if key in updated:
            log.warning("STAC asset key already in use, not adding an asset", asset=key, href=href)
            continue
        updated[key] = asset
        referenced.add(path)

    for path in written:
        if path in referenced:
            continue
        key = asset_key(path)
        if key in updated:
            # key taken by a CPM asset of another array (e.g. GRD `slant_range_time` -> `sr0`)
            key = asset_key(path, with_group=True)
        if key in updated:
            log.warning(
                "STAC asset key already in use, not adding an asset", asset=key, href=f"/{path}"
            )
            continue
        updated[key] = {"href": f"/{path}", "title": key}

    assets.clear()
    assets.update(updated)

    # subroots carry the root assets that lie inside them
    for subroot in subroots:
        sub_stac = dt_input[subroot].attrs.get("stac_discovery")
        if not isinstance(sub_stac, dict):
            continue
        sub_stac["assets"] = {
            key: asset
            for key, asset in updated.items()
            if isinstance(asset, Mapping)
            and utils._is_store_href(asset.get("href"))
            and subroot_of(asset["href"].strip("/")) == subroot
        }


def create_generic_geozarr_dataset(
    dt_input: xr.DataTree,
    output_path: str,
    spatial_chunk: int,
    enable_sharding: bool,
    compression_level: int = 3,
    scale_offset_codec: bool = False,
) -> xr.DataTree:
    """
    Create a GeoZarr-spec compliant dataset from EOPF data with CPM 3.0.0.
    Possibly backward compatabile but not verified and not required, as this generic converter is (so far) only used by EODC which uses solely CPM 3.0.0

    Parameters
    ----------
    dt_input : xr.DataTree
        Input EOPF DataTree
    output_path : str
        Output path for the Zarr store
    spatial_chunk : int, default 1024 (loaded in the writer.py script)
        Spatial chunk size for encoding
    enable_sharding : bool
        Enable zarr sharding for spatial dimensions of each variable
    compression_level: int, default 3

    Returns
    -------
    xr.DataTree
        DataTree containing the GeoZarr compliant data
    """
    start_time = time.time()

    output_group = zarr.open_group(output_path)
    processed_groups: dict[str, xr.Dataset | None] = {}

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

        encoding = utils.create_uniform_encoding(
            dataset,
            enable_sharding=enable_sharding,
            compression_level=compression_level,
            scale_offset_codec=scale_offset_codec,
        )

        # Write dataset -> does NOT add geo metadata
        ds_out = utils.stream_write_dataset(
            dataset,
            path=group_path,
            group=output_group,
            encoding=encoding,
        )
        processed_groups[group_path] = ds_out

    update_generic_stac_assets(dt_input, processed_groups)

    # root/sub-root level consolidation and attribute handling
    utils.updated_root_consolidation(dt_input, output_path, processed_groups)

    # Create result DataTree
    result_dt = utils.create_result_datatree(output_path)

    total_time = time.time() - start_time
    log.info("Optimization complete", duration_seconds=round(total_time, 2))

    utils.optimization_summary(dt_input, result_dt, output_path)

    return result_dt
