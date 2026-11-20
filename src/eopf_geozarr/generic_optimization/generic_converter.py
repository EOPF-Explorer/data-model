"""
GeoZarr compliant conversion tools for EOPF datasets.

This module provides functions to convert EOPF datasets to GeoZarr format
without any application of multiscales. Dataset are rechunked and sharded,
with geozarr attributes only applied where feasible.

"""

import re
import time
from collections.abc import Mapping

import structlog
import xarray as xr
import zarr

from eopf_geozarr.conversion import utils

log = structlog.get_logger()

#: End of an S1 burst subroot name: `_<swath>_<burst id>`, e.g. `..._IW1_258175`.
_BURST_SUFFIX = re.compile(r"_((?:IW|EW|WV|S)\d)_(\d+)$")


def update_generic_stac_assets(dt_input: xr.DataTree, datasets: Mapping[str, object]) -> None:
    """Make the `stac_discovery` assets match the written arrays (hotfix for incomplete CPM metadata).

    CPM keys the root assets by variable name, so for products with several subroots
    (e.g. the bursts of S1 SLC) the assets of all but one subroot overwrite each other,
    and arrays such as `conditions/gcp/*` get no asset at all. This function

    1. drops assets whose href points neither at a written array nor at a group above one,
    2. adds an asset for every written array that none references yet,
    3. gives each subroot the root assets that lie inside it.

    An added asset is keyed by a fixed rule that only depends on its path:
    `<var>_<group>[_<swath>_<burst>]`, where `<group>` is the name of the group holding the
    array (left out if `<var>` already ends with `_<group>`, e.g. `azimuth_time_gcp`) and
    `_<swath>_<burst>` is the end of an S1 burst subroot name
    (e.g. `.../S01SIWSLC_..._IW1_258175/conditions/gcp/height` -> `height_gcp_IW1_258175`).
    Subroots not named after a burst (e.g. S1 GRD) add no suffix. Existing (CPM) assets keep
    their keys, except a bare variable-name key of an array in a burst subroot (e.g. `slc`),
    which collides between the bursts and is re-keyed by the same rule.

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

    def burst_of(path: str) -> re.Match[str] | None:
        return next((m for c in path.split("/") if (m := _BURST_SUFFIX.search(c))), None)

    def asset_key(path: str) -> str:
        group, _, var = path.rpartition("/")
        group_name = group.rpartition("/")[2]
        # no doubled group name, e.g. `gcp/azimuth_time_gcp` -> `azimuth_time_gcp`
        parts = [var] if var.endswith(f"_{group_name}") else [var, group_name]
        burst = burst_of(path)
        if burst is not None:
            parts.extend(burst.groups())
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
        if key == path.rpartition("/")[2] and burst_of(path) is not None:
            # a bare variable-name key collides between the bursts (CPM keeps only one of
            # them, e.g. `slc` for the last burst), so it gets the fixed key as well
            new_key = asset_key(path)
            asset = {**asset, "title": new_key} if asset.get("title") == key else asset
            key = new_key
        if key in updated:
            log.warning("STAC asset key already in use, not keeping an asset", asset=key, href=href)
            continue
        updated[key] = asset
        referenced.add(path)

    for path in written:
        if path in referenced:
            continue
        key = asset_key(path)
        if key in updated:
            log.warning(
                "STAC asset key already in use, not adding an asset", asset=key, href=f"/{path}"
            )
            continue
        updated[key] = {"href": f"/{path}", "title": key}

    assets.clear()
    assets.update(updated)

    # subroots carry the root assets that lie inside them. Merged into the existing subroot
    # assets (e.g. URL hrefs are kept); an existing asset of the same array is replaced, so
    # an array never has two assets (e.g. a bare `slc` next to `slc_measurements_IW1_258175`)
    for subroot in sorted(s.strip("/") for s in utils.get_subroots(dt_input.groups) or ()):
        sub_stac = dt_input[subroot].attrs.get("stac_discovery")
        if not isinstance(sub_stac, dict):
            continue
        sub_assets = sub_stac.get("assets")
        if not isinstance(sub_assets, dict):
            sub_assets = sub_stac["assets"] = {}
        subroot_assets = {
            key: asset
            for key, asset in updated.items()
            if isinstance(asset, Mapping)
            and utils._is_store_href(asset.get("href"))
            and asset["href"].strip("/").startswith(f"{subroot}/")
        }
        subroot_hrefs = {asset["href"].strip("/") for asset in subroot_assets.values()}
        for key in [
            key
            for key, asset in sub_assets.items()
            if isinstance(asset, Mapping)
            and utils._is_store_href(asset.get("href"))
            and asset["href"].strip("/") in subroot_hrefs
        ]:
            del sub_assets[key]
        sub_assets.update(subroot_assets)


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
