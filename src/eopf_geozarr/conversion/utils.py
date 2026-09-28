"""Utility functions for GeoZarr conversion."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
import rasterio  # Import to enable .rio accessor
import rasterio.transform
import structlog
import xarray as xr
import zarr
import zarr_cm
from pyproj import CRS
from zarr_cm import GeoProjAttrs, MultiConventionAttrs, MultiscalesAttrs, SpatialAttrs
from zarr_cm import geo_proj as geo_proj_cm
from zarr_cm import spatial as spatial_cm

from eopf_geozarr.conversion import fs_utils
from eopf_geozarr.data_api.geozarr.types import (
    CF_SCALE_OFFSET_KEYS,
    XARRAY_ENCODING_KEYS,
    XarrayDataArrayEncoding,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from affine import Affine

from importlib.util import find_spec

DISTRIBUTED_AVAILABLE = find_spec("distributed") is not None


# Dimension names that represent a "band-like" axis (polarization) to allow a per-"band" sharding if they extend beyond ram
# purposedly doeStn inlcude the 'band' option to not impleemnt on small enOugh arrays
BAND_LIKE_DIM_NAMES = frozenset({"polarization"})

_LEGACY_CODEC_ENCODING_KEYS = {"compressor", "compressors", "filters"}

log = structlog.get_logger()

CF_STANDARD_NAME_URL = "https://raw.githubusercontent.com/cf-convention/cf-convention.github.io/master/Data/cf-standard-names/current/src/cf-standard-name-table.xml"


@lru_cache(maxsize=1)
def _cf_standard_names() -> frozenset[str]:
    """Fetch + cache the CF standard-name table once, lazily, on first use (never at import)."""
    import urllib.request

    from cf_xarray.utils import parse_cf_standard_name_table

    try:
        with urllib.request.urlopen(CF_STANDARD_NAME_URL, timeout=5) as resp:
            _info, table, _aliases = parse_cf_standard_name_table(source=resp)
        return frozenset(table)
    except Exception as e:  # offline, GitHub down, etc. — never block a write over this
        log.warning(
            "Could not fetch CF standard-name table; skipping standard_name checks", error=str(e)
        )
        return frozenset()


def warn_if_not_cf_standard_name(name: str | None) -> None:
    """Log (never raise) if *name* isn't a recognised CF standard name."""
    table = _cf_standard_names()
    if name and table and name not in table:
        log.warning("standard_name not in CF standard name table", standard_name=name)


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


def simple_root_consolidation(
    dt_input: xr.DataTree, output_path: str, datasets: Mapping[str, object]
) -> None:
    """Simple root-level metadata consolidation with proper zarr group creation."""
    # create missing intermediary groups (/conditions, /quality, etc.)
    # using the keys of the datasets dict
    missing_groups = set()
    for group_path in datasets:
        # extract all the parent paths
        parts = group_path.strip("/").split("/")
        for i in range(1, len(parts)):
            parent_path = "/" + "/".join(parts[:i])
            if parent_path not in datasets:
                missing_groups.add(parent_path)

    for group_path in missing_groups:
        dt_parent = xr.DataTree()

        # check if the parent root (eg per burst for slc) has root attributes which need to be added to root -> consolidated metadata
        ref_root = dt_input[group_path]
        group_attrs = ref_root.attrs
        if len(group_attrs) > 0:
            root_attrs = ["stac_discovery", "other_metadata", "processing_history"]
            for attr_key in group_attrs:
                if attr_key in root_attrs:
                    dt_parent.attrs.update({attr_key: group_attrs[attr_key]})
                else:
                    log.warning(
                        "Couldnt allocate available root attribute to those usually found at nested roots",
                        not_found_key=attr_key,
                        available_root_keys=root_attrs,
                    )

        dt_parent.to_zarr(
            output_path + group_path,
            mode="a",
            zarr_format=3,
            consolidated=False,
        )

        # also add some geo root metadata if its a parent root
        if len(group_attrs) > 0 and "stac_discovery" in group_attrs:
            write_store_root_geo_metadata(output_path + group_path, input_root_attrs=group_attrs)  # type: ignore[arg-type]

    # Create root zarr group if it doesn't exist
    log.info("Creating root zarr group")
    dt_root = xr.DataTree()
    dt_root.to_zarr(
        output_path,
        mode="a",
        consolidated=False,
        zarr_format=3,
    )
    dt_root = xr.DataTree()
    for group_path in datasets:
        dt_root[group_path] = xr.DataTree()

    dt_root.to_zarr(
        output_path,
        mode="r+",
        consolidated=False,
        zarr_format=3,
    )
    log.info("Root zarr group created")

    # Write the store-root spatial footprint (geozarr minispec, Store Root section).
    # Aggregates child-group `spatial:bbox` values, reprojects them to EPSG:4326
    # and writes the union on the root `zarr.json`.
    write_store_root_geo_metadata(output_path, input_root_attrs=dt_input.attrs)  # type: ignore[arg-type]

    if dt_input and dt_input.attrs:
        # this can be used to add multiscale paths to the stac attributes
        # wether we want that or not has to be discussed
        # -> For now this data is not added, as we dont want to expose the additional multiscale arrays for users in the stac assets, this comes at the possibility of confusion for users, but we accept that risk
        # as users wont need the multiscale, but they are just used for visualisation
        # the code is currently commented out, as this discussion is not 100% final yet and changes might apply

        # updated_stac_attrs = add_multiscale_pyramids_to_stac_metadata(datasets, dt_input.attrs)
        # utils.write_store_root_stac_metadata(
        #     output_path,
        #     root_attrs=cast("dict[str, dict[str, Any]]", updated_stac_attrs),
        # )

        write_store_root_stac_metadata(
            output_path,
            root_attrs=cast("dict[str, dict[str, Any]]", dt_input.attrs),
        )

    # consolidate measurement group metadata
    # check if its available from root -> SLC has none! (only nested)
    if "/measurements" in dt_root.groups:
        zarr.consolidate_metadata(output_path + "/measurements", zarr_format=3)
    if "/multiscales" in dt_root.groups:
        zarr.consolidate_metadata(output_path + "/multiscales", zarr_format=3)
    else:
        log.info(
            "Couldnt find a '/measurement' or '/multiscales' group in root -> trying to find measurements or multiscales in children"
        )
        consolidated_groups = []
        for group in dt_input.groups:
            if "/measurements" in str(group) or "/multiscales" in str(group):
                zarr.consolidate_metadata(output_path + group, zarr_format=3)
                consolidated_groups.append(group)
        if len(consolidated_groups) > 0:
            log.info(
                "consolidating other '/measurement' or '/multiscales' groups: ",
                groups=consolidated_groups,
            )
        else:
            log.warning(
                "Couldnt find a '/measurement' or '/multiscales' group at all -> nothing consolidated"
            )

    # consolidate root group metadata
    zarr.consolidate_metadata(output_path, zarr_format=3)


def updated_root_consolidation(
    dt_input: xr.DataTree, output_path: str, datasets: Mapping[str, object]
) -> None:
    """Simple root-level and sub-root level metadata consolidation with proper zarr group creation."""
    # catch all recently added groups (eg. multiscales)
    missing_groups = set()

    # add all measurement groups to (poTentially) consolidate them
    # -> separate from `subroot_groups` as it functions as it being empyt
    # functions as the distinction between consolidating subroots and roots
    measurements_groups = set()

    # create missing intermediary groups (/conditions, /quality, etc.) using the keys of the datasets dict
    # also gather potential measurement groups for consolidating
    for group_path in datasets:
        # extract all the parent paths and names
        parent, _, name = group_path.rstrip("/").rpartition("/")
        if parent not in datasets:
            missing_groups.add(parent)
        if name == "measurements":
            measurements_groups.add(group_path)

    # instantiate missing groups
    for group_path in missing_groups:
        dt_parent = xr.DataTree()

        dt_parent.to_zarr(
            output_path + group_path,
            mode="a",
            zarr_format=3,
            consolidated=False,
        )

    # check and get possible subroots for consolidation
    subroot_groups = get_subroots(dt_input.groups)

    # also add some geo- and stac-root metadata
    if subroot_groups:
        for subroot in subroot_groups:
            write_store_root_geo_metadata(
                output_path + subroot,
                input_root_attrs=cast("dict[str, dict[str, Any]]", dt_input[subroot].attrs),
            )

            write_store_root_stac_metadata(
                output_path + subroot,
                root_attrs=cast("dict[str, dict[str, Any]]", dt_input[subroot].attrs),
            )

    # Create root zarr group if it doesn't existand add the subroot/subgroup groups as zarr arrays
    log.info("Creating root zarr group")
    dt_root = xr.DataTree()
    dt_root.to_zarr(
        output_path,
        mode="a",
        consolidated=False,
        zarr_format=3,
    )
    dt_root = xr.DataTree()
    for group_path in datasets:
        dt_root[group_path] = xr.DataTree()

    dt_root.to_zarr(
        output_path,
        mode="r+",
        consolidated=False,
        zarr_format=3,
    )
    log.info("Root zarr group created")

    # Write the store-root spatial footprint (geozarr minispec, Store Root section).
    # Aggregates child-group `spatial:bbox` values, reprojects them to EPSG:4326
    # and writes the union on the root `zarr.json`.
    write_store_root_geo_metadata(output_path, input_root_attrs=dt_input.attrs)  # type: ignore[arg-type]

    if dt_input and dt_input.attrs:
        # this can be used to add multiscale paths to the stac attributes
        # wether we want that or not has to be discussed
        # -> For now this data is not added, as we dont want to expose the additional multiscale arrays for users in the stac assets, this comes at the possibility of confusion for users, but we accept that risk
        # as users wont need the multiscale, but they are just used for visualisation
        # the code is currently commented out, as this discussion is not 100% final yet and changes might apply

        # updated_stac_attrs = add_multiscale_pyramids_to_stac_metadata(datasets, dt_input.attrs)
        # utils.write_store_root_stac_metadata(gi
        #     output_path,
        #     root_attrs=cast("dict[str, dict[str, Any]]", updated_stac_attrs),
        # )

        write_store_root_stac_metadata(
            output_path,
            root_attrs=cast("dict[str, dict[str, Any]]", dt_input.attrs),
        )

    if measurements_groups:
        for consolidate_measurement in measurements_groups:
            zarr.consolidate_metadata(output_path + consolidate_measurement, zarr_format=3)
    # consolidate metadata in root OR in each subroot
    if subroot_groups:
        for consolidate_subroot in subroot_groups:
            zarr.consolidate_metadata(output_path + consolidate_subroot, zarr_format=3)
    else:
        zarr.consolidate_metadata(output_path, zarr_format=3)


def get_subroots(
    groups: tuple[str, ...], subroot_markers: set[str] | None = None
) -> set[str] | None:
    if not subroot_markers:
        subroot_markers = {"measurements", "conditions", "quality"}
    subroots = set()
    for path in groups:
        parent, _, name = path.rstrip("/").rpartition("/")
        if name in subroot_markers and parent and parent not in subroots:
            subroots.add(parent)
    return subroots or None


def add_multiscale_pyramids_to_stac_metadata(
    datasets: Mapping[str, object], dt_attributes: dict[Hashable, Any]
) -> dict[Hashable, Any]:
    stac_attrs = dt_attributes["stac_discovery"]["assets"]

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

    # replace attrs
    dt_attributes["stac_discovery"]["assets"] = stac_attrs
    return dt_attributes


def transform_from_coordinates(
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


def rio_transform_matches_coordinates(
    transform: tuple[float, float, float, float, float, float] | None,
    coordinate_transform: tuple[float, float, float, float, float, float] | None,
) -> bool:
    """Check whether rio-derived metadata matches the current x/y grid."""
    if transform is None or coordinate_transform is None:
        return False

    return all(np.isclose(a, b) for a, b in zip(transform, coordinate_transform, strict=False))


def preferred_spatial_transform(
    dataset: xr.Dataset,
) -> tuple[float, float, float, float, float, float] | None:
    """Prefer rio metadata only when it matches the current coordinate grid."""
    coordinate_transform = transform_from_coordinates(dataset)
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
        and rio_transform_matches_coordinates(rio_transform, coordinate_transform)
    ):
        return rio_transform

    return coordinate_transform or rio_transform


def remove_geozarr_attrs(ds: xr.Dataset) -> None:
    remove_conventions = {"spatial", "proj", "multiscale", "zarr_conventions"}

    attrs = ds.attrs.copy()
    for attr in attrs:
        for conv in remove_conventions:
            if conv in attr:
                ds.attrs.pop(attr)

    for var in ds.data_vars.values():
        vattrs = var.attrs.copy()
        for attr in vattrs:
            for conv in remove_conventions:
                if conv in attr:
                    ds.attrs.pop(attr)


def _half_pixel(coords: np.ndarray) -> float:
    """Half the grid spacing of a coordinate array, or 0.0 when it has no spacing."""
    if len(coords) < 2:
        log.warning("Changing half-pixel offset and it triggerd len(coords) < 2")
        return 0.0
    return float(np.abs(coords[1] - coords[0])) / 2


def write_geo_metadata(
    dataset: xr.Dataset,
    grid_mapping_var_name: str = "spatial_ref",
    crs: CRS | None = None,
    input_is_image_array: bool = True,  # describes if the datainput is a transform based image array -> S1 and S3 are likely swaths or irregular grids
) -> None:
    """
    Write geographic metadata to the dataset.

    Args:
        dataset: Dataset to write metadata to
        grid_mapping_var_name: Name for grid mapping variable
        crs: Coordinate Reference System to use (if None, attempts to detect from dataset)
    """

    # Use provided CRS or try to detect from dataset
    def _epsg_from_ds_attrs(epsg: int | str) -> CRS:
        if isinstance(epsg, str) and ("epsg:" in epsg or "EPSG:" in epsg):
            return CRS.from_string(epsg)
        return CRS.from_epsg(epsg)

    if crs is None:
        # check parent dataset
        if "proj:code" in dataset.attrs:
            epsg = dataset.attrs["proj:code"]
            crs = _epsg_from_ds_attrs(epsg)

            # assert same set for children - i think it would be against spec, ut jsut to be sure
            for var in dataset.data_vars.values():
                if "proj:code" in var.attrs:
                    if var.attrs["proj:code"] == epsg:
                        # aligns with parent - thats alright
                        continue
                    # not aligning with parent -> problem!
                    crs = None
                    log.warning(
                        "CRS of children data variable doesnt align with dataset parent",
                        child_crs=var.attrs["proj:code"],
                        parent_crs=epsg,
                    )
                    remove_geozarr_attrs(dataset)
                    return

        else:
            for var in dataset.data_vars.values():
                if hasattr(var, "rio") and var.rio.crs:
                    crs = var.rio.crs
                    break
                if "proj:code" in var.attrs:
                    epsg = var.attrs["proj:code"]
                    crs = _epsg_from_ds_attrs(epsg)
                    break

    if crs is None:
        # introducing  here to raise warning for non-aligment of geospatial metadata
        log.warning("No CRS set.")
        remove_geozarr_attrs(dataset)
        return

    # Write CRS using rioxarray
    # NOTE: for now rioxarray only supports writing grid mapping using CF conventions
    dataset.rio.write_crs(crs, grid_mapping_name=grid_mapping_var_name, inplace=True)
    dataset.rio.write_grid_mapping(grid_mapping_var_name, inplace=True)
    dataset.attrs["grid_mapping"] = grid_mapping_var_name

    for var in dataset.data_vars.values():
        var.rio.write_grid_mapping(grid_mapping_var_name, inplace=True)
        var.attrs["grid_mapping"] = grid_mapping_var_name

    # catch the case of having a real image with y/x coords and capable of deribving an affine transform
    if input_is_image_array:
        assert "x" in dataset.coords
        assert "y" in dataset.coords

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

            # this introduces an error in the calculated bbox as it uses pixel center, the corresponding transform uses pixel edges
            # which would lead to inconsitencies when comapring them (halfpixel narrower!)
            # x_min, x_max = float(x_coords.min()), float(x_coords.max())
            # y_min, y_max = float(y_coords.min()), float(y_coords.max())

            # `spatial:registration` below declares "pixel", so the bbox covers the
            # pixel edges. Coordinates are centres, hence the half-pixel outset —
            # without it the footprint is a half-pixel narrower than the raster and
            # disagrees with the arrays' own `proj:bbox`.
            half_x = _half_pixel(x_coords)
            half_y = _half_pixel(y_coords)
            x_min, x_max = float(x_coords.min()) - half_x, float(x_coords.max()) + half_x
            y_min, y_max = float(y_coords.min()) - half_y, float(y_coords.max()) + half_y

            spatial_data["spatial:bbox"] = [x_min, y_min, x_max, y_max]

            spatial_transform = preferred_spatial_transform(dataset)

            # Only add spatial:transform if we have valid transform data (not all zeros)
            if spatial_transform is not None and not all(t == 0 for t in spatial_transform):
                spatial_data["spatial:transform"] = list(spatial_transform)

            # Add spatial shape if data variables exist
            # if dataset.data_vars:
            #     first_var = next(iter(dataset.data_vars.values()))
            #     if first_var.ndim >= 2:
            #         _set = True
            #         height, width = first_var.shape[-2:]
            #         spatial_data["spatial:shape"] = [height, width]

            # new
            # Shape comes from the raster dims themselves: the first data variable
            # need not be 2-D, nor have (y, x) as its trailing dims.
            if "y" in dataset.sizes and "x" in dataset.sizes:
                spatial_data["spatial:shape"] = [
                    int(dataset.sizes["y"]),
                    int(dataset.sizes["x"]),
                ]

        # Build validated spatial + proj convention attrs (data + CMOs) via zarr-cm
        dataset.attrs.update(build_convention_attrs(spatial=spatial_data, crs=crs))

        return

    # irregular grids of S3 or swath based data from S1
    # assert "latitude" in dataset.coords and "longitude" in dataset.coords and dataset.coords["latitude"].ndim == 2
    dataset.attrs.update(build_convention_attrs(spatial=None, crs=crs))
    return


def stream_write_dataset(
    dataset: xr.Dataset,
    *,
    path: str,
    group: zarr.Group,
    encoding: dict[str, XarrayDataArrayEncoding],
    enable_sharding: bool,
    chunk_and_shard_coords=True,
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

    # Sanitize NaN values in dataset attributes before writing
    dataset = fs_utils.sanitize_dataset_attributes(dataset)

    # Write with streaming computation and progress tracking
    # The to_zarr operation will trigger all lazy computations

    write_job = dataset.to_zarr(
        store=group.store,
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


def overview_levels(rows: int, cols: int, min_dimension: int) -> int:
    """Return the number of /2 decimations before min(rows, cols) drops below min_dimension.

    A level is generated only when the *post*-decimation minimum spatial
    dimension is at least *min_dimension*.  For example, a 512x480 dataset
    with min_dimension=256 yields zero levels because 480//2=240 < 256, while
    a 1024x1024 dataset with min_dimension=256 yields two levels (512x512,
    then 256x256).
    """
    levels = 0
    r, c = rows, cols
    while min(r, c) // 2 >= min_dimension:
        r, c = r // 2, c // 2
        levels += 1
    return levels


def clear_encoding(ds: xr.Dataset) -> xr.Dataset:
    """Return *ds* with all inherited source encoding cleared.

    When the input DataTree was opened from a Zarr v2 store, xarray carries
    ``numcodecs.Blosc`` compressors (and potentially scale-offset filters) in
    each variable's ``.encoding``.  Passing that encoding to
    ``Dataset.to_zarr(zarr_format=3)`` raises::

        TypeError: Expected a BytesBytesCodec. Got <class 'numcodecs.blosc.Blosc'>

    because numcodecs codecs are not valid Zarr v3 BytesBytesCodecs.  Clearing
    the encoding lets the Zarr v3 writer choose its own default codecs.

    This converter expects raw (non-mask-scaled) input: the caller must open
    the source DataTree with ``mask_and_scale=False`` so that CF
    ``scale_factor``/``add_offset`` stay in ``.attrs`` and integer fill pixels
    are identified via ``attrs["_FillValue"]``.  Only Zarr v2 *codec* encoding
    (e.g. ``numcodecs.Blosc`` compressors) is stripped here — CF metadata is
    untouched.
    """
    ds = ds.copy()
    for key in _LEGACY_CODEC_ENCODING_KEYS:
        ds.encoding.pop(key, None)
    for var in list(ds.data_vars) + list(ds.coords):
        for key in _LEGACY_CODEC_ENCODING_KEYS:
            ds[var].encoding.pop(key, None)
    return ds


def coarsen_variable(
    var_name: str, var_data: xr.DataArray, factor: int, other_fill_value: int | None = None
) -> xr.DataArray:
    """Coarsen a single variable using type-aware resampling.

    Dispatches to the appropriate coarsen reduction (mean, max, subsample)
    based on `determine_variable_type`.  Preserves encoding and dtype.
    """
    # some data products have several 'nodata' values:
    # S1 GRDH has a fill value of 65535 due to the reprojection. The other 'fill_value' (-> 0) is already present in the data as such, and should be kept in the data to distinguish it from reprojection-based nodata
    # to also consider these, the attribute 'other_fill_value' is added, which also functions similar to the fill_value, but gets reinserted as such at the end of the coarsening

    coarsened = var_data.coarsen({"x": factor, "y": factor}, boundary="trim")
    # Cast the input array to float and ignore nans during the .coarsen() operation, which could not be considered in int array with "nan-value" == 0.
    # This prohibits the inclusion of 0 values in the mean calculation of multiscales, mainly impacting the boder regions of arrays

    # nan values are later refilled again with 0s (or fillna values) to conform with int array requirements
    fill_value = var_data.attrs.get("fill_value")

    # resort to _FillValue from encoding (-> likely empty anyway, as encodings are set later) is not found via fill_value
    if not fill_value:
        fill_value = var_data.encoding.get("_FillValue")
    if fill_value is not None:
        # mask all 0 as nan in float array
        masked = var_data.where(var_data != fill_value)

        # redefine coarsen operation to ignore nans and fill up with fill_value later
        result = (
            masked.coarsen({"x": factor, "y": factor}, boundary="trim")
            .mean(skipna=True)  # type: ignore[attr-defined]
            .fillna(fill_value)
        )
    else:
        result = coarsened.mean()  # type: ignore[attr-defined]

    # `xr.DataArray.astype` clears `.encoding`, so we capture it first and
    # restore it on the cast result. Without this, downstream code that
    # inspects encoding (e.g. to push CF scale-offset into a codec pipeline)
    # would see an empty encoding on every coarsened level.
    encoding = var_data.encoding
    cast_result: xr.DataArray = result.astype(var_data.dtype)
    cast_result.encoding = encoding
    return cast_result


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


def _band_like_dim_index(var_data: xr.DataArray) -> int | None:
    """Index of the band-like dimension, if var_data has one. Falls back to
    None (caller should then skip band-axis sharding, not guess)."""
    if len(var_data.dims) <= 2:
        log.info(
            "Not sharding along bandlike dim if its not dim>2", band_like_dim=BAND_LIKE_DIM_NAMES
        )
    else:
        for i, dim in enumerate(var_data.dims):
            if dim in BAND_LIKE_DIM_NAMES:
                return i
    return None


def _rechunk_ds(ds: xr.Dataset, spatial_chunk: int, chunk_data: tuple | None = None) -> xr.Dataset:
    if chunk_data:
        if len(ds.sizes) != len(chunk_data):
            log.warning(
                "chunk_data not same length as data variables:",
                data_vars=list(ds.data_vars.keys()),
                chunk_keys=chunk_data,
            )

        chunks_ = {}
        for (dim, size), chunk_ in zip(ds.sizes.items(), chunk_data, strict=True):
            chunks_[dim] = chunk_ if chunk_ != -1 else size
        return ds.chunk(chunks_)
    chunks = {dim: (min(spatial_chunk, size)) for dim, size in ds.sizes.items()}
    return ds.chunk(chunks)


def rechunk_dataset_for_encoding(
    dataset: xr.Dataset,
    encoding: dict[str, XarrayDataArrayEncoding],
    chunk_and_shard_coords: bool = False,
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

    if chunk_and_shard_coords:
        rechunked_coords: dict[Hashable, xr.DataArray] = {}

        for coord_name, coord_data in dataset.coords.items():
            if str(coord_name) in encoding:
                coord_encoding = encoding[str(coord_name)]

                # If sharding is enabled, rechunk based on shard dimensions
                if "shards" in coord_encoding and coord_encoding["shards"] is not None:
                    target_chunks = coord_encoding["shards"]  # Use shard dimensions for rechunking
                elif "chunks" in coord_encoding:
                    target_chunks = coord_encoding["chunks"]  # Fallback to chunk dimensions
                else:
                    # No specific chunking needed, use original coordiable
                    rechunked_coords[coord_name] = coord_data
                    continue

                # Create chunk dict using the actual dimensions of the coordiable
                coord_dims = coord_data.dims
                chunk_dict = {}
                for i, dim in enumerate(coord_dims):
                    if i < len(target_chunks):
                        chunk_dict[dim] = target_chunks[i]

                # Rechunk the coordiable to match the target dimensions
                rechunked_coords[coord_name] = coord_data.chunk(chunk_dict)
            else:
                # No specific chunking needed, use original coordiable
                rechunked_coords[coord_name] = coord_data

        # Create new dataset with rechunked variables, also sharding coordinates
        return xr.Dataset(rechunked_vars, coords=rechunked_coords, attrs=dataset.attrs)

    # Create new dataset with rechunked variables, preserving coordinates
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

        # consider the occurance of 1dim arrays, provide the encoding chunk ndim times
        return (max_chunksizes[0],) if var_data.ndim == 1 else tuple(max_chunksizes)
    raise ValueError(
        f"Datavariable {var_data.name!r} is not chunked already, cannot derive Zarr encoding chunks -> will lead to unchunked array"
    )


def create_uniform_encoding(
    dataset: xr.Dataset,
    *,
    spatial_chunk: int,
    enable_sharding: bool = True,
    shard_along_smallest_dimension: bool = False,
    keep_scale_offset: bool = True,
    compression_level: int = 3,
    chunk_and_shard_coords: bool = False,
) -> dict[str, XarrayDataArrayEncoding]:
    """
    Create encoding (compression, chunking, sharding) for a dataset.

    Chunking is taken from the input dataset's existing chunks when present
    (e.g. a group that's already been rechunked/aggregated, such as a
    pyramid level or a group written with `preferred_chunks`). Only when a
    variable has no chunks at all do we compute a chunk grid from
    `spatial_chunk`. Sharding always covers the *entire* array along every
    dimension, sized as the smallest multiple of that dimension's chunk size
    that is >= the array's shape — so a shard always contains a whole number
    of chunks and there is exactly one shard per array. This avoids partial
    edge chunks ending up in their own oddly-sized shard when e.g. shape=1830
    and chunk=1024 (shard becomes 2048, i.e. 2 chunks, not some 1830-based
    value that would clip/overlap the second chunk).
    """
    import math

    from zarr.codecs import BloscCodec

    encoding: dict[str, XarrayDataArrayEncoding] = {}
    compressor = BloscCodec(cname="zstd", clevel=compression_level, shuffle="shuffle", blocksize=0)

    for var_name, var_data in dataset.data_vars.items():
        var_encoding: XarrayDataArrayEncoding = {}

        encoding_chunks = get_chunking_for_encoding(var_data, shard_along_smallest_dimension)

        var_encoding["chunks"] = encoding_chunks
        var_encoding["compressors"] = (compressor,)

        # --- Shards: cover the whole array, one shard per array -----------
        if enable_sharding:
            # select next largest mutliple of chunksize to fit full array
            shards_ = [
                math.ceil(shape / chunk) * chunk
                for shape, chunk in zip(var_data.shape, encoding_chunks, strict=True)
            ]
            if shard_along_smallest_dimension:
                band_dim = _band_like_dim_index(var_data)
                if band_dim is None:
                    log.warning(
                        "shard_along_smallest_dimension=True but %s has no "
                        "recognized band-like dimension (%s); falling back to "
                        "whole-array sharding",
                        var_data.name,
                        list(var_data.dims),
                    )
                else:
                    shards_[band_dim] = encoding_chunks[band_dim]
            var_encoding["shards"] = tuple(shards_)
        else:
            var_encoding["shards"] = None

        # --- Forward-propagate remaining encoding keys ---------------------
        keep_keys = XARRAY_ENCODING_KEYS - {"compressors", "shards", "chunks"}

        # Whether to inject a CF _FillValue attribute for xarray issue #11345.
        # The injection itself happens after sanitize_array_attrs below, which
        # would otherwise strip it.
        inject_nan_fillvalue = False

        if not keep_scale_offset:
            # When stripping scale/offset, also strip _FillValue since the original
            # _FillValue is in raw integer units and meaningless for decoded float data.
            keep_keys = keep_keys - CF_SCALE_OFFSET_KEYS - {"_FillValue"}
            var_encoding["fill_value"] = "NaN"
            inject_nan_fillvalue = True
        else:
            # Not stripping scale/offset: pick an explicit zarr-level fill_value
            # rather than letting xarray infer one differently across versions.
            keep_keys = keep_keys - {"fill_value"}
            fv = explicit_fill_value(var_data)
            if fv is not UNSET:
                var_encoding["fill_value"] = fv
            else:
                # We need to pass _FillValue in the encoding to allow decode_cf to read it..
                # either this, or it gets removed from everywhere else during the sanitize_array_attrs() call
                if "fill_value" in var_data.attrs and "_FillValue" not in var_encoding:
                    var_encoding["_FillValue"] = var_data.attrs["fill_value"]
                else:
                    pass

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
        var_data.attrs = sanitize_array_attrs(var_data.attrs, is_decoded_float=is_float)
        if inject_nan_fillvalue:
            var_data.attrs["_FillValue"] = np.nan

            # need to validate this logic here - not tested but kept from orignal encoding function
            # original_encoding = var_data.encoding
            # dataset[var_name] = var_data.astype(np.float32)
            # dataset[var_name].encoding = original_encoding

        encoding[str(var_name)] = var_encoding

    for coord_name, coord_data in dataset.coords.items():
        coord_encoding: XarrayDataArrayEncoding = {}

        if chunk_and_shard_coords:
            if (
                coord_name in dataset.xindexes
            ):  # skip indexed coords which are likely not chunked -> check for chunked
                continue

            encoding_chunks = get_chunking_for_encoding(coord_data, shard_along_smallest_dimension)

            coord_encoding["chunks"] = encoding_chunks
            coord_encoding["compressors"] = (compressor,)

            # --- Shards: cover the whole array, one shard per array -----------
            if enable_sharding:
                # select next largest mutliple of chunksize to fit full array
                shards_ = [
                    math.ceil(shape / chunk) * chunk
                    for shape, chunk in zip(coord_data.shape, encoding_chunks, strict=True)
                ]
                if shard_along_smallest_dimension:
                    band_dim = _band_like_dim_index(coord_data)
                    if band_dim is None:
                        log.warning(
                            "shard_along_smallest_dimension=True but %s has no "
                            "recognized band-like dimension (%s); falling back to "
                            "whole-array sharding",
                            coord_data.name,
                            list(coord_data.dims),
                        )
                    else:
                        shards_[band_dim] = encoding_chunks[band_dim]

                coord_encoding["shards"] = tuple(shards_)
            else:
                coord_encoding["shards"] = None
        else:
            coord_encoding["compressors"] = (compressor,)

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


# Sentinel: distinguish "no explicit fill_value" from a legitimate `None`.
UNSET: Any = object()


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
        for key in ("dtype", "fill_value", "valid_min", "valid_max"):
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


def _as_bbox(value: object) -> tuple[float, float, float, float] | None:
    """Return *value* as a 4-tuple of floats, or ``None`` if it is not one.

    ``spatial:bbox`` is read from stored metadata, so its type is not known
    statically; this verifies the shape at runtime rather than asserting it.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(v, (int, float)) for v in value):
        return None
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _crs_from_attrs(attrs: dict[str, Any]) -> Any | None:
    """Resolve a pyproj CRS from a group's ``proj:*`` attributes, else ``None``.

    Tries ``proj:code``, then ``proj:wkt2``, then ``proj:projjson``. Malformed
    values are logged and treated as unresolvable rather than raised, since the
    attributes come from stored metadata.
    """
    from pyproj import CRS as PyprojCRS

    code = attrs.get("proj:code")
    if isinstance(code, str):
        try:
            return PyprojCRS.from_user_input(code)
        except Exception as e:  # malformed stored metadata
            log.warning("Unresolvable proj:code; skipping group bbox", code=code, error=str(e))
            return None
    wkt2 = attrs.get("proj:wkt2")
    if isinstance(wkt2, str):
        try:
            return PyprojCRS.from_wkt(wkt2)
        except Exception as e:
            log.warning("Unresolvable proj:wkt2; skipping group bbox", error=str(e))
            return None
    projjson = attrs.get("proj:projjson")
    if isinstance(projjson, dict):
        try:
            return PyprojCRS.from_json_dict(projjson)
        except Exception as e:
            log.warning("Unresolvable proj:projjson; skipping group bbox", error=str(e))
            return None
    return None


def write_store_root_geo_metadata(
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

    bboxes_4326: list[tuple[float, float, float, float]] = []

    def _walk(group: zarr.Group) -> None:
        attrs = dict(group.attrs)
        corners = _as_bbox(attrs.get("spatial:bbox"))
        if corners is not None:
            crs = _crs_from_attrs(attrs)
            if crs is None:
                if any(k in attrs for k in ("proj:code", "proj:wkt2", "proj:projjson")):
                    # warning already logged by _crs_from_attrs
                    pass
                else:
                    log.warning(
                        "Group has spatial:bbox but no proj:* CRS; skipping it "
                        "for the store-root footprint",
                        group=group.path,
                    )
            elif crs.to_epsg() == 4326:
                bboxes_4326.append(corners)
            else:
                try:
                    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
                    # transform_bounds densifies the edges, which corner-wise
                    # transformation misses (projected edges curve in lon/lat).
                    bboxes_4326.append(transformer.transform_bounds(*corners, densify_pts=21))
                except Exception as e:  # never abort the write for one group
                    log.warning(
                        "Failed to reproject group bbox; skipping it",
                        group=group.path,
                        error=str(e),
                    )
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
                from shapely.geometry import shape

                geoms = stac_attrs["geometry"]
                coords = shape(geoms)
                bboxes_4326.append(coords.bounds)
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


def write_store_root_stac_metadata(
    output_path: str,
    root_attrs: dict[str, dict[str, Any]],
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
        new_attrs = set(root_attrs.keys())
        for int_attr in original_attrs.intersection(new_attrs):
            root_attrs.pop(int_attr)

    root.attrs.update(root_attrs)
    log.info(
        "Updated root metadata attributes for STAC ingestion", root_attrs=list(root_attrs.keys())
    )
