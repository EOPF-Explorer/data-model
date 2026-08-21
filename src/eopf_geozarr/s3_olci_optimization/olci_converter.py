"""Top-level Sentinel-3 OLCI L1 EFR -> GeoZarr conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import rioxarray  # noqa: F401
import structlog
import xarray as xr
import zarr
from rasterio.crs import CRS

from eopf_geozarr.conversion.utils import build_convention_attrs
from eopf_geozarr.data_api.s3_olci import Sentinel3OlciRoot
from eopf_geozarr.s3_olci_optimization.olci_band_mapping import OLCI_BANDS
from eopf_geozarr.s3_olci_optimization.olci_multiscale import (
    SWATH_DIMS,
    grid_spatial_attrs,
    reduce_swath,
    swath_spatial_attrs,
)
from eopf_geozarr.s3_olci_optimization.olci_reproject import GRID_DIMS, reproject_olci

if TYPE_CHECKING:
    from zarr.core.common import JSON
    from zarr_cm import LayoutObject, MultiscalesAttrs, SpatialAttrs, Transform

log = structlog.get_logger()


def _sanitize_olci_array_attrs_keep_fill(attrs: dict[str, object]) -> dict[str, object]:
    """Return a copy of *attrs* with stale source-only keys removed.

    Strips ``_eopf_attrs``, ``dtype``, ``valid_min``, and ``valid_max`` (source
    provenance and raw-integer-domain metadata that is misleading in GeoZarr
    output).  Unlike the shared :func:`~eopf_geozarr.conversion.utils.sanitize_array_attrs`,
    this helper intentionally **preserves** ``_FillValue`` because OLCI input is
    opened with ``mask_and_scale=False`` (raw uint16) and downstream code (e.g.
    ``reduce_swath``) needs ``_FillValue`` in ``.attrs`` to identify fill pixels
    without CF decoding.

    CF keys ``scale_factor``, ``add_offset``, ``units``, ``standard_name``,
    ``coordinates``, and ``long_name`` are always preserved.
    """
    _strip = frozenset(("_eopf_attrs", "dtype", "valid_min", "valid_max"))
    return {k: v for k, v in attrs.items() if k not in _strip}


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


def _overview_levels(rows: int, cols: int, min_dimension: int) -> int:
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


def _clear_encoding(ds: xr.Dataset) -> xr.Dataset:
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
    ds.encoding = {}
    for var in list(ds.data_vars) + list(ds.coords):
        ds[var].encoding.clear()
    return ds


def _sanitize_data_vars(ds: xr.Dataset) -> xr.Dataset:
    """Return *ds* with stale source attrs stripped from all data variables.

    Applies :func:`_sanitize_olci_array_attrs_keep_fill` to every data variable in *ds*.
    Coordinate variable attrs are left intact.

    This removes ``_eopf_attrs``, ``dtype``, ``valid_min``, and ``valid_max``
    (source-only / misleading) while preserving CF attrs
    (``scale_factor``, ``add_offset``, ``_FillValue``, ``units``,
    ``standard_name``, ``coordinates``).

    Note: ``xr.DataArray.assign_attrs`` *merges* (update semantics), so we
    copy the DataArray and replace ``.attrs`` in-place to ensure stale keys
    are actually removed rather than retained from the old dict.
    """
    new_vars: dict[str, xr.DataArray] = {}
    for name in ds.data_vars:
        var = ds[name]
        new_var = var.copy(data=var.data)
        new_var.attrs = _sanitize_olci_array_attrs_keep_fill(dict(var.attrs))
        new_vars[str(name)] = new_var
    return ds.assign(new_vars)


def _copy_subtree(node: xr.DataTree, output_path: str, *, root_group: str) -> None:
    """Write every Dataset in *node*'s subtree to the Zarr store at *output_path*.

    Each path is mapped relative to the node's own path: the node root becomes
    *root_group*, and child nodes are placed at ``root_group/<relative_path>``.

    ``DataTree.to_zarr`` does not yet support a ``group`` keyword for
    specifying a root offset, so we write each leaf's :py:meth:`~xarray.DataTree.to_dataset`
    using ``xr.Dataset.to_zarr`` instead.
    """
    node_path = node.path  # e.g. "/conditions"
    for child in node.subtree:
        ds = child.to_dataset()
        if not ds.data_vars and not ds.coords:
            continue
        # Strip any inherited Zarr v2 encoding before writing to a v3 store.
        ds = _clear_encoding(ds)
        # Build the group path: strip the ancestor prefix and prepend root_group.
        relative = child.path[len(node_path) :]  # "" for root, "/sub" for children
        group_path = root_group + relative
        log.info("Copying ancillary subgroup", group=group_path)
        ds.to_zarr(
            output_path,
            group=group_path,
            mode="a",
            consolidated=False,
            zarr_format=3,
        )


def convert_olci_optimized(
    dt_input: xr.DataTree,
    *,
    output_path: str,
    enable_sharding: bool = False,
    spatial_chunk: int = 1024,
    compression_level: int = 3,
    min_dimension: int = 256,
    keep_scale_offset: bool = False,
    output_grid: str = "native",
) -> xr.DataTree:
    """Convert an EOPF OLCI L1 EFR DataTree to a GeoZarr multiscale store.

    Writes the measurements pyramid to ``measurements/r0`` (+ ``r2``, ``r4``,
    … siblings). By default the instrument grid is preserved; pass
    ``output_grid=<CRS>`` to warp once onto a regular grid with per-level CRS
    metadata.

    Parameters
    ----------
    dt_input:
        Input OLCI L1 EFR DataTree (must contain a ``/measurements`` node).
    output_path:
        Filesystem path for the output Zarr v3 store.
    enable_sharding:
        Enable Zarr v3 sharding on measurement arrays.
        Not yet wired into encoding for this minimal pass; accepted as a
        typed parameter for forward-compatibility (follow-up task).
    spatial_chunk:
        Target spatial chunk size (pixels per side).
        Not yet wired into encoding for this minimal pass; accepted as a
        typed parameter for forward-compatibility (follow-up task).
    compression_level:
        Blosc/zstd compression level.
        Not yet wired into encoding for this minimal pass; accepted as a
        typed parameter for forward-compatibility (follow-up task).
    min_dimension:
        Stop generating overview levels once either spatial dimension would
        drop below this value after /2 decimation.
    keep_scale_offset:
        When ``True``, preserve CF ``scale_factor``/``add_offset`` in the
        output encoding rather than decoding to float32.
        Not yet wired into encoding for this minimal pass; accepted as a
        typed parameter for forward-compatibility (follow-up task).
    output_grid:
        ``"native"`` (default) preserves the instrument swath geometry:
        no warp, 2-D lat/lon geolocation, per-row ``time_stamp`` kept,
        and no CRS metadata.  Any other value is parsed as a CRS
        (``"EPSG:4326"``, WKT, …) and the swath is warped once onto a
        regular grid in that CRS before the pyramid builds.

    Returns
    -------
    xr.DataTree
        The opened output DataTree (lazy; backed by the written Zarr store).
        Opened with ``mask_and_scale=False``, mirroring the raw store and the
        converter's input: radiance is packed ``uint16`` with its CF
        ``scale_factor``/``_FillValue`` attrs intact, not decoded floats.
        Native-resolution arrays live at ``measurements/r0`` with overview
        levels (``r2``, ``r4``, …) as sibling groups, all on the instrument
        grid (default) or a regular ``output_grid`` grid with 1-D ``y``/``x``
        coordinates and a declared CRS; ``measurements`` itself holds only
        the multiscales/spatial convention metadata, so the whole store
        opens cleanly with ``xr.open_datatree``.

    Notes
    -----
    Parameters ``enable_sharding``, ``spatial_chunk``, ``compression_level``,
    and ``keep_scale_offset`` are accepted but not yet applied to the on-disk
    encoding.  Wiring them through the existing ``conversion`` helpers
    (``create_measurements_encoding``, sharding codec, etc.) is left for a
    follow-up task so as not to block the integration test.  A warning is
    logged when a non-default value is passed for any of them, so callers
    aren't silently handed default-encoded output.
    """
    # Fail fast before any store mutation: _overview_levels floor-halves the
    # dimensions, and min(r, c) // 2 >= min_dimension never becomes false for
    # min_dimension <= 0 once the sizes decay to zero (infinite loop).
    if min_dimension < 1:
        raise ValueError(f"min_dimension must be >= 1; got {min_dimension}")

    unwired: dict[str, tuple[object, object]] = {
        "enable_sharding": (enable_sharding, False),
        "spatial_chunk": (spatial_chunk, 1024),
        "compression_level": (compression_level, 3),
        "keep_scale_offset": (keep_scale_offset, False),
    }
    ignored = [name for name, (value, default) in unwired.items() if value != default]
    if ignored:
        log.warning(
            "Options not yet applied by the OLCI converter; output uses default encoding",
            ignored_options=ignored,
        )

    measurements = dt_input["/measurements"].to_dataset()
    # Structural detection does not constrain dimension names, but the whole
    # swath pipeline (reduce_swath, decimate_swath, SWATH_DIMS) assumes
    # rows/columns; fail with a clear error instead of a bare KeyError below.
    missing_dims = [d for d in ("rows", "columns") if d not in measurements.sizes]
    if missing_dims:
        raise ValueError(
            "OLCI converter requires swath dimensions ('rows', 'columns') on the "
            f"measurements group; missing {missing_dims}. Use the generic convert "
            "path for products with different dimension names."
        )
    # Strip any inherited Zarr v2 encoding (e.g. numcodecs.Blosc compressors)
    # so the v3 writer can choose its own default codecs without raising a
    # "Expected a BytesBytesCodec" error.  The caller is expected to have opened
    # the source with mask_and_scale=False, so CF attrs (scale_factor,
    # add_offset, _FillValue) live in .attrs and are preserved here.
    measurements = _clear_encoding(measurements)
    # Sanitize radiance variable attrs: strip source-only / misleading attrs
    # (_eopf_attrs, dtype, valid_min, valid_max) while keeping CF scale/offset
    # and _FillValue so that downstream readers and reduce_swath can work
    # correctly with raw integer data.
    measurements = _sanitize_data_vars(measurements)

    # Resolve the output-grid mode once. "native" keeps instrument geometry;
    # anything else must parse as a CRS and selects the warp pipeline.
    crs_obj: CRS | None
    if output_grid == "native":
        crs_obj = None
    else:
        try:
            crs_obj = CRS.from_string(output_grid)
        except Exception as e:
            raise ValueError(
                f"output_grid must be 'native' or a CRS string "
                f"(e.g. 'EPSG:4326'); got {output_grid!r}"
            ) from e

    if crs_obj is not None:
        # Warp the curvilinear swath onto a regular grid (1-D y/x coords,
        # spatial_ref + grid_mapping on every variable) at (approximately)
        # native resolution. Everything downstream — the pyramid, spatial
        # attrs, and CRS metadata — operates on this gridded dataset.
        measurements = reproject_olci(measurements, target_crs=output_grid)
        # rioxarray's write_crs records grid_mapping in both .attrs and
        # .encoding; xarray's to_zarr refuses to serialize a variable whose
        # attrs and encoding disagree on an encoding-owned key, so clear the
        # inherited encoding once more after the warp.
        measurements = _clear_encoding(measurements)
        pyramid_dims = GRID_DIMS
    else:
        pyramid_dims = SWATH_DIMS

    # Truncate any pre-existing store first: the writes below are per-group
    # (mode="w" scoped to measurements/r0, mode="a" for overviews/ancillary),
    # so a prior run with more overview levels or extra ancillary groups would
    # otherwise leave stale sibling groups behind, and the returned DataTree
    # (built by re-scanning the store) would surface them.
    zarr.open_group(output_path, mode="w", zarr_format=3)

    # The native-resolution arrays go in a named child group (r0) alongside the
    # overview groups (r2, r4, …) rather than directly in ``measurements``.
    # If the parent held the full-res coordinates itself, every overview child
    # would inherit them over the shared y/x dims at mismatched sizes and
    # ``xr.open_datatree`` (and any generic GeoZarr reader) would reject the
    # store with an alignment error.
    log.info("Writing native-resolution measurements", shape=dict(measurements.sizes))
    measurements.to_zarr(
        output_path,
        group="measurements/r0",
        mode="w",
        consolidated=False,
        zarr_format=3,
    )

    # Write /2 reduced overview subgroups: r2, r4, r8, …
    rows = measurements.sizes[pyramid_dims[0]]
    cols = measurements.sizes[pyramid_dims[1]]
    n_levels = _overview_levels(rows, cols, min_dimension)
    log.info("Generating overview levels", n_levels=n_levels)

    level_datasets: dict[str, xr.Dataset] = {"r0": measurements}
    base_transform = measurements.rio.transform(recalc=True) if crs_obj is not None else None
    current = measurements
    for level in range(1, n_levels + 1):
        current = reduce_swath(current, factor=2, dims=pyramid_dims)
        current = _clear_encoding(current)
        # Attrs already sanitized at native level and passed through by
        # reduce_swath; no second sanitize pass needed.
        if base_transform is not None:
            # Radiance is block-AVERAGED, so an overview coordinate is the
            # CENTER of the 2^level x 2^level base-pixel block it aggregates.
            # Stride-decimated coords (the first fine pixel's center) would
            # shift every level's recomputed transform/bbox by
            # (2^level - 1)/2 base pixels and contradict the multiscales
            # layout's declared {scale: [2, 2], translation: [0, 0]}.
            # Derive edge-aligned coords from the r0 transform instead.
            step = float(2**level)
            xs = base_transform.c + base_transform.a * step * (np.arange(current.sizes["x"]) + 0.5)
            ys = base_transform.f + base_transform.e * step * (np.arange(current.sizes["y"]) + 0.5)
            current = current.assign_coords(
                x=("x", xs, dict(measurements["x"].attrs)),
                y=("y", ys, dict(measurements["y"].attrs)),
            )
        group_name = f"r{2**level}"
        level_datasets[group_name] = current
        log.info("Writing overview", group=f"measurements/{group_name}", shape=dict(current.sizes))
        current.to_zarr(
            output_path,
            group=f"measurements/{group_name}",
            mode="a",
            consolidated=False,
            zarr_format=3,
        )

    # Build and attach GeoZarr convention metadata (spatial + multiscales CMO)
    # to the measurements group attrs.
    layout: list[LayoutObject] = [{"asset": "r0"}]
    for lvl in range(1, n_levels + 1):
        transform: Transform = {"scale": [2.0, 2.0], "translation": [0.0, 0.0]}
        lo: LayoutObject = {
            "asset": f"r{2**lvl}",
            "derived_from": f"r{2 ** (lvl - 1)}" if lvl > 1 else "r0",
            "transform": transform,
            "resampling_method": "average",
        }
        layout.append(lo)

    root_rw = zarr.open_group(output_path, mode="a")

    def _level_spatial(level_ds: xr.Dataset) -> SpatialAttrs:
        if crs_obj is None:
            return swath_spatial_attrs()
        return grid_spatial_attrs(
            level_ds.rio.transform(recalc=True),
            (level_ds.sizes["y"], level_ds.sizes["x"]),
        )

    base_spatial = _level_spatial(measurements)
    for group_name, level_ds in level_datasets.items():
        level_conv = build_convention_attrs(spatial=_level_spatial(level_ds), crs=crs_obj)
        root_rw[f"measurements/{group_name}"].attrs.update(cast("dict[str, JSON]", level_conv))

    if n_levels > 0:
        ms: MultiscalesAttrs = {"layout": layout, "resampling_method": "average"}
        conv = build_convention_attrs(multiscales=ms, spatial=base_spatial, crs=crs_obj)
    else:
        conv = build_convention_attrs(spatial=base_spatial, crs=crs_obj)

    root_rw["measurements"].attrs.update(cast("dict[str, JSON]", conv))

    # Copy conditions/quality through unchanged (if present).
    # DataTree.to_zarr does not support a root ``group`` argument, so we
    # iterate the subtree and write each leaf Dataset individually.
    for grp in ("conditions", "quality"):
        try:
            node_item = dt_input[f"/{grp}"]
        except KeyError:
            continue
        if not isinstance(node_item, xr.DataTree):
            continue
        log.info("Copying ancillary group", group=grp)
        _copy_subtree(node_item, output_path, root_group=grp)

    # Copy any child subgroups of measurements (e.g. orphans) through unchanged.
    try:
        meas_node = dt_input["/measurements"]
    except KeyError:
        meas_node = None
    if isinstance(meas_node, xr.DataTree):
        for child in meas_node.children.values():
            log.info("Copying measurements subgroup", group=f"measurements/{child.name}")
            _copy_subtree(child, output_path, root_group=f"measurements/{child.name}")

    # The r0 named-sibling layout keeps every parent group free of arrays, so
    # the whole store — overview levels and nested ancillary groups included —
    # opens directly as a DataTree. mask_and_scale=False so the returned tree
    # mirrors the raw store (packed uint16 + CF attrs), matching how the
    # converter opened its input, rather than handing callers CF-decoded
    # floats.
    return xr.open_datatree(
        output_path,
        engine="zarr",
        chunks={},
        consolidated=False,
        mask_and_scale=False,
    )
