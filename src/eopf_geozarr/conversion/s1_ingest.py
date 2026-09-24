"""S1 GRD RTC GeoTIFF → GeoZarr V3 ingestion pipeline.

Converts S1Tiling γ⁰ RTC (GammaNaughtRTC) GeoTIFF outputs into a sharded Zarr V3 store
with multiscale overviews, spatial coordinate arrays, and full GeoZarr
convention metadata.

Public API:
    - extract_geotiff_metadata(path) -> S1TilingMetadata
    - ingest_s1tiling_acquisition(vv_path, vh_path, border_mask_path, store_path, orbit_direction) -> int
    - ingest_s1tiling_conditions(store_path, orbit_direction, relative_orbit, ...) -> None
    - consolidate_s1_store(store_path, orbit_direction) -> None
    - discover_s1tiling_acquisitions(input_dir) -> list[dict]
    - discover_s1tiling_conditions(input_dir) -> list[dict]
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import re
import struct
from dataclasses import dataclass
from math import ceil, isclose
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import rasterio
import structlog
import zarr
import zarr.codecs
from pyproj import CRS as PyprojCRS
from rasterio.errors import RasterioIOError
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr_cm import geo_proj
from zarr_cm import multiscales as multiscales_cm
from zarr_cm import spatial as spatial_cm

from eopf_geozarr.conversion import fs_utils, utils
from eopf_geozarr.conversion.utils import calculate_aligned_chunk_size
from eopf_geozarr.types import make_bounding_box, make_crs_code

if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractContextManager

    from zarr.core.common import JSON

    from eopf_geozarr.data_api.geozarr.types import S1BackscatterAttrsJSON, S1TimeCoordAttrsJSON
    from eopf_geozarr.types import BoundingBox2D, CRSCode

log = structlog.get_logger()


# =============================================================================
# Zarr member-access helpers
# =============================================================================
#
# Under pyright, ``group[name]`` is typed ``zarr.Array | zarr.Group``. The store layout
# guarantees which one a given child is, so these helpers narrow (and assert) the type at
# the single point of access. The ``raise`` branches encode a store-layout invariant and
# are effectively unreachable — the established idiom in this codebase (see geozarr.py and
# s2_multiscale.py).


def _child_group(group: zarr.Group, name: str) -> zarr.Group:
    member = group[name]
    if not isinstance(member, zarr.Group):
        raise TypeError(f"expected a group at {name!r}, got {type(member).__name__}")
    return member


def _child_array(group: zarr.Group, name: str) -> zarr.Array:
    member = group[name]
    if not isinstance(member, zarr.Array):
        raise TypeError(f"expected an array at {name!r}, got {type(member).__name__}")
    return member


# =============================================================================
# Constants
# =============================================================================

MULTISCALES_UUID = multiscales_cm.UUID
GEO_PROJ_UUID = geo_proj.UUID
SPATIAL_UUID = spatial_cm.UUID

ZARR_CONVENTIONS = [multiscales_cm.CMO, geo_proj.CMO, spatial_cm.CMO]

# Generation stamp for the on-disk layout, written to the root as `eopf:writer_schema`.
#
# 1 (implicit, unstamped) = stores written by eopf-geozarr <= 0.10.2.
# 2 = this writer: complete root attributes and per-group convention declarations.
#
# Bump this whenever a change alters array geometry, coordinate values or array names, so that a
# reader can tell generations apart. `_open_store_for_write` compares against it. The comparison
# only *warns* today because generation 2 is layout-compatible with generation 1 -- it adds
# metadata and changes nothing about array geometry, so appending a generation-2 orbit group to a
# generation-1 store is safe and the conformance migration heals the metadata. The pending chunk
# and coordinate changes (fix-plan F6/F10/F12/B1) are NOT layout-compatible; the release that
# lands them must bump this to 3 and turn the warning into a hard refusal, otherwise one cube can
# carry an ascending group with cell-centre coordinates and a descending group with edge
# coordinates under a single STAC item.
#
# It covers LAYOUT, not pixel semantics, and it is written ONLY at store creation (`create_s1_store`)
# -- never on append. So it cannot mark a cube whose time slices carry mixed semantics: bumping it
# would stamp the cubes built fresh under the new writer (uniformly new, not mixed) and leave every
# cube that actually becomes mixed by appending unstamped. Exactly inverted. The F11 overview-mask
# rule change (`nearest` -> block-`max`) therefore deliberately does NOT bump it; the conformance
# migration rewrites the affected overviews from r10m, which is idempotent and needs no marker.
WRITER_SCHEMA: Final[int] = 2

# Overview chain: (level_name, parent_name, downsample_factor)
OVERVIEW_CHAIN = [
    ("r10m", None, 1),
    ("r20m", "r10m", 2),
    ("r60m", "r20m", 3),
    ("r120m", "r60m", 2),
    ("r360m", "r120m", 3),
    ("r720m", "r360m", 2),
]

# S1Tiling filename pattern
# e.g. s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif
# Multi-frame products mask the time as 'txxxxxx' (no single shared acquisition time); those are
# accepted here and the real stamp is resolved from the GeoTIFF ACQUISITION_DATETIME tag (#183).
S1TILING_FILENAME_PATTERN = re.compile(
    r"(?P<platform>s1[abc])_"
    r"(?P<tile>[0-9]{2}[A-Z]{3})_"
    r"(?P<pol>vv|vh)_"
    r"(?P<orbit_dir>ASC|DES)_"
    r"(?P<rel_orbit>\d{3})_"
    r"(?P<acq_stamp>\d{8}t(?:\d{6}|x{6}))_"
    r"(?P<product>GammaNaughtRTC)"
    r"(?P<mask>_BorderMask)?\.tif$"
)

# The orbit-direction group name each short filename code maps to. Discovery reads the short
# form out of the filename; the store, the ingester and STAC all speak the long form.
ORBIT_DIR_TO_GROUP: Final = {"ASC": "ascending", "DES": "descending"}
# The long forms `ingest_s1tiling_acquisition` accepts, derived so the two cannot drift.
VALID_ORBIT_DIRECTIONS: Final = tuple(ORBIT_DIR_TO_GROUP.values())

# The bundle keys the ingester actually reads. `vh_mask` is discovered and returned for
# completeness but never consumed, so requiring it would condemn ingestable bundles.
REQUIRED_ACQUISITION_KEYS: Final = ("vv", "vh", "vv_mask")

# S1Tiling conditions filename patterns
# e.g. GAMMA_AREA_31TCH_008.tif or GAMMA_AREA_s1a_31TCH_ASC_008.tif
S1TILING_GAMMA_AREA_PATTERN = re.compile(
    r"^GAMMA_AREA_(?:s1[abc]_)?(?P<tile>[A-Z0-9]+)_(?:(?:ASC|DES)_)?(?P<orbit>\d{3})\.tif$",
    re.IGNORECASE,
)
# e.g. sin_LIA_31TCH_008.tif or LIA_31TCH_008.tif
S1TILING_LIA_PATTERN = re.compile(
    r"^(?P<kind>sin_LIA|LIA)_(?P<tile>[A-Z0-9]+)_(?P<orbit>\d{3})\.tif$",
    re.IGNORECASE,
)


# =============================================================================
# Data Transfer Object
# =============================================================================


@dataclass(frozen=True)
class S1TilingMetadata:
    """Metadata extracted from an S1Tiling GeoTIFF."""

    crs: CRSCode
    spatial_transform: list[float]
    shape: list[int]
    bounds: BoundingBox2D
    datetime: str
    absolute_orbit: int
    relative_orbit: int
    platform: str
    calibration: str
    input_s1_images: str


# =============================================================================
# Metadata Extraction
# =============================================================================


def _normalise_s1tiling_datetime(dt_str: str, source: str | Path | None = None) -> str:
    """Normalise an S1Tiling ``ACQUISITION_DATETIME`` tag to ISO 8601.

    Input:  "2025:02:10T06:09:20Z" (S1Tiling uses colons in the date part)
    Output: "2025-02-10T06:09:20"

    Also accepts the space-separated form ("2025:02:10 06:09:20"), which real S1Tiling output
    does emit -- GDAL writes it that way when the tag round-trips through a TIFF `DateTime`.

    Anything else RAISES here, naming the raw tag and the file it came from. Previously an
    unrecognised form was passed through untouched and only failed ~560 lines later inside
    `np.datetime64(...)`, from a traceback that named neither the tag nor the GeoTIFF -- after
    the ingest had already opened and read the rasters.

    Raises
    ------
    ValueError
        If the normalised value is not a valid ISO 8601 datetime.
    """
    # ' ' is the RFC 3339 / TIFF DateTime separator for the same value, so fold it to 'T' first.
    # The date part uses ':' as its separator and the time part legitimately does too, so only
    # the field before the separator is rewritten.
    dt_normalised = dt_str.strip().removesuffix("Z").replace(" ", "T", 1)
    date_part, sep, time_part = dt_normalised.partition("T")
    if sep:
        dt_normalised = f"{date_part.replace(':', '-')}T{time_part}"

    # Return what the parser produced, NOT the input string. `fromisoformat` is a different
    # parser from the `np.datetime64` that ultimately consumes this value, and merely *checking*
    # with one while passing the other the raw text blesses forms the two disagree about:
    #   '20230115'        -> np.datetime64 reads dtype datetime64[Y], the YEAR 20230115, which
    #                        lands in the store as 2206-09-06 with no error and then makes the
    #                        out-of-order guard reject every later append to that orbit.
    #   '20230115T061234' -> fromisoformat accepts, np.datetime64 raises -- the late failure this
    #                        function exists to prevent.
    # Canonicalising through `.isoformat()` leaves exactly one spelling, which both parsers agree
    # on, so the gate and the consumer can no longer diverge.
    try:
        parsed = dt.datetime.fromisoformat(dt_normalised)
    except ValueError as exc:
        where = f" in {source}" if source is not None else ""
        raise ValueError(
            f"Unparseable ACQUISITION_DATETIME{where}: {dt_str!r} "
            f"(normalised to {dt_normalised!r}). Expected an S1Tiling stamp such as "
            "'2025:02:10T06:09:20Z' or '2025:02:10 06:09:20'."
        ) from exc
    return parsed.isoformat()


def extract_geotiff_metadata(path: str | Path) -> S1TilingMetadata:
    """Extract CRS, transform, bounds, and custom tags from an S1Tiling GeoTIFF.

    Raises
    ------
    ValueError
        If critical tags (ACQUISITION_DATETIME, ORBIT_NUMBER,
        RELATIVE_ORBIT_NUMBER, FLYING_UNIT_CODE) are missing.
    """
    with _rasterio_env(path), rasterio.open(str(path)) as src:
        tags = src.tags()
        t = src.transform
        spatial_transform = [t.a, t.b, t.c, t.d, t.e, t.f]

        # Validate critical tags
        required_tags = [
            "ACQUISITION_DATETIME",
            "ORBIT_NUMBER",
            "RELATIVE_ORBIT_NUMBER",
            "FLYING_UNIT_CODE",
        ]
        missing = [tag for tag in required_tags if tag not in tags]
        if missing:
            raise ValueError(f"GeoTIFF {path} missing required tags: {missing}")

        dt_raw = tags["ACQUISITION_DATETIME"]
        dt_normalised = _normalise_s1tiling_datetime(dt_raw, source=path)

        metadata = S1TilingMetadata(
            crs=make_crs_code(str(src.crs)),
            spatial_transform=spatial_transform,
            shape=[src.height, src.width],
            bounds=make_bounding_box(
                [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
            ),
            datetime=dt_normalised,
            absolute_orbit=int(tags["ORBIT_NUMBER"]),
            relative_orbit=int(tags["RELATIVE_ORBIT_NUMBER"]),
            platform=tags["FLYING_UNIT_CODE"],
            calibration=tags.get("CALIBRATION", ""),
            input_s1_images=tags.get("INPUT_S1_IMAGES", ""),
        )

    log.info(
        "Extracted GeoTIFF metadata",
        path=str(path),
        crs=metadata.crs,
        shape=metadata.shape,
        datetime=metadata.datetime,
    )
    return metadata


def parse_s1tiling_filename(filename: str) -> dict | None:
    """Parse an S1Tiling filename into component fields.

    Returns None if the filename does not match the expected pattern.
    """
    m = S1TILING_FILENAME_PATTERN.match(filename)
    if not m:
        return None
    return {
        "platform": m.group("platform"),
        "tile": m.group("tile"),
        "pol": m.group("pol"),
        "orbit_dir": m.group("orbit_dir"),
        "rel_orbit": m.group("rel_orbit"),
        "acq_stamp": m.group("acq_stamp"),
        "is_mask": m.group("mask") is not None,
    }


# =============================================================================
# Multiscales Layout
# =============================================================================


def compute_multiscales_layout(
    native_shape: list[int],
    native_transform: list[float],
) -> list[dict]:
    """Build the multiscales layout array for all resolution levels."""
    layout: list[dict] = []
    current_shape = native_shape[:]
    current_transform = native_transform[:]

    for level_name, parent_name, factor in OVERVIEW_CHAIN:
        if parent_name is not None:
            current_shape = [
                ceil(current_shape[0] / factor),
                ceil(current_shape[1] / factor),
            ]
            current_transform = [
                current_transform[0] * factor,  # a: pixel width
                current_transform[1],  # b: rotation (0)
                current_transform[2],  # c: x origin
                current_transform[3],  # d: rotation (0)
                current_transform[4] * factor,  # e: pixel height (negative)
                current_transform[5],  # f: y origin
            ]

        entry: dict = {
            "asset": level_name,
            "spatial:shape": current_shape[:],
            "spatial:transform": current_transform[:],
        }
        if parent_name is None:
            entry["transform"] = {"scale": [1.0, 1.0]}
        else:
            entry["derived_from"] = parent_name
            entry["transform"] = {
                "scale": [float(factor), float(factor)],
                "translation": [0.0, 0.0],
            }

        layout.append(entry)

    return layout


# =============================================================================
# Store Creation
# =============================================================================


def _create_spatial_coordinate_arrays(
    level_group: zarr.Group,
    level_h: int,
    level_w: int,
    level_transform: list[float],
    only: Sequence[str] | None = None,
) -> None:
    """Create 1D x and y spatial coordinate arrays at a resolution level.

    ``only`` restricts creation to the named coordinates, for backfilling a group that already
    carries one of them -- recreating an existing coordinate raises ``ContainsArrayError``.
    """
    wanted = ("x", "y") if only is None else tuple(only)
    pixel_w = level_transform[0]  # a: pixel width
    x_origin = level_transform[2]  # c: x origin (left edge)
    pixel_h = level_transform[4]  # e: pixel height (negative)
    y_origin = level_transform[5]  # f: y origin (top edge)

    x_coords = np.linspace(
        x_origin, x_origin + level_w * pixel_w, level_w, endpoint=False, dtype="float64"
    )
    y_coords = np.linspace(
        y_origin, y_origin + level_h * pixel_h, level_h, endpoint=False, dtype="float64"
    )

    if "x" in wanted:
        _create_x_coordinate(level_group, x_coords, level_w)
    if "y" in wanted:
        _create_y_coordinate(level_group, y_coords, level_h)


def _create_x_coordinate(level_group: zarr.Group, x_coords: np.ndarray, level_w: int) -> None:
    x_arr = level_group.create_array(
        "x",
        data=x_coords,
        chunks=(level_w,),
        fill_value=float("nan"),
        dimension_names=["x"],
    )
    x_arr.attrs.update(
        {
            "units": "m",
            "long_name": "x coordinate of projection",
            "standard_name": "projection_x_coordinate",
            "_ARRAY_DIMENSIONS": ["x"],
        }
    )


def _create_y_coordinate(level_group: zarr.Group, y_coords: np.ndarray, level_h: int) -> None:
    y_arr = level_group.create_array(
        "y",
        data=y_coords,
        chunks=(level_h,),
        fill_value=float("nan"),
        dimension_names=["y"],
    )
    y_arr.attrs.update(
        {
            "units": "m",
            "long_name": "y coordinate of projection",
            "standard_name": "projection_y_coordinate",
            "_ARRAY_DIMENSIONS": ["y"],
        }
    )


# CF datetime encoding for the `time` coordinate. Without it the array is a bare int64 and readers
# (xarray / TiTiler's `open_datatree(decode_times=True)`) cannot expose `time` as a datetime index, so
# per-acquisition previews can only select positionally (`sel=time={index}`) — fragile once a cube's
# time axis goes non-monotonic. With these attrs `time` decodes to datetime64 and previews can render by
# `sel=time={datetime}` (order-immune). The stored dtype stays int64 nanoseconds. See data-model #192.
TIME_CF_ATTRS: Final[S1TimeCoordAttrsJSON] = {
    "units": "nanoseconds since 1970-01-01",
    "calendar": "proleptic_gregorian",
    "standard_name": "time",
}

# CF `_FillValue` for the float32 arrays, mirroring the S1 GRD converter (geozarr.py)
# and S2 (data-model #172). This is what lets xarray mask NaN nodata via
# `use_zarr_fill_value_as_mask=True` despite xarray #11345 — the zarr-level `fill_value`
# field alone is not surfaced through xarray's encoding. The stored attribute must match the
# base64 form S2 writes; the S2 path gets it for free via `to_zarr`, but this store is written
# zarr-direct, so the attribute is set explicitly.
# CF `_FillValue` for the float32 arrays: the base64 IEEE-754 NaN bit pattern, which is what
# lets xarray mask NaN nodata via `use_zarr_fill_value_as_mask=True` despite xarray #11345 (the
# zarr-level `fill_value` field alone is not surfaced through xarray's encoding). The S2 path
# gets this for free via `to_zarr`; this store is written zarr-direct, so it is set explicitly.
#
# Encoded with the stdlib rather than xarray's `FillValueCoder`. That encoder is PRIVATE
# (`xarray.backends.zarr`), and importing it at module scope meant any xarray reorganisation
# raised from `import eopf_geozarr` itself, taking down the S2 and OLCI paths that never touch
# S1. Guarding the import only covered the rename case -- a signature or return-type change
# still raised at module scope, since the value is needed at import time either way. Eight bytes
# of `struct` cannot break. `test_float32_nan_fill_value_matches_xarray` keeps the two pinned
# together, so if xarray ever encodes it differently that surfaces as a CI failure here.
FLOAT32_NAN_FILL_VALUE: Final = base64.b64encode(struct.pack("<d", float("nan"))).decode()
# CF metadata for the backscatter bands (vv/vh).
BACKSCATTER_CF_ATTRS: S1BackscatterAttrsJSON = {
    "standard_name": "surface_backwards_scattering_coefficient_of_radar_wave",
    "units": "1",
    "_FillValue": FLOAT32_NAN_FILL_VALUE,
}


def _create_time_coordinate_array(level_group: zarr.Group) -> None:
    """Create the CF-encoded ``time`` coordinate (length 0, grown on append) on one level group.

    Replicated at EVERY multiscale level (with identical values) so datetime ``.sel`` resolves at
    whatever level a reader renders — TiTiler picks a coarse level for previews, and a level lacking a
    ``time`` coordinate cannot be selected by datetime. Keeping the dtype/values consistent across
    levels is also required for the datatree to open (mixed int64/datetime64 fails alignment).
    """
    t_arr = level_group.create_array(
        "time",
        shape=(0,),
        dtype="int64",
        chunks=(512,),
        fill_value=0,
        dimension_names=["time"],
    )
    # cast: zarr's `attrs.update` is typed for its JSON union, which the TypedDict spread
    # (inferred `str | object | list[str]` values) doesn't satisfy; the values are JSON-safe.
    t_arr.attrs.update(cast("dict", {**TIME_CF_ATTRS, "_ARRAY_DIMENSIONS": ["time"]}))


def _add_grid_mapping(group: zarr.Group, crs_string: CRSCode) -> None:
    """Add a CF ``spatial_ref`` grid-mapping coordinate to a group holding (y, x) arrays.

    rioxarray -- and TiTiler's GeoZarr reader -- resolve the CRS from a CF
    ``spatial_ref``/``crs_wkt`` coordinate; the GeoZarr ``proj:code`` attribute alone
    is not read. This mirrors the S2 converter (which writes a ``spatial_ref``
    grid-mapping variable via ``rio.write_crs``). The CF attributes come from
    ``pyproj.CRS.to_cf()`` -- the same source rioxarray uses -- so the projection is
    described correctly for any CRS rather than hard-coded.

    The scalar ``spatial_ref`` array is created (if absent) and every (y, x) data
    array in the group is given ``grid_mapping = "spatial_ref"``.
    """
    cf_attrs = PyprojCRS.from_user_input(crs_string).to_cf()
    # rioxarray writes both ``crs_wkt`` and a ``spatial_ref`` attr holding the WKT.
    cf_attrs["spatial_ref"] = cf_attrs["crs_wkt"]
    cf_attrs["_ARRAY_DIMENSIONS"] = []

    if "spatial_ref" in group:
        sref = group["spatial_ref"]
    else:
        sref = group.create_array("spatial_ref", shape=(), dtype="int64", fill_value=0)
        sref[...] = 0
    sref.attrs.update(cf_attrs)

    for name, arr in group.arrays():
        # This store is always Zarr V3; only V3 metadata carries ``dimension_names``.
        dimension_names = (
            arr.metadata.dimension_names if isinstance(arr.metadata, ArrayV3Metadata) else None
        )
        if name != "spatial_ref" and {"y", "x"}.issubset(dimension_names or ()):
            arr.attrs["grid_mapping"] = "spatial_ref"


def _create_band_arrays(level_group: zarr.Group, level_h: int, level_w: int) -> None:
    """Create the (time, y, x) data bands (vv, vh, border_mask) for one multiscale level.

    Float backscatter bands (vv/vh) get the CF ``_FillValue``/``standard_name``/``units``
    attributes so readers can mask NaN nodata (xarray #11345) — parity with the S1 GRD
    converter and S2 (data-model #172). Shared by ``create_s1_store`` (new store) and
    ``ingest_s1tiling_acquisition`` (new orbit added to an existing store) so the two
    creation paths cannot drift.
    """
    inner_chunks = (
        1,
        calculate_aligned_chunk_size(level_h, 512),
        calculate_aligned_chunk_size(level_w, 512),
    )
    shard_shape = (1, level_h, level_w)
    for name, dtype, fill in [
        ("vv", "float32", float("nan")),
        ("vh", "float32", float("nan")),
        ("border_mask", "uint8", 0),
    ]:
        band = level_group.create_array(
            name,
            shape=(0, level_h, level_w),
            dtype=dtype,
            chunks=inner_chunks,
            shards=shard_shape,
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=5),
            fill_value=fill,
            dimension_names=["time", "y", "x"],
        )
        if name in ("vv", "vh"):
            # cast: zarr's `attrs.update` is typed for its JSON union, which a TypedDict
            # (invariant) doesn't satisfy structurally; the values are JSON-safe.
            band.attrs.update(cast("dict", BACKSCATTER_CF_ATTRS))
        else:
            # The orbit group declares `multiscales.resampling_method: "average"`, which is true
            # for vv/vh and has never been true for the mask -- it was `nearest` before F11 and is
            # block-`max` after. Declare the real rule on the array itself rather than leaving the
            # group-level claim silently wrong. Its ABSENCE on a legacy cube correctly reads as
            # "unknown", which is more than the writer stamp could ever say (see WRITER_SCHEMA).
            band.attrs.update({"resampling_method": "max"})


def _open_for_write(group_path: str, credentials_for: str) -> zarr.Group:
    """Open a group read-write, ignoring any consolidated block.

    ``credentials_for`` is the store URI the storage options are derived from, which is not
    ``group_path`` when opening a subgroup.
    """
    return zarr.open_group(
        group_path,
        mode="r+",
        zarr_format=3,
        use_consolidated=False,
        storage_options=cast(
            "dict[str, object] | None", fs_utils.get_storage_options(credentials_for)
        ),
    )


def _open_store_for_write(store_path: str) -> zarr.Group:
    """Open a cube root for writing, ignoring any consolidated metadata block.

    ``use_consolidated=False`` is required for *membership* to be correct: an orbit group created
    since the last consolidation is otherwise invisible, so a second-orbit append silently lands
    in the wrong place and `consolidate_s1_store` skips the new group entirely.

    Note this fixes membership only. Array *shapes* resolve through the enclosing orbit group's
    own consolidated block, so a caller that needs a true shape must also open that orbit with
    `_open_orbit_for_write` -- the root flag alone leaves shapes stale.
    """
    root = _open_for_write(store_path, store_path)
    stamp = root.attrs.get("eopf:writer_schema")
    if not isinstance(stamp, int) or stamp < WRITER_SCHEMA:
        log.warning(
            "Store predates this writer generation",
            store_path=store_path,
            store_schema=stamp,
            writer_schema=WRITER_SCHEMA,
            detail=(
                "generation 2 is layout-compatible with earlier stores, so the append proceeds; "
                "run the conformance migration to bring root and group metadata up to date"
            ),
        )
    return root


def _open_orbit_for_write(store_path: str, orbit_direction: str) -> zarr.Group:
    """Open one orbit group for writing, ignoring its consolidated metadata block.

    `consolidate_s1_store` writes a block on every orbit group, and `use_consolidated` applies
    only to the group actually opened -- so array shapes read through the root stay stale after a
    consolidation even when the root's own block is bypassed. Reading a stale shape makes every
    post-consolidation append target the same time index.
    """
    return _open_for_write(f"{store_path}/{orbit_direction}", store_path)


def _strip_consolidated_metadata(store_path: str, orbit_direction: str) -> None:
    """Drop the consolidated block from the root and one orbit group after a write.

    Hygiene for external readers: a stale block is worse than none, because it reports the
    pre-append shapes with no indication it is out of date. The write path itself never relies on
    this having run -- it always opens with ``use_consolidated=False`` -- since an append that
    crashes mid-way leaves the block in place.
    """
    fs = fs_utils.get_filesystem(store_path)
    for group_path in (store_path, f"{store_path}/{orbit_direction}"):
        meta_path = f"{group_path}/zarr.json"
        try:
            if not fs.exists(meta_path):
                continue
            meta = json.loads(fs.cat_file(meta_path))
            if "consolidated_metadata" not in meta:
                continue
            del meta["consolidated_metadata"]
            fs.pipe_file(meta_path, json.dumps(meta, indent=2).encode())
        except (OSError, ValueError) as exc:
            log.warning("Could not strip consolidated metadata", path=meta_path, error=str(exc))


def _assert_grid_matches(root: zarr.Group, metadata: S1TilingMetadata) -> None:
    """Raise unless an incoming GeoTIFF sits on exactly the grid the store already holds.

    Compares ``proj:code`` and, against any orbit group already present, the native
    ``spatial:shape`` and all six terms of ``spatial:transform``.

    The transform is the load-bearing comparison. Two adjacent MGRS tiles in the same UTM zone
    share a CRS and a shape and differ only in origin, so a CRS+shape check passes them both and
    slice 1's pixels are served 100 km from where the item says they are, with every downstream
    signal still looking valid. `build_s1_rtc_stac_item` then unions the per-orbit bboxes while
    deriving `grid:code` from the filename, producing a two-tile footprint under one MGRS code.

    Called on both the new-orbit and the append branch: the new-orbit branch writes ``proj:code``,
    the multiscales layout and ``spatial:bbox`` straight from the incoming metadata with no
    comparison at all, so a wrong tile there poisons the whole orbit group.
    """
    store_crs = root.attrs.get("proj:code")
    for orbit_name, orbit_group in root.groups():
        attrs = dict(orbit_group.attrs)
        store_crs = attrs.get("proj:code", store_crs)
        if store_crs is not None and store_crs != metadata.crs:
            raise ValueError(f"CRS mismatch: store has {store_crs}, GeoTIFF has {metadata.crs}")

        multiscales = attrs.get("multiscales")
        layout = multiscales.get("layout", []) if isinstance(multiscales, dict) else []
        if not (isinstance(layout, list) and layout and isinstance(layout[0], dict)):
            continue
        native = layout[0]

        store_shape = native.get("spatial:shape")
        if isinstance(store_shape, list):
            incoming_shape = [int(v) for v in metadata.shape]
            if [int(v) for v in cast("list[float]", store_shape)] != incoming_shape:
                raise ValueError(
                    f"Shape mismatch against orbit '{orbit_name}': store has {store_shape}, "
                    f"GeoTIFF has {incoming_shape}"
                )

        store_transform = native.get("spatial:transform")
        if not isinstance(store_transform, list):
            continue
        incoming_transform = [float(v) for v in metadata.spatial_transform]
        stored_transform = [float(v) for v in cast("list[float]", store_transform)]
        # Compare with a sub-pixel tolerance rather than exactly. The real failure mode is an
        # adjacent MGRS tile, whose origin is ~100 km away -- five orders of magnitude outside
        # any tolerance -- so nothing is lost by allowing float drift between rasterio's computed
        # transform and the value round-tripped through the store's JSON. An exact comparison
        # would risk rejecting every legitimate append across the existing archive to catch a
        # difference that cannot occur in practice.
        pixel = abs(stored_transform[0])
        if len(stored_transform) != len(incoming_transform) or not all(
            isclose(a, b, rel_tol=1e-9, abs_tol=pixel * 1e-3)
            for a, b in zip(stored_transform, incoming_transform, strict=True)
        ):
            raise ValueError(
                f"Grid mismatch against orbit '{orbit_name}': the GeoTIFF is on a different grid "
                f"than the store. Store transform {store_transform}, GeoTIFF transform "
                f"{incoming_transform}. Same CRS and shape with a different origin means a "
                "different MGRS tile -- ingest it into its own store."
            )
        return


def _build_orbit_group(
    root: zarr.Group, orbit_direction: str, metadata: S1TilingMetadata
) -> zarr.Group:
    """Create one orbit group (asc/desc) with full GeoZarr metadata.

    Builds the multiscale level groups (bands + spatial coordinates + grid-mapping +
    per-level ``time``) and the native-resolution per-acquisition metadata coordinates.
    Shared by ``create_s1_store`` (new store) and ``ingest_s1tiling_acquisition`` (new
    orbit added to an existing store) so the two creation paths cannot drift — the inline
    path previously omitted ``proj:code`` on level groups.
    """
    layout = compute_multiscales_layout(metadata.shape, metadata.spatial_transform)
    orbit_group = root.create_group(orbit_direction)
    orbit_group.attrs.update(
        {
            "zarr_conventions": ZARR_CONVENTIONS,
            "multiscales": {
                "layout": layout,
                "resampling_method": "average",
            },
            "proj:code": metadata.crs,
            "spatial:dimensions": ["y", "x"],
            "spatial:bbox": metadata.bounds,
        }
    )

    for level_entry in layout:
        level_name = level_entry["asset"]
        level_h, level_w = level_entry["spatial:shape"]

        level_group = orbit_group.create_group(level_name)
        # A level group must declare its OWN conventions. The validator passes inherited
        # convention UUIDs down only to a group's direct child *arrays*, never to a child group
        # (`validation._walk`), so a subgroup can never inherit them from the orbit group — it
        # was writing `spatial:*` and `proj:*` keys while declaring neither convention, which is
        # 6 of the store's validation issues per level. `spatial:dimensions` is likewise required
        # and was absent; `build_convention_attrs` refuses to build without it.
        level_attrs = utils.build_convention_attrs(
            spatial={
                "spatial:dimensions": ["y", "x"],
                "spatial:shape": [level_h, level_w],
                "spatial:transform": level_entry["spatial:transform"],
            },
            crs=PyprojCRS.from_user_input(metadata.crs),
        )
        level_group.attrs.update(cast("dict[str, JSON]", level_attrs))

        _create_band_arrays(level_group, level_h, level_w)

        _create_spatial_coordinate_arrays(
            level_group, level_h, level_w, level_entry["spatial:transform"]
        )
        _add_grid_mapping(level_group, metadata.crs)
        # `time` coordinate on every level so datetime `.sel` resolves at any rendered scale (#192).
        _create_time_coordinate_array(level_group)

    # Per-time metadata coordinates at native resolution only (not selected on by readers).
    r10m = _child_group(orbit_group, "r10m")
    for name, dtype, fill in [
        ("absolute_orbit", "int32", 0),
        ("relative_orbit", "int32", 0),
    ]:
        r10m.create_array(
            name,
            shape=(0,),
            dtype=dtype,
            chunks=(512,),
            fill_value=fill,
            dimension_names=["time"],
        )
    r10m.create_array(
        "platform",
        shape=(0,),
        dtype="<U4",
        chunks=(512,),
        fill_value="",
        dimension_names=["time"],
    )
    return orbit_group


def create_s1_store(
    store_path: str | Path,
    orbit_direction: str,
    metadata: S1TilingMetadata,
) -> zarr.Group:
    """Create a new S1 GRD RTC Zarr V3 store with full conventions metadata.

    Returns the root group.
    """
    store_path = fs_utils.normalize_path(str(store_path))
    root = zarr.open_group(
        store_path,
        mode="w-",
        zarr_format=3,
        storage_options=cast("dict[str, object] | None", fs_utils.get_storage_options(store_path)),
    )
    # Write a COMPLETE root at creation, not a partial one refined at consolidation.
    # `GeoZarrStoreAttrs.bbox` (alias `spatial:bbox`) has no default, so a store that is never
    # consolidated fails root validation outright. The minispec does not require EPSG:4326 here
    # -- `proj:code` only has to match `^[A-Z]+:[0-9]+$` and `spatial:bbox` only has to be
    # correctly ordered -- and the native CRS and bounds are both already in hand, so the root
    # is written in the native UTM CRS.
    #
    # Note this store's root is NOT passed through `utils.write_store_root_geo_metadata`, which
    # the S2 and OLCI paths call. That helper rewrites `proj:code` to EPSG:4326 and replaces the
    # bbox with a reprojected lon/lat union of the child groups. Adopting it here would change
    # the root's CRS semantics for every reader, so it is a deliberate behaviour decision rather
    # than part of making the store minispec-compliant -- the native-CRS root already validates.
    root.attrs.update(
        {
            "zarr_conventions": ZARR_CONVENTIONS,
            "proj:code": metadata.crs,
            "spatial:bbox": list(metadata.bounds),
            "eopf:writer_schema": WRITER_SCHEMA,
        }
    )
    _build_orbit_group(root, orbit_direction, metadata)

    log.info(
        "Created S1 store",
        store_path=store_path,
        orbit_direction=orbit_direction,
        crs=metadata.crs,
        native_shape=metadata.shape,
    )
    return root


# =============================================================================
# Downsampling (private helper)
# =============================================================================


def _downsample_2d(data: np.ndarray, factor: int, method: str = "average") -> np.ndarray:
    """Downsample a 2D array by the given integer factor.

    ``average`` block-means, ``max`` block-maxes, and both handle non-divisible sizes via edge
    padding on the same ``ceil(h / factor)`` grid -- so a mask downsampled with ``max`` stays
    aligned with backscatter downsampled with ``average``, pixel for pixel.

    ``nearest`` is stride-based subsampling. It is NOT valid for the border mask: see the
    overview loop, which uses ``max``.
    """
    h, w = data.shape
    new_h = ceil(h / factor)
    new_w = ceil(w / factor)

    if method == "nearest":
        return data[::factor, ::factor][:new_h, :new_w]

    # Block reduction with edge padding for non-divisible sizes. Edge padding replicates real
    # border values, so it can never invent validity the source did not have.
    pad_h = new_h * factor - h
    pad_w = new_w * factor - w
    padded = np.pad(data, ((0, pad_h), (0, pad_w)), mode="edge") if pad_h > 0 or pad_w > 0 else data

    reshaped = padded.reshape(new_h, factor, new_w, factor)
    if method == "max":
        return reshaped.max(axis=(1, 3)).astype(data.dtype)
    if method != "average":
        # Anything unrecognised used to fall through to the block mean, which for the uint8 mask
        # silently reproduces the exact defect this branch exists to fix: a block of [1,0,0,0]
        # means 0.25 and truncates to 0, marking real data invalid. A typo must not do that.
        raise ValueError(f"Unknown downsample method {method!r}: expected average, max or nearest")
    if np.issubdtype(data.dtype, np.floating):
        return np.nanmean(reshaped, axis=(1, 3)).astype(data.dtype)
    return reshaped.mean(axis=(1, 3)).astype(data.dtype)


# =============================================================================
# Acquisition Ingestion
# =============================================================================


def ingest_s1tiling_acquisition(
    vv_path: str | Path,
    vh_path: str | Path,
    border_mask_path: str | Path,
    store_path: str | Path,
    orbit_direction: str,
    allow_out_of_order: bool = False,
) -> int:
    """Ingest one S1Tiling acquisition into a GeoZarr V3 store.

    Creates the store if it does not exist, or appends to an existing store.
    Returns the time index of the ingested acquisition.

    Parameters
    ----------
    vv_path : str or Path
        Path to the VV polarisation GeoTIFF.
    vh_path : str or Path
        Path to the VH polarisation GeoTIFF.
    border_mask_path : str or Path
        Path to the VV border mask GeoTIFF.
    store_path : str or Path
        Path to the output Zarr V3 store.
    orbit_direction : str
        Orbit direction group name (e.g. "ascending", "descending").
    allow_out_of_order : bool, default False
        Accept an acquisition older than (or equal to) the cube's last slice, writing a
        non-monotonic time axis. The default refuses: `.sel(time=slice(...))` raises on such a
        cube, so accepting one silently makes it unqueryable by time. Set this only to recover a
        cube whose ordering is already broken.

    Returns
    -------
    int
        The time index (0-based) of the newly ingested acquisition.

    Raises
    ------
    FileNotFoundError
        If any of the input GeoTIFF paths do not exist.
    ValueError
        If ``orbit_direction`` is not one of ``VALID_ORBIT_DIRECTIONS``, or if the GeoTIFF CRS
        or shape does not match the existing store.
    """
    # FIRST, before any path coercion or I/O. The only previous guard was argparse `choices`,
    # and no CLI command consumes discovery -- so the library path IS the production path. A
    # caller passing discovery's short `orbit_dir` ("ASC") reached `create_group(orbit_direction)`
    # unchecked and built `<store>/ASC/`, which STAC only reported as `No acquisitions found`
    # after the whole multi-hour ingest had completed. Fail in the first second instead.
    if orbit_direction not in VALID_ORBIT_DIRECTIONS:
        raise ValueError(
            f"orbit_direction must be one of {list(VALID_ORBIT_DIRECTIONS)}, "
            f"got {orbit_direction!r}. Discovery exposes this as `orbit_direction`; "
            "`orbit_dir` is the short filename form (ASC/DES) and is not accepted here."
        )

    vv_path = _coerce_input_path(vv_path)
    vh_path = _coerce_input_path(vh_path)
    border_mask_path = _coerce_input_path(border_mask_path)
    # Never `Path()` the store URI: `Path("s3://bucket/x.zarr")` collapses to
    # `PosixPath("s3:/bucket/x.zarr")`, whose `.exists()` is always False, so the create branch
    # wrote a LocalStore under `./s3:/bucket/...`, logged success and exited 0 with nothing in S3.
    store_path = fs_utils.normalize_path(str(store_path))

    for p in [vv_path, vh_path, border_mask_path]:
        if not _input_path_exists(p):
            raise FileNotFoundError(f"GeoTIFF not found: {p}")

    # Extract metadata from VV file
    meta = extract_geotiff_metadata(vv_path)

    log.info(
        "Ingesting S1 acquisition",
        vv_path=str(vv_path),
        orbit_direction=orbit_direction,
    )

    # Create-or-open store. Probe `<store>/zarr.json` rather than the prefix itself: on object
    # storage a "directory" exists only implicitly, so a prefix test is unreliable.
    if not fs_utils.path_exists(f"{store_path}/zarr.json"):
        root = create_s1_store(store_path, orbit_direction, meta)
    else:
        root = _open_store_for_write(store_path)
        # Check the grid on BOTH branches. The new-orbit branch below calls `_build_orbit_group`,
        # which writes `proj:code`, the multiscales layout and `spatial:bbox` straight from the
        # incoming metadata without comparing anything.
        _assert_grid_matches(root, meta)
        if orbit_direction not in root:
            # Create the new orbit group in the existing store (same builder as a fresh
            # store, so per-level metadata — incl. `proj:code` — stays consistent).
            _build_orbit_group(root, orbit_direction, meta)

    orbit = _open_orbit_for_write(store_path, orbit_direction)

    # Read GeoTIFF pixel data
    with _rasterio_env(vv_path):
        with rasterio.open(str(vv_path)) as src:
            vv_data = src.read(1)
        with rasterio.open(str(vh_path)) as src:
            vh_data = src.read(1)
        with rasterio.open(str(border_mask_path)) as src:
            mask_data = src.read(1).astype(np.uint8)

    log.info(
        "GeoTIFF read complete",
        vv_min=float(np.nanmin(vv_data)),
        vv_max=float(np.nanmax(vv_data)),
    )

    # nodata → NaN: s1tiling writes 0 out of swath, which titiler treats as valid data and renders
    # opaque black. Store NaN there instead so it masks transparent like the S2 reference. The
    # border mask is the authoritative valid-data mask (0 = no-data); `_downsample_2d` uses
    # `np.nanmean` for floats, so NaN propagates to every overview level below.
    vv_data = np.where(mask_data == 0, np.nan, vv_data).astype("float32")
    vh_data = np.where(mask_data == 0, np.nan, vh_data).astype("float32")

    # Determine time index
    r10m = _child_group(orbit, "r10m")
    current_size = _child_array(r10m, "vv").shape[0]
    new_size = current_size + 1

    # Generate overviews
    data_by_level: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "r10m": (vv_data, vh_data, mask_data)
    }
    prev_vv, prev_vh, prev_mask = vv_data, vh_data, mask_data
    for level_name, _, factor in OVERVIEW_CHAIN[1:]:
        prev_vv = _downsample_2d(prev_vv, factor, "average")
        prev_vh = _downsample_2d(prev_vh, factor, "average")
        # `max`, NOT `nearest`. The mask is documented as the authoritative valid-data mask and
        # STAC advertises it with `nodata: 0`, but subsampling it while vv/vh were block-averaged
        # broke that: on a diagonal swath edge, 60 pixels at r20m (20 at r60m, 10 at r120m) had
        # `border_mask == 0` over finite `vv`, so applying the mask at preview zoom erased a band
        # of real data. Block-max over the same padded grid guarantees the direction that
        # matters -- `isfinite(vv) => mask != 0` -- by construction: a block's vv is finite iff
        # some source pixel was finite, and every such pixel has `mask != 0`, so the block's max
        # mask is non-zero. NOT the converse: real γ⁰ can be NaN inside the swath (a dropout with
        # `border_mask == 1`), where the two already disagree at r10m and no downsampling rule
        # can repair it. Do not "restore" a bidirectional invariant here; real data violates it.
        prev_mask = _downsample_2d(prev_mask, factor, "max")
        data_by_level[level_name] = (prev_vv, prev_vh, prev_mask)

    log.info("Overviews generated", levels=len(data_by_level))

    dt_ns = np.datetime64(meta.datetime).astype("datetime64[ns]").astype(np.int64)

    # Heal a multiscale level missing `time` before the per-level resize below. A cube built before
    # #192 -- or left half-built by an interrupted append -- can carry `r10m/time` yet lack it at a
    # coarser level; the unconditional `level["time"].resize` then raises `KeyError: 'time'` and,
    # because the consistency check above validates only CRS + shape, the append is non-convergent.
    # Recreate the missing-level coordinate from `r10m/time` (backfilling the existing slices so prior
    # timestamps are preserved), or raise if the cube is inconsistent in a way a backfill cannot fix.
    if "time" in r10m:
        ref_time = np.asarray(_child_array(r10m, "time")[:])
        ref_len = ref_time.shape[0]

        # Assert the cube is square BEFORE writing anything. A transport or OOM failure inside the
        # per-level write loop below leaves r10m at N+1 while a coarser level is still at N; the
        # next append then reads `current_size` from r10m, writes the new slice at index N+1 on the
        # short level and leaves index N as NaN fill with `time[N] == 0` -- 1970 -- and exits 0.
        # This check must cover the levels that DO carry `time`, which is precisely the set the
        # heal loop's `continue` skips, so it cannot live inside that loop.
        ragged = []
        for level_name in data_by_level:
            level = _child_group(orbit, level_name)
            lengths = {
                array_name: _child_array(level, array_name).shape[0]
                for array_name in ("vv", "vh", "border_mask")
                if array_name in level
            }
            if "time" in level:
                lengths["time"] = _child_array(level, "time").shape[0]
            if any(length != ref_len for length in lengths.values()):
                ragged.append(f"{level_name}={lengths}")
        for coord_name in ("absolute_orbit", "relative_orbit", "platform"):
            if coord_name in r10m:
                coord_len = _child_array(r10m, coord_name).shape[0]
                if coord_len != ref_len:
                    ragged.append(f"r10m/{coord_name}={coord_len}")
        if ragged:
            raise ValueError(
                f"Cannot append to {orbit_direction}: the cube is half-built -- these arrays "
                f"disagree with r10m/time ({ref_len} slice(s)): {'; '.join(ragged)}. A previous "
                "append failed part-way through, so `time` cannot be safely backfilled; re-run "
                "that append to completion or wipe + reingest, but do not append on top."
            )

        # Every level is now known to be the same length as r10m/time, so any level missing `time`
        # is healable by backfilling from r10m -- the length check that used to live here is
        # subsumed by the ragged check above.
        healed = []
        for level_name in data_by_level:
            level = _child_group(orbit, level_name)
            if level_name == "r10m" or "time" in level:
                continue
            _create_time_coordinate_array(level)
            _child_array(level, "time").resize((ref_len,))
            _child_array(level, "time")[:] = ref_time
            healed.append(level_name)
        if healed:
            log.info("Healed missing per-level `time`", levels=healed)

        # Keep the time axis monotonic. `.sel(time=slice(...))` raises on a non-monotonic index,
        # so a cube that accepts an out-of-order append is silently unqueryable by time
        # afterwards. This is the only place the invariant can be enforced:
        # `discover_s1tiling_acquisitions` now returns acquisitions chronologically, but the real
        # caller iterates that list and other callers never go through discovery at all.
        #
        # Checked AFTER the ragged assertion above: on a half-built cube the last timestamp is
        # not a reliable thing to compare against, and raggedness is the more fundamental fault.
        #
        # An equal timestamp is rejected as a duplicate. The append is positional, so re-ingesting
        # the same acquisition adds a second slice carrying the same instant rather than replacing
        # the first.
        if current_size > 0:
            last_ns = int(ref_time[current_size - 1])
            if int(dt_ns) <= last_ns:
                # `np.datetime64` takes a plain int but rejects np.int64 (numpy 2.4), and `dt_ns`
                # is np.int64 — without the cast this formatting raises instead of reporting.
                previous = np.datetime64(last_ns, "ns")
                incoming = np.datetime64(int(dt_ns), "ns")
                relation = "duplicates" if int(dt_ns) == last_ns else "precedes"
                message = (
                    f"Out-of-order append to {orbit_direction}: incoming acquisition {incoming} "
                    f"{relation} the last slice already in the cube ({previous}). Ingest "
                    "acquisitions chronologically, or pass allow_out_of_order=True to accept a "
                    "non-monotonic time axis (`.sel(time=slice(...))` then raises on this cube)."
                )
                if not allow_out_of_order:
                    raise ValueError(message)
                log.warning("Appending out of order", detail=message)
    elif current_size > 0:
        raise ValueError(
            f"Cannot append to {orbit_direction}: r10m has {current_size} slice(s) but no `time` "
            "coordinate (no backfill source -- wipe + reingest)"
        )

    # Write data + the `time` coordinate at all levels (time is replicated per level so datetime
    # `.sel` resolves at any rendered scale, #192).
    for level_name, (vv_lev, vh_lev, mask_lev) in data_by_level.items():
        level = _child_group(orbit, level_name)
        h, w = vv_lev.shape

        _child_array(level, "vv").resize((new_size, h, w))
        _child_array(level, "vh").resize((new_size, h, w))
        _child_array(level, "border_mask").resize((new_size, h, w))

        _child_array(level, "vv")[current_size, :, :] = vv_lev
        _child_array(level, "vh")[current_size, :, :] = vh_lev
        _child_array(level, "border_mask")[current_size, :, :] = mask_lev

        _child_array(level, "time").resize((new_size,))
        _child_array(level, "time")[current_size] = dt_ns

    # Per-time metadata coordinates at native resolution only.
    for coord_name in ["absolute_orbit", "relative_orbit", "platform"]:
        _child_array(r10m, coord_name).resize((new_size,))
    _child_array(r10m, "absolute_orbit")[current_size] = meta.absolute_orbit
    _child_array(r10m, "relative_orbit")[current_size] = meta.relative_orbit
    _child_array(r10m, "platform")[current_size] = meta.platform

    # Drop the now-stale consolidated block so external readers list the hierarchy instead of
    # trusting pre-append shapes. The write path does not depend on this having run.
    _strip_consolidated_metadata(store_path, orbit_direction)

    log.info(
        "Zarr write complete",
        time_index=current_size,
        levels_written=len(data_by_level),
    )
    return current_size


# =============================================================================
# Consolidation
# =============================================================================


def consolidate_s1_store(store_path: str | Path, orbit_direction: str) -> None:
    """Consolidate metadata for every orbit-direction group and the root.

    Must be called AFTER all ingestions complete — consolidated metadata
    caches array shapes and will become stale if called mid-ingestion.

    Consolidates *every* orbit group present, not just ``orbit_direction``: the
    pipeline ingests acquisitions one orbit at a time, so consolidating only the
    passed orbit would leave the other orbit's group unconsolidated on disk
    (readers opening that orbit standalone then fall back to a listing).
    ``orbit_direction`` is retained for logging / caller compatibility.

    Enumerates orbits through a root opened with ``use_consolidated=False``: an orbit group
    created since the last consolidation is absent from a stale root block, so consolidating
    through it would silently skip the newest orbit.
    """
    store_path = fs_utils.normalize_path(str(store_path))

    # Refine the root to the same geographic form S2 and OLCI publish: `proj:code = EPSG:4326`
    # and a lon/lat `spatial:bbox` covering every orbit group. `create_s1_store` writes a
    # complete root in the native UTM CRS so that a never-consolidated store still validates;
    # this replaces it once the orbit groups exist and their extents are known.
    #
    # The union is rebuilt from the child groups and never reads the root's own bbox, and all
    # three keys are replaced in a single `attrs.update`, so the pair cannot go half-updated.
    # It reprojects with `transform_bounds(..., densify_pts=21)` rather than transforming the
    # corners, which also avoids the near-global bbox a corner-only conversion produces in UTM
    # zones 1 and 60. `eopf:writer_schema` survives because this is an update, not a replace.
    utils.write_store_root_geo_metadata(
        store_path,
        storage_options=cast("dict[str, object] | None", fs_utils.get_storage_options(store_path)),
    )

    root = _open_store_for_write(store_path)
    for orbit_name, _ in root.groups():
        zarr.consolidate_metadata(store_path, path=orbit_name, zarr_format=3)
    zarr.consolidate_metadata(store_path, zarr_format=3)
    log.info(
        "Metadata consolidated",
        store_path=store_path,
        orbit_direction=orbit_direction,
    )


# =============================================================================
# S3 / local filesystem helpers
# =============================================================================


def _list_tifs(input_dir: str | Path) -> list[str | Path]:
    """List *.tif files; supports local paths and s3:// URIs."""
    s = str(input_dir).rstrip("/")
    if s.startswith("s3://"):
        import s3fs as _s3

        fs = _s3.S3FileSystem()
        bucket_key = s[len("s3://") :]
        return [f"s3://{p}" for p in sorted(fs.glob(f"{bucket_key}/*.tif"))]
    return sorted(Path(input_dir).glob("*.tif"))


def _coerce_input_path(p: str | Path) -> str | Path:
    """Return str for s3:// URIs (preserves double-slash); Path otherwise."""
    s = str(p)
    return s if s.startswith("s3://") else Path(s)


def _input_path_exists(p: str | Path) -> bool:
    """Existence check for both local Path and s3:// URI."""
    s = str(p)
    if s.startswith("s3://"):
        import s3fs as _s3

        return _s3.S3FileSystem().exists(s.removeprefix("s3://"))
    return Path(p).exists()


def _rasterio_env(path: str | Path) -> AbstractContextManager[object]:
    """rasterio.Env context for S3 paths; no-op context manager for local paths.

    rasterio 1.5 passes endpoint_url verbatim as GDAL's AWS_S3_ENDPOINT.
    GDAL expects hostname only (no scheme), so strip https:// from
    AWS_ENDPOINT_URL if that is what the environment provides.
    """
    import contextlib
    import os

    if not str(path).startswith("s3://"):
        return contextlib.nullcontext()

    import boto3
    import rasterio
    from rasterio.session import AWSSession

    raw = os.environ.get("AWS_S3_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL", "")
    endpoint = raw.split("://", 1)[-1] if "://" in raw else raw
    session = boto3.Session()
    return rasterio.Env(AWSSession(session, endpoint_url=endpoint or None))


# =============================================================================
# File Discovery
# =============================================================================


def _acq_stamp_from_geotiff(path: str | Path) -> str:
    """Resolve a ``YYYYMMDDtHHMMSS`` acquisition stamp from a GeoTIFF's ACQUISITION_DATETIME tag.

    Used when an S1Tiling filename carries a masked multi-frame time (e.g. ``…txxxxxx``), which
    has no real timestamp to parse from the name. See #183.
    """
    iso = extract_geotiff_metadata(path).datetime  # e.g. "2023-01-15T06:12:34"
    date_part, time_part = iso.split("T")
    return f"{date_part.replace('-', '')}t{time_part.replace(':', '')}"


def _bundle_key(parsed: dict) -> tuple:
    """The grouping key WITHOUT the stamp -- identifies one acquisition's files."""
    return (parsed["platform"], parsed["tile"], parsed["orbit_dir"], parsed["rel_orbit"])


def _resolve_masked_stamps(parsed_files: list[tuple[str | Path, dict]]) -> dict[tuple, str]:
    """Resolve the real acquisition stamp for each masked multi-frame bundle, once per bundle.

    Multi-frame products mask the filename time (`...txxxxxx`), so the real stamp has to come
    from a GeoTIFF's ACQUISITION_DATETIME tag (#183).

    Resolved PER BUNDLE, trying each file until one carries the tag, because `acq_stamp` is part
    of the grouping key. Resolving per file split one acquisition in two as soon as a single file
    failed: derived `_BorderMask` products commonly carry no ACQUISITION_DATETIME, so vv/vh keyed
    under the resolved stamp and the masks under the masked one -- two half-bundles that the
    `skip_incomplete` default then discarded, so discovery returned `[]` with every file present
    on disk and the ingest loop reported success having written nothing. Any sibling's stamp is
    authoritative: they are the same acquisition, which is exactly what the rest of the key says.

    A bundle absent from the result is one where no file carried a usable tag.
    """
    resolved: dict[tuple, str] = {}
    failures: dict[tuple, list[str]] = {}
    for path, parsed in parsed_files:
        if "x" not in parsed["acq_stamp"]:
            continue
        bundle = (*_bundle_key(parsed), parsed["acq_stamp"])
        if bundle in resolved:
            continue
        try:
            resolved[bundle] = _acq_stamp_from_geotiff(path)
        except (ValueError, KeyError, RasterioIOError) as exc:
            failures.setdefault(bundle, []).append(f"{path}: {exc}")

    for bundle, errors in failures.items():
        if bundle in resolved:
            continue
        # A GeoTIFF with missing or unparseable tags used to raise straight out of discovery, so
        # ONE bad file aborted an entire archive -- while an unparseable *filename* is silently
        # skipped. Report the whole bundle at once and let the caller mark it incomplete.
        log.warning(
            "No file in this bundle carries a usable ACQUISITION_DATETIME; "
            "it cannot be stamped and is marked incomplete",
            bundle=bundle,
            errors=errors,
        )
    return resolved


def discover_s1tiling_acquisitions(
    input_dir: str | Path, skip_incomplete: bool = True
) -> list[dict]:
    """Discover and group S1Tiling GeoTIFF files into acquisition bundles.

    Returns a list of dicts, each with keys:
        platform, tile, orbit_dir, orbit_direction, rel_orbit, acq_stamp, complete,
        vv, vh, vv_mask, vh_mask

    ``orbit_direction`` is the long form the ingester accepts; ``orbit_dir`` keeps the short
    ``ASC``/``DES`` form that filename reconstruction needs.

    Parameters
    ----------
    skip_incomplete : bool, default True
        Drop bundles that cannot actually be ingested. Previously every bundle was returned
        regardless, so a VV-only acquisition reached `ingest_s1tiling_acquisition` and died on
        `KeyError: 'vh'` -- after the warning that should have prevented it. Pass ``False`` to
        get them all back and triage yourself; each carries ``complete``.

    Notes
    -----
    Completeness means the keys the ingester actually reads: ``vv``, ``vh`` and ``vv_mask``.
    ``vh_mask`` is discovered and returned but never consumed -- requiring it wrongly condemned
    perfectly ingestable bundles.
    """
    files = _list_tifs(input_dir)
    groups: dict[tuple, dict] = {}

    parsed_files = []
    for f in files:
        parsed = parse_s1tiling_filename(Path(str(f)).name)
        if parsed is not None:
            parsed_files.append((f, parsed))

    masked_stamps = _resolve_masked_stamps(parsed_files)

    for f, parsed in parsed_files:
        acq_stamp = parsed["acq_stamp"]
        unstamped = False
        if "x" in acq_stamp:
            bundle = (*_bundle_key(parsed), acq_stamp)
            resolved = masked_stamps.get(bundle)
            if resolved is None:
                unstamped = True
            else:
                acq_stamp = resolved

        key = (
            parsed["platform"],
            parsed["tile"],
            parsed["orbit_dir"],
            parsed["rel_orbit"],
            acq_stamp,
        )

        if key not in groups:
            groups[key] = {
                "platform": parsed["platform"],
                "tile": parsed["tile"],
                # Short form ("ASC"/"DES"), kept because filename reconstruction needs it.
                "orbit_dir": parsed["orbit_dir"],
                # Long form ("ascending"/"descending") -- the zarr group name, and the ONLY
                # value `ingest_s1tiling_acquisition` accepts. Discovery used to emit the short
                # form alone, so a caller passing `acq["orbit_dir"]` straight through built
                # `<store>/ASC/` and STAC then failed with `No acquisitions found` after a
                # multi-hour ingest.
                "orbit_direction": ORBIT_DIR_TO_GROUP[parsed["orbit_dir"]],
                "rel_orbit": parsed["rel_orbit"],
                "acq_stamp": acq_stamp,
            }
        if unstamped:
            groups[key]["_unstamped"] = True

        pol = parsed["pol"]
        is_mask = parsed["is_mask"]

        if is_mask:
            groups[key][f"{pol}_mask"] = f
        else:
            groups[key][pol] = f

    acquisitions = []
    # Order by acquisition time FIRST. The key is (platform, tile, orbit_dir, rel_orbit,
    # acq_stamp), so sorting it verbatim lets `platform` dominate the timestamp: a mixed
    # S1A/S1C archive comes back as all the S1A dates then all the S1C dates, and ingesting in
    # that order writes a non-monotonic time axis on which `.sel(time=slice(...))` raises.
    skipped = 0
    for key, acq in sorted(groups.items(), key=lambda item: (item[1]["acq_stamp"], item[0])):
        # Only the keys the ingester reads. `vh_mask` is returned but never consumed.
        missing = [k for k in REQUIRED_ACQUISITION_KEYS if k not in acq]
        unstamped = acq.pop("_unstamped", False)
        acq["complete"] = not missing and not unstamped
        if not acq["complete"]:
            log.warning(
                "Incomplete acquisition",
                key=key,
                missing=missing,
                unstamped=unstamped,
                skipped=skip_incomplete,
            )
            if skip_incomplete:
                skipped += 1
                continue
        acquisitions.append(acq)

    log.info(
        "Discovered acquisitions",
        count=len(acquisitions),
        skipped=skipped,
        input_dir=str(input_dir),
    )
    return acquisitions


# =============================================================================
# Conditions Ingestion
# =============================================================================


def ingest_s1tiling_conditions(
    store_path: str | Path,
    orbit_direction: str,
    relative_orbit: int,
    gamma_area_path: str | Path | None = None,
    lia_path: str | Path | None = None,
    incidence_angle_path: str | Path | None = None,
) -> None:
    """Write time-invariant condition arrays into the conditions group.

    Conditions are per-orbit (not per-acquisition) and have shape (Y, X) only.
    The conditions group carries its own proj: and spatial: conventions.

    Parameters
    ----------
    store_path : str or Path
        Path to an existing Zarr V3 store (must already have the orbit group).
    orbit_direction : str
        Orbit direction group name (e.g. "ascending", "descending").
    relative_orbit : int
        Relative orbit number, used to suffix array names (e.g. 8 → "gamma_area_008").
    gamma_area_path : str, Path, or None
        Path to gamma area GeoTIFF. At least one condition path must be provided.
    lia_path : str, Path, or None
        Path to LIA (sin(LIA)) GeoTIFF.
    incidence_angle_path : str, Path, or None
        Path to incidence angle GeoTIFF.

    Raises
    ------
    ValueError
        If no condition paths are provided, or the store/orbit group doesn't exist.
    FileNotFoundError
        If any provided condition path does not exist.
    """
    condition_inputs: list[tuple[str, str | Path]] = []
    for label, path in [
        ("gamma_area", gamma_area_path),
        ("lia", lia_path),
        ("incidence_angle", incidence_angle_path),
    ]:
        if path is not None:
            p = _coerce_input_path(path)
            if not _input_path_exists(p):
                raise FileNotFoundError(f"Condition GeoTIFF not found: {p}")
            condition_inputs.append((label, p))

    if not condition_inputs:
        raise ValueError("At least one condition path must be provided")

    store_path = fs_utils.normalize_path(str(store_path))
    if not fs_utils.path_exists(f"{store_path}/zarr.json"):
        raise ValueError(f"Store does not exist: {store_path}")

    orbit_suffix = f"{relative_orbit:03d}"

    root = _open_store_for_write(store_path)
    if orbit_direction not in root:
        raise ValueError(
            f"Orbit direction '{orbit_direction}' not found in store. "
            "Ingest at least one acquisition first."
        )

    # Open the ORBIT with its own consolidated block bypassed, not as a child of `root`.
    # `_open_store_for_write` only fixes membership at the root; every membership test below --
    # `"conditions" not in orbit`, and the x/y test in the else branch -- resolves through the
    # orbit group, whose stale block still lists the pre-consolidation members. Reading those
    # answers a question about disk from a snapshot: the conditions group looks absent (so
    # `create_group` raises `ContainsGroupError`), or its coordinates look absent when they are
    # present (so a "repair" tries to recreate them and raises `ContainsArrayError`).
    orbit = _open_orbit_for_write(store_path, orbit_direction)

    # Read reference metadata from the first condition file
    _ref_label, ref_path = condition_inputs[0]
    with _rasterio_env(ref_path), rasterio.open(str(ref_path)) as src:
        ref_crs = make_crs_code(str(src.crs))
        t = src.transform
        ref_transform = [t.a, t.b, t.c, t.d, t.e, t.f]
        ref_shape = [src.height, src.width]

    # Validate CRS consistency with orbit group
    store_crs = dict(orbit.attrs).get("proj:code")
    if store_crs and store_crs != ref_crs:
        raise ValueError(f"CRS mismatch: store has {store_crs}, condition GeoTIFF has {ref_crs}")

    # Create or open conditions group
    if "conditions" not in orbit:
        conditions = orbit.create_group("conditions")
        # Declare the conventions this group's own attributes use. A subgroup never inherits
        # them: the validator hands inherited convention UUIDs only to a group's direct child
        # *arrays*, so `conditions` writing `spatial:*` and `proj:*` while declaring neither is
        # two of the store's validation issues.
        conditions_attrs = utils.build_convention_attrs(
            spatial={
                "spatial:dimensions": ["y", "x"],
                "spatial:transform": ref_transform,
                "spatial:shape": ref_shape,
            },
            crs=PyprojCRS.from_user_input(ref_crs),
        )
        conditions.attrs.update(cast("dict[str, JSON]", conditions_attrs))
        # 1-D `x`/`y` so the condition rasters are georeferenced at all. Without them the group
        # opens with "dimensions without coordinates" and rioxarray infers an identity transform
        # -- the georeferencing is simply absent, and the validator reports one issue per
        # dimension per condition array. Created once, from the same reference grid the group's
        # attributes record; the guard matters because this function is called once per orbit
        # into a shared conditions group.
        _create_spatial_coordinate_arrays(conditions, ref_shape[0], ref_shape[1], ref_transform)
        log.info("Created conditions group", orbit_direction=orbit_direction)
    else:
        conditions = _child_group(orbit, "conditions")
        # The group's grid is fixed by whichever file created it, and every later orbit shares
        # one `x`/`y` pair. A condition raster on a different grid would therefore be written
        # under coordinates that do not describe it -- silently mis-georeferenced rather than
        # merely mislabelled, and the group stops opening at all once the shapes diverge
        # ("conflicting sizes for dimension 'y'"). Reject it instead.
        existing = dict(conditions.attrs)
        existing_shape = existing.get("spatial:shape")
        if isinstance(existing_shape, list) and [
            int(v) for v in cast("list[float]", existing_shape)
        ] != [int(v) for v in ref_shape]:
            raise ValueError(
                f"Conditions grid mismatch in {orbit_direction}: the group was created with "
                f"shape {existing_shape} but this GeoTIFF is {ref_shape}. All condition rasters "
                "in one orbit must share a grid -- they are stored against a single x/y pair."
            )
        existing_transform = existing.get("spatial:transform")
        if isinstance(existing_transform, list):
            stored = [float(v) for v in cast("list[float]", existing_transform)]
            pixel = abs(stored[0])
            if not all(
                isclose(a, float(b), rel_tol=1e-9, abs_tol=pixel * 1e-3)
                for a, b in zip(stored, ref_transform, strict=False)
            ):
                raise ValueError(
                    f"Conditions grid mismatch in {orbit_direction}: the group was created with "
                    f"transform {existing_transform} but this GeoTIFF has {ref_transform}. All "
                    "condition rasters in one orbit must share a grid."
                )

        # Heal a conditions group written before this function created x/y -- AFTER the grid
        # checks above, so a mismatched raster raises instead of writing coordinates that do not
        # describe it. `_add_grid_mapping` below already runs unconditionally, so re-running
        # `ingest-s1-conditions` restored `spatial_ref`; x/y were only ever created on the branch
        # above, so an older group could not be repaired at all and the data model (which now
        # requires them) rejected the store with no route back short of a full reingest.
        #
        # Per-coordinate, not both-or-nothing: a run interrupted between the two writes leaves
        # `x` present and `y` absent, and recreating the pair would raise `ContainsArrayError`
        # on `x` -- turning a half-repaired store into an unrepairable one.
        #
        # Only when the group's recorded grid was actually checked above. Both guards are
        # conditional on the `spatial:*` attrs being well-formed, so a group missing them
        # validates nothing -- and a group missing its coordinates is exactly the population
        # most likely to also be missing its attrs. Backfilling there would write x/y from this
        # call's raster against arrays of an unrelated size (a 5490-long pair over 10980 arrays),
        # which is the silent mis-georeferencing the guard above exists to reject. Refuse instead.
        missing_coords = [name for name in ("x", "y") if name not in conditions]
        if missing_coords and not isinstance(existing.get("spatial:shape"), list):
            raise ValueError(
                f"Conditions group in {orbit_direction} is missing {missing_coords} and records "
                "no 'spatial:shape', so its grid cannot be verified against this GeoTIFF. "
                "Refusing to guess: delete the conditions group and re-run ingest-s1-conditions."
            )
        if missing_coords:
            log.info(
                "Backfilling conditions coordinates",
                orbit_direction=orbit_direction,
                missing=missing_coords,
            )
            _create_spatial_coordinate_arrays(
                conditions, ref_shape[0], ref_shape[1], ref_transform, only=missing_coords
            )

    # Write each condition array
    for label, cond_path in condition_inputs:
        array_name = f"{label}_{orbit_suffix}"

        with _rasterio_env(cond_path), rasterio.open(str(cond_path)) as src:
            # nodata → NaN via the GeoTIFF's declared nodata (border_mask is N/A for static
            # conditions), so out-of-coverage pixels mask transparent like vv/vh. A no-op when
            # the GeoTIFF declares no nodata.
            # Cast to float32 BEFORE filling: `.filled(np.nan)` on an integer-dtype masked array
            # raises, so an integer condition GeoTIFF aborted the whole ingest. NaN is only a
            # legal fill value once the array is floating point.
            data = src.read(1, masked=True).astype(np.float32).filled(np.nan)

        h, w = data.shape

        if array_name in conditions:
            # Overwrite existing array
            _child_array(conditions, array_name)[:, :] = data
            log.info("Overwrote condition array", array_name=array_name)
        else:
            # Shard like the vv/vh pyramid: one shard over the full (y, x) extent so a 10980²
            # condition array is a single object, not ~900 tiny 366²-chunk objects.
            # calculate_aligned_chunk_size returns a divisor of the dimension, so (h, w) is a clean
            # multiple of the inner chunk — the Zarr v3 shard-divisibility requirement.
            inner_chunks = (
                calculate_aligned_chunk_size(h, 512),
                calculate_aligned_chunk_size(w, 512),
            )
            arr = conditions.create_array(
                array_name,
                shape=(h, w),
                dtype="float32",
                chunks=inner_chunks,
                shards=(h, w),
                compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=5),
                fill_value=float("nan"),
                dimension_names=["y", "x"],
            )
            # CF `_FillValue` so readers mask NaN nodata (xarray #11345), like vv/vh (#172).
            arr.attrs["_FillValue"] = FLOAT32_NAN_FILL_VALUE
            arr[:, :] = data
            log.info(
                "Wrote condition array",
                array_name=array_name,
                shape=list(data.shape),
                min=float(np.nanmin(data)),
                max=float(np.nanmax(data)),
            )

    _add_grid_mapping(conditions, ref_crs)

    # Drop the now-stale consolidated blocks, exactly as the acquisition path does. Without this
    # the repair above is INVISIBLE: `stac._open_root`, xarray and TiTiler all read the
    # consolidated view, which still lists the pre-repair members -- so a store whose x/y were
    # just restored keeps being rejected, and the next conditions call reads "x is absent" from
    # the same stale block and raises `ContainsArrayError` over the array on disk. Writing new
    # condition arrays makes the block stale on its own, repair or no repair.
    _strip_consolidated_metadata(store_path, orbit_direction)

    log.info(
        "Conditions ingestion complete",
        orbit_direction=orbit_direction,
        relative_orbit=orbit_suffix,
        arrays=[f"{label}_{orbit_suffix}" for label, _ in condition_inputs],
    )


# =============================================================================
# Conditions File Discovery
# =============================================================================


def discover_s1tiling_conditions(input_dir: str | Path) -> list[dict]:
    """Discover S1Tiling condition GeoTIFF files (gamma_area, LIA).

    Returns a list of dicts, each with keys:
        tile, orbit, gamma_area (Path), lia (Path or None)

    Groups by (tile, orbit).
    """
    files = _list_tifs(input_dir)
    groups: dict[tuple[str, str], dict] = {}

    for f in files:
        m = S1TILING_GAMMA_AREA_PATTERN.match(Path(str(f)).name)
        if m:
            tile = m.group("tile")
            orbit = m.group("orbit")
            key = (tile, orbit)
            if key not in groups:
                groups[key] = {"tile": tile, "orbit": orbit}
            groups[key]["gamma_area"] = f
            continue

        m = S1TILING_LIA_PATTERN.match(Path(str(f)).name)
        if m:
            tile = m.group("tile")
            orbit = m.group("orbit")
            key = (tile, orbit)
            if key not in groups:
                groups[key] = {"tile": tile, "orbit": orbit}
            groups[key]["lia"] = f

    conditions = list(groups.values())
    log.info("Discovered conditions", count=len(conditions), input_dir=str(input_dir))
    return conditions
