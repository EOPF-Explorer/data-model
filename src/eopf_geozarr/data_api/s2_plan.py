"""
Spec-level planning of the Sentinel-2 GeoZarr hierarchy.

This module is the metadata plane of the Sentinel-2 conversion:
`load_sentinel2` reads a Zarr store into the validated `Sentinel2Root` input
model — keeping the model as the carrier of everything the conversion needs,
instead of validating and discarding it — and `plan_geozarr` is a pure
function from that model to the GeoZarr hierarchy we want to write, as a
Zarr V3 `GroupSpec` tree. No chunk data is read or written here: realizing
the plan (copying and coarsening chunks into the planned arrays) is a
separate, data-plane step.

Everything the plan emits is derived from the input model's own metadata:
EOPF products carry `long_name`, the `proj:*` attributes, and the CF packing
attributes on every band, so the output documents (band arrays retyped and
sharded, per-level coordinate arrays, the spatial/proj convention
attributes, and the multiscales layout) are all
computable from the input document alone.

The reflectance pyramid follows the converter's policy
(`eopf_geozarr.s2_optimization`): the scale levels are
``(10, 20, 60, 120, 360, 720)`` metres, levels present in the input are
reused, coarser levels are derived from the level before them, and finer
bands are injected into the coarser native levels (`b08` everywhere; for L1C
also the other 10 m and 20 m bands) so they propagate through the overview
chain. The `conditions` and `quality` subtrees are carried: their own
multiscale groups (the classification masks and the cloud probabilities) are
planned by `AUX_MULTISCALE_LEVELS`, georeferenced datasets get their spatial
attributes, and the quicklook groups and scalar arrays are dropped, as the
converter drops them. The `proj` and `spatial` conventions are the only
georeferencing: no CF `grid_mapping` attribute or `spatial_ref` grid-mapping
array is written (EOPF-Explorer/data-model#352).
"""

from __future__ import annotations

import base64
import re
import struct
from collections.abc import Callable, Mapping
from math import ceil
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import structlog
import xarray as xr
from pydantic_zarr.v2 import GroupSpec as GroupSpecV2
from pydantic_zarr.v3 import GroupSpec as GroupSpecV3
from pydantic_zarr.v3 import parse_dtype_v3
from typing_extensions import TypedDict
from zarr_cm import ConventionMetadataObject
from zarr_cm.multiscales import CMO as MULTISCALES_CMO
from zarr_cm.multiscales import MultiscalesAttrs
from zarr_cm.proj import CMO as PROJ_CMO
from zarr_cm.spatial import CMO as SPATIAL_CMO

from eopf_geozarr.conversion.utils import normalize_cf_array_attrs
from eopf_geozarr.data_api.geozarr.scope import GEOZARR_SCOPE
from eopf_geozarr.data_api.geozarr.v3 import DataArray
from eopf_geozarr.data_api.s2 import Sentinel2Root
from eopf_geozarr.pyz.datatree import group_spec_from_datatree
from eopf_geozarr.s2_optimization.s2_multiscale import SCL_ATTRS, SCL_NODATA

if TYPE_CHECKING:
    import zarr
    from pydantic_zarr.v3 import AnyGroupSpec, FillValue
    from zarr_cm.multiscales import LayoutObject
    from zarr_metadata.v3.chunk_grid.regular import RegularChunkGridObject
    from zarr_metadata.v3.chunk_key_encoding.default import DefaultChunkKeyEncodingObject
    from zarr_metadata.v3.codec.blosc import BloscCodecObject
    from zarr_metadata.v3.codec.bytes import BytesCodecObject
    from zarr_metadata.v3.codec.crc32c import Crc32cCodecObject
    from zarr_metadata.v3.codec.sharding_indexed import ShardingIndexedCodecObject

log = structlog.get_logger()

S2Level = Literal["L1C", "L2A"]

#: Reflectance pyramid levels in metres, finest first (the converter's policy).
REFLECTANCE_SCALE_LEVELS: tuple[int, ...] = (10, 20, 60, 120, 360, 720)

#: Bands injected into coarser *native* levels, per product level, keyed by
#: target resolution in metres; generated levels inherit the previous level's
#: bands. Mirrors the converter's `inject_missing_bands` calls.
INJECTED_BANDS: dict[S2Level, dict[int, frozenset[str]]] = {
    "L2A": {20: frozenset({"b08"}), 60: frozenset({"b08"})},
    "L1C": {
        20: frozenset({"b02", "b03", "b04", "b08"}),
        60: frozenset({"b02", "b03", "b04", "b08", "b05", "b06", "b07", "b8a", "b11", "b12"}),
    },
}

#: Inner (sub-shard) chunk edge length for band arrays.
BAND_CHUNK = 1024

#: Band attributes carried from the input document, in the order they are
#: carried; everything else (xarray bookkeeping such as `_ARRAY_DIMENSIONS`
#: and `_eopf_attrs`, and the per-band `proj:wkt2`, which the enclosing
#: group's `proj:code` replaces) is dropped.
_BAND_ATTR_CARRY = (
    "add_offset",
    "long_name",
    "proj:bbox",
    "proj:epsg",
    "proj:shape",
    "proj:transform",
    "scale_factor",
    "valid_max",
    "valid_min",
)

# base64 of a little-endian float64 NaN: the `_FillValue` spelling xarray
# stores for the float coordinate arrays
_NAN_FLOAT64_B64 = base64.b64encode(struct.pack("<d", float("nan"))).decode()

_CHUNK_KEY_ENCODING: Final[DefaultChunkKeyEncodingObject] = {
    "name": "default",
    "configuration": {"separator": "/"},
}

#: The little-endian `bytes` codec, the array -> bytes codec of every planned array.
_BYTES_LITTLE: Final[BytesCodecObject] = {"name": "bytes", "configuration": {"endian": "little"}}

_CRC32C: Final[Crc32cCodecObject] = {"name": "crc32c"}


def load_sentinel2(source: zarr.Group | xr.DataTree) -> Sentinel2Root:
    """
    Read a Zarr group or an xarray DataTree into the validated
    `Sentinel2Root` input model.

    The model is the carrier for the conversion: it holds exactly the members
    and attributes the GeoZarr plan reads, validated (member structure by the
    closed TypedDicts, attributes by the attribute models). A DataTree is
    read through its GroupSpec-shaped view (`group_spec_from_datatree`), so a
    tree built in memory or backed by a non-V2 store validates like a V2
    store does. Raises `pydantic.ValidationError` when the source is not a
    Sentinel-2 product.
    """
    spec = (
        group_spec_from_datatree(source)
        if isinstance(source, xr.DataTree)
        else GroupSpecV2.from_zarr(source)
    )
    return Sentinel2Root.model_validate(spec.model_dump())


def product_level(root: Sentinel2Root) -> S2Level:
    """
    The product level (L1C or L2A) of a Sentinel-2 product, read from the
    `stac_discovery` root attribute (the STAC id and `product:type` both
    carry it). Raises `ValueError` when neither names a known level.
    """
    stac = root.attributes.stac_discovery
    candidates: list[object] = [stac.get("id")]
    properties = stac.get("properties")
    if isinstance(properties, dict):
        candidates.append(properties.get("product:type"))
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        for level in ("L2A", "L1C"):
            # "MSIL2A" in product ids; "S2MSI2A" in product:type
            if level in candidate or candidate.endswith(level[1:]):
                return level
    msg = "Could not determine the Sentinel-2 product level (L1C or L2A) from stac_discovery"
    raise ValueError(msg)


def _product_proj(root: Sentinel2Root) -> tuple[tuple[float, float], int] | None:
    """
    The product-level georeferencing, as (origin, EPSG), from the `proj:*`
    keys of `stac_discovery` `properties` — the only place the Zarr V3
    baselines (N0512) declare it. The origin is the bbox's (min-x, max-y)
    corner. Returns None when either is missing or malformed.
    """
    properties = root.attributes.stac_discovery.get("properties")
    if not isinstance(properties, dict):
        return None
    bbox = properties.get("proj:bbox")
    code = properties.get("proj:code")
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) < 4
        or not all(isinstance(value, (int, float)) for value in bbox[:4])
        or not isinstance(code, str)
        or not code.startswith("EPSG:")
    ):
        return None
    return (float(bbox[0]), float(bbox[3])), int(code.removeprefix("EPSG:"))


def _shard_and_inner(shape: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    The (shard, inner chunk) shapes for a band array: inner chunks of
    `BAND_CHUNK` per side (the whole dimension when smaller), shards rounded
    up to whole inner chunks.
    """
    inner = tuple(min(dim, BAND_CHUNK) for dim in shape)
    shard = tuple(
        dim if dim <= i else ceil(dim / i) * i for dim, i in zip(shape, inner, strict=True)
    )
    return shard, inner


def _regular_grid(chunk_shape: tuple[int, ...]) -> RegularChunkGridObject:
    """The regular chunk grid metadata for one chunk shape."""
    return {"name": "regular", "configuration": {"chunk_shape": chunk_shape}}


def _plan_band(
    source_attrs: Mapping[str, object],
    shape: tuple[int, ...],
    data_type: str,
    fill_value: FillValue,
    dimension_names: tuple[str, ...],
    typesize: int,
) -> DataArray:
    """One band array of one *native* level: input attributes mapped and carried."""
    attributes: dict[str, object] = {
        "_FillValue": source_attrs.get("fill_value", fill_value),
        "units": "1",
    }
    for key in _BAND_ATTR_CARRY:
        if key in source_attrs:
            attributes[key] = source_attrs[key]
    attributes = normalize_cf_array_attrs(attributes, np.dtype(data_type))
    return _band_array(attributes, shape, data_type, fill_value, dimension_names, typesize)


def _band_array(
    attributes: dict[str, object],
    shape: tuple[int, ...],
    data_type: str | Mapping[str, object],
    fill_value: FillValue,
    dimension_names: tuple[str, ...],
    typesize: int,
) -> DataArray:
    """
    One band array document with the blosc policy: sharded, unless the whole
    array fits in one inner chunk, where a shard would hold that one chunk and
    cost a reader an index request for nothing (#357); such an array is
    written in plain chunks with the same codecs.
    """
    shard, inner = _shard_and_inner(shape)
    compressor: BloscCodecObject = {
        "name": "blosc",
        "configuration": {
            "typesize": typesize,
            "cname": "zstd",
            "clevel": 3,
            "shuffle": "shuffle",
            "blocksize": 0,
        },
    }
    if shard == inner:
        return DataArray.model_validate(
            {
                "attributes": attributes,
                "shape": shape,
                "data_type": data_type,
                "chunk_grid": _regular_grid(inner),
                "chunk_key_encoding": _CHUNK_KEY_ENCODING,
                "fill_value": fill_value,
                "codecs": (_BYTES_LITTLE, compressor),
                "dimension_names": dimension_names,
            }
        )
    sharding: ShardingIndexedCodecObject = {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": inner,
            "codecs": (_BYTES_LITTLE, compressor),
            "index_codecs": (_BYTES_LITTLE, _CRC32C),
            "index_location": "end",
        },
    }
    return DataArray.model_validate(
        {
            "attributes": attributes,
            "shape": shape,
            "data_type": data_type,
            "chunk_grid": _regular_grid(shard),
            "chunk_key_encoding": _CHUNK_KEY_ENCODING,
            "fill_value": fill_value,
            "codecs": (sharding,),
            "dimension_names": dimension_names,
        }
    )


def _plan_coordinate(name: str, size: int) -> DataArray:
    """The `x` or `y` array of a generated level: float64, NaN-filled, one chunk."""
    # create_default supplies the one-chunk grid, the `/` chunk key encoding
    # and the little-endian bytes codec
    return DataArray.create_default(
        attributes={"_FillValue": _NAN_FLOAT64_B64},
        shape=(size,),
        data_type="float64",
        fill_value="NaN",
        dimension_names=(name,),
    )


SpatialGroupAttrs = TypedDict(
    "SpatialGroupAttrs",
    {
        "proj:code": str,
        "spatial:bbox": list[float],
        "spatial:dimensions": list[str],
        "spatial:registration": str,
        "spatial:shape": list[int],
        "spatial:transform": list[float],
        "zarr_conventions": list[ConventionMetadataObject],
    },
)
"""The spatial/proj convention attributes of one georeferenced group (a
resolution dataset or a pyramid level), which always carries every key;
the per-key value types are zarr-cm's."""

LatLonGroupAttrs = TypedDict(
    "LatLonGroupAttrs",
    {
        "proj:code": str,
        "spatial:bbox": list[float],
        "spatial:dimensions": list[str],
        "spatial:registration": str,
        "spatial:shape": list[int],
        "spatial:transform": list[float],
        "zarr_conventions": list[ConventionMetadataObject],
    },
)
"""The spatial/proj convention attributes of a group on a regular geographic
grid (`_lat_lon_attributes`), which always carries every key."""

MultiscaleGroupAttrs = TypedDict(
    "MultiscaleGroupAttrs",
    {
        "multiscales": MultiscalesAttrs,
        "proj:code": str,
        "spatial:bbox": list[float],
        "spatial:dimensions": list[str],
        "spatial:registration": str,
        "spatial:shape": list[int],
        "spatial:transform": list[float],
        "zarr_conventions": list[ConventionMetadataObject],
    },
)
"""The attributes of a multiscale pyramid's root group: zarr-cm's multiscales
attributes plus the finest level's georeferencing, its `spatial:shape` with
its `spatial:transform` so the transform says what grid it describes
(EOPF-Explorer/data-model#355)."""


def _spatial_dataset_attributes(
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, ...],
    epsg: int,
) -> SpatialGroupAttrs:
    """The spatial/proj convention attributes of one spatial dataset's group."""
    res_x, _, x0, _, res_y, y0 = transform
    rows, cols = shape
    return {
        "proj:code": f"EPSG:{epsg}",
        "spatial:bbox": [x0, y0 + rows * res_y, x0 + cols * res_x, y0],
        "spatial:dimensions": ["y", "x"],
        "spatial:registration": "pixel",
        "spatial:shape": list(shape),
        "spatial:transform": list(transform),
        "zarr_conventions": [SPATIAL_CMO, PROJ_CMO],
    }


def _level_spatial_attributes(
    *,
    meters: int,
    shape: tuple[int, ...],
    origin: tuple[float, float],
    epsg: int,
) -> SpatialGroupAttrs:
    """The spatial/proj convention attributes of one pyramid level's group."""
    x0, y0 = origin
    resolution = float(meters)
    return _spatial_dataset_attributes((resolution, 0.0, x0, 0.0, -resolution, y0), shape, epsg)


_TileGeoreferencing = tuple[tuple[float, float], int]
"""The tile's (origin, EPSG), as the reflectance pyramid is georeferenced."""

GEOMETRY_PATH = "conditions/geometry"
"""The group of the sun and viewing angle grids: node-registered, in the tile CRS."""


def plan_reflectance(root: Sentinel2Root) -> AnyGroupSpec:
    """
    Plan the GeoZarr `measurements/reflectance` multiscale group of a
    Sentinel-2 product: one level group per entry of
    `REFLECTANCE_SCALE_LEVELS`, native levels carried from the input (with
    the product level's finer bands injected), coarser levels derived from
    the level before them, each holding sharded uint16 band arrays and `x`/`y`
    coordinates (the input's own arrays on a native level, float64 on a
    generated one), under a
    multiscales layout attribute.
    """
    return _plan_reflectance(root)[0]


def _plan_reflectance(root: Sentinel2Root) -> tuple[AnyGroupSpec, _TileGeoreferencing]:
    """`plan_reflectance`, with the tile georeferencing the pyramid was planned in."""
    level = product_level(root)
    reflectance = root.members["measurements"].members["reflectance"]
    input_levels: dict[int, Any] = {}
    for meters in REFLECTANCE_SCALE_LEVELS:
        member = reflectance.members.get(f"r{meters}m")
        if member is not None:
            input_levels[meters] = member

    if not input_levels:
        msg = "The product's reflectance group holds none of the known resolution datasets"
        raise ValueError(msg)

    # the georeferencing of the pyramid: origin and EPSG from any finest-level
    # band, else from the product-level declaration (Zarr V3 baselines carry
    # proj attributes only on stac_discovery, not on the band arrays)
    finest_meters_in = min(input_levels)
    finest = input_levels[finest_meters_in]
    first_band_name = next((k for k in finest.members if k.startswith("b")), None)
    if first_band_name is None:
        msg = f"The r{finest_meters_in}m resolution dataset holds no band arrays"
        raise ValueError(msg)
    first_band_attrs: Mapping[str, object] = finest.members[first_band_name].attributes
    transform = first_band_attrs.get("proj:transform")
    band_epsg = first_band_attrs.get("proj:epsg")
    if (
        isinstance(transform, (list, tuple))
        and len(transform) >= 6
        and isinstance(band_epsg, (int, str))
    ):
        origin = (float(transform[2]), float(transform[5]))
        epsg = int(band_epsg)
    else:
        product = _product_proj(root)
        if product is None:
            msg = (
                f"Band {first_band_name!r} carries no usable proj:transform/proj:epsg "
                "and the product declares no stac_discovery proj georeferencing"
            )
            raise ValueError(msg)
        origin, epsg = product

    # per-level planned bands: name -> (source attrs, shape, dtype itemsize)
    planned_levels: dict[int, dict[str, DataArray]] = {}
    level_attributes: dict[int, SpatialGroupAttrs] = {}
    previous_meters: int | None = None
    layout: list[LayoutObject] = []
    members_out: dict[str, AnyGroupSpec] = {}

    for meters in REFLECTANCE_SCALE_LEVELS:
        bands: dict[str, DataArray] = {}
        native = meters in input_levels
        if native:
            source = input_levels[meters]
            for name, member in source.members.items():
                if not name.startswith("b"):
                    continue
                dims = tuple(member.attributes.get("_ARRAY_DIMENSIONS", ("y", "x")))
                bands[name] = _plan_band(
                    member.attributes,
                    tuple(member.shape),
                    "uint16",
                    0,
                    dims,
                    typesize=2,
                )
            if not bands:
                msg = f"The r{meters}m resolution dataset holds no band arrays"
                raise ValueError(msg)
            level_shape = next(iter(bands.values())).shape
            # inject finer bands so they propagate through the overview chain
            for band_name in sorted(INJECTED_BANDS.get(level, {}).get(meters, ())):
                if band_name in bands:
                    continue
                for src_meters in sorted(input_levels):
                    if src_meters >= meters:
                        break
                    src_member = input_levels[src_meters].members.get(band_name)
                    if src_member is None:
                        continue
                    dims = tuple(src_member.attributes.get("_ARRAY_DIMENSIONS", ("y", "x")))
                    bands[band_name] = _plan_band(
                        src_member.attributes, level_shape, "uint16", 0, dims, typesize=2
                    )
                    break
        else:
            if previous_meters is None:
                msg = (
                    f"The finest pyramid level (r{meters}m) is not in the input, "
                    "so there is nothing to derive the coarser levels from"
                )
                raise ValueError(msg)
            factor = meters // previous_meters
            for name, previous in planned_levels[previous_meters].items():
                shape = tuple(dim // factor for dim in previous.shape)
                # a coarsened band keeps its attributes verbatim
                dims = previous.dimension_names or ("y", "x")
                bands[name] = _band_array(
                    previous.attributes.model_dump(),
                    shape,
                    "uint16",
                    0,
                    dims,
                    typesize=2,
                )
            level_shape = next(iter(bands.values())).shape

        planned_levels[meters] = bands
        rows, cols = level_shape
        shape2d = (rows, cols)
        level_members: dict[str, DataArray] = dict(bands)
        for axis, size in (("x", cols), ("y", rows)):
            coordinate = input_levels[meters].members.get(axis) if native else None
            level_members[axis] = _plan_level_coordinate(
                axis, size, None if coordinate is None else coordinate.model_dump()
            )

        level_attributes[meters] = _level_spatial_attributes(
            meters=meters, shape=shape2d, origin=origin, epsg=epsg
        )
        members_out[f"r{meters}m"] = GroupSpecV3(
            attributes=level_attributes[meters],
            members=level_members,
        )

        level_shape2d = list(shape2d)
        level_transform = [float(meters), 0.0, origin[0], 0.0, -float(meters), origin[1]]
        entry: LayoutObject = {
            "asset": f"r{meters}m",
            "spatial:shape": level_shape2d,
            "spatial:transform": level_transform,
        }
        if previous_meters is not None:
            factor = meters // previous_meters
            entry = {
                "asset": f"r{meters}m",
                "derived_from": f"r{previous_meters}m",
                "spatial:shape": level_shape2d,
                "spatial:transform": level_transform,
                "transform": {"scale": [factor, factor], "translation": [0.0, 0.0]},
            }
        layout.append(entry)
        previous_meters = meters

    finest_meters = REFLECTANCE_SCALE_LEVELS[0]
    finest_level_attrs = level_attributes[min(input_levels)]
    group_attributes: MultiscaleGroupAttrs = {
        "multiscales": {"layout": layout, "resampling_method": "average"},
        "proj:code": f"EPSG:{epsg}",
        "spatial:bbox": finest_level_attrs["spatial:bbox"],
        "spatial:dimensions": ["y", "x"],
        "spatial:registration": "pixel",
        "spatial:shape": level_attributes[finest_meters]["spatial:shape"],
        "spatial:transform": [
            float(finest_meters),
            0.0,
            origin[0],
            0.0,
            -float(finest_meters),
            origin[1],
        ],
        "zarr_conventions": [MULTISCALES_CMO, SPATIAL_CMO, PROJ_CMO],
    }
    return GroupSpecV3(attributes=group_attributes, members=members_out), (origin, epsg)


# ---------------------------------------------------------------------------
# conditions and quality

#: Groups that get a multiscale pyramid, with their levels in metres (the
#: converter's aux multiscale specs); levels present in the input are reused,
#: the rest are coarsened from the level before them.
AUX_MULTISCALE_LEVELS: dict[str, tuple[int, ...]] = {
    "conditions/mask/l1c_classification": (60, 120, 360, 720),
    "conditions/mask/l2a_classification": (20, 60, 120, 360, 720),
    "quality/probability": (20, 60, 120, 360, 720),
    "quality/atmosphere": (10, 20, 60, 120, 360, 720),
}

#: Groups the output drops: the quicklooks duplicate RGB data derivable from
#: the reflectance bands (EOPF-Explorer/data-model#81).
_DROPPED_GROUPS = frozenset({"quality/l1c_quicklook", "quality/l2a_quicklook"})

#: Attribute keys never carried into the output: xarray's dimension
#: bookkeeping, the EOPF source bag, the fill value (which lives in the array
#: document, re-added as a CF attribute only for floats), and the geo-proj
#: convention keys that belong on the enclosing group.
_DROPPED_ATTRS = frozenset(
    {"_ARRAY_DIMENSIONS", "_eopf_attrs", "_FillValue", "proj:code", "proj:wkt2", "proj:projjson"}
)

#: Group attribute keys never carried: the source's CF `Conventions` claim,
#: which the output does not make (`proj` and `spatial` are its
#: georeferencing; EOPF-Explorer/data-model#353).
_DROPPED_GROUP_ATTRS = frozenset({"Conventions"})

CoordinateValues = Callable[[str], "np.ndarray | None"]
"""Looks up the stored values of the input array at an output-plan path (None
when the input holds no array there); the plan reads only the small 1-D
coordinate arrays a group's georeferencing is derived from."""


def _dims_of(doc: _ArrayDoc) -> tuple[str, ...]:
    """The dimension names an input array declares (`_ARRAY_DIMENSIONS`)."""
    dims = _doc_attrs(doc).get("_ARRAY_DIMENSIONS")
    return tuple(str(dim) for dim in dims) if isinstance(dims, (list, tuple)) else ()


def _carried_group_attributes(group: Mapping[str, Any]) -> dict[str, object]:
    """A carried group's attributes, without the keys never carried."""
    return {
        key: value for key, value in _doc_attrs(group).items() if key not in _DROPPED_GROUP_ATTRS
    }


def _in_longitude_range(longitude: float) -> float:
    """`longitude` moved by whole turns into [-180, 180] when outside it, else as it is."""
    if -180.0 <= longitude <= 180.0:
        return longitude
    wrapped = (longitude + 180.0) % 360.0 - 180.0
    # a positive multiple of 180 lands on the antimeridian's eastern spelling
    return 180.0 if wrapped == -180.0 and longitude > 0 else wrapped


def _regular_axis(values: np.ndarray, name: str) -> tuple[float, float]:
    """
    The (first value, step) of a regularly spaced 1-D coordinate. The step is
    taken from the end points, which averages out the rounding of coordinates
    stored as float32, and spacing within a thousandth of it counts as regular.
    Raises `ValueError` for fewer than two values, NaN, or irregular spacing,
    which no affine transform describes.
    """
    values = np.asarray(values, dtype="float64")
    if values.ndim != 1 or values.size < 2:
        msg = f"The {name} coordinate needs at least two values, got shape {values.shape}"
        raise ValueError(msg)
    step = float(values[-1] - values[0]) / (values.size - 1)
    if step == 0 or not np.allclose(np.diff(values), step, rtol=1e-3, atol=0.0):
        msg = f"The {name} coordinate is not regularly spaced, so no transform describes it"
        raise ValueError(msg)
    return float(values[0]), step


def _lat_lon_attributes(latitude: np.ndarray, longitude: np.ndarray) -> LatLonGroupAttrs:
    """
    The `proj` and `spatial` attributes of a group on a regular geographic
    grid, from its 1-D `latitude` and `longitude` coordinate values
    (EOPF-Explorer/data-model#353). The values are grid points, so the
    registration is `node`: the transform maps an index to a point, its
    origin the first point, and a decreasing latitude gives a negative step;
    the bbox spans the outer points. Longitudes are unwrapped first, so a grid
    crossing the antimeridian (or 0/360 degrees) keeps a regular step. Refuses
    irregular spacing, which no affine transform describes, and NaN values.
    """
    lat = np.asarray(latitude, dtype="float64")
    lon = np.asarray(longitude, dtype="float64")
    if lon.ndim == 1 and lon.size > 1 and not np.isnan(lon).any():
        lon = np.unwrap(lon, period=360.0)
    lat0, dlat = _regular_axis(lat, "latitude")
    lon0, dlon = _regular_axis(lon, "longitude")
    # the bbox longitudes back in [-180, 180]: a grid crossing the antimeridian
    # gets west > east, as STAC and GeoJSON spell the crossing
    west, east = (_in_longitude_range(float(value)) for value in (lon.min(), lon.max()))
    return {
        "proj:code": "EPSG:4326",
        "spatial:bbox": [west, float(lat.min()), east, float(lat.max())],
        "spatial:dimensions": ["latitude", "longitude"],
        "spatial:registration": "node",
        "spatial:shape": [int(lat.size), int(lon.size)],
        "spatial:transform": [dlon, 0.0, lon0, 0.0, dlat, lat0],
        "zarr_conventions": [SPATIAL_CMO, PROJ_CMO],
    }


# The conditions/quality planner walks the input model's *dump* — plain Zarr
# V2 documents — because the input model types those subtrees loosely, so
# their nodes are a mix of spec models and raw mappings. A node holding
# "members" is a group; one holding "shape" is an array.
_ArrayDoc = Mapping[str, Any]
_GroupDoc = Mapping[str, Any]


def _doc_attrs(doc: Mapping[str, Any]) -> Mapping[str, object]:
    attributes = doc.get("attributes") or {}
    if not isinstance(attributes, Mapping):
        msg = f"Attributes are not a mapping: {type(attributes)!r}"
        raise TypeError(msg)
    return attributes


_LEVEL_GROUP = re.compile(r"r(\d+)m")
"""The name of a resolution level group (`r10m`, `r720m`), its metres captured."""


def level_meters(path: str) -> int | None:
    """The resolution in metres of the level group at `path` (`r10m`, or a path
    ending in it), or None if it is not a level group."""
    match = _LEVEL_GROUP.fullmatch(path.rsplit("/", 1)[-1])
    return int(match.group(1)) if match else None


def _with_level_short_name(
    attributes: dict[str, object], name: str, meters: int | None
) -> dict[str, object]:
    """
    `attributes` with a carried `short_name` written for the level the array
    is in, `<variable>_r<resolution>m`: a source `short_name` names the source
    level (and a coarsened level copied it verbatim), and the source styles
    are mixed (EOPF-Explorer/data-model#351). Outside a level group, or
    without a `short_name`, the attributes are returned as they are.
    """
    if meters is None or "short_name" not in attributes:
        return attributes
    return {**attributes, "short_name": f"{name}_r{meters}m"}


def _carried_array_attributes(
    attrs: Mapping[str, object],
    *,
    is_float: bool,
    name: str,
    dtype: np.dtype[Any],
    meters: int | None = None,
) -> dict[str, object]:
    """The output attributes of a carried array: sanitized, the CF attributes
    put right for its data type, with the CF float fill, the SCL flag table
    and a per-level `short_name`."""
    out = normalize_cf_array_attrs(
        {key: value for key, value in attrs.items() if key not in _DROPPED_ATTRS}, dtype
    )
    if is_float:
        out["_FillValue"] = _NAN_FLOAT64_B64
    if name == "scl":
        # the converter's SCL flag table and NO_DATA fill declaration
        out.update(SCL_ATTRS)
        out["_FillValue"] = SCL_NODATA
    return _with_level_short_name(out, name, meters)


def _plan_carried_array(name: str, doc: _ArrayDoc, *, meters: int | None = None) -> DataArray:
    """
    One carried (non-reflectance) array, translated to its output document:
    coordinate arrays (an array whose only dimension is its own name, `x`
    and `y` included) keep their data type uncompressed, and data arrays are
    sharded with the blosc policy.
    """
    attrs = _doc_attrs(doc)
    dims_raw = attrs.get("_ARRAY_DIMENSIONS")
    if not isinstance(dims_raw, (list, tuple)):
        msg = f"Array {name!r} carries no _ARRAY_DIMENSIONS attribute"
        raise TypeError(msg)
    dims = tuple(str(dim) for dim in dims_raw)
    shape = tuple(doc["shape"])
    parsed = np.dtype(str(doc["dtype"]))
    # the Zarr V3 form zarr-python writes for the V2 dtype, in the GeoZarr scope
    data_type = parse_dtype_v3(parsed, scope=GEOZARR_SCOPE)
    fill_value: FillValue = "NaN" if parsed.kind == "f" else "" if parsed.kind in "US" else 0
    attributes = _carried_array_attributes(
        attrs,
        is_float=parsed.kind == "f",
        name=name,
        dtype=parsed,
        meters=meters,
    )
    if parsed.kind in "US":
        # fixed-length unicode or bytes labels (the geometry `angle`, `band`
        # and `detector` coordinates) are stored as the registered `string`
        # data type, which every Zarr V3 reader can read, instead of the
        # unregistered `fixed_length_utf32` / `null_terminated_bytes` (#356);
        # create_default supplies its `vlen-utf8` codec, the empty fill and
        # one chunk
        return DataArray.create_default(
            attributes=attributes, shape=shape, data_type="string", dimension_names=dims
        )
    if dims == (name,):
        # a coordinate array: one chunk, uncompressed (create_default supplies
        # the one-chunk grid and the `/` chunk key encoding)
        return DataArray.create_default(
            attributes=attributes,
            shape=shape,
            data_type=data_type,
            fill_value=fill_value,
            codecs=(_BYTES_LITTLE,),
            dimension_names=dims,
        )
    return _band_array(attributes, shape, data_type, fill_value, dims, typesize=parsed.itemsize)


def _plan_level_coordinate(name: str, size: int, source: _ArrayDoc | None) -> DataArray:
    """
    The `x` or `y` array of one pyramid level. A native level carries the
    input's own coordinate array, data type included (the int64 metres of an
    ESA product stay int64); a generated level, which has no input array,
    gets the float64 coordinate array.
    """
    if source is None:
        return _plan_coordinate(name, size)
    return _plan_carried_array(name, source)


def _carried_arrays(group: _GroupDoc) -> dict[str, _ArrayDoc]:
    """The group's non-scalar arrays (scalar arrays are dropped)."""
    members = group.get("members") or {}
    return {
        name: member
        for name, member in members.items()
        if isinstance(member, Mapping) and "shape" in member and len(member["shape"]) > 0
    }


def _spatial_anchor(arrays: Mapping[str, _ArrayDoc]) -> tuple[str, _ArrayDoc] | None:
    """The first array carrying proj attributes, with its name, if any."""
    for name, member in arrays.items():
        if "proj:epsg" in _doc_attrs(member):
            return name, member
    return None


_Transform6 = tuple[float, float, float, float, float, float]


def _proj_of(name: str, doc: _ArrayDoc) -> tuple[_Transform6, int]:
    """The (transform, EPSG) a proj-bearing input array carries."""
    attrs = _doc_attrs(doc)
    transform = attrs.get("proj:transform")
    if not isinstance(transform, (list, tuple)) or len(transform) < 6:
        msg = f"Array {name!r} carries no usable proj:transform"
        raise ValueError(msg)
    a, b, c, d, e, f = (float(part) for part in transform[:6])
    return (
        (a, b, c, d, e, f),
        int(attrs["proj:epsg"]),  # type: ignore[call-overload]
    )


def _node_grid_attributes(x: np.ndarray, y: np.ndarray, epsg: int) -> SpatialGroupAttrs:
    """
    The spatial/proj convention attributes of a node-registered grid in the
    tile CRS (the geometry angle grids, EOPF-Explorer/data-model#346): its
    values are at the grid nodes, so the transform's origin is `(x[0], y[0])`,
    not moved by a half cell, and the bbox spans the outer nodes.
    """
    x0, dx = _regular_axis(x, "x")
    y0, dy = _regular_axis(y, "y")
    x_last, y_last = x0 + dx * (len(x) - 1), y0 + dy * (len(y) - 1)
    return {
        "proj:code": f"EPSG:{epsg}",
        "spatial:bbox": [min(x0, x_last), min(y0, y_last), max(x0, x_last), max(y0, y_last)],
        "spatial:dimensions": ["y", "x"],
        "spatial:registration": "node",
        "spatial:shape": [len(y), len(x)],
        "spatial:transform": [dx, 0.0, x0, 0.0, dy, y0],
        "zarr_conventions": [SPATIAL_CMO, PROJ_CMO],
    }


def _derived_attributes(
    group_path: str,
    values: tuple[np.ndarray | None, np.ndarray | None],
    build: Callable[[np.ndarray, np.ndarray], Mapping[str, object]],
) -> Mapping[str, object] | None:
    """
    The attributes `build` derives from a group's coordinate `values`, or None
    when a value is missing or `build` refuses them: a grid no transform
    describes is carried without georeferencing, with a warning, rather than
    stopping the conversion.
    """
    first, second = values
    if first is None or second is None:
        return None
    try:
        return build(first, second)
    except ValueError as error:
        log.warning(
            "Carrying a group without the georeferencing its coordinates do not describe",
            group=group_path,
            reason=str(error),
        )
        return None


def _value_attributes(
    arrays: Mapping[str, _ArrayDoc],
    path: str,
    coordinate_values: CoordinateValues | None,
    tile: _TileGeoreferencing | None,
) -> Mapping[str, object] | None:
    """
    The `proj` and `spatial` attributes a carried dataset without proj
    attributes of its own derives from its coordinate values, or None: a
    regular geographic grid (the meteorology groups, #353) from its 1-D
    latitude/longitude, and the geometry angle grids (#346) from their x/y
    nodes, in the tile CRS.
    """
    if coordinate_values is None:
        return None

    def coordinates(first: str, second: str) -> tuple[np.ndarray | None, np.ndarray | None] | None:
        """The values of the group's 1-D `first` and `second` coordinates, if it has them."""
        docs = arrays.get(first), arrays.get(second)
        if docs[0] is None or docs[1] is None:
            return None
        if (_dims_of(docs[0]), _dims_of(docs[1])) != ((first,), (second,)):
            return None
        return coordinate_values(f"{path}/{first}"), coordinate_values(f"{path}/{second}")

    lat_lon = coordinates("latitude", "longitude")
    if lat_lon is not None:
        return _derived_attributes(path, lat_lon, _lat_lon_attributes)
    if tile is not None and path == GEOMETRY_PATH:
        x_y = coordinates("x", "y")
        if x_y is not None:
            epsg = tile[1]
            return _derived_attributes(path, x_y, lambda x, y: _node_grid_attributes(x, y, epsg))
    return None


def _tile_level_attributes(
    arrays: Mapping[str, _ArrayDoc], path: str, tile: _TileGeoreferencing | None
) -> SpatialGroupAttrs | None:
    """
    The attributes of a resolution level under `conditions/mask` or `quality`
    whose arrays carry no proj attributes (as in native Zarr V3 products):
    pixel-registered in the tile CRS from the tile origin at its resolution
    (EOPF-Explorer/data-model#346), or None.
    """
    meters = level_meters(path)
    if tile is None or meters is None or not path.startswith(("conditions/mask/", "quality/")):
        return None
    shape = next(
        (
            tuple(doc["shape"])
            for doc in arrays.values()
            if len(doc["shape"]) == 2 and _dims_of(doc) == ("y", "x")
        ),
        None,
    )
    if shape is None:
        return None
    origin, epsg = tile
    return _level_spatial_attributes(meters=meters, shape=shape, origin=origin, epsg=epsg)


def _plan_dataset(
    group: _GroupDoc,
    path: str,
    coordinate_values: CoordinateValues | None = None,
    tile: _TileGeoreferencing | None = None,
) -> AnyGroupSpec | None:
    """
    Plan one carried leaf dataset: every non-scalar array translated, and —
    when the dataset is georeferenced — the group's spatial attributes. A
    dataset is georeferenced by an array that carries proj attributes; failing
    that, from its coordinate values (the meteorology and geometry grids, see
    `_value_attributes`), or from the tile (`_tile_level_attributes`).
    Returns `None` for a dataset with no arrays to carry.
    """
    arrays = _carried_arrays(group)
    if not arrays:
        return None
    anchor = _spatial_anchor(arrays)
    meters = level_meters(path)
    members: dict[str, DataArray] = {
        name: _plan_carried_array(name, member, meters=meters) for name, member in arrays.items()
    }
    if anchor is not None:
        transform, epsg = _proj_of(*anchor)
        return GroupSpecV3(
            attributes=_spatial_dataset_attributes(transform, tuple(anchor[1]["shape"]), epsg),
            members=members,
        )
    derived = _value_attributes(arrays, path, coordinate_values, tile) or _tile_level_attributes(
        arrays, path, tile
    )
    attributes = _carried_group_attributes(group)
    if derived is not None:
        attributes = {**attributes, **derived}
    return GroupSpecV3(attributes=attributes, members=members)


def _plan_aux_multiscale(
    group: _GroupDoc,
    path: str,
    scale_levels: tuple[int, ...],
    product_proj: tuple[tuple[float, float], int] | None = None,
) -> AnyGroupSpec:
    """
    Plan an auxiliary multiscale group (a classification mask or the cloud
    probabilities): levels present in the input are carried, coarser levels
    derive from the level before them with their arrays' attributes verbatim,
    and the group gets the multiscales layout — resampled `nearest` when the
    arrays carry a flag table, `average` otherwise.
    """
    group_members = group.get("members") or {}
    input_levels: dict[int, _GroupDoc] = {}
    for meters in scale_levels:
        member = group_members.get(f"r{meters}m")
        if member is not None:
            input_levels[meters] = member
    if not input_levels:
        msg = f"The {path} group holds none of the known resolution datasets"
        raise ValueError(msg)

    finest_arrays = {
        name: member
        for name, member in _carried_arrays(input_levels[min(input_levels)]).items()
        if name not in ("x", "y")
    }
    anchor = _spatial_anchor(finest_arrays)
    if anchor is not None:
        transform, epsg = _proj_of(*anchor)
        origin = (transform[2], transform[5])
    elif product_proj is not None:
        # Zarr V3 baselines carry no proj attributes on the mask arrays; the
        # pyramid is georeferenced from the product-level declaration instead
        origin, epsg = product_proj
    else:
        msg = f"The {path} group's finest level carries no proj attributes"
        raise ValueError(msg)

    planned_levels: dict[int, dict[str, DataArray]] = {}
    level_attributes: dict[int, SpatialGroupAttrs] = {}
    members_out: dict[str, AnyGroupSpec] = {}
    layout: list[LayoutObject] = []
    previous_meters: int | None = None
    uses_flags = False

    for meters in scale_levels:
        variables: dict[str, DataArray] = {}
        native = meters in input_levels
        if native:
            for name, member in _carried_arrays(input_levels[meters]).items():
                if name in ("x", "y"):
                    continue
                variables[name] = _plan_carried_array(name, member, meters=meters)
            if not variables:
                msg = f"The {path} r{meters}m dataset holds no arrays"
                raise ValueError(msg)
        else:
            if previous_meters is None:
                msg = (
                    f"The finest {path} level (r{meters}m) is not in the input, "
                    "so there is nothing to derive the coarser levels from"
                )
                raise ValueError(msg)
            factor = meters // previous_meters
            for name, previous in planned_levels[previous_meters].items():
                shape = tuple(dim // factor for dim in previous.shape)
                data_type = previous.data_type
                if not isinstance(data_type, str):
                    msg = f"Cannot coarsen {path} variable {name!r} of data type {data_type!r}"
                    raise TypeError(msg)
                # a coarsened variable keeps its attributes, its short_name
                # written for this level
                variables[name] = _band_array(
                    _with_level_short_name(previous.attributes.model_dump(), name, meters),
                    shape,
                    data_type,
                    previous.fill_value,
                    previous.dimension_names or ("y", "x"),
                    typesize=np.dtype(data_type).itemsize,
                )
        uses_flags = uses_flags or any(
            not {"flag_values", "flag_meanings", "flag_masks"}.isdisjoint(
                variable.attributes.model_dump()
            )
            for variable in variables.values()
        )
        planned_levels[meters] = variables
        rows, cols = next(iter(variables.values())).shape
        level_members: dict[str, DataArray] = dict(variables)
        carried = _carried_arrays(input_levels[meters]) if native else {}
        for axis, size in (("x", cols), ("y", rows)):
            level_members[axis] = _plan_level_coordinate(axis, size, carried.get(axis))
        level_attributes[meters] = _level_spatial_attributes(
            meters=meters, shape=(rows, cols), origin=origin, epsg=epsg
        )
        members_out[f"r{meters}m"] = GroupSpecV3(
            attributes=level_attributes[meters], members=level_members
        )

        level_shape2d = [rows, cols]
        level_transform = [float(meters), 0.0, origin[0], 0.0, -float(meters), origin[1]]
        entry: LayoutObject = {
            "asset": f"r{meters}m",
            "spatial:shape": level_shape2d,
            "spatial:transform": level_transform,
        }
        if previous_meters is not None:
            factor = meters // previous_meters
            entry = {
                "asset": f"r{meters}m",
                "derived_from": f"r{previous_meters}m",
                "spatial:shape": level_shape2d,
                "spatial:transform": level_transform,
                "transform": {"scale": [factor, factor], "translation": [0.0, 0.0]},
            }
        layout.append(entry)
        previous_meters = meters

    finest_meters = scale_levels[0]
    group_attributes: MultiscaleGroupAttrs = {
        "multiscales": {
            "layout": layout,
            "resampling_method": "nearest" if uses_flags else "average",
        },
        "proj:code": f"EPSG:{epsg}",
        "spatial:bbox": level_attributes[min(input_levels)]["spatial:bbox"],
        "spatial:dimensions": ["y", "x"],
        "spatial:registration": "pixel",
        "spatial:shape": level_attributes[finest_meters]["spatial:shape"],
        "spatial:transform": [
            float(finest_meters),
            0.0,
            origin[0],
            0.0,
            -float(finest_meters),
            origin[1],
        ],
        "zarr_conventions": [MULTISCALES_CMO, SPATIAL_CMO, PROJ_CMO],
    }
    return GroupSpecV3(attributes=group_attributes, members=members_out)


def _plan_subtree(
    group: _GroupDoc,
    path: str,
    product_proj: tuple[tuple[float, float], int] | None = None,
    coordinate_values: CoordinateValues | None = None,
    tile: _TileGeoreferencing | None = None,
) -> AnyGroupSpec | None:
    """
    Plan one conditions/quality subtree: aux multiscale groups by their specs,
    leaf datasets carried, dropped and empty groups skipped.
    """
    if path in _DROPPED_GROUPS:
        return None
    if path in AUX_MULTISCALE_LEVELS:
        return _plan_aux_multiscale(group, path, AUX_MULTISCALE_LEVELS[path], product_proj)
    members = group.get("members") or {}
    subgroups = {
        name: member
        for name, member in members.items()
        if isinstance(member, Mapping) and "members" in member
    }
    if not subgroups:
        return _plan_dataset(group, path, coordinate_values, tile)
    planned: dict[str, AnyGroupSpec] = {}
    for name, member in subgroups.items():
        sub = _plan_subtree(member, f"{path}/{name}", product_proj, coordinate_values, tile)
        if sub is not None:
            planned[name] = sub
    # a group holding arrays next to its subgroups keeps those arrays too
    own = _plan_dataset(group, path, coordinate_values, tile)
    if own is not None:
        return GroupSpecV3(attributes=own.attributes, members={**dict(own.members), **planned})
    if not planned:
        return None
    return GroupSpecV3(attributes=_carried_group_attributes(group), members=planned)


def plan_geozarr(
    root: Sentinel2Root, *, coordinate_values: CoordinateValues | None = None
) -> AnyGroupSpec:
    """
    Plan the GeoZarr hierarchy to write for a Sentinel-2 product.

    Pure: reads only the validated input model, returns the output tree as a
    Zarr V3 `GroupSpec` — `measurements/reflectance` as the sharded multiscale
    pyramid, and the `conditions` and `quality` subtrees carried (their own
    multiscale groups planned by `AUX_MULTISCALE_LEVELS`, the quicklooks and
    scalar arrays dropped).

    The plan reads metadata only, except where georeferencing must be derived
    from coordinate values: a carried group on a regular geographic grid
    (1-D `latitude` and `longitude` coordinates, no `proj:*` attributes,
    such as the meteorology groups) and the geometry angle grids (x/y nodes,
    in the tile CRS) get their `proj` and `spatial` attributes from the values
    `coordinate_values` looks up, and none without it.
    """
    reflectance, tile = _plan_reflectance(root)
    members: dict[str, AnyGroupSpec] = {
        "measurements": GroupSpecV3(attributes={}, members={"reflectance": reflectance}),
    }
    product_proj = _product_proj(root)
    for top in ("conditions", "quality"):
        member = root.members.get(top)
        if member is None:
            continue
        planned = _plan_subtree(member.model_dump(), top, product_proj, coordinate_values, tile)
        if planned is not None:
            members[top] = planned
    return GroupSpecV3(attributes={}, members=members)
