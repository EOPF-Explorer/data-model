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
sharded, per-level coordinate arrays, `spatial_ref` grid-mapping arrays, the
spatial/proj convention attributes, and the multiscales layout) are all
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
attributes and `spatial_ref` grid mappings, and the quicklook groups and
scalar arrays are dropped, as the converter drops them.
"""

from __future__ import annotations

import base64
import struct
from collections.abc import Mapping
from math import ceil
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import xarray as xr
from pydantic_zarr.v2 import GroupSpec as GroupSpecV2
from pydantic_zarr.v3 import ArraySpec as ArraySpecV3
from pydantic_zarr.v3 import GroupSpec as GroupSpecV3
from pyproj import CRS
from typing_extensions import TypedDict
from zarr_cm import ConventionMetadataObject
from zarr_cm.multiscales import CMO as MULTISCALES_CMO
from zarr_cm.multiscales import MultiscalesAttrs
from zarr_cm.proj import CMO as PROJ_CMO
from zarr_cm.spatial import CMO as SPATIAL_CMO
from zarr_metadata.typed_json import check

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
    from zarr_metadata.v3.codec.zstd import ZstdCodecObject

    from eopf_geozarr.data_api.geozarr.scope import (
        FixedLengthUtf32Object,
        NullTerminatedBytesObject,
    )

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
#: and `_eopf_attrs`, and the per-band `proj:wkt2`, which the per-level
#: `spatial_ref` array carries instead) is dropped.
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


def _product_proj(root: Sentinel2Root) -> tuple[tuple[float, float], int, str] | None:
    """
    The product-level georeferencing, as (origin, EPSG, WKT2), from the
    `proj:*` keys of `stac_discovery` `properties` — the only place the Zarr
    V3 baselines (N0512) declare it. The origin is the bbox's (min-x, max-y)
    corner. Returns None when any of the three is missing or malformed.
    """
    properties = root.attributes.stac_discovery.get("properties")
    if not isinstance(properties, dict):
        return None
    bbox = properties.get("proj:bbox")
    code = properties.get("proj:code")
    wkt2 = properties.get("proj:wkt2")
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) < 4
        or not all(isinstance(value, (int, float)) for value in bbox[:4])
        or not isinstance(code, str)
        or not code.startswith("EPSG:")
        or not isinstance(wkt2, str)
    ):
        return None
    return (float(bbox[0]), float(bbox[3])), int(code.removeprefix("EPSG:")), wkt2


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
        "grid_mapping": "spatial_ref",
        "units": "1",
    }
    for key in _BAND_ATTR_CARRY:
        if key in source_attrs:
            attributes[key] = source_attrs[key]
    return _band_array(attributes, shape, data_type, fill_value, dimension_names, typesize)


def _band_array(
    attributes: dict[str, object],
    shape: tuple[int, ...],
    data_type: str | FixedLengthUtf32Object | NullTerminatedBytesObject,
    fill_value: FillValue,
    dimension_names: tuple[str, ...],
    typesize: int,
) -> DataArray:
    """One band array document: sharded uint16 with the blosc policy."""
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
    """A 1-D coordinate array (`x` or `y`): float64, NaN-filled, one chunk."""
    return DataArray.model_validate(
        {
            "attributes": {"_FillValue": _NAN_FLOAT64_B64},
            "shape": (size,),
            "data_type": "float64",
            "chunk_grid": _regular_grid((size,)),
            "chunk_key_encoding": _CHUNK_KEY_ENCODING,
            "fill_value": "NaN",
            "codecs": (_BYTES_LITTLE,),
            "dimension_names": (name,),
        }
    )


class GridMappingAttrs(TypedDict, closed=True):
    """
    The attributes of a Sentinel-2 `spatial_ref` grid-mapping array: pyproj's
    CF encoding of the tile's UTM (transverse mercator) CRS, plus the WKT1
    spellings rioxarray stores (CF's `crs_wkt` and GDAL's `spatial_ref`).
    """

    crs_wkt: str
    false_easting: float
    false_northing: float
    geographic_crs_name: str
    grid_mapping_name: str
    horizontal_datum_name: str
    inverse_flattening: float
    latitude_of_projection_origin: float
    longitude_of_central_meridian: float
    longitude_of_prime_meridian: float
    prime_meridian_name: str
    projected_crs_name: str
    reference_ellipsoid_name: str
    scale_factor_at_central_meridian: float
    semi_major_axis: float
    semi_minor_axis: float
    spatial_ref: str


def _grid_mapping_attributes(wkt2: str) -> GridMappingAttrs:
    """
    The CF grid-mapping attributes of a CRS, rioxarray-style, derived from
    the product's own `proj:wkt2` (rather than re-deriving from the EPSG
    code, which is hostage to the installed PROJ database's naming).
    """
    # round-trip through CF: the stored grid mapping is exactly what a reader
    # reconstructs from the CF attributes, with the authority bookkeeping the
    # CF form cannot carry dropped from the WKT
    crs = CRS.from_cf(CRS.from_wkt(wkt2).to_cf())
    attributes: dict[str, object] = dict(crs.to_cf())
    wkt = crs.to_wkt("WKT1_GDAL")
    attributes["crs_wkt"] = wkt
    attributes["spatial_ref"] = wkt
    checked, problems = check(attributes, GridMappingAttrs)
    if checked is None or problems:
        details = "; ".join(f"{'.'.join(map(str, p.loc))}: {p.message}" for p in problems)
        msg = f"The product CRS is not the expected transverse-mercator grid mapping: {details}"
        raise ValueError(msg)
    return checked


def _plan_spatial_ref(wkt2: str, *, native: bool) -> ArraySpecV3:
    """
    The dimensionless `spatial_ref` grid-mapping array of one level.

    A plain `pydantic_zarr.v3.ArraySpec` rather than a GeoZarr `DataArray`:
    the grid-mapping array has no dimensions, which `DataArray` requires.
    The levels written natively carry a zstd codec; the coarsened levels are
    written uncompressed (mirroring the two write paths of the converter).
    """
    codecs: tuple[BytesCodecObject | ZstdCodecObject, ...] = (_BYTES_LITTLE,)
    if native:
        codecs = (*codecs, {"name": "zstd", "configuration": {"level": 0, "checksum": False}})
    return ArraySpecV3.model_validate(
        {
            "attributes": _grid_mapping_attributes(wkt2),
            "shape": (),
            "data_type": "int64",
            "chunk_grid": _regular_grid(()),
            "chunk_key_encoding": _CHUNK_KEY_ENCODING,
            "fill_value": 0,
            "codecs": codecs,
        }
    )


SpatialGroupAttrs = TypedDict(
    "SpatialGroupAttrs",
    {
        "grid_mapping": str,
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

MultiscaleGroupAttrs = TypedDict(
    "MultiscaleGroupAttrs",
    {
        "multiscales": MultiscalesAttrs,
        "proj:code": str,
        "spatial:bbox": list[float],
        "spatial:dimensions": list[str],
        "spatial:registration": str,
        "spatial:transform": list[float],
        "zarr_conventions": list[ConventionMetadataObject],
    },
)
"""The attributes of a multiscale pyramid's root group: zarr-cm's multiscales
attributes plus the finest level's georeferencing (no `grid_mapping` or
`spatial:shape`, which belong to the level groups)."""


def _spatial_dataset_attributes(
    transform: tuple[float, float, float, float, float, float],
    shape: tuple[int, ...],
    epsg: int,
) -> SpatialGroupAttrs:
    """The spatial/proj convention attributes of one spatial dataset's group."""
    res_x, _, x0, _, res_y, y0 = transform
    rows, cols = shape
    return {
        "grid_mapping": "spatial_ref",
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


def plan_reflectance(root: Sentinel2Root) -> AnyGroupSpec:
    """
    Plan the GeoZarr `measurements/reflectance` multiscale group of a
    Sentinel-2 product: one level group per entry of
    `REFLECTANCE_SCALE_LEVELS`, native levels carried from the input (with
    the product level's finer bands injected), coarser levels derived from
    the level before them, each holding sharded uint16 band arrays, float64
    `x`/`y` coordinates and a `spatial_ref` grid-mapping array, under a
    multiscales layout attribute.
    """
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
    band_wkt2 = first_band_attrs.get("proj:wkt2")
    if (
        isinstance(transform, (list, tuple))
        and len(transform) >= 6
        and isinstance(band_epsg, (int, str))
        and isinstance(band_wkt2, str)
    ):
        origin = (float(transform[2]), float(transform[5]))
        epsg = int(band_epsg)
        wkt2 = band_wkt2
    else:
        product = _product_proj(root)
        if product is None:
            msg = (
                f"Band {first_band_name!r} carries no usable proj:transform/proj:epsg/proj:wkt2 "
                "and the product declares no stac_discovery proj georeferencing"
            )
            raise ValueError(msg)
        origin, epsg, wkt2 = product

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
        level_members: dict[str, DataArray | ArraySpecV3] = dict(bands)
        level_members["x"] = _plan_coordinate("x", cols)
        level_members["y"] = _plan_coordinate("y", rows)
        level_members["spatial_ref"] = _plan_spatial_ref(wkt2, native=native)

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


# ---------------------------------------------------------------------------
# conditions and quality

#: Groups that get a multiscale pyramid, with their levels in metres (the
#: converter's aux multiscale specs); levels present in the input are reused,
#: the rest are coarsened from the level before them.
AUX_MULTISCALE_LEVELS: dict[str, tuple[int, ...]] = {
    "conditions/mask/l1c_classification": (60, 120, 360, 720),
    "conditions/mask/l2a_classification": (20, 60, 120, 360, 720),
    "quality/probability": (20, 60, 120, 360, 720),
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


def _v3_data_type(dtype_v2: str) -> str | FixedLengthUtf32Object | NullTerminatedBytesObject:
    """The Zarr V3 data type of a Zarr V2 dtype string."""
    parsed = np.dtype(dtype_v2)
    if parsed.kind == "U":
        return {
            "name": "fixed_length_utf32",
            "configuration": {"length_bytes": parsed.itemsize},
        }
    if parsed.kind == "S":
        return {
            "name": "null_terminated_bytes",
            "configuration": {"length_bytes": parsed.itemsize},
        }
    return parsed.name


def _carried_array_attributes(
    attrs: Mapping[str, object], *, is_float: bool, grid_mapping: bool, name: str
) -> dict[str, object]:
    """The output attributes of a carried array: sanitized, with the CF float
    fill, the grid-mapping reference and the SCL flag table added."""
    out = {key: value for key, value in attrs.items() if key not in _DROPPED_ATTRS}
    if is_float:
        out["_FillValue"] = _NAN_FLOAT64_B64
    if grid_mapping:
        out["grid_mapping"] = "spatial_ref"
    if name == "scl":
        # the converter's SCL flag table and NO_DATA fill declaration
        out.update(SCL_ATTRS)
        out["_FillValue"] = SCL_NODATA
    return out


def _plan_carried_array(name: str, doc: _ArrayDoc, *, spatial: bool) -> DataArray:
    """
    One carried (non-reflectance) array, translated to its output document:
    `x`/`y` become the float64 coordinate arrays, other coordinate arrays
    (an array whose only dimension is its own name) keep their data type
    uncompressed, and data arrays are sharded with the blosc policy.
    """
    attrs = _doc_attrs(doc)
    dims_raw = attrs.get("_ARRAY_DIMENSIONS")
    if not isinstance(dims_raw, (list, tuple)):
        msg = f"Array {name!r} carries no _ARRAY_DIMENSIONS attribute"
        raise TypeError(msg)
    dims = tuple(str(dim) for dim in dims_raw)
    shape = tuple(doc["shape"])
    if name in ("x", "y") and dims == (name,):
        return _plan_coordinate(name, shape[0])
    parsed = np.dtype(str(doc["dtype"]))
    data_type = _v3_data_type(str(doc["dtype"]))
    fill_value: FillValue = "NaN" if parsed.kind == "f" else "" if parsed.kind in "US" else 0
    attributes = _carried_array_attributes(
        attrs,
        is_float=parsed.kind == "f",
        grid_mapping=spatial and "proj:epsg" in attrs,
        name=name,
    )
    if dims == (name,):
        # a coordinate array: one chunk, uncompressed
        return DataArray.model_validate(
            {
                "attributes": attributes,
                "shape": shape,
                "data_type": data_type,
                "chunk_grid": _regular_grid(shape),
                "chunk_key_encoding": _CHUNK_KEY_ENCODING,
                "fill_value": fill_value,
                "codecs": (_BYTES_LITTLE,),
                "dimension_names": dims,
            }
        )
    return _band_array(attributes, shape, data_type, fill_value, dims, typesize=parsed.itemsize)


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


def _proj_of(name: str, doc: _ArrayDoc) -> tuple[_Transform6, int, str]:
    """The (transform, EPSG, WKT2) a proj-bearing input array carries."""
    attrs = _doc_attrs(doc)
    transform = attrs.get("proj:transform")
    if not isinstance(transform, (list, tuple)) or len(transform) < 6:
        msg = f"Array {name!r} carries no usable proj:transform"
        raise ValueError(msg)
    a, b, c, d, e, f = (float(part) for part in transform[:6])
    return (
        (a, b, c, d, e, f),
        int(attrs["proj:epsg"]),  # type: ignore[call-overload]
        str(attrs["proj:wkt2"]),
    )


def _plan_dataset(group: _GroupDoc, path: str) -> AnyGroupSpec | None:
    """
    Plan one carried leaf dataset: every non-scalar array translated, and —
    when the dataset is georeferenced (an array carries proj attributes) —
    the group's spatial attributes and a `spatial_ref` grid-mapping array.
    Returns `None` for a dataset with no arrays to carry.
    """
    arrays = _carried_arrays(group)
    if not arrays:
        return None
    anchor = _spatial_anchor(arrays)
    members: dict[str, DataArray | ArraySpecV3] = {
        name: _plan_carried_array(name, member, spatial=anchor is not None)
        for name, member in arrays.items()
    }
    if anchor is None:
        return GroupSpecV3(attributes=dict(_doc_attrs(group)), members=members)
    transform, epsg, wkt2 = _proj_of(*anchor)
    members["spatial_ref"] = _plan_spatial_ref(wkt2, native=True)
    return GroupSpecV3(
        attributes=_spatial_dataset_attributes(transform, tuple(anchor[1]["shape"]), epsg),
        members=members,
    )


def _plan_aux_multiscale(
    group: _GroupDoc,
    path: str,
    scale_levels: tuple[int, ...],
    product_proj: tuple[tuple[float, float], int, str] | None = None,
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
        transform, epsg, wkt2 = _proj_of(*anchor)
        origin = (transform[2], transform[5])
    elif product_proj is not None:
        # Zarr V3 baselines carry no proj attributes on the mask arrays; the
        # pyramid is georeferenced from the product-level declaration instead
        origin, epsg, wkt2 = product_proj
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
                variables[name] = _plan_carried_array(name, member, spatial=True)
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
                # a coarsened variable keeps its attributes verbatim
                variables[name] = _band_array(
                    previous.attributes.model_dump(),
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
        level_members: dict[str, DataArray | ArraySpecV3] = dict(variables)
        level_members["x"] = _plan_coordinate("x", cols)
        level_members["y"] = _plan_coordinate("y", rows)
        level_members["spatial_ref"] = _plan_spatial_ref(wkt2, native=native)
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
    product_proj: tuple[tuple[float, float], int, str] | None = None,
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
        return _plan_dataset(group, path)
    planned: dict[str, AnyGroupSpec] = {}
    for name, member in subgroups.items():
        sub = _plan_subtree(member, f"{path}/{name}", product_proj)
        if sub is not None:
            planned[name] = sub
    # a group holding arrays next to its subgroups keeps those arrays too
    own = _plan_dataset(group, path)
    if own is not None:
        return GroupSpecV3(attributes=own.attributes, members={**dict(own.members), **planned})
    if not planned:
        return None
    return GroupSpecV3(attributes=dict(_doc_attrs(group)), members=planned)


def plan_geozarr(root: Sentinel2Root) -> AnyGroupSpec:
    """
    Plan the GeoZarr hierarchy to write for a Sentinel-2 product.

    Pure: reads only the validated input model, returns the output tree as a
    Zarr V3 `GroupSpec` — `measurements/reflectance` as the sharded multiscale
    pyramid, and the `conditions` and `quality` subtrees carried (their own
    multiscale groups planned by `AUX_MULTISCALE_LEVELS`, the quicklooks and
    scalar arrays dropped).
    """
    members: dict[str, AnyGroupSpec] = {
        "measurements": GroupSpecV3(attributes={}, members={"reflectance": plan_reflectance(root)}),
    }
    product_proj = _product_proj(root)
    for top in ("conditions", "quality"):
        member = root.members.get(top)
        if member is None:
            continue
        planned = _plan_subtree(member.model_dump(), top, product_proj)
        if planned is not None:
            members[top] = planned
    return GroupSpecV3(attributes={}, members=members)
