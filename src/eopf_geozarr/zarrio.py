"""
This module contains zarr-specific IO routines
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, NotRequired, Protocol, TypedDict

import numcodecs
import structlog
import zarr
from zarr.core.group import GroupMetadata
from zarr.core.metadata.v3 import ArrayV3Metadata
from zarr.core.sync import sync
from zarr.dtype import Float32, ZDType
from zarr.storage._common import make_store_path

from eopf_geozarr.data_api.geozarr.common import extract_scale_offset

if TYPE_CHECKING:
    from collections.abc import Mapping

    from zarr.core.metadata.v2 import ArrayV2Metadata


class ChunkEncodingSpec(TypedDict):
    write_chunks: tuple[int, ...]
    read_chunks: NotRequired[tuple[int, ...]]


def convert_compression(
    compressor: numcodecs.abc.Codec | dict[str, object], *, compression_level: int | None = None
) -> tuple[dict[str, Any], ...]:
    """
    Convert the compression parameters for a Zarr V2 array into a tuple of Zarr V3 codecs.

    Parameters
    ----------
    compressor : numcodecs.abc.Codec | dict[str, object]
        The compressor of the zarr v3 array
    compression_level: int | None = None
        The new compression level to use.
        Only applies to the specific case of blosc -> blosc compression.
    """

    if compressor is None:
        return ()

    SHUFFLE = ("noshuffle", "shuffle", "bitshuffle")
    if isinstance(compressor, numcodecs.abc.Codec):
        old_config = compressor.get_config()
    else:
        old_config = compressor
    if old_config["id"] == "blosc":
        new_level = old_config["clevel"] if compression_level is None else compression_level
        new_codec = {
            "name": "blosc",
            "configuration": {
                "cname": old_config["cname"],
                "clevel": new_level,
                "blocksize": old_config["blocksize"],
                "shuffle": SHUFFLE[old_config["shuffle"]],
            },
        }
        return (new_codec,)

    raise ValueError(f"Only blosc -> blosc or None -> () is supported. Got {compressor=}")


class ArrayReencoderConfig(TypedDict):
    keep_scale_offset: NotRequired[bool]
    spatial_chunk: NotRequired[int]
    enable_sharding: NotRequired[bool]
    compression_level: NotRequired[int]


def default_array_reencoder(
    key: str,
    metadata: ArrayV2Metadata,
) -> ArrayV3Metadata:
    """
    Re-encode a zarr array into a new zarr v3 array.
    """
    keep_scale_offset = False
    new_codecs = convert_compression(metadata.compressor)
    if metadata.fill_value is None:
        fill_value = metadata.dtype.default_scalar()
    else:
        fill_value = metadata.fill_value
    attrs, maybe_scale_offset = extract_scale_offset(metadata.attributes.copy())
    out_dtype: ZDType[Any, Any]
    # If the array used CF scale/offset encoding, we change the output dtype to Float32
    out_dtype = metadata.dtype
    if not keep_scale_offset and len(maybe_scale_offset) > 0:
        out_dtype = Float32()

    dimension_names = attrs.pop("_ARRAY_DIMENSIONS", None)
    chunk_grid_shape = metadata.chunks
    codecs = ({"name": "bytes"}, *new_codecs)

    return ArrayV3Metadata(
        shape=metadata.shape,
        data_type=out_dtype,
        chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
        chunk_grid={
            "name": "regular",
            "configuration": {"chunk_shape": chunk_grid_shape},
        },
        fill_value=fill_value,
        dimension_names=dimension_names,
        codecs=codecs,
        attributes=attrs,
    )


def replace_json_invalid_floats(obj: object) -> object:
    """
    Recursively replace NaN and Infinity float values in a JSON-like object
    with their string representations, to make them JSON-compliant.

    Parameters
    ----------
    obj : object
        The JSON-like object to process

    Returns
    -------
    object
        The processed object with NaN and Infinity replaced
    """
    if isinstance(obj, float):
        if obj != obj:  # NaN check
            return "NaN"
        if obj == float("inf"):
            return "Infinity"
        if obj == float("-inf"):
            return "-Infinity"
        return obj
    if isinstance(obj, dict):
        return {k: replace_json_invalid_floats(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [replace_json_invalid_floats(item) for item in obj]
    return obj


class ArrayReencoder(Protocol):
    @staticmethod
    def __call__(
        key: str,
        metadata: ArrayV2Metadata,
    ) -> ArrayV3Metadata: ...


def reencode_group(
    group: zarr.Group,
    store: zarr.storage.StoreLike,
    path: str,
    *,
    overwrite: bool = False,
    use_consolidated_for_children: bool = False,
    omit_nodes: set[str] | None = None,
    array_reencoder: ArrayReencoder = default_array_reencoder,
    allow_json_nan: bool = False,
) -> zarr.Group:
    """
    Re-encode a Zarr group, applying a re-encoding to all sub-groups and sub-arrays.

    Parameters
    ----------
    group : zarr.Group
        The Zarr group to re-encode
    store : zarr.storage.StoreLike
        The store to write into
    path : str
        The path in the new store to use
    overwrite : bool, default = False
        Whether to overwrite contents of the new store
    omit_nodes : set[str], default = {}
        The names of groups or arrays to omit from re-encoding.
    chunk_reencoder : Callable[[zarr.Array[Any], ChunkEncodingSpec]] | None, default = None
        A function that takes a Zarr array object and returns a ChunkEncodingSpec, which is a dict
        that defines a new chunk encoding. Use this parameter to define per-array chunk encoding
        logic.
    allow_json_nan : bool, default = False
        Whether to allow NaN and Infinity values in JSON metadata. If False, float("NaN") will be
        encoded as the string "Nan", float("Inf") as "Infinity", and float("-Inf") as "-Infinity".
    """
    if omit_nodes is None:
        omit_nodes = set()

    log = structlog.get_logger()

    # Convert store-like to a proper Store object
    store_path = sync(make_store_path(store))
    store = store_path.store

    members = dict(
        group.members(max_depth=None, use_consolidated_for_children=use_consolidated_for_children)
    )

    log.info("Begin re-encoding Zarr group %s", group)
    new_members: dict[str, ArrayV3Metadata | GroupMetadata] = {
        path: GroupMetadata(zarr_format=3, attributes=group.attrs.asdict())
    }
    chunks_to_encode: list[str] = []
    for name in omit_nodes:
        if not any(k.startswith(name) for k in members):
            log.warning(
                "The name %s was provided in omit_nodes but no such array or group exists.", name
            )
    for name, member in members.items():
        if any(name.startswith(v) for v in omit_nodes):
            log.info(
                "Skipping node %s because it is contained in a subgroup declared in the omit_groups parameter",
                name,
            )
            continue

        log.info("Re-encoding member %s", name)
        new_path = f"{path}/{name}"
        source_meta = member.metadata

        if not allow_json_nan:
            log.info("Replacing any nan and inf float values with string equivalents")
            # replace any invalid floats with string encodings using the
            # zarr v3 fill value encoding for floats
            new_attrs = replace_json_invalid_floats(source_meta.attributes)
            source_meta = replace(source_meta, attributes=new_attrs)

        if isinstance(member, zarr.Array):
            new_meta = array_reencoder(member.path, source_meta)
            new_members[new_path] = new_meta
            chunks_to_encode.append(name)
        else:
            new_members[new_path] = GroupMetadata(
                zarr_format=3,
                attributes=member.attrs.asdict(),
            )
    log.info("Creating new Zarr hierarchy structure at %s", f"{store}/{path}")
    tree = dict(zarr.create_hierarchy(store=store, nodes=new_members, overwrite=overwrite))
    new_group: zarr.Group = tree[path]
    for name in chunks_to_encode:
        log.info("Re-encoding chunks for array %s", name)
        old_array = group[name]
        new_array = new_group[name]

        new_array[...] = old_array[...]

    # return the root group
    return tree[path]


class ArrayEncoding(TypedDict):
    dimension_names: NotRequired[None | tuple[str | None, ...]]
    attributes: NotRequired[Mapping[str, object] | None]
