"""
A GroupSpec-shaped view of an xarray DataTree.

`group_spec_from_datatree` is the DataTree analogue of
`pydantic_zarr.v2.GroupSpec.from_zarr`: it extracts the metadata plane of a
DataTree — named members, shapes, dtypes, and attributes — as the Zarr V2
documents xarray's own zarr writer would store. A typed input model (such as
the Sentinel-2 `Sentinel2Root`) can then validate a tree exactly as it
validates a store, without materializing the tree to a scratch store first.

The documents describe the variables as *stored*, not as decoded: every
variable goes through xarray's `encode_zarr_variable` (CF packing applied,
`_FillValue` masked, dtypes cast), with the storage-layout encoding keys a
tree read from a store carries (chunks, compressors, a V3 `serializer`, ...)
dropped first, so no source-store layout leaks into the documents. The
storage fields of the emitted array documents are neutral for the same
reason: whole-array chunks, no compressor or filters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_zarr.v2 import ArraySpec, GroupSpec
from xarray.backends.zarr import encode_zarr_attr_value, encode_zarr_variable
from xarray.conventions import encode_dataset_coordinates

if TYPE_CHECKING:
    import xarray as xr
    from pydantic_zarr.v2 import AnyArraySpec, AnyGroupSpec

#: Per-variable encoding keys that tie a variable to its source store's zarr
#: format and layout; everything else (dtype, CF packing, fill) is kept.
ZARR_STORAGE_ENCODING_KEYS = (
    "chunks",
    "preferred_chunks",
    "compressor",
    "compressors",
    "filters",
    "serializer",
    "shards",
)


def encoded_variable(name: str, variable: xr.Variable) -> xr.Variable:
    """
    The variable as xarray's zarr writer would store it: CF-encoded (packed
    dtype, `_FillValue` applied, attributes serialized), with the
    storage-layout encoding keys dropped so the layout of the store the tree
    was read from does not leak through.
    """
    cleaned = variable.copy(deep=False)
    cleaned.encoding = {
        key: value
        for key, value in cleaned.encoding.items()
        if key not in ZARR_STORAGE_ENCODING_KEYS
    }
    return encode_zarr_variable(cleaned, name=name)


def array_spec_from_variable(name: str, variable: xr.Variable) -> AnyArraySpec:
    """
    The Zarr V2 array document describing one variable as stored: the encoded
    dtype and attributes (with `_ARRAY_DIMENSIONS` carrying the dims and the
    `_FillValue` attribute lifted into the document's fill value, as xarray's
    zarr writer stores them), and neutral storage fields.
    """
    encoded = encoded_variable(name, variable)
    attributes: dict[str, object] = {
        str(key): encode_zarr_attr_value(value) for key, value in encoded.attrs.items()
    }
    fill = attributes.pop("_FillValue", None)
    if not (fill is None or isinstance(fill, bool | int | float | str)):
        msg = f"The _FillValue of variable {name!r} is not storable in a document: {fill!r}"
        raise TypeError(msg)
    attributes["_ARRAY_DIMENSIONS"] = [str(dim) for dim in encoded.dims]
    return ArraySpec(
        shape=tuple(encoded.shape),
        chunks=tuple(encoded.shape),
        dtype=encoded.dtype.str,
        fill_value=fill,
        order="C",
        filters=None,
        compressor=None,
        dimension_separator="/",
        attributes=attributes,
    )


def group_spec_from_datatree(tree: xr.DataTree) -> AnyGroupSpec:
    """
    The DataTree analogue of `GroupSpec.from_zarr`: the tree's metadata plane
    as a Zarr V2 `GroupSpec`, each node a group document (its own attributes
    and variables only — inherited coordinates stay with their ancestor, as a
    written store keeps them).
    """
    dataset = tree.to_dataset(inherit=False)
    variables, group_attrs = encode_dataset_coordinates(dataset)
    members: dict[str, AnyArraySpec | AnyGroupSpec] = {
        str(name): array_spec_from_variable(str(name), variable)
        for name, variable in variables.items()
    }
    for child_name, child in tree.children.items():
        members[str(child_name)] = group_spec_from_datatree(child)
    attributes = {str(key): encode_zarr_attr_value(value) for key, value in group_attrs.items()}
    return GroupSpec(attributes=attributes, members=members)
