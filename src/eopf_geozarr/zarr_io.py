"""
Reading and writing pydantic-zarr specs with zarr-python.

pydantic-zarr models metadata documents and, from 0.12, does no IO of its own:
a spec's ``to_stored_metadata_json`` gives the documents a store holds, keyed
as the store keys them, and ``from_stored_metadata_json`` reads such a mapping
back. These two functions move the documents in and out of a store with
zarr-python, for Zarr V2 and V3 alike. They follow the ``examples/zarr_io.py``
module of pydantic-zarr, which documents this split.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import zarr
from pydantic_zarr import v2, v3
from zarr.core.buffer import default_buffer_prototype
from zarr.core.group import GroupMetadata
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

if TYPE_CHECKING:
    from zarr.abc.store import Store

#: The store key of a Zarr V2 hierarchy's consolidated metadata.
_ZMETADATA = ".zmetadata"


def write_hierarchy(
    spec: v2.AnyArraySpec | v2.AnyGroupSpec | v3.AnyArraySpec | v3.AnyGroupSpec,
    store: Store,
    path: str = "",
    *,
    overwrite: bool = False,
) -> zarr.Array | zarr.Group:
    """
    Write the metadata documents of `spec` to `store` at `path` and return the
    node written there. No array data is written.

    `zarr.create_hierarchy` writes every node, refusing to overwrite an
    existing node unless `overwrite` is set.
    """
    flat = v2.to_flat(spec) if spec.zarr_format == 2 else v3.to_flat(spec)
    nodes: dict[str, ArrayV2Metadata | ArrayV3Metadata | GroupMetadata] = {}
    for node_path, node in flat.items():
        # the document as JSON holds it, with tuples as arrays
        document = json.loads(json.dumps(node.to_metadata_json()))
        key = "/".join(part for part in (path.strip("/"), node_path.strip("/")) if part)
        if "shape" not in document:
            nodes[key] = GroupMetadata.from_dict(document)
        elif document["zarr_format"] == 2:
            nodes[key] = ArrayV2Metadata.from_dict(document)
        else:
            nodes[key] = ArrayV3Metadata.from_dict(document)
    written = dict(zarr.create_hierarchy(store=store, nodes=nodes, overwrite=overwrite))
    return written[path.strip("/")]


def _own_document(document: object) -> object:
    """
    `document` without the ``consolidated_metadata`` of a Zarr V3 group:
    zarr-python's copy of the documents below the group, which
    `read_documents` reads from each node instead. (zarr-python before 3.4
    also writes ``"consolidated_metadata": null``, which the readers refuse.)
    """
    if isinstance(document, dict) and "consolidated_metadata" in document:
        return {key: value for key, value in document.items() if key != "consolidated_metadata"}
    return document


def read_documents(node: zarr.Array | zarr.Group) -> dict[str, object]:
    """
    The metadata documents of `node` and, for a group, of every node under it,
    keyed by store key relative to `node`: the mapping
    ``from_stored_metadata_json`` takes.

    Consolidated metadata is left out (a Zarr V2 ``.zmetadata`` document, a
    Zarr V3 group's ``consolidated_metadata`` field): it repeats the documents
    of the nodes below, which are read here one by one. Lists in the documents
    stay lists, as JSON holds them.
    """
    prototype = default_buffer_prototype()
    nodes: list[tuple[str, zarr.Array | zarr.Group]] = [("", node)]
    if isinstance(node, zarr.Group):
        nodes.extend(node.members(max_depth=None))
    documents: dict[str, object] = {}
    for member_path, member in nodes:
        for name, buffer in member.metadata.to_buffer_dict(prototype).items():
            if name == _ZMETADATA:
                continue
            key = f"{member_path}/{name}" if member_path else name
            documents[key] = _own_document(json.loads(buffer.to_bytes()))
    return documents
