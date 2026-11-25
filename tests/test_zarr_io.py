"""Tests for the zarr-python glue in `eopf_geozarr.zarr_io`."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import zarr
from pydantic_zarr import v2, v3
from zarr.errors import ContainsGroupError
from zarr.storage import MemoryStore

from eopf_geozarr.zarr_io import read_documents, write_hierarchy


def _spec(
    zarr_format: Literal[2, 3], kind: Literal["array", "group"]
) -> v2.AnyArraySpec | v2.AnyGroupSpec | v3.AnyArraySpec | v3.AnyGroupSpec:
    """A small hierarchy: a group with an array and a nested group, or one array."""
    data = np.zeros((4, 4), dtype="uint8")
    if zarr_format == 2:
        array_v2 = v2.ArraySpec.from_array(data, attributes={"flag_values": [0, 1]})
        if kind == "array":
            return array_v2
        return v2.GroupSpec(
            attributes={"title": "t"},
            members={"x": array_v2, "sub": v2.GroupSpec(members={"y": array_v2})},
        )
    array_v3 = v3.ArraySpec.from_array(data, attributes={"flag_values": [0, 1]})
    if kind == "array":
        return array_v3
    return v3.GroupSpec(
        attributes={"title": "t"},
        members={"x": array_v3, "sub": v3.GroupSpec(members={"y": array_v3})},
    )


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    ("kind", "consolidated"), [("array", False), ("group", False), ("group", True)]
)
@pytest.mark.parametrize("path", ["", "a/b"])
@pytest.mark.parametrize("overwrite", [False, True])
def test_write_then_read_round_trips(
    zarr_format: Literal[2, 3],
    kind: Literal["array", "group"],
    consolidated: bool,
    path: str,
    overwrite: bool,
) -> None:
    """
    `read_documents` of the node `write_hierarchy` wrote gives back the spec
    written, at the root or a nested path, with or without consolidated
    metadata, and over an existing node when `overwrite` is set.
    """
    spec = _spec(zarr_format, kind)
    store = MemoryStore()
    if overwrite:
        # a different node already at the path, which overwrite replaces
        other = _spec(zarr_format, "group" if kind == "array" else "array")
        write_hierarchy(other, store, path)

    written = write_hierarchy(spec, store, path, overwrite=overwrite)
    assert isinstance(written, zarr.Group if kind == "group" else zarr.Array)
    if consolidated:
        zarr.consolidate_metadata(store, path=path or None, zarr_format=zarr_format)

    node = zarr.open(store, path=path, mode="r", zarr_format=zarr_format)
    documents = read_documents(node)
    assert ".zmetadata" not in documents
    assert all(
        "consolidated_metadata" not in document
        for document in documents.values()
        if isinstance(document, dict)
    )
    assert type(spec).from_stored_metadata_json(documents) == spec


def test_write_refuses_an_existing_node() -> None:
    """Without `overwrite`, writing where a node already exists raises."""
    store = MemoryStore()
    write_hierarchy(_spec(3, "group"), store, "a")
    with pytest.raises(ContainsGroupError):
        write_hierarchy(_spec(3, "array"), store, "a")
