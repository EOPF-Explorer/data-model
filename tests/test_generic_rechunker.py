"""Tests for the generic rechunking pipeline (eopf_geozarr.generic_optimization).

These call ``create_generic_geozarr_dataset`` directly, so they need no ``eopf``
install. The CPM writer's ``generic_rechunker=True`` option is tested in
``tests/test_cpm_writer.py``.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pytest
import xarray as xr
import zarr
from structlog.testing import capture_logs
from zarr.codecs import BloscCodec

from eopf_geozarr.generic_optimization.generic_converter import create_generic_geozarr_dataset

from .conftest import create_zarrv3_group_from_json, get_stem, read_json

s1_slc_example_json_paths = tuple(pathlib.Path("tests/_test_data/s1_slc_examples").glob("*.json"))

#: Cap for every array dimension of the S1 SLC fixture. A full burst holds a
#: (2, ~1500, ~24000) complex64 array, far too large to convert in a unit test.
SLC_MAX_DIM_SIZE = 64


def build_synthetic_tree() -> xr.DataTree:
    """A small product that is neither S2 nor OLCI.

    Two leaf groups with data (one larger than the test chunk size, one
    smaller) and one empty group.
    """
    tree = xr.DataTree()
    tree["measurements"] = xr.Dataset(
        {
            "amplitude": (
                ("azimuth_time", "slant_range"),
                np.arange(300 * 500, dtype="float32").reshape(300, 500),
            )
        },
        coords={"azimuth_time": np.arange(300), "slant_range": np.arange(500)},
    )
    tree["conditions/geometry"] = xr.Dataset(
        {
            "incidence_angle": (
                ("y", "x"),
                np.linspace(20.0, 45.0, 40 * 60, dtype="float32").reshape(40, 60),
            )
        }
    )
    tree["quality/empty"] = xr.Dataset()
    return tree


def build_scaled_tree() -> xr.DataTree:
    """A tree whose only variable is stored on disk as scaled uint16."""
    variable = xr.DataArray(
        np.linspace(1.0, 100.0, 40 * 60, dtype="float32").reshape(40, 60), dims=("y", "x")
    )
    variable.encoding = {"dtype": "uint16", "scale_factor": 0.5, "add_offset": 1.0, "_FillValue": 0}
    tree = xr.DataTree()
    tree["measurements"] = xr.Dataset({"reflectance": variable})
    return tree


def leaf_groups_with_data(tree: xr.DataTree) -> list[str]:
    """Groups the rechunker converts: no child groups, at least one data variable."""
    return sorted(
        path
        for path in tree.groups
        if path != "/" and not tree[path].children and tree[path].data_vars
    )


def _cap_array_shapes(node: dict[str, Any], max_dim_size: int) -> None:
    """Cap every array dimension (and its chunk size) at `max_dim_size`, in place.

    Arrays that share a dimension name share its original size, so they still
    agree after capping.
    """
    if node["node_type"] == "array":
        node["shape"] = [min(size, max_dim_size) for size in node["shape"]]
        config = node["chunk_grid"]["configuration"]
        config["chunk_shape"] = [
            min(chunk, size)
            for chunk, size in zip(config["chunk_shape"], node["shape"], strict=True)
        ]
        return
    for member in (node.get("members") or {}).values():
        _cap_array_shapes(member, max_dim_size)


def open_capped_s1_slc_example(source_path: pathlib.Path, tmp_path: pathlib.Path) -> xr.DataTree:
    """Build the S1 SLC fixture as a Zarr V3 store with capped array sizes and open it."""
    spec = read_json(source_path)
    _cap_array_shapes(spec, SLC_MAX_DIM_SIZE)
    capped_json = tmp_path / "capped" / source_path.name
    capped_json.parent.mkdir()
    capped_json.write_text(json.dumps(spec))
    store = create_zarrv3_group_from_json(capped_json, tmp_path)
    return xr.open_datatree(store, engine="zarr", chunks={})


def read_array(output_path: pathlib.Path, path: str) -> zarr.Array:
    """Open one written array of the output store."""
    array = zarr.open_group(str(output_path), mode="r")[path]
    assert isinstance(array, zarr.Array), f"{path} is not an array"
    return array


def convert(
    tree: xr.DataTree, tmp_path: pathlib.Path, **overrides: Any
) -> tuple[xr.DataTree, pathlib.Path]:
    """Run the rechunker with small-array defaults; return its result and the output path."""
    options: dict[str, Any] = {
        "spatial_chunk": 128,
        "enable_sharding": True,
        "compression_level": 3,
        "keep_scale_offset": True,
    } | overrides
    output_path = tmp_path / "out.zarr"
    with capture_logs():
        result = create_generic_geozarr_dataset(tree, str(output_path), **options)
    return result, output_path


def test_writes_every_leaf_group_with_data(tmp_path: pathlib.Path) -> None:
    """Every leaf group with data is written; empty groups are skipped."""
    _, output_path = convert(build_synthetic_tree(), tmp_path)

    root = zarr.open_group(str(output_path), mode="r")
    assert "amplitude" in root["measurements"]
    assert "incidence_angle" in root["conditions/geometry"]
    assert "quality/empty" not in root


def test_chunks_are_capped_at_spatial_chunk(tmp_path: pathlib.Path) -> None:
    """Chunks are `min(spatial_chunk, size)` along each dimension."""
    _, output_path = convert(build_synthetic_tree(), tmp_path, spatial_chunk=128)

    assert read_array(output_path, "measurements/amplitude").chunks == (128, 128)
    # Smaller than spatial_chunk in both dimensions: one chunk for the whole array.
    assert read_array(output_path, "conditions/geometry/incidence_angle").chunks == (40, 60)


@pytest.mark.parametrize(
    ("enable_sharding", "expected_shards"),
    [
        # One shard per array, rounded up to a whole number of 128-chunks.
        (True, (384, 512)),
        (False, None),
    ],
)
def test_sharding(
    tmp_path: pathlib.Path, enable_sharding: bool, expected_shards: tuple[int, ...] | None
) -> None:
    _, output_path = convert(build_synthetic_tree(), tmp_path, enable_sharding=enable_sharding)

    assert read_array(output_path, "measurements/amplitude").shards == expected_shards


def test_compression_is_zstd_at_requested_level(tmp_path: pathlib.Path) -> None:
    _, output_path = convert(build_synthetic_tree(), tmp_path, compression_level=7)

    (compressor,) = read_array(output_path, "measurements/amplitude").compressors
    assert isinstance(compressor, BloscCodec)
    assert compressor.cname == "zstd"
    assert compressor.clevel == 7


def test_data_round_trips(tmp_path: pathlib.Path) -> None:
    tree = build_synthetic_tree()
    _, output_path = convert(tree, tmp_path)

    for path in ("measurements/amplitude", "conditions/geometry/incidence_angle"):
        np.testing.assert_array_equal(read_array(output_path, path)[:], tree[path].values)


def test_keep_scale_offset_true_keeps_packed_integers(tmp_path: pathlib.Path) -> None:
    _, output_path = convert(build_scaled_tree(), tmp_path, keep_scale_offset=True)

    reflectance = read_array(output_path, "measurements/reflectance")
    assert reflectance.dtype == np.dtype("uint16")
    assert reflectance.attrs["scale_factor"] == 0.5
    assert reflectance.attrs["add_offset"] == 1.0


def test_keep_scale_offset_false_writes_decoded_floats(tmp_path: pathlib.Path) -> None:
    _, output_path = convert(build_scaled_tree(), tmp_path, keep_scale_offset=False)

    reflectance = read_array(output_path, "measurements/reflectance")
    assert reflectance.dtype == np.dtype("float32")
    assert "scale_factor" not in reflectance.attrs
    assert "add_offset" not in reflectance.attrs
    assert np.isnan(reflectance.fill_value)


@pytest.mark.parametrize("dtype", ["int32", "complex64"])
def test_keep_scale_offset_false_handles_non_float_variables(
    tmp_path: pathlib.Path, dtype: str
) -> None:
    """Variables that are not floats are written with their dtype intact.

    Covers the S1 SLC `measurements/slc` (complex64) array; a NaN fill value
    cannot be encoded for integer or complex dtypes.
    """
    tree = xr.DataTree()
    tree["measurements"] = xr.Dataset({"values": (("y", "x"), np.ones((40, 60), dtype=dtype))})

    _, output_path = convert(tree, tmp_path, keep_scale_offset=False)

    assert read_array(output_path, "measurements/values").dtype == np.dtype(dtype)


def test_root_metadata_is_consolidated(tmp_path: pathlib.Path) -> None:
    _, output_path = convert(build_synthetic_tree(), tmp_path)

    root = zarr.open_group(str(output_path), mode="r")
    assert root.metadata.consolidated_metadata is not None


def test_returns_the_written_output(tmp_path: pathlib.Path) -> None:
    """The result is the tree read back from the output, not the input tree."""
    result, _ = convert(build_synthetic_tree(), tmp_path, spatial_chunk=128)

    assert result["measurements/amplitude"].encoding["chunks"] == (128, 128)


@pytest.mark.xfail(
    strict=True,
    reason="Groups that have child groups are skipped, so their own variables are not written.",
)
def test_writes_variables_of_groups_with_children(tmp_path: pathlib.Path) -> None:
    tree = xr.DataTree()
    tree["measurements"] = xr.Dataset({"parent_var": (("y", "x"), np.ones((4, 4), "float32"))})
    tree["measurements/child"] = xr.Dataset({"child_var": (("y", "x"), np.ones((4, 4), "float32"))})

    _, output_path = convert(tree, tmp_path)

    root = zarr.open_group(str(output_path), mode="r")
    assert "child_var" in root["measurements/child"]
    assert "parent_var" in root["measurements"]


@pytest.mark.filterwarnings("ignore:.*:UserWarning")
@pytest.mark.parametrize("keep_scale_offset", [True, False])
@pytest.mark.parametrize("source_path", s1_slc_example_json_paths, ids=get_stem)
def test_s1_slc_example_converts(
    source_path: pathlib.Path, tmp_path: pathlib.Path, keep_scale_offset: bool
) -> None:
    """Every data group of every burst of a real S1 SLC layout is written."""
    tree = open_capped_s1_slc_example(source_path, tmp_path)
    expected_groups = leaf_groups_with_data(tree)
    assert expected_groups, "fixture has no data groups"

    _, output_path = convert(tree, tmp_path, spatial_chunk=32, keep_scale_offset=keep_scale_offset)

    root = zarr.open_group(str(output_path), mode="r")
    missing = [group for group in expected_groups if group.lstrip("/") not in root]
    assert missing == []
    for burst in tree.children:
        assert read_array(output_path, f"{burst}/measurements/slc").dtype == np.dtype("complex64")
