"""Tests for the zarrio module"""

from __future__ import annotations

import numcodecs
import pytest
import zarr
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

from eopf_geozarr.zarrio import (
    convert_compression,
    default_array_reencoder,
    reencode_group,
    replace_json_invalid_floats,
)


def test_convert_compression_blosc() -> None:
    """
    Test that convert_compression properly maps zarr
    v2 blosc to zarr v3 blosc
    """
    source = {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2, "blocksize": 0}

    expected = (
        {
            "name": "blosc",
            "configuration": {
                "cname": "zstd",
                "clevel": 3,
                "shuffle": "bitshuffle",
                "blocksize": 0,
            },
        },
    )

    observed = convert_compression(source)
    assert observed == expected


def test_convert_compression_none() -> None:
    """
    Test that convert_compression maps None to an empty tuple
    """
    source = None
    expected = ()
    observed = convert_compression(source)
    assert observed == expected


def test_convert_compression_fails() -> None:
    """
    Test that convert compression will fail on out of band input
    """
    source = {"id": "gzip"}
    with pytest.raises(ValueError, match=r"Only blosc -> blosc or None -> ()"):
        convert_compression(source)


def test_reencode_array_fill_value_none() -> None:
    """
    Test that None fill_value is converted to dtype default
    """
    array_a = zarr.create_array(
        {}, shape=(10,), dtype="uint8", zarr_format=2, compressors=None, fill_value=None
    )

    meta = default_array_reencoder("test_array", array_a.metadata)
    assert meta.fill_value == 0


def test_reencode_array_fill_value_custom() -> None:
    """
    Test that custom fill_value is preserved
    """
    array_a = zarr.create_array(
        {}, shape=(10,), dtype="uint8", zarr_format=2, compressors=None, fill_value=255
    )

    meta = default_array_reencoder("test_array", array_a.metadata)
    assert meta.fill_value == 255


def test_reencode_array_blosc_compression_converts() -> None:
    """
    Test that blosc compression is properly converted from v2 to v3
    """
    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

    # Verify the conversion function works
    converted = convert_compression(compressor)
    assert len(converted) == 1
    assert converted[0]["name"] == "blosc"
    assert converted[0]["configuration"]["cname"] == "zstd"
    assert converted[0]["configuration"]["clevel"] == 5
    assert converted[0]["configuration"]["shuffle"] == "bitshuffle"


def test_reencode_group_basic() -> None:
    """
    Test that reencode_group creates a zarr v3 group from a v2 group
    """
    group_v2 = zarr.create_group({}, zarr_format=2)
    group_v3 = reencode_group(group_v2, {}, "")

    assert group_v3.metadata.zarr_format == 3


def test_reencode_group_preserves_attributes() -> None:
    """
    Test that group attributes are preserved during re-encoding
    """
    group_v2 = zarr.create_group({}, zarr_format=2)
    group_v2.attrs["foo"] = "bar"
    group_v2.attrs["number"] = 42

    group_v3 = reencode_group(group_v2, {}, "")

    assert group_v3.attrs["foo"] == "bar"
    assert group_v3.attrs["number"] == 42


def test_reencode_group_with_array() -> None:
    """
    Test that arrays within a group are re-encoded
    """
    group_v2 = zarr.create_group({}, zarr_format=2)
    group_v2.create_array("data", shape=(10,), dtype="float32")

    group_v3 = reencode_group(group_v2, {}, "")

    assert "data" in group_v3
    assert group_v3["data"].metadata.zarr_format == 3


def test_reencode_group_array_data_preserved() -> None:
    """
    Test that array data is preserved during re-encoding
    """
    group_v2 = zarr.create_group({}, zarr_format=2)
    array_v2 = group_v2.create_array("data", shape=(5,), dtype="int32")
    array_v2[:] = [1, 2, 3, 4, 5]

    group_v3 = reencode_group(group_v2, {}, "")

    assert (group_v3["data"][:] == [1, 2, 3, 4, 5]).all()


def test_reencode_group_array_attributes_preserved() -> None:
    """
    Test that array attributes are preserved during re-encoding
    """
    group_v2 = zarr.create_group({}, zarr_format=2)
    array_v2 = group_v2.create_array("data", shape=(5,), dtype="int32")
    array_v2.attrs["units"] = "meters"

    group_v3 = reencode_group(group_v2, {}, "")

    assert group_v3["data"].attrs["units"] == "meters"


def test_reencode_group_with_nested_group() -> None:
    """
    Test that nested groups are re-encoded
    """
    group_v2 = zarr.create_group({}, zarr_format=2)
    group_v2.create_group("subgroup")

    group_v3 = reencode_group(group_v2, {}, "")

    assert "subgroup" in group_v3
    assert isinstance(group_v3["subgroup"], zarr.Group)


def test_reencode_group_nested_group_attributes() -> None:
    """
    Test that nested group attributes are preserved
    """
    group_v2 = zarr.create_group({}, zarr_format=2)
    sub = group_v2.create_group("subgroup")
    sub.attrs["nested"] = "value"

    group_v3 = reencode_group(group_v2, {}, "")

    assert group_v3["subgroup"].attrs["nested"] == "value"


def test_reencode_group_with_array_dimensions() -> None:
    """
    Test that _ARRAY_DIMENSIONS attribute is converted to dimension_names
    """
    group_v2 = zarr.create_group({}, zarr_format=2)
    array_v2 = group_v2.create_array("data", shape=(10, 20), dtype="float32")
    array_v2.attrs["_ARRAY_DIMENSIONS"] = ("x", "y")

    group_v3 = reencode_group(group_v2, {}, "")

    assert group_v3["data"].metadata.dimension_names == ("x", "y")


def test_reencode_group_removes_array_dimensions_from_attrs() -> None:
    """
    Test that _ARRAY_DIMENSIONS is removed from attributes after conversion
    """
    group_v2 = zarr.create_group({}, zarr_format=2)
    array_v2 = group_v2.create_array("data", shape=(10,), dtype="float32")
    array_v2.attrs["_ARRAY_DIMENSIONS"] = ("x",)
    array_v2.attrs["other"] = "value"

    group_v3 = reencode_group(group_v2, {}, "")

    assert "_ARRAY_DIMENSIONS" not in group_v3["data"].attrs
    assert group_v3["data"].attrs["other"] == "value"


def test_reencode_group_deep_hierarchy() -> None:
    """
    Test that deeply nested hierarchies are fully re-encoded
    """
    group_v2 = zarr.create_group({}, zarr_format=2)
    level1 = group_v2.create_group("level1")
    level2 = level1.create_group("level2")
    level2.create_array("deep_array", shape=(5,), dtype="int32")

    group_v3 = reencode_group(group_v2, {}, "")

    assert "level1/level2/deep_array" in group_v3
    assert group_v3["level1/level2/deep_array"].metadata.zarr_format == 3


def test_reencode_group_with_chunk_reencoder() -> None:
    """
    Test that custom chunk_reencoder function is applied
    """
    group_v2 = zarr.create_group({}, zarr_format=2)
    old_node = group_v2.create_array(
        "data", shape=(100,), chunks=(10,), dtype="float32", attributes={"foo": 10}
    )
    new_chunks = (25,)

    def custom_array_encoder(key: str, metadata: ArrayV2Metadata) -> ArrayV3Metadata:
        return ArrayV3Metadata(
            shape=metadata.shape,
            data_type=metadata.dtype,
            fill_value=metadata.fill_value,
            chunk_grid={"name": "regular", "configuration": {"chunk_shape": new_chunks}},
            chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
            codecs=({"name": "bytes"},),
            dimension_names=None,
            attributes=metadata.attributes,
        )

    group_v3 = reencode_group(group_v2, {}, "", array_reencoder=custom_array_encoder)

    new_node = group_v3["data"]
    assert new_node.metadata.chunk_grid.chunk_shape == new_chunks
    assert new_node.attrs.asdict() == old_node.attrs.asdict()


def test_replace_json_invalid_floats() -> None:
    data: dict[str, object] = {
        "nan": float("nan"),
        "nested_nan": {"nan": float("nan")},
        "nan_in_list": [float("nan")],
        "inf": float("inf"),
        "-inf": float("-inf"),
    }
    expected = {
        "nan": "NaN",
        "nested_nan": {"nan": "NaN"},
        "nan_in_list": ["NaN"],
        "inf": "Infinity",
        "-inf": "-Infinity",
    }
    observed = replace_json_invalid_floats(data)
    assert observed == expected


if __name__ == "__main__":
    pytest.main([__file__])
