"""Tests for the zarrio module"""

from __future__ import annotations

import numcodecs
import pytest
import zarr

from eopf_geozarr.zarrio import (
    ChunkEncodingSpec,
    convert_compression,
    reencode_array,
    reencode_group,
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


@pytest.mark.parametrize("dimension_names", [None, ("a",)])
def test_reencode_array_dimension_names(
    dimension_names: None | tuple[str | None, ...],
) -> None:
    """
    Test that dimension names are handled correctly when re-encoding an array
    """
    array_a = zarr.create_array({}, shape=(1,), dtype="uint8", zarr_format=2, compressors=None)
    assert (
        reencode_array(array_a, dimension_names=dimension_names).dimension_names == dimension_names
    )


def test_reencode_array_attributes_default() -> None:
    """
    Test that original attributes are preserved when no custom attributes provided
    """
    array_a = zarr.create_array({}, shape=(1,), dtype="uint8", zarr_format=2, compressors=None)
    array_a.attrs["original"] = "value"

    meta = reencode_array(array_a)
    assert meta.attributes == {"original": "value"}


def test_reencode_array_attributes_custom() -> None:
    """
    Test that custom attributes override original attributes
    """
    array_a = zarr.create_array({}, shape=(1,), dtype="uint8", zarr_format=2, compressors=None)
    array_a.attrs["original"] = "value"

    custom_attrs = {"custom": "new_value", "foo": "bar"}
    meta = reencode_array(array_a, attributes=custom_attrs)
    assert meta.attributes == custom_attrs


def test_reencode_array_chunking_write_only() -> None:
    """
    Test that chunking with only write_chunks updates chunk shape
    """
    array_a = zarr.create_array(
        {}, shape=(100,), dtype="uint8", zarr_format=2, chunks=(10,), compressors=None
    )

    chunking: ChunkEncodingSpec = {"write_chunks": (25,)}
    meta = reencode_array(array_a, chunking=chunking)

    assert meta.chunk_grid.chunk_shape == (25,)


def test_reencode_array_chunking_no_sharding_when_read_chunks_missing() -> None:
    """
    Test that sharding codec is not used when read_chunks not specified
    """
    array_a = zarr.create_array(
        {}, shape=(100,), dtype="uint8", zarr_format=2, chunks=(10,), compressors=None
    )

    chunking: ChunkEncodingSpec = {"write_chunks": (25,)}
    meta = reencode_array(array_a, chunking=chunking)

    codec_names = [codec["name"] for codec in meta.codecs if isinstance(codec, dict)]
    assert "sharding-indexed" not in codec_names


def test_reencode_array_chunking_with_sharding_codec_present() -> None:
    """
    Test that sharding codec is created when read_chunks specified
    """
    array_a = zarr.create_array(
        {}, shape=(100,), dtype="uint8", zarr_format=2, chunks=(10,), compressors=None
    )

    chunking: ChunkEncodingSpec = {"write_chunks": (25,), "read_chunks": (5,)}
    meta = reencode_array(array_a, chunking=chunking)

    assert len(meta.codecs) == 1


def test_reencode_array_chunking_with_sharding_grid_shape() -> None:
    """
    Test that chunk grid uses write_chunks when sharding is enabled
    """
    array_a = zarr.create_array(
        {}, shape=(100,), dtype="uint8", zarr_format=2, chunks=(10,), compressors=None
    )

    chunking: ChunkEncodingSpec = {"write_chunks": (25,), "read_chunks": (5,)}
    meta = reencode_array(array_a, chunking=chunking)

    assert meta.chunk_grid.chunk_shape == (25,)


def test_reencode_array_fill_value_none() -> None:
    """
    Test that None fill_value is converted to dtype default
    """
    array_a = zarr.create_array(
        {}, shape=(10,), dtype="uint8", zarr_format=2, compressors=None, fill_value=None
    )

    meta = reencode_array(array_a)
    assert meta.fill_value == 0


def test_reencode_array_fill_value_custom() -> None:
    """
    Test that custom fill_value is preserved
    """
    array_a = zarr.create_array(
        {}, shape=(10,), dtype="uint8", zarr_format=2, compressors=None, fill_value=255
    )

    meta = reencode_array(array_a)
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
    group_v2.create_array("data", shape=(100,), chunks=(10,), dtype="float32")

    def custom_chunker(array: zarr.Array) -> ChunkEncodingSpec:
        return {"write_chunks": (25,)}

    group_v3 = reencode_group(group_v2, {}, "", chunk_reencoder=custom_chunker)

    assert group_v3["data"].metadata.chunk_grid.chunk_shape == (25,)
