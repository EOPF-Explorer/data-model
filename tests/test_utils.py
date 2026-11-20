"""Tests for the Zarr encoding helpers in `eopf_geozarr.conversion.utils`."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.conversion.utils import build_dataarray_encoding, create_uniform_encoding


def _chunked(
    shape: tuple[int, ...], chunks: tuple[int, ...], dims: tuple[str, ...] | None = None
) -> xr.DataArray:
    dims = dims or tuple(f"d{i}" for i in range(len(shape)))
    da = xr.DataArray(np.zeros(shape, dtype="float32"), dims=dims)
    return da.chunk(dict(zip(dims, chunks, strict=True)))


@pytest.mark.parametrize(
    ("shape", "chunks", "expected_chunks"),
    [
        # Chunk shape equals the array shape: a single chunk.
        pytest.param((2, 4), (2, 4), (2, 4), id="2d-exact-fit"),
        # Requested chunk is larger than the array: dask clips it to the shape.
        pytest.param((2, 4), (1000, 1000), (2, 4), id="2d-chunk-larger-than-array"),
        # Small metadata-like grid, typical of conditions/geometry arrays.
        pytest.param((23, 23), (1024, 1024), (23, 23), id="2d-angle-grid"),
        pytest.param((10,), (10,), (10,), id="1d"),
        pytest.param((3, 5, 7), (3, 5, 7), (3, 5, 7), id="3d"),
        # Length-1 dimensions must not trigger sharding.
        pytest.param((1, 1), (1, 1), (1, 1), id="single-element"),
    ],
)
def test_single_chunk_array_is_not_sharded(
    shape: tuple[int, ...], chunks: tuple[int, ...], expected_chunks: tuple[int, ...]
) -> None:
    """An array held in one chunk gets no shard, even with sharding enabled."""
    result = build_dataarray_encoding(_chunked(shape, chunks), enable_sharding=True)

    assert "shards" in result
    assert result["shards"] is None
    # The chunk encoding and compressor are still set for the unsharded array.
    assert result.get("chunks") == expected_chunks
    compressors = result.get("compressors")
    assert compressors is not None
    assert len(compressors) == 1


@pytest.mark.parametrize(
    ("shape", "chunks", "expected_shards"),
    [
        # Split along one dimension only: still sharded, over the whole array.
        pytest.param((2, 4), (2, 2), (2, 4), id="split-along-last-dim"),
        pytest.param((4, 2), (2, 2), (4, 2), id="split-along-first-dim"),
        # Array one element larger than the chunk: smallest case that shards.
        pytest.param((1025, 10), (1024, 10), (2048, 10), id="one-element-over"),
        # Real S2 10 m tile: 1830 is rounded up to a whole number of 1024-chunks.
        pytest.param((1830, 1830), (1024, 1024), (2048, 2048), id="s2-1830"),
        # Shape is an exact multiple of the chunk: no padding needed.
        pytest.param((2048, 2048), (1024, 1024), (2048, 2048), id="exact-multiple"),
        pytest.param((100,), (30,), (120,), id="1d"),
        pytest.param((3, 100, 7), (3, 30, 7), (3, 120, 7), id="3d-split-along-middle-dim"),
    ],
)
def test_multi_chunk_array_is_sharded_over_whole_array(
    shape: tuple[int, ...], chunks: tuple[int, ...], expected_shards: tuple[int, ...]
) -> None:
    """An array spanning several chunks still gets one shard covering the whole array."""
    result = build_dataarray_encoding(_chunked(shape, chunks), enable_sharding=True)

    shards = result.get("shards")
    chunks_ = result.get("chunks")
    assert shards == expected_shards
    assert shards is not None
    assert chunks_ is not None
    # Each shard holds a whole number of chunks along every dimension.
    assert all(s % c == 0 for s, c in zip(shards, chunks_, strict=True))


@pytest.mark.parametrize(
    ("shape", "chunks"),
    [
        pytest.param((2, 4), (2, 4), id="single-chunk"),
        pytest.param((1830, 1830), (1024, 1024), id="multi-chunk"),
    ],
)
def test_sharding_disabled_never_shards(shape: tuple[int, ...], chunks: tuple[int, ...]) -> None:
    result = build_dataarray_encoding(_chunked(shape, chunks), enable_sharding=False)

    assert "shards" in result
    assert result["shards"] is None


def test_unchunked_array_raises() -> None:
    """The fit check relies on dask chunks, which must be set beforehand."""
    with pytest.raises(ValueError, match="not chunked"):
        build_dataarray_encoding(xr.DataArray(np.zeros((2, 4)), dims=("y", "x")))


def test_shard_along_band_dimension_for_large_array() -> None:
    """With `shard_along_smallest_dimension`, each band gets its own whole-array shard."""
    var = _chunked((2, 300, 500), (2, 128, 128), dims=("polarization", "y", "x"))

    result = build_dataarray_encoding(
        var, enable_sharding=True, shard_along_smallest_dimension=True
    )

    assert result.get("chunks") == (1, 128, 128)
    assert result.get("shards") == (1, 384, 512)


def test_create_uniform_encoding_mixes_sharded_and_unsharded_variables() -> None:
    """In one dataset, only the variables spanning several chunks are sharded."""
    ds = xr.Dataset(
        {
            "large": (("y", "x"), np.zeros((1830, 1830), dtype="float32")),
            "small": (("ay", "ax"), np.zeros((23, 23), dtype="float32")),
        }
    ).chunk({"y": 1024, "x": 1024, "ay": 23, "ax": 23})

    encoding = create_uniform_encoding(ds, enable_sharding=True)

    small = encoding["small"]
    assert encoding["large"].get("shards") == (2048, 2048)
    assert "shards" in small
    assert small["shards"] is None
