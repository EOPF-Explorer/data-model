"""Tests for utils."""

import numpy as np
import xarray as xr

from eopf_geozarr.conversion.utils import build_dataarray_encoding


def test_sanitize_dataset_attributes() -> None:
    """ """
    small_da = xr.DataArray(
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), dims=("x", "y"), coords={"x": [10, 20]}
    )

    # same chunk shape as array
    small_da_noshard = small_da.chunk({"x": 2, "y": 4})
    result = build_dataarray_encoding(var=small_da_noshard, enable_sharding=True)
    assert "shards" in result
    assert result["shards"] is None

    # chunked larger than array
    small_da_irregular_noshard = small_da.chunk({"x": 1000, "y": 1000})
    result = build_dataarray_encoding(var=small_da_irregular_noshard, enable_sharding=True)
    assert "shards" in result
    assert result["shards"] is None

    # sharded along one dimension
    small_da_shard = small_da.chunk({"x": 2, "y": 2})
    result = build_dataarray_encoding(var=small_da_shard, enable_sharding=True)
    assert "shards" in result
    assert result["shards"] == (2, 4)

    # real appliation 1830 array
    large_da = xr.DataArray(np.random.randn(1830, 1830), dims=("x", "y"))

    large_da_shard = large_da.chunk({"x": 1024, "y": 1024})
    result = build_dataarray_encoding(var=large_da_shard, enable_sharding=True)
    assert "shards" in result
    assert result["shards"] == (2048, 2048)
