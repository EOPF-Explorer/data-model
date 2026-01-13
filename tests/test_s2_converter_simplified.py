"""
Unit tests for simplified S2 converter implementation.

Tests the new simplified approach that uses only xarray and proper metadata consolidation.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

from eopf_geozarr.s2_optimization.s2_converter import (
    convert_s2_optimized,
    simple_root_consolidation,
)
from eopf_geozarr.s2_optimization.s2_multiscale import auto_chunks


@pytest.fixture
def mock_s2_dataset() -> xr.DataTree:
    """Create a mock S2 dataset for testing."""
    # Create test data arrays
    coords = {
        "x": (["x"], np.linspace(0, 1000, 100)),
        "y": (["y"], np.linspace(0, 1000, 100)),
        "time": (["time"], [np.datetime64("2023-01-01")]),
    }

    # Create test variables
    data_vars = {
        "b02": (["time", "y", "x"], np.random.rand(1, 100, 100)),
        "b03": (["time", "y", "x"], np.random.rand(1, 100, 100)),
        "b04": (["time", "y", "x"], np.random.rand(1, 100, 100)),
    }

    ds = xr.Dataset(data_vars, coords=coords)

    # Add rioxarray CRS
    ds = ds.rio.write_crs("EPSG:32632")

    # Create datatree
    dt = xr.DataTree(ds)
    dt.attrs = {"stac_discovery": {"properties": {"mission": "sentinel-2"}}}

    return dt


@pytest.mark.filterwarnings("ignore:Object at .*:zarr.errors.ZarrUserWarning")
@pytest.mark.filterwarnings(
    "ignore:Consolidated metadata is currently .*:zarr.errors.ZarrUserWarning"
)
def test_simple_root_consolidation_success(tmp_path: Path) -> None:
    """
    Test that simple_root_consolidation produces consolidated metadata at the root, and for the
    measurements/reflectance group, but not for other groups.
    """

    datasets = {
        "/measurements/reflectance/r10m": xr.Dataset(),
        "/quality/atmosphere": xr.Dataset(),
    }

    [
        v.to_zarr(
            str(tmp_path / f"test.zarr{k}/"),
            mode="a",
            zarr_format=3,
            consolidated=False,
        )
        for k, v in datasets.items()
    ]

    simple_root_consolidation(str(tmp_path / "test.zarr"), datasets=datasets)

    root_z_meta = json.loads((tmp_path / "test.zarr/zarr.json").read_text())
    reflectance_zmeta = json.loads(
        (tmp_path / "test.zarr/measurements/reflectance/zarr.json").read_text()
    )
    atmos_zmeta = json.loads((tmp_path / "test.zarr/quality/zarr.json").read_text())

    assert "consolidated_metadata" in root_z_meta
    assert isinstance(root_z_meta["consolidated_metadata"], dict)
    assert "consolidated_metadata" in reflectance_zmeta
    assert isinstance(reflectance_zmeta["consolidated_metadata"], dict)
    if "consolidated_metadata" in atmos_zmeta:
        assert atmos_zmeta["consolidated_metadata"] is None


@pytest.mark.filterwarnings("ignore:.*Expected 'MISSING' sentinel:UserWarning")
@pytest.mark.filterwarnings("ignore:.*:zarr.errors.UnstableSpecificationWarning")
@pytest.mark.filterwarnings("ignore:.*:xarray.coding.common.SerializationWarning")
@pytest.mark.filterwarnings(
    "ignore:Failed to open Zarr store with consolidated metadata:RuntimeWarning"
)
@pytest.mark.filterwarnings("ignore:.*:zarr.errors.ZarrUserWarning")
@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
class TestConvenienceFunction:
    """Test the convenience function."""

    @pytest.mark.parametrize("compression_level", [3])
    @pytest.mark.parametrize("spatial_chunk", [256])
    @pytest.mark.parametrize("enable_sharding", [False, True])
    def test_convert_s2_optimized_convenience_function(
        self,
        s2_group_example: Path,
        tmp_path: Path,
        spatial_chunk: int,
        compression_level: int,
        enable_sharding: bool,
    ) -> None:
        """Test the convenience function with real S2 data."""
        # Open the S2 example as a DataTree
        dt_input = xr.open_datatree(
            s2_group_example,
            engine="zarr",
            consolidated=False,
            decode_timedelta=True,
        )
        output_path = str(tmp_path / "test_output.zarr")

        # Run the conversion
        result = convert_s2_optimized(
            dt_input,
            output_path=output_path,
            enable_sharding=enable_sharding,
            spatial_chunk=spatial_chunk,
            compression_level=compression_level,
            max_retries=5,
            validate_output=False,
            keep_scale_offset=False,
        )

        # Verify the output was created
        assert Path(output_path).exists()

        # Verify the result is a DataTree
        assert isinstance(result, xr.DataTree)

        # Verify basic structure - output should have multiscale groups
        output_dt = xr.open_datatree(output_path, engine="zarr")
        assert len(output_dt.groups) > 0

        # Open the Zarr store to verify metadata
        # We know the S2 data has measurements/reflectance group structure
        output_zarr = zarr.open_group(output_path, mode="r")

        reflectance_group = output_zarr["measurements/reflectance"]
        reencoded_array = reflectance_group["r10m/b03"]
        downsampled_array = reflectance_group["r720m/b03"]

        # For re-encoded arrays, chunks should match output of auto_chunks
        assert reencoded_array.chunks == auto_chunks(reencoded_array.shape, spatial_chunk)

        # For downsampled arrays, chunks should be min(spatial_chunk, array_shape)
        # because the array might be smaller than the chunk size
        expected_downsampled_chunks = (
            min(spatial_chunk, downsampled_array.shape[0]),
            min(spatial_chunk, downsampled_array.shape[1]),
        )
        assert downsampled_array.chunks == expected_downsampled_chunks

        # Sharding should only be present on arrays large enough for the chunk size
        # Reencoded arrays are typically large (10980x10980), so should have sharding
        if enable_sharding:
            assert reencoded_array.shards is not None, "Large reencoded array should have sharding"
        else:
            assert reencoded_array.shards is None, (
                "Reencoded array should not have sharding when disabled"
            )

        # Downsampled arrays (r720m) might be too small for sharding
        # Only check for sharding if the array is large enough (shape >= spatial_chunk in both dims)
        if (
            enable_sharding
            and downsampled_array.shape[0] >= spatial_chunk
            and downsampled_array.shape[1] >= spatial_chunk
        ):
            assert downsampled_array.shards is not None, (
                f"Downsampled array with shape {downsampled_array.shape} should have sharding"
            )
        elif (
            not enable_sharding
            or downsampled_array.shape[0] < spatial_chunk
            or downsampled_array.shape[1] < spatial_chunk
        ):
            assert downsampled_array.shards is None, (
                f"Downsampled array with shape {downsampled_array.shape} should not have sharding "
                f"(enable_sharding={enable_sharding}, spatial_chunk={spatial_chunk})"
            )

        assert reencoded_array.compressors[0].to_dict()["name"] == "blosc"
        assert downsampled_array.compressors[0].to_dict()["name"] == "blosc"

        assert (
            reencoded_array.compressors[0].to_dict()["configuration"]["clevel"] == compression_level
        )
        assert (
            downsampled_array.compressors[0].to_dict()["configuration"]["clevel"]
            == compression_level
        )


if __name__ == "__main__":
    pytest.main([__file__])
