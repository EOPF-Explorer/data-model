"""
Tests for S2 multiscale pyramid creation with xy-aligned sharding.
"""

import json
import pathlib
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import GroupSpec
from structlog.testing import capture_logs

from eopf_geozarr.s2_optimization.s2_multiscale import (
    _coarsen_variable,
    calculate_aligned_chunk_size,
    calculate_simple_shard_dimensions,
    create_downsampled_resolution_group,
    create_measurements_encoding,
    create_multiscale_from_datatree,
    inject_missing_bands,
)


@pytest.fixture
def sample_dataset(s2_group_example: pathlib.Path) -> xr.Dataset:
    """Create a sample xarray dataset for testing."""
    with pytest.warns((RuntimeWarning, FutureWarning)):
        return xr.open_datatree(s2_group_example, engine="zarr")[
            "measurements/reflectance/r10m"
        ].to_dataset()


def test_create_downsampled_resolution_group_quality_mask() -> None:
    """Quality-mask downsampling should not crash and should preserve dtype."""
    x = np.arange(8)
    y = np.arange(6)
    quality = xr.DataArray(
        np.random.randint(0, 2, (6, 8), dtype=np.uint8),
        dims=["y", "x"],
        coords={"y": y, "x": x},
        name="quality_clouds",
    )
    ds = xr.Dataset({"quality_clouds": quality})

    out = create_downsampled_resolution_group(ds, factor=2)

    assert "quality_clouds" in out.data_vars
    assert out["quality_clouds"].dtype == np.uint8
    assert out["quality_clouds"].shape == (3, 4)


def test_calculate_simple_shard_dimensions() -> None:
    """Test simplified shard dimensions calculation."""
    # Test 3D data (time, y, x) - shards are multiples of chunks
    data_shape = (5, 1024, 1024)
    chunks = (1, 256, 256)

    shard_dims = calculate_simple_shard_dimensions(data_shape, chunks)

    assert len(shard_dims) == 3
    assert shard_dims[0] == 1  # Time dimension should be 1
    assert shard_dims[1] == 1024  # Y dimension matches exactly (divisible by 256)
    assert shard_dims[2] == 1024  # X dimension matches exactly (divisible by 256)

    # Test 2D data (y, x) with non-divisible dimensions
    data_shape = (1000, 1000)
    chunks = (256, 256)

    shard_dims = calculate_simple_shard_dimensions(data_shape, chunks)

    assert len(shard_dims) == 2
    # Should use largest multiple of chunk_size that fits
    assert shard_dims[0] == 768  # 3 * 256 = 768 (largest multiple that fits in 1000)
    assert shard_dims[1] == 768  # 3 * 256 = 768


@pytest.mark.parametrize("keep_scale_offset", [True, False])
def test_create_measurements_encoding(keep_scale_offset: bool, sample_dataset: xr.Dataset) -> None:
    """Test measurements encoding creation with xy-aligned sharding."""
    encoding = create_measurements_encoding(
        sample_dataset,
        enable_sharding=True,
        spatial_chunk=1024,
        keep_scale_offset=keep_scale_offset,
    )

    # Check that encoding is created for all variables
    for var_name in sample_dataset.data_vars:
        assert var_name in encoding
        var_encoding = encoding[var_name]

        # Check basic encoding structure
        assert "chunks" in var_encoding
        # Zarr v3 uses 'compressors' (plural)
        assert "compressors" in var_encoding or "compressor" in var_encoding

        # Check sharding is included when enabled
        assert "shards" in var_encoding

    # Check coordinate encoding
    for coord_name in sample_dataset.coords:
        if coord_name in encoding:
            # Coordinates may have either compressor or compressors set to None
            assert (
                encoding[coord_name].get("compressor") is None
                or encoding[coord_name].get("compressors") is None
            )
    # Store data and check that we are conditionally applying the scale-offset transformation
    # based on the request passed to the encoding
    stored = sample_dataset.to_zarr({}, encoding=encoding)
    zg = stored.zarr_group
    for var_name in sample_dataset.data_vars:
        if "add_offset" in sample_dataset[var_name].encoding:
            if keep_scale_offset:
                assert zg[var_name].dtype != sample_dataset[var_name].dtype
            else:
                assert zg[var_name].dtype == sample_dataset[var_name].dtype


def test_create_measurements_encoding_time_chunking(sample_dataset: xr.Dataset) -> None:
    """Test that time dimension is chunked to 1 for single file per time."""
    encoding = create_measurements_encoding(
        sample_dataset, enable_sharding=True, spatial_chunk=1024
    )

    for var_name in sample_dataset.data_vars:
        if sample_dataset[var_name].ndim == 3:  # 3D variable with time
            chunks = encoding[var_name]["chunks"]
            assert chunks[0] == 1  # Time dimension should be chunked to 1


def test_calculate_aligned_chunk_size() -> None:
    """Test aligned chunk size calculation."""
    # Test with spatial_chunk that divides evenly
    chunk_size = calculate_aligned_chunk_size(1024, 256)
    assert chunk_size == 256

    # Test with spatial_chunk that doesn't divide evenly
    chunk_size = calculate_aligned_chunk_size(1000, 256)
    # Should return a value that divides evenly into 1000
    assert 1000 % chunk_size == 0


@pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:.*:FutureWarning")
@pytest.mark.filterwarnings("ignore:.*:UserWarning")
def test_create_multiscale_from_datatree(
    s2_group_example: zarr.Group,
    tmp_path: pathlib.Path,
) -> None:
    """Test multiscale creation from DataTree."""
    output_path = str(tmp_path / "output.zarr")
    input_group = zarr.open_group(s2_group_example)
    output_group = zarr.create_group(output_path)
    dt_input = xr.open_datatree(input_group.store, engine="zarr", chunks="auto")

    # Capture log output using structlog's testing context manager
    with capture_logs():
        create_multiscale_from_datatree(
            dt_input,
            output_group=output_group,
            enable_sharding=True,
            spatial_chunk=256,
            keep_scale_offset=False,
        )

    observed_group = zarr.open_group(output_path, use_consolidated=False)

    observed_structure_json = GroupSpec.from_zarr(observed_group).model_dump()

    # Comparing JSON objects is sensitive to the difference between tuples and lists, but we
    # don't care about that here, so we convert all lists to tuples before creating the GroupSpec
    observed_structure = GroupSpec(**tuplify_json(observed_structure_json))
    observed_structure_flat = observed_structure.to_flat()
    expected_structure_path = Path("tests/_test_data/optimized_geozarr_examples/") / (
        s2_group_example.stem + ".json"
    )

    # Uncomment this section to write out the expected structure from the observed structure
    # This is useful when the expected structure needs to be updated
    # expected_structure_path.write_text(
    #    json.dumps(observed_structure_json, indent=2, sort_keys=True)
    # )

    expected_structure_json = tuplify_json(json.loads(expected_structure_path.read_text()))
    expected_structure = GroupSpec(**expected_structure_json)
    expected_structure_flat = expected_structure.to_flat()

    # check that all multiscale levels have the same data type
    # this check is redundant with the later check, but it's expedient to check this here.
    # eventually this check should be spun out into its own test
    _, res_groups = zip(*observed_group["measurements/reflectance"].groups(), strict=False)

    dtype_mismatch: set[object] = set()
    for group_a, group_b in pairwise(res_groups):
        ds_a = xr.open_dataset(group_a.store, engine="zarr", group=group_a.path)
        ds_b = xr.open_dataset(group_b.store, engine="zarr", group=group_b.path)

        for name in ds_a.data_vars:
            dtype_a = ds_a[name].dtype
            if name in ds_b.data_vars:
                dtype_b = ds_b[name].dtype
                if dtype_a != dtype_b:
                    dtype_mismatch.add(
                        (f"{group_a.path}/{name}::{dtype_a}", f"{group_b.path}/{name}::{dtype_b}")
                    )
    assert dtype_mismatch == set()

    o_keys = set(observed_structure_flat.keys())
    e_keys = set(expected_structure_flat.keys())

    # Check that all of the keys are the same
    assert o_keys == e_keys
    # Check that all values are the same
    assert [k for k in o_keys if expected_structure_flat[k] != observed_structure_flat[k]] == []


# ---------------------------------------------------------------------------
# _coarsen_variable
# ---------------------------------------------------------------------------


def test_coarsen_variable_classification() -> None:
    """Classification variables should be downsampled via subsample."""
    data = np.arange(16, dtype="uint8").reshape(4, 4)
    var = xr.DataArray(data, dims=["y", "x"], coords={"y": np.arange(4.0), "x": np.arange(4.0)})
    result = _coarsen_variable("scl", var, factor=2)
    assert result.shape == (2, 2)
    assert result.dtype == np.uint8
    # subsample picks top-left of each 2x2 block
    np.testing.assert_array_equal(result.values, data[::2, ::2])


def test_coarsen_variable_quality_mask() -> None:
    """Quality mask variables should be downsampled via max."""
    data = np.array([[0, 1], [2, 3]], dtype="uint8")
    var = xr.DataArray(data, dims=["y", "x"], coords={"y": np.arange(2.0), "x": np.arange(2.0)})
    result = _coarsen_variable("quality_cirrus", var, factor=2)
    assert result.shape == (1, 1)
    assert result.values.item() == 3


# ---------------------------------------------------------------------------
# inject_missing_bands
# ---------------------------------------------------------------------------


def _make_reflectance_datatree() -> xr.DataTree:
    """Build a minimal DataTree with /measurements/reflectance/r10m and r20m."""
    size_10m = 120  # must be divisible by 2 (→60) and 6 (→20)
    x10 = np.arange(size_10m, dtype="float64")
    y10 = np.arange(size_10m, dtype="float64")

    r10m_ds = xr.Dataset(
        {
            "b02": (["y", "x"], np.ones((size_10m, size_10m), dtype="uint16")),
            "b03": (["y", "x"], np.ones((size_10m, size_10m), dtype="uint16")),
            "b04": (["y", "x"], np.ones((size_10m, size_10m), dtype="uint16")),
            "b08": (["y", "x"], np.full((size_10m, size_10m), 42, dtype="uint16")),
        },
        coords={"x": x10, "y": y10},
    )

    size_20m = size_10m // 2
    x20 = np.arange(size_20m, dtype="float64")
    y20 = np.arange(size_20m, dtype="float64")
    r20m_ds = xr.Dataset(
        {
            "b05": (["y", "x"], np.ones((size_20m, size_20m), dtype="uint16")),
        },
        coords={"x": x20, "y": y20},
    )

    dt = xr.DataTree()
    dt["measurements/reflectance/r10m"] = xr.DataTree(r10m_ds)
    dt["measurements/reflectance/r20m"] = xr.DataTree(r20m_ds)
    return dt


def test_inject_missing_bands_respects_bands_filter() -> None:
    """With bands={"b08"}, only b08 should be injected even when others are eligible."""
    dt = _make_reflectance_datatree()
    r20m_ds = dt["measurements/reflectance/r20m"].to_dataset()

    result = inject_missing_bands(r20m_ds, dt, target_resolution=20, bands={"b08"})

    assert "b08" in result.data_vars
    assert result["b08"].shape == (60, 60)
    assert result["b08"].dtype == np.uint16
    # b02/b03/b04 are also eligible (10m native, missing from r20m) but excluded
    for excluded in ("b02", "b03", "b04"):
        assert excluded not in result.data_vars


def test_inject_missing_bands_skips_existing() -> None:
    """Bands already present in the dataset should not be overwritten."""
    dt = _make_reflectance_datatree()
    r20m_ds = dt["measurements/reflectance/r20m"].to_dataset()

    # Pre-populate b08 with a sentinel value so we can verify it is NOT replaced.
    sentinel = np.full((60, 60), 999, dtype="uint16")
    r20m_ds["b08"] = (["y", "x"], sentinel)

    result = inject_missing_bands(r20m_ds, dt, target_resolution=20, bands={"b08"})

    # b08 was already present — inject_missing_bands must leave it untouched.
    np.testing.assert_array_equal(result["b08"].values, sentinel)


def test_inject_missing_bands_noop_when_no_source() -> None:
    """If the source group is missing from the DataTree, return dataset unchanged."""
    dt = xr.DataTree()
    ds = xr.Dataset({"b05": (["y", "x"], np.ones((60, 60)))})

    result = inject_missing_bands(ds, dt, target_resolution=20)

    assert "b08" not in result.data_vars


def test_inject_missing_bands_default_injects_all() -> None:
    """With bands=None (default), all eligible finer bands should be injected."""
    size_10m = 120
    x10 = np.arange(size_10m, dtype="float64")
    y10 = np.arange(size_10m, dtype="float64")

    r10m_ds = xr.Dataset(
        {
            "b02": (["y", "x"], np.ones((size_10m, size_10m), dtype="uint16")),
            "b03": (["y", "x"], np.ones((size_10m, size_10m), dtype="uint16")),
            "b04": (["y", "x"], np.ones((size_10m, size_10m), dtype="uint16")),
            "b08": (["y", "x"], np.full((size_10m, size_10m), 42, dtype="uint16")),
        },
        coords={"x": x10, "y": y10},
    )

    size_20m = 60
    x20 = np.arange(size_20m, dtype="float64")
    y20 = np.arange(size_20m, dtype="float64")
    r20m_ds = xr.Dataset(
        {
            "b05": (["y", "x"], np.ones((size_20m, size_20m), dtype="uint16")),
            "b06": (["y", "x"], np.ones((size_20m, size_20m), dtype="uint16")),
        },
        coords={"x": x20, "y": y20},
    )

    size_60m = 20
    x60 = np.arange(size_60m, dtype="float64")
    y60 = np.arange(size_60m, dtype="float64")
    r60m_ds = xr.Dataset(
        {
            "b01": (["y", "x"], np.ones((size_60m, size_60m), dtype="uint16")),
        },
        coords={"x": x60, "y": y60},
    )

    dt = xr.DataTree()
    dt["measurements/reflectance/r10m"] = xr.DataTree(r10m_ds)
    dt["measurements/reflectance/r20m"] = xr.DataTree(r20m_ds)
    dt["measurements/reflectance/r60m"] = xr.DataTree(r60m_ds)

    result = inject_missing_bands(r60m_ds, dt, target_resolution=60)

    # All 10m bands (b02, b03, b04, b08) and 20m bands (b05, b06) should be injected
    for band in ("b02", "b03", "b04", "b08", "b05", "b06"):
        assert band in result.data_vars, f"{band} missing from r60m"
        assert result[band].shape == (size_60m, size_60m)
