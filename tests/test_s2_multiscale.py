"""
Tests for S2 multiscale pyramid creation with xy-aligned sharding.
"""

import json
import pathlib
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
import zarr
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import GroupSpec
from structlog.testing import capture_logs
from zarr.codecs import BloscCodec, CastValue, ScaleOffset
from zarr.core.dtype import Int16

from eopf_geozarr.s2_optimization.s2_multiscale import (
    _coarsen_variable,
    add_multiscales_metadata_to_parent,
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


def test_add_multiscales_metadata_prefers_coordinate_transform_for_inconsistent_rio(
    tmp_path: pathlib.Path,
) -> None:
    """Derived levels should not reuse a stale rio transform."""

    def _dataset(resolution: int, size: int, x0: float, y0: float) -> xr.Dataset:
        x = x0 + np.arange(size, dtype="float64") * resolution
        y = y0 - np.arange(size, dtype="float64") * resolution
        ds = xr.Dataset(
            {"band": (["y", "x"], np.ones((size, size), dtype=np.uint16))},
            coords={"x": x, "y": y},
        )
        return ds.rio.write_crs("EPSG:32631")

    r10m = _dataset(10, 12, 600000.0, 4900020.0)
    r120m = _dataset(120, 3, 600030.0, 4899990.0)

    parent_group = zarr.create_group(tmp_path / "multiscales.zarr")

    def stale_transform() -> tuple[float, float, float, float, float, float]:
        return (60.0, 0.0, 600030.0, 0.0, -60.0, 4899990.0)

    with patch.object(r120m.rio, "transform", stale_transform):
        add_multiscales_metadata_to_parent(
            parent_group,
            {"r10m": r10m, "r120m": r120m},
        )

    layout = parent_group.attrs["multiscales"]["layout"]
    derived_level = next(level for level in layout if level["asset"] == "r120m")
    assert tuple(derived_level["spatial:transform"]) == (
        120.0,
        0.0,
        600030.0,
        0.0,
        -120.0,
        4899990.0,
    )


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


def test_create_measurements_encoding_experimental_scale_offset_codec() -> None:
    """Test that experimental_scale_offset_codec adds ScaleOffset + CastValue filters."""
    # Create a dataset with CF-style scale-offset encoding, as xarray would
    # produce when reading a CF-encoded zarr/netCDF variable.
    data = xr.DataArray(
        np.arange(0, 100, dtype="float64").reshape(10, 10),
        dims=["y", "x"],
    )
    data.encoding = {
        "scale_factor": 0.01,
        "add_offset": 273.15,
        "dtype": np.dtype("int16"),
    }
    ds = xr.Dataset({"temperature": data})

    encoding = create_measurements_encoding(
        ds,
        enable_sharding=True,
        spatial_chunk=256,
        keep_scale_offset=False,
        experimental_scale_offset_codec=True,
    )

    int16_min = int(np.iinfo(np.int16).min)
    assert encoding == {
        "temperature": {
            "chunks": (10, 10),
            "compressors": (BloscCodec(cname="zstd", clevel=3, shuffle="shuffle", blocksize=0),),
            "shards": (10, 10),
            "filters": (
                ScaleOffset(offset=273.15, scale=100.0),
                CastValue(
                    data_type=Int16(endianness="little"),
                    rounding="nearest-even",
                    out_of_range=None,
                    scalar_map={
                        "encode": [("NaN", int16_min)],
                        "decode": [(int16_min, "NaN")],
                    },
                ),
            ),
            "fill_value": "NaN",
        }
    }


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
    """Snapshot test: a single canonical parametrization (keep_scale_offset=False,
    experimental_scale_offset_codec=False) compared against a stored fixture.

    Behavior under other parametrizations is exercised by
    `test_create_multiscale_from_datatree_behavior` below, which uses a small
    in-memory dataset with explicit, easily-verified expectations.
    """
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
            experimental_scale_offset_codec=False,
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


def _make_minimal_s2_datatree() -> xr.DataTree:
    """Build a tiny CF-encoded reflectance DataTree for behavior tests.

    Three levels (r10m, r20m, r60m), one band each, with `scale_factor`,
    `add_offset`, and `dtype` set in the variable encoding so that the
    measurement-encoding logic recognises CF-encoded variables.
    """
    rng = np.random.default_rng(0)

    def _band(size: int) -> xr.DataArray:
        # Decoded floats in a small range so the codec round-trips cleanly.
        data = rng.uniform(0.0, 1.0, size=(size, size)).astype("float64")
        da = xr.DataArray(
            data,
            dims=["y", "x"],
            coords={
                "x": np.arange(size, dtype="float64"),
                "y": np.arange(size, dtype="float64"),
            },
        )
        # CF encoding: stored as uint16, decoded via scale_factor/add_offset.
        da.encoding = {
            "scale_factor": 0.0001,
            "add_offset": 0.0,
            "dtype": np.dtype("uint16"),
        }
        return da

    r10m = xr.Dataset({"b02": _band(120)})
    r20m = xr.Dataset({"b05": _band(60)})
    r60m = xr.Dataset({"b01": _band(20)})

    dt = xr.DataTree()
    dt["measurements/reflectance/r10m"] = xr.DataTree(r10m)
    dt["measurements/reflectance/r20m"] = xr.DataTree(r20m)
    dt["measurements/reflectance/r60m"] = xr.DataTree(r60m)
    return dt


# Spatial chunk small enough that no padding is needed for the 20x20 r60m band.
_BEHAVIOR_SPATIAL_CHUNK = 16

# Original groups in the synthetic datatree — those that have CF encoding on input.
_ORIGINAL_GROUPS = {
    "measurements/reflectance/r10m": "b02",
    "measurements/reflectance/r20m": "b05",
    "measurements/reflectance/r60m": "b01",
}

# Downsampled groups added by `create_multiscale_from_datatree`, derived from the
# r60m level by successive coarsening. `_coarsen_variable` preserves the source
# variable's CF encoding, so these levels see the same encoding flags as the
# original groups.
_DOWNSAMPLED_GROUPS = (
    "measurements/reflectance/r120m",
    "measurements/reflectance/r360m",
    "measurements/reflectance/r720m",
)


@pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:.*:FutureWarning")
@pytest.mark.filterwarnings("ignore:.*:UserWarning")
@pytest.mark.parametrize("experimental_scale_offset_codec", [False, True])
@pytest.mark.parametrize("keep_scale_offset", [False, True])
def test_create_multiscale_from_datatree_behavior(
    keep_scale_offset: bool,
    experimental_scale_offset_codec: bool,
    tmp_path: pathlib.Path,
) -> None:
    """Verify per-parameter behavior of multiscale generation on a tiny dataset.

    This test asserts the *meaning* of the two flags directly, instead of
    snapshotting an entire pyramid. Both flags apply uniformly to every level
    of the pyramid (original groups + downsampled groups). The expectations
    per (keep_scale_offset, experimental_scale_offset_codec) combo:

    * `keep_scale_offset=True`: every level preserves the packed integer dtype
      on disk.
    * `keep_scale_offset=False, experimental_scale_offset_codec=False`:
      every level is written as decoded floats, CF attributes are stripped.
    * `keep_scale_offset=False, experimental_scale_offset_codec=True`: every
      level carries a `[scale_offset, cast_value]` filter pipeline; logical
      dtype is float32 (decoded), chunks store packed integers.
    """
    dt_input = _make_minimal_s2_datatree()
    output_path = str(tmp_path / "output.zarr")
    output_group = zarr.create_group(output_path)

    with capture_logs():
        create_multiscale_from_datatree(
            dt_input,
            output_group=output_group,
            enable_sharding=False,
            spatial_chunk=_BEHAVIOR_SPATIAL_CHUNK,
            keep_scale_offset=keep_scale_offset,
            experimental_scale_offset_codec=experimental_scale_offset_codec,
        )

    # ------------------------------------------------------------------
    # Original groups (those that had CF encoding on input)
    # ------------------------------------------------------------------
    for group_path, var_name in _ORIGINAL_GROUPS.items():
        arr = zarr.open_array(output_path, path=f"{group_path}/{var_name}")
        codec_names = [type(c).__name__ for c in arr.metadata.codecs]

        if keep_scale_offset:
            # Source CF encoding is preserved as-is: packed integer on disk,
            # CF attributes round-tripped via xarray.
            assert arr.dtype == np.uint16, (
                f"{group_path}/{var_name}: expected uint16 on disk, got {arr.dtype}"
            )
        elif experimental_scale_offset_codec:
            # CF encoding is pushed into a codec pipeline. The logical dtype
            # is the cast-to-float32 array dtype; the codec stores uint16.
            assert arr.dtype == np.float32, (
                f"{group_path}/{var_name}: expected float32 logical dtype, got {arr.dtype}"
            )
            assert "ScaleOffset" in codec_names, f"{group_path}/{var_name}: codecs={codec_names}"
            assert "CastValue" in codec_names, f"{group_path}/{var_name}: codecs={codec_names}"
        else:
            # CF encoding is stripped; data is written as decoded floats.
            assert arr.dtype == np.float32, (
                f"{group_path}/{var_name}: expected float32, got {arr.dtype}"
            )
            assert "ScaleOffset" not in codec_names
            assert "CastValue" not in codec_names

    # ------------------------------------------------------------------
    # Downsampled groups: `_coarsen_variable` preserves the source variable's
    # CF encoding, so the same per-combo expectations as the original groups
    # apply here too — every level in the pyramid should be encoded
    # consistently.
    # ------------------------------------------------------------------
    parent_group = zarr.open_group(output_path, path="measurements/reflectance")
    for group_path in _DOWNSAMPLED_GROUPS:
        # All three downsampled levels should exist for the minimal datatree
        # (10/20/60 → 120/360/720).
        sub = group_path.removeprefix("measurements/reflectance/")
        assert sub in dict(parent_group.groups()), f"missing downsampled group {group_path}"
        ds = xr.open_dataset(output_path, engine="zarr", group=group_path)
        assert ds.data_vars, f"{group_path} has no variables"
        for name in ds.data_vars:
            arr = zarr.open_array(output_path, path=f"{group_path}/{name}")
            codec_names = [type(c).__name__ for c in arr.metadata.codecs]

            if keep_scale_offset:
                assert arr.dtype == np.uint16, (
                    f"{group_path}/{name}: expected uint16 on disk, got {arr.dtype}"
                )
                assert "ScaleOffset" not in codec_names
                assert "CastValue" not in codec_names
            elif experimental_scale_offset_codec:
                assert arr.dtype == np.float32, (
                    f"{group_path}/{name}: expected float32 logical dtype, got {arr.dtype}"
                )
                assert "ScaleOffset" in codec_names, f"{group_path}/{name}: codecs={codec_names}"
                assert "CastValue" in codec_names, f"{group_path}/{name}: codecs={codec_names}"
            else:
                assert arr.dtype == np.float32, (
                    f"{group_path}/{name}: expected float32, got {arr.dtype}"
                )
                assert "ScaleOffset" not in codec_names
                assert "CastValue" not in codec_names

    # ------------------------------------------------------------------
    # Decoded-value invariance: regardless of which storage path the
    # converter chose, the values xarray hands back must be the same — the
    # input quantised onto the source variable's scale_factor/add_offset
    # grid (every storage path round-trips through that grid).
    # ------------------------------------------------------------------
    for group_path, var_name in _ORIGINAL_GROUPS.items():
        da_in = dt_input[group_path].to_dataset()[var_name]
        sf = float(da_in.encoding["scale_factor"])
        ao = float(da_in.encoding["add_offset"])
        packed = np.round((da_in.values - ao) / sf).astype("uint16")
        expected = (packed.astype("float64") * sf + ao).astype("float32")

        observed = xr.open_dataset(output_path, engine="zarr", group=group_path)[var_name].values
        # CF quantisation rounds to the nearest multiple of `scale_factor`,
        # so values agree with `expected` to within sf/2 in exact arithmetic.
        # On the r10m level the values round-trip through a float64 → float32
        # cast in the converter, which can push a handful of points up to one
        # full step off (observed worst case: ~1.0002 * sf due to float32
        # ULP slop at this magnitude). Tolerate one full step.
        np.testing.assert_allclose(observed.astype("float32"), expected, atol=sf, rtol=1e-3)


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
