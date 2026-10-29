"""
Tests for the S2 optimized converter pipeline: the uniform encoding it stores
and the structure `convert_s2_optimized` produces end to end. The pyramids
themselves are planned by `data_api.s2_plan` and written by
`data_api.s2_write`, which carry their own tests; the golden snapshots are
pinned by `tests/test_data_api/test_s2_plan.py`.
"""

import pathlib
from itertools import pairwise

import numpy as np
import pytest
import xarray as xr
import zarr
from structlog.testing import capture_logs
from zarr.core.metadata import ArrayV3Metadata

from eopf_geozarr.conversion.utils import (
    _rechunk_ds,
    calculate_aligned_chunk_size,
    create_uniform_encoding,
    rechunk_dataset_for_encoding,
)
from eopf_geozarr.s2_optimization.s2_converter import convert_s2_optimized
from eopf_geozarr.s2_optimization.s2_multiscale import _AUX_MULTISCALES, S2Type


def _codec_names(metadata: ArrayV3Metadata) -> list[str]:
    """Codec class names, including those nested in a sharding codec."""
    names: list[str] = []
    for codec in metadata.codecs:
        names.append(type(codec).__name__)
        names.extend(type(inner).__name__ for inner in getattr(codec, "codecs", ()))
    return names


@pytest.fixture
def sample_dataset(s2_group_example: pathlib.Path) -> xr.Dataset:
    """Create a sample xarray dataset for testing."""
    with pytest.warns((RuntimeWarning, FutureWarning)):
        return xr.open_datatree(
            s2_group_example, engine="zarr", mask_and_scale=False, decode_coords="all"
        )["measurements/reflectance/r10m"].to_dataset()


@pytest.mark.parametrize("mask_and_scale", [False, True], ids=["raw", "decoded"])
@pytest.mark.parametrize("scale_offset_codec", [True, False], ids=["codec", "cf"])
def test_create_measurements_encoding(
    scale_offset_codec: bool, mask_and_scale: bool, s2_group_example: pathlib.Path
) -> None:
    """Test measurements encoding creation with xy-aligned sharding.

    Raw input (`mask_and_scale=False`, as in the CPM path) has the CF values in
    `.attrs`; decoded input (as in the CLI) has them in `.encoding`. Both must
    give the same stored layout for each encoding mode.
    """
    with pytest.warns((RuntimeWarning, FutureWarning)):
        sample_dataset = xr.open_datatree(
            s2_group_example, engine="zarr", mask_and_scale=mask_and_scale, decode_coords="all"
        )["measurements/reflectance/r10m"].to_dataset()
    sample_dataset = _rechunk_ds(sample_dataset, 1024)

    encoding = create_uniform_encoding(
        sample_dataset,
        enable_sharding=True,
        scale_offset_codec=scale_offset_codec,
    )

    # Check that encoding is created for all variables
    for var_name in sample_dataset.data_vars:
        assert str(var_name) in encoding
        var_encoding = encoding[str(var_name)]

        # Check basic encoding structure
        assert "chunks" in var_encoding
        # Zarr v3 uses 'compressors' (plural)
        assert "compressors" in var_encoding or "compressor" in var_encoding

        # Check sharding is included when enabled
        assert "shards" in var_encoding

    # Check coordinate encoding
    for coord_name in sample_dataset.coords:
        if str(coord_name) in encoding:
            # Coordinates may have either compressor or compressors set to None
            assert (
                encoding[str(coord_name)].get("compressor") is None
                or encoding[str(coord_name)].get("compressors") is None
            )
    # rechunk before write
    output_dataset = rechunk_dataset_for_encoding(sample_dataset, encoding)

    stored = output_dataset.to_zarr({}, encoding=encoding)
    zg = stored.zarr_group
    reflectance_bands = [name for name in output_dataset.data_vars if str(name).startswith("b")]
    assert reflectance_bands
    for var_name in reflectance_bands:
        array = zg[str(var_name)]
        assert isinstance(array, zarr.Array)
        assert isinstance(array.metadata, ArrayV3Metadata)
        codec_names = _codec_names(array.metadata)
        if not scale_offset_codec:
            assert array.dtype == np.uint16
            assert array.attrs["scale_factor"] == pytest.approx(0.0001)
            assert array.attrs["add_offset"] == pytest.approx(-0.1)
            assert array.attrs["_FillValue"] == 0
            assert "ScaleOffset" not in codec_names
        else:
            assert array.dtype == np.float32
            assert {"ScaleOffset", "CastValue"} <= set(codec_names)
            assert "scale_factor" not in array.attrs
            assert "add_offset" not in array.attrs


def test_create_measurements_encoding_time_chunking(sample_dataset: xr.Dataset) -> None:
    """Test that time dimension is chunked to 1 for single file per time."""
    # rechunk
    sample_dataset = _rechunk_ds(sample_dataset, 1024)

    encoding = create_uniform_encoding(sample_dataset, enable_sharding=True)

    for var_name in sample_dataset.data_vars:
        if sample_dataset[var_name].ndim == 3:  # 3D variable with time
            chunks = encoding[str(var_name)].get("chunks")
            assert chunks is not None
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


# implemented as reading in from Fiztures wont include spatial_refs and other variables
# Arrays that aren't spectral bands — excluded when comparing a group's band set between input and output
_NON_BAND_ARRAYS = {"x", "y", "time", "spatial_ref"}


def _band_names(group: zarr.Group) -> set[str]:
    """Names of the spectral-band arrays directly under `group`."""
    return {name for name in group.array_keys() if name not in _NON_BAND_ARRAYS}


# L1C r60m is filled with every finer band, so it and its overviews carry all 13.
_L1C_ALL_BANDS = {f"b{i:02d}" for i in range(1, 13)} | {"b8a"}


def test_create_multiscale_from_datatree(
    s2_group_example: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """The default encoding exercised through the full `convert_s2_optimized` workflow.

    This asserts the structural properties the multiscale pipeline is
    actually responsible for — every band on an original resolution level
    survives conversion, the downsampled overview levels exist and carry the
    same bands as their r60m source, and dtypes are consistent across the
    whole pyramid — rather than diffing the output against a golden JSON
    snapshot. A byte-exact snapshot is both brittle (any deliberate metadata
    change breaks it regardless of correctness) and, for this fixture
    specifically, partly unverifiable: `s2_group_example` is built from a
    JSON zarr-metadata spec with no real chunk data, so anything derived
    from real coordinate values (bbox, transforms, geotransforms) can never
    match a snapshot built from a real product.

    Behavior under other parametrizations is exercised by
    `test_create_multiscale_from_datatree_behavior` below, which uses a small
    in-memory dataset with explicit, easily-verified expectations.
    """

    # changed to not have a Scrict comparison between fixtures read in from zarrv2 jsons (out of data, impossible to maintain)
    output_path = str(tmp_path / "output.zarr")
    input_group = zarr.open_group(s2_group_example)
    # xarray's open_datatree accepts a zarr store at runtime, but its stub does
    # not list Store among the accepted input types.
    dt_input = xr.open_datatree(
        input_group.store,  # pyright: ignore[reportArgumentType]
        engine="zarr",
        chunks={},
        mask_and_scale=False,
        decode_coords="all",
    )

    # Capture log output using structlog's testing context manager
    with capture_logs():
        convert_s2_optimized(
            dt_input,
            output_path=output_path,
            enable_sharding=True,
            spatial_chunk=1024,
            validate_output=False,
            compression_level=3,
        )

    observed_group = zarr.open_group(output_path, use_consolidated=False)

    # Every top-level group present on input (measurements/quality/conditions)
    # must survive conversion.
    for top_level in input_group.group_keys():
        assert top_level in observed_group, f"missing top-level group '{top_level}'"

    input_reflectance = input_group["measurements/reflectance"]
    reflectance_group = observed_group["measurements/reflectance"]
    assert isinstance(input_reflectance, zarr.Group)
    assert isinstance(reflectance_group, zarr.Group)

    # Every band on an original resolution level (r10m/r20m/r60m) must be
    # present after conversion (conversion may *add* bands here — e.g.
    # `inject_missing_bands` backfills bands missing at coarser native
    # resolutions — but must never drop one).
    original_levels = list(input_reflectance.group_keys())
    for level in original_levels:
        input_level = input_reflectance[level]
        observed_level = reflectance_group[level]
        assert isinstance(input_level, zarr.Group)
        assert isinstance(observed_level, zarr.Group)
        input_bands = _band_names(input_level)
        observed_bands = _band_names(observed_level)
        assert input_bands <= observed_bands, (
            f"{level}: missing bands {input_bands - observed_bands}"
        )

    # The pyramid must extend the finest coarsened level (r60m) down through
    # r120m/r360m/r720m, carrying the same band set at every level.
    r60m_group = reflectance_group["r60m"]
    assert isinstance(r60m_group, zarr.Group)
    r60m_bands = _band_names(r60m_group)
    for level in ("r120m", "r360m", "r720m"):
        assert level in reflectance_group, f"missing downsampled group '{level}'"
        level_group = reflectance_group[level]
        assert isinstance(level_group, zarr.Group)
        level_bands = _band_names(level_group)
        assert level_bands == r60m_bands, (
            f"{level}: band mismatch with r60m; expected {r60m_bands}, got {level_bands}"
        )

    # Bands injected from finer resolutions must be present. The product level
    # comes from `stac_discovery`, because a tree opened from a store has no name.
    s2_type = S2Type.from_datatree(dt_input)
    assert s2_type is not None
    for level in ("r20m", "r60m", "r120m", "r360m", "r720m"):
        level_group = reflectance_group[level]
        assert isinstance(level_group, zarr.Group)
        assert "b08" in _band_names(level_group), f"{level}: b08 not injected"
    if s2_type == S2Type.L1C:
        for level in ("r60m", "r120m", "r360m", "r720m"):
            level_group = reflectance_group[level]
            assert isinstance(level_group, zarr.Group)
            assert _band_names(level_group) == _L1C_ALL_BANDS, f"{level}: incomplete L1C bands"

    # Auxiliary pyramids (L1C cloud mask for every product; SCL and cloud/snow
    # probabilities for L2A): every level exists with the variables and dtypes of
    # the finest level, and the parent group's layout lists exactly those levels.
    for aux in _AUX_MULTISCALES:
        if not aux.applies_to(s2_type):
            continue
        base_path, finest_key, scale_levels = aux.base_path, aux.finest_key, aux.scale_levels
        attrs = aux.additional_attributes
        base = base_path.lstrip("/")
        aux_group = observed_group[base]
        assert isinstance(aux_group, zarr.Group), f"missing auxiliary pyramid '{base}'"
        finest_group = aux_group[finest_key]
        assert isinstance(finest_group, zarr.Group)
        finest_vars = _band_names(finest_group)
        assert finest_vars, f"{base}/{finest_key} has no variables"
        level_names = [f"r{level}m" for level in scale_levels]
        for level in level_names:
            assert level in aux_group, f"missing level '{base}/{level}'"
            level_group = aux_group[level]
            assert isinstance(level_group, zarr.Group)
            assert _band_names(level_group) == finest_vars, f"{base}/{level}: variable mismatch"
            for name in finest_vars:
                level_array = level_group[name]
                finest_array = finest_group[name]
                assert isinstance(level_array, zarr.Array)
                assert isinstance(finest_array, zarr.Array)
                assert level_array.dtype == finest_array.dtype, f"{base}/{level}/{name}: dtype"
                if attrs is not None:
                    for key, value in attrs.items():
                        assert level_array.attrs[key] == value, f"{base}/{level}/{name}: {key}"
        layout = aux_group.attrs["multiscales"]["layout"]  # type: ignore[index, call-overload]
        assert [lo["asset"] for lo in layout] == level_names, f"{base}: layout assets"

    # All multiscale levels must agree on dtype for the bands they share.
    _, res_groups = zip(*reflectance_group.groups(), strict=False)
    dtype_mismatch: set[object] = set()
    for group_a, group_b in pairwise(res_groups):
        ds_a = xr.open_dataset(
            group_a.store, engine="zarr", group=group_a.path, decode_coords="all"
        )
        ds_b = xr.open_dataset(
            group_b.store, engine="zarr", group=group_b.path, decode_coords="all"
        )

        for name in ds_a.data_vars:
            dtype_a = ds_a[name].dtype
            if name in ds_b.data_vars:
                dtype_b = ds_b[name].dtype
                if dtype_a != dtype_b:
                    dtype_mismatch.add(
                        (f"{group_a.path}/{name}::{dtype_a}", f"{group_b.path}/{name}::{dtype_b}")
                    )
    assert dtype_mismatch == set()
