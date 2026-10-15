"""
Tests for S2 multiscale pyramid creation with xy-aligned sharding.
"""

import json
import os
import pathlib
from collections.abc import Mapping, Sequence
from itertools import pairwise

import numpy as np
import pytest
import xarray as xr
import zarr
from affine import Affine
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import GroupSpec
from pyproj import CRS
from structlog.testing import capture_logs
from zarr.core.metadata import ArrayV3Metadata

from eopf_geozarr.conversion.utils import (
    _rechunk_ds,
    create_uniform_encoding,
    rechunk_dataset_for_encoding,
)
from eopf_geozarr.s2_optimization.s2_converter import convert_s2_optimized
from eopf_geozarr.s2_optimization.s2_multiscale import (
    _AUX_MULTISCALES,
    S2Type,
    _coarsen_variable,
    calculate_aligned_chunk_size,
    calculate_simple_shard_dimensions,
    create_downsampled_resolution_group,
    create_multiscale_from_datatree,
    generic_multiscales,
    inject_missing_bands,
    write_geo_metadata,
)

from .conftest import create_zarrv2_group_from_json, get_stem, s2_example_json_paths


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


def test_create_downsampled_resolution_group_quality_mask() -> None:
    """Quality-mask downsampling is disabled, so it should raise."""
    x = np.arange(8)
    y = np.arange(6)
    quality = xr.DataArray(
        np.random.randint(0, 2, (6, 8), dtype=np.uint8),
        dims=["y", "x"],
        coords={"y": y, "x": x},
        name="quality_clouds",
    )
    ds = xr.Dataset({"quality_clouds": quality})

    with pytest.raises(ValueError, match="quality_mask"):
        create_downsampled_resolution_group(ds, factor=2)


def test_generic_multiscales_prefers_coordinate_transform_for_inconsistent_rio(
    tmp_path: pathlib.Path,
) -> None:
    """Derived levels should not reuse the stale rio transform inherited from the finest level."""
    crs = CRS.from_epsg(32631)
    size = 24  # 24 px at 10 m -> 12 px at 20 m -> 4 px at 60 m -> 2 px at 120 m
    x = 600000.0 + np.arange(size, dtype="float64") * 10
    y = 4900020.0 - np.arange(size, dtype="float64") * 10
    r10m = _rechunk_ds(
        xr.Dataset(
            {"band": (["y", "x"], np.ones((size, size), dtype=np.uint16))},
            coords={"x": x, "y": y},
        ),
        1024,
    )
    write_geo_metadata(r10m, crs=crs)
    # A GeoTransform on `spatial_ref` is carried into every coarsened level, where
    # the rio transform then no longer matches the coordinate grid.
    r10m.rio.write_transform(Affine(10.0, 0.0, 599995.0, 0.0, -10.0, 4900025.0), inplace=True)

    output_path = tmp_path / "multiscales.zarr"
    base_path = "/measurements/reflectance"
    with capture_logs():
        generic_multiscales(
            base_path=base_path,
            src_processed_groups={"r10m": r10m},
            output_path=str(output_path),
            output_group=zarr.create_group(output_path),
            scale_levels=(10, 20, 60, 120),
            finest_dataset_key="r10m",
            crs=crs,
        )

    parent_group = zarr.open_group(output_path, path=base_path)
    multiscales = parent_group.attrs["multiscales"]
    assert isinstance(multiscales, Mapping)
    layout = multiscales["layout"]
    assert isinstance(layout, Sequence)
    assert [level["asset"] for level in layout if isinstance(level, Mapping)] == [
        "r10m",
        "r20m",
        "r60m",
        "r120m",
    ]
    derived_level = next(
        level for level in layout if isinstance(level, Mapping) and level["asset"] == "r120m"
    )
    assert isinstance(derived_level, Mapping)
    assert derived_level["derived_from"] == "r60m"
    assert derived_level["spatial:shape"] == [2, 2]
    transform = derived_level["spatial:transform"]
    assert isinstance(transform, Sequence)
    # Origin is the outer pixel edge, half of the 120 m pixel outside the first centre.
    assert tuple(transform) == (
        120.0,
        0.0,
        599995.0,
        0.0,
        -120.0,
        4900025.0,
    )

    # The parent footprint covers the pixel edges of the finest level (#266).
    assert parent_group.attrs["spatial:bbox"] == [599995.0, 4899785.0, 600235.0, 4900025.0]


def test_calculate_simple_shard_dimensions() -> None:
    """Test simplified shard dimensions calculation."""
    # Test 3D data (time, y, x) - shards are multiples of chunks
    data_shape: tuple[int, ...] = (5, 1024, 1024)
    chunks: tuple[int, ...] = (1, 256, 256)

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

# Footprint of an S2 tile (109.8 km square) in UTM, used to give the JSON fixtures
# real coordinates: they carry no chunk data, so every x/y value is the fill value.
_TILE_CRS = CRS.from_epsg(32632)
_TILE_X_MIN = 300000.0
_TILE_Y_MAX = 5000040.0
_TILE_EXTENT = 109800.0


def _with_tile_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """Replace the x/y coordinates with pixel centres covering `_TILE_EXTENT`."""
    if "x" not in ds.dims or "y" not in ds.dims:
        return ds
    res_x = _TILE_EXTENT / ds.sizes["x"]
    res_y = _TILE_EXTENT / ds.sizes["y"]
    return ds.assign_coords(
        x=_TILE_X_MIN + (np.arange(ds.sizes["x"]) + 0.5) * res_x,
        y=_TILE_Y_MAX - (np.arange(ds.sizes["y"]) + 0.5) * res_y,
    )


@pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:.*:FutureWarning")
@pytest.mark.filterwarnings("ignore:.*:UserWarning")
@pytest.mark.parametrize("source_path", s2_example_json_paths, ids=get_stem)
def test_create_multiscale_from_datatree_snapshot(
    source_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Compare the output structure of a Zarr v2 S2 input against a stored snapshot.

    Any change to array metadata (dtype, fill value, chunks, codecs, attributes)
    or to the set of arrays shows up here. When a change is intended, regenerate
    the snapshots with `REGENERATE_SNAPSHOTS=1 pytest -k snapshot` and review
    the JSON diff.
    """
    input_group = zarr.open_group(create_zarrv2_group_from_json(source_path, tmp_path))
    dt_input = xr.open_datatree(
        input_group.store,  # pyright: ignore[reportArgumentType]
        engine="zarr",
        chunks={},
        mask_and_scale=False,
        decode_coords="all",
    ).map_over_datasets(_with_tile_coordinates)
    output_path = str(tmp_path / "output.zarr")
    with capture_logs():
        create_multiscale_from_datatree(
            dt_input,
            output_group=zarr.create_group(output_path),
            output_path=output_path,
            enable_sharding=True,
            spatial_chunk=1024,
            crs=_TILE_CRS,
        )

    observed_json = GroupSpec.from_zarr(
        zarr.open_group(output_path, use_consolidated=False)
    ).model_dump()
    snapshot_path = pathlib.Path("tests/_test_data/optimized_geozarr_examples") / (
        source_path.stem + ".json"
    )
    if os.environ.get("REGENERATE_SNAPSHOTS"):
        snapshot_path.write_text(json.dumps(observed_json, indent=2, sort_keys=True) + "\n")

    observed = GroupSpec(**tuplify_json(observed_json)).to_flat()
    expected = GroupSpec(**tuplify_json(json.loads(snapshot_path.read_text()))).to_flat()
    assert set(observed) == set(expected)

    # Compare as JSON: a NaN fill value never equals itself as a float.
    def canonical(node: object) -> str:
        return json.dumps(node.model_dump(), sort_keys=True)  # type: ignore[attr-defined]

    assert [key for key in observed if canonical(observed[key]) != canonical(expected[key])] == []


@pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:.*:FutureWarning")
@pytest.mark.filterwarnings("ignore:.*:UserWarning")
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
    for base_path, finest_key, scale_levels, attrs, product_levels in _AUX_MULTISCALES:
        if product_levels is not None and s2_type not in product_levels:
            continue
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


# A 6x6 nodata block in r60m covers r120m[0:3, 0:3] and r360m[0, 0] entirely.
_NODATA_BLOCK = 6
_SCALE_FACTOR = 0.0001
_ADD_OFFSET = -0.1


def _make_minimal_s2_datatree(*, raw: bool) -> xr.DataTree:
    """Build a tiny reflectance DataTree for behavior tests.

    Three levels (r10m, r20m, r60m), one band each, packed like ESA S2 L2A
    reflectance (uint16, `scale_factor` 0.0001, `add_offset` -0.1, nodata 0).
    The r60m band has a nodata block of `_NODATA_BLOCK` pixels in its top-left
    corner.

    `raw=False` gives decoded floats with the CF values in `.encoding`, as the
    CLI opens products. `raw=True` gives the packed integers with the CF values
    in `.attrs` and the nodata value only in the EOPF `fill_value` attribute,
    as CPM Zarr v3 products opened with `mask_and_scale=False`.
    """
    rng = np.random.default_rng(0)

    def _band(size: int, *, nodata_block: int = 0) -> xr.DataArray:
        data = rng.uniform(0.0, 1.0, size=(size, size)).astype("float64")
        data[:nodata_block, :nodata_block] = np.nan
        coords = {
            "x": np.arange(size, dtype="float64"),
            "y": np.arange(size, dtype="float64"),
        }
        if raw:
            packed = np.round((data - _ADD_OFFSET) / _SCALE_FACTOR)
            packed = np.where(np.isnan(data), 0, packed).astype("uint16")
            return xr.DataArray(
                packed,
                dims=["y", "x"],
                coords=coords,
                attrs={"scale_factor": _SCALE_FACTOR, "add_offset": _ADD_OFFSET, "fill_value": 0},
            )
        da = xr.DataArray(data, dims=["y", "x"], coords=coords)
        da.encoding = {
            "scale_factor": _SCALE_FACTOR,
            "add_offset": _ADD_OFFSET,
            "_FillValue": 0,
            "dtype": np.dtype("uint16"),
        }
        return da

    r10m = xr.Dataset({"b02": _band(120)})
    r20m = xr.Dataset({"b05": _band(60)})
    r60m = xr.Dataset({"b01": _band(20, nodata_block=_NODATA_BLOCK)})
    # L1C cloud classification bitmask (OPAQUE | CIRRUS | SNOW_ICE), unpacked uint8
    # on the r60m grid, as in ESA products. Required by `create_multiscale_from_datatree`.
    l1c_classification = xr.Dataset(
        {
            "b00": xr.DataArray(
                rng.integers(0, 8, size=(20, 20), dtype="uint8"),
                dims=["y", "x"],
                coords=r60m.coords,
                attrs={"flag_masks": [1, 2, 4], "flag_meanings": ["OPAQUE", "CIRRUS", "SNOW_ICE"]},
            )
        }
    )

    dt = xr.DataTree()
    dt["measurements/reflectance/r10m"] = xr.DataTree(r10m)
    dt["measurements/reflectance/r20m"] = xr.DataTree(r20m)
    dt["measurements/reflectance/r60m"] = xr.DataTree(r60m)
    dt["conditions/mask/l1c_classification/r60m"] = xr.DataTree(l1c_classification)
    return dt


def _decoded(da: xr.DataArray) -> np.ndarray:
    """Values of a synthetic band as the source product defines them (NaN for nodata)."""
    values = da.values.astype("float64")
    if np.issubdtype(da.dtype, np.integer):
        return np.where(values == 0, np.nan, values * _SCALE_FACTOR + _ADD_OFFSET)
    return values


# Spatial chunk small enough that no padding is needed for the 20x20 r60m band.
_BEHAVIOR_SPATIAL_CHUNK = 16

# Original groups in the synthetic datatree — those that are packed on input.
_ORIGINAL_GROUPS = {
    "measurements/reflectance/r10m": "b02",
    "measurements/reflectance/r20m": "b05",
    "measurements/reflectance/r60m": "b01",
}

# Downsampled groups added by `create_multiscale_from_datatree`, derived from the
# r60m level by successive coarsening. `_coarsen_variable` preserves the source
# variable's packing, so these levels are encoded like the original groups.
_DOWNSAMPLED_GROUPS = (
    "measurements/reflectance/r120m",
    "measurements/reflectance/r360m",
    "measurements/reflectance/r720m",
)


@pytest.mark.filterwarnings("ignore:.*:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:.*:FutureWarning")
@pytest.mark.filterwarnings("ignore:.*:UserWarning")
@pytest.mark.parametrize("raw", [False, True], ids=["decoded", "raw"])
@pytest.mark.parametrize("scale_offset_codec", [True, False], ids=["codec", "cf"])
def test_create_multiscale_from_datatree_behavior(
    scale_offset_codec: bool,
    raw: bool,
    tmp_path: pathlib.Path,
) -> None:
    """Verify both encoding modes on every level of a tiny pyramid, for both input forms.

    * default: every level keeps the source packing in Zarr `scale_offset` +
      `cast_value` codecs; the logical dtype is float32 (decoded) and there are
      no CF scale attributes.
    * `scale_offset_codec=False`: every level is uint16 on disk with the CF
      `scale_factor`, `add_offset` and `_FillValue` attributes, as in the ESA
      product.

    In both modes, decoded values match the source within one packing step and
    nodata stays nodata through the pyramid.
    """
    dt_input = _make_minimal_s2_datatree(raw=raw)
    output_path = str(tmp_path / "output.zarr")

    with capture_logs():
        create_multiscale_from_datatree(
            dt_input,
            output_group=zarr.create_group(output_path),
            output_path=output_path,
            enable_sharding=False,
            spatial_chunk=_BEHAVIOR_SPATIAL_CHUNK,
            # The spatial convention attrs that the pyramid layout needs are only written with a CRS.
            crs=CRS.from_epsg(32632),
            scale_offset_codec=scale_offset_codec,
        )

    def _check_array(path: str) -> None:
        arr = zarr.open_array(output_path, path=path)
        assert isinstance(arr.metadata, ArrayV3Metadata)
        codec_names = _codec_names(arr.metadata)
        cf_attrs = {"scale_factor", "add_offset"} & set(arr.attrs)
        if not scale_offset_codec:
            assert arr.dtype == np.uint16, f"{path}: expected uint16 on disk, got {arr.dtype}"
            assert "ScaleOffset" not in codec_names
            assert "CastValue" not in codec_names
            assert arr.attrs["scale_factor"] == pytest.approx(_SCALE_FACTOR)
            assert arr.attrs["add_offset"] == pytest.approx(_ADD_OFFSET)
            assert arr.attrs["_FillValue"] == 0
        else:
            assert arr.dtype == np.float32, (
                f"{path}: expected float32 logical dtype, got {arr.dtype}"
            )
            assert "ScaleOffset" in codec_names, f"{path}: codecs={codec_names}"
            assert "CastValue" in codec_names, f"{path}: codecs={codec_names}"
            assert cf_attrs == set(), f"{path}: unexpected CF attrs {cf_attrs}"

    for group_path, var_name in _ORIGINAL_GROUPS.items():
        _check_array(f"{group_path}/{var_name}")

    parent_group = zarr.open_group(output_path, path="measurements/reflectance")
    for group_path in _DOWNSAMPLED_GROUPS:
        # All three downsampled levels should exist for the minimal datatree
        # (10/20/60 → 120/360/720).
        sub = group_path.removeprefix("measurements/reflectance/")
        assert sub in dict(parent_group.groups()), f"missing downsampled group {group_path}"
        ds = xr.open_dataset(output_path, engine="zarr", group=group_path, decode_coords="all")
        assert ds.data_vars, f"{group_path} has no variables"
        for name in ds.data_vars:
            _check_array(f"{group_path}/{name}")

    # Decoded values match the source, quantised onto the packing grid.
    for group_path, var_name in _ORIGINAL_GROUPS.items():
        expected = _decoded(dt_input[group_path].to_dataset()[var_name])
        observed = xr.open_dataset(
            output_path, engine="zarr", group=group_path, decode_coords="all"
        )[var_name].values
        # Quantisation plus the float32 round trip stay within one packing step.
        np.testing.assert_allclose(observed, expected, atol=_SCALE_FACTOR, rtol=0)

    # Nodata stays nodata through the pyramid: pixels fully covered by the
    # source nodata block decode to NaN, the others do not.
    for group_path, nodata_extent in (
        ("measurements/reflectance/r60m", _NODATA_BLOCK),
        ("measurements/reflectance/r120m", _NODATA_BLOCK // 2),
        ("measurements/reflectance/r360m", _NODATA_BLOCK // 6),
    ):
        decoded = xr.open_dataset(
            output_path, engine="zarr", group=group_path, decode_coords="all"
        )["b01"].values
        block = decoded[:nodata_extent, :nodata_extent]
        assert np.isnan(block).all(), f"{group_path}: nodata block not NaN"
        assert np.isnan(decoded).sum() == block.size, f"{group_path}: unexpected NaN outside block"


# ---------------------------------------------------------------------------
# _coarsen_variable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("var_name", ["scl", "b00"])
def test_coarsen_variable_classification(var_name: str) -> None:
    """Classification variables (SCL, L1C cloud bitmask) should be downsampled via subsample."""
    data = np.arange(16, dtype="uint8").reshape(4, 4)
    var = xr.DataArray(data, dims=["y", "x"], coords={"y": np.arange(4.0), "x": np.arange(4.0)})
    result = _coarsen_variable(var_name, var, factor=2)
    assert result.shape == (2, 2)
    assert result.dtype == np.uint8
    # subsample picks top-left of each 2x2 block
    np.testing.assert_array_equal(result.values, data[::2, ::2])


def test_coarsen_variable_quality_mask() -> None:
    """Quality mask downsampling is disabled, so it should raise."""
    data = np.array([[0, 1], [2, 3]], dtype="uint8")
    var = xr.DataArray(data, dims=["y", "x"], coords={"y": np.arange(2.0), "x": np.arange(2.0)})
    with pytest.raises(ValueError, match="quality_mask"):
        _coarsen_variable("quality_cirrus", var, factor=2)


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


@pytest.mark.parametrize("scale_offset_codec", [True, False])
def test_inject_missing_bands_skips_unmasked_nodata(scale_offset_codec: bool) -> None:
    """Decoded Zarr v3 input keeps its nodata unmasked; injection must still skip it.

    xarray only masks `_FillValue`, so a decoded CPM Zarr v3 band shows nodata
    as `0 * scale_factor + add_offset` and declares it only in the EOPF
    `fill_value` attribute. The codec mode injects decoded float32 values; the
    ESA layout injects packed integers, with no cast to float.
    """
    dt = _make_reflectance_datatree()
    decoded = np.full((120, 120), 0.3)
    decoded[0, 0] = -0.1  # nodata (DN 0) in the top-left 2x2 block
    decoded[2:4, 2:4] = -0.1  # a fully nodata 2x2 block
    b08 = xr.DataArray(
        decoded,
        dims=["y", "x"],
        coords=dt["measurements/reflectance/r10m"].to_dataset().coords,
        attrs={"fill_value": 0},
    )
    b08.encoding = {"scale_factor": 0.0001, "add_offset": -0.1, "dtype": np.dtype("uint16")}
    r10m = dt["measurements/reflectance/r10m"].to_dataset().assign(b08=b08)
    dt["measurements/reflectance/r10m"] = xr.DataTree(r10m)
    r20m_ds = dt["measurements/reflectance/r20m"].to_dataset()

    result = inject_missing_bands(
        r20m_ds,
        dt,
        target_resolution=20,
        bands={"b08"},
        spatial_chunk=1024,
        scale_offset_codec=scale_offset_codec,
    )

    values = result["b08"].values
    if scale_offset_codec:
        assert values.dtype == np.float32
        assert values[0, 0] == pytest.approx(0.3)  # the nodata pixel is not averaged in
        assert np.isnan(values[1, 1])  # a fully nodata block stays nodata
        assert np.isnan(values).sum() == 1
    else:
        assert values.dtype == np.uint16
        assert result["b08"].attrs["scale_factor"] == pytest.approx(0.0001)
        assert result["b08"].attrs["add_offset"] == pytest.approx(-0.1)
        assert result["b08"].encoding["_FillValue"] == 0
        assert values[0, 0] == 4000  # (0.3 + 0.1) / 0.0001: nodata is not averaged in
        assert values[1, 1] == 0  # a fully nodata block stays nodata
        assert (values == 0).sum() == 1


def test_inject_missing_bands_respects_bands_filter() -> None:
    """With bands={"b08"}, only b08 should be injected even when others are eligible."""
    dt = _make_reflectance_datatree()
    r20m_ds = dt["measurements/reflectance/r20m"].to_dataset()

    result = inject_missing_bands(
        r20m_ds, dt, target_resolution=20, bands={"b08"}, spatial_chunk=1024
    )

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

    result = inject_missing_bands(
        r20m_ds, dt, target_resolution=20, bands={"b08"}, spatial_chunk=1024
    )

    # b08 was already present — inject_missing_bands must leave it untouched.
    np.testing.assert_array_equal(result["b08"].values, sentinel)


def test_inject_missing_bands_noop_when_no_source() -> None:
    """If the source group is missing from the DataTree, return dataset unchanged."""
    dt = xr.DataTree()
    ds = xr.Dataset({"b05": (["y", "x"], np.ones((60, 60)))})

    result = inject_missing_bands(ds, dt, target_resolution=20, spatial_chunk=1024)

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

    result = inject_missing_bands(r60m_ds, dt, target_resolution=60, spatial_chunk=1024)

    # All 10m bands (b02, b03, b04, b08) and 20m bands (b05, b06) should be injected
    for band in ("b02", "b03", "b04", "b08", "b05", "b06"):
        assert band in result.data_vars, f"{band} missing from r60m"
        assert result[band].shape == (size_60m, size_60m)
