"""
Tests for the plan-driven writer (`eopf_geozarr.data_api.s2_write`).

`write_geozarr` replaced the DataTree-based writer after reproducing its
stores byte for byte, file by file, on these fixtures (the parity test lived
here until that writer was removed). These tests pin what the writer
guarantees on its own: the stored documents are exactly the plan, and the
chunk data follows the copy / inject / coarsen semantics, checked against
independently formulated references.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
import zarr
from pydantic_zarr.v3 import GroupSpec as GroupSpecV3

from eopf_geozarr.data_api.s2_plan import load_sentinel2
from eopf_geozarr.data_api.s2_write import _coarsen, write_geozarr

from ..conftest import create_zarrv2_group_from_json, get_stem, s2_example_json_paths

if TYPE_CHECKING:
    import pathlib

    from pydantic_zarr.v3 import AnyGroupSpec

# Fallback footprint for coordinate arrays in groups that carry no proj
# attributes (geometry, meteorology), where the writer only copies the values.
_TILE_X_MIN = 300000.0
_TILE_Y_MAX = 5000040.0
_TILE_EXTENT = 109800.0

#: Arrays at most this many elements get deterministic test data, so the
#: resampling arithmetic is exercised on values, not only on nodata fill.
_PATTERN_MAX_ELEMENTS = 1830 * 1830

#: Larger arrays seeded anyway: the scene classification and one finest band,
#: so the window-centre subsampling and the band injection are exercised on
#: real values.
_PATTERNED_PATHS = (
    "conditions/mask/l2a_classification/r20m/scl",
    "conditions/mask/l1c_classification/r60m/b00",
    "measurements/reflectance/r10m/b08",
)

#: The L2A product used for the value-semantics checks.
_L2A_PATH = next(path for path in s2_example_json_paths if "MSIL2A" in path.stem)


def _walk_groups(group: zarr.Group, prefix: str = "") -> list[tuple[str, zarr.Group]]:
    found: list[tuple[str, zarr.Group]] = [(prefix, group)]
    for name, subgroup in group.groups():
        found.extend(_walk_groups(subgroup, f"{prefix}{name}/"))
    return found


def _group_transform(group: zarr.Group) -> tuple[float, ...] | None:
    """The proj:transform any of the group's arrays carries, if any."""
    for _, array in group.arrays():
        transform = array.attrs.get("proj:transform")
        if not isinstance(transform, (list, tuple)) or len(transform) < 6:
            continue
        values = [value for value in transform[:6] if isinstance(value, (int, float))]
        if len(values) == 6:
            return tuple(float(value) for value in values)
    return None


def _pattern(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Deterministic 2-D test values, with occasional zeros (the nodata fill)."""
    modulus = min(999, int(np.iinfo(dtype).max))
    rows = np.arange(shape[0], dtype="uint16")[:, None]
    cols = np.arange(shape[1], dtype="uint16")[None, :]
    return ((rows * np.uint16(7) + cols) % np.uint16(modulus)).astype(dtype)


def _seed_input_store(group: zarr.Group) -> None:
    """Give the metadata-only fixture store coordinates and array values.

    Coordinates are the pixel centres of each group's own `proj:transform`
    (real products store them as float64, so the fixtures' int64 declarations
    are recreated in that form), and selected arrays get deterministic values.
    """
    patterned = set(_PATTERNED_PATHS)
    for group_path, subgroup in _walk_groups(group):
        transform = _group_transform(subgroup)
        for name, array in subgroup.arrays():
            path = f"{group_path}{name}"
            if name in ("x", "y") and array.ndim == 1:
                centers = np.arange(array.shape[0]) + 0.5
                if transform is not None:
                    res_x, _, x0, _, res_y, y0 = transform
                    values = x0 + centers * res_x if name == "x" else y0 + centers * res_y
                else:
                    res = _TILE_EXTENT / array.shape[0]
                    values = (
                        _TILE_X_MIN + centers * res if name == "x" else _TILE_Y_MAX - centers * res
                    )
                replacement = zarr.create_array(
                    group.store,
                    name=path,
                    shape=array.shape,
                    dtype="float64",
                    chunks=array.chunks,
                    zarr_format=2,
                    fill_value=float("nan"),
                    attributes=dict(array.attrs),
                    overwrite=True,
                )
                replacement[:] = values
            elif array.ndim == 2 and np.issubdtype(array.dtype, np.integer):
                if array.size <= _PATTERN_MAX_ELEMENTS or path in patterned:
                    array[...] = _pattern(array.shape, array.dtype)


def _written_store(
    source_path: pathlib.Path, tmp_path: pathlib.Path
) -> tuple[AnyGroupSpec, zarr.Group, zarr.Group]:
    """Seed the fixture store, write it with `write_geozarr`, open both."""
    input_path = create_zarrv2_group_from_json(source_path, tmp_path)
    _seed_input_store(zarr.open_group(input_path, mode="r+", zarr_format=2))
    source = zarr.open_group(input_path, mode="r", zarr_format=2)
    output_path = tmp_path / "planned.zarr"
    plan = write_geozarr(load_sentinel2(source), source, str(output_path))
    return plan, source, zarr.open_group(output_path, mode="r", use_consolidated=False)


def _reference_block_mean(values: np.ndarray, factor: int, fill: int) -> np.ndarray:
    """The converter's masked block mean, formulated independently (nanmean)."""
    rows = values.shape[0] // factor * factor
    cols = values.shape[1] // factor * factor
    windows = values[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor)
    masked = np.where(windows == fill, np.nan, windows.astype("float64"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(masked, axis=(1, 3))
    return np.where(np.isnan(mean), fill, np.round(mean)).astype(values.dtype)


@pytest.mark.parametrize("source_path", s2_example_json_paths, ids=get_stem)
def test_write_geozarr_realizes_the_plan(source_path: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """
    The written store holds exactly the planned hierarchy: reading it back
    gives documents equal to the plan, for every group and array.
    """
    plan, _, output = _written_store(source_path, tmp_path)
    observed = GroupSpecV3.from_zarr(output).model_dump(mode="json")
    assert observed == plan.model_dump(mode="json")


def test_write_geozarr_data_semantics(tmp_path: pathlib.Path) -> None:
    """
    The written chunk data follows the writer's semantics, checked on seeded
    values: native arrays are copied, injected bands are the masked block
    mean of the finest input level holding them, generated levels chain from
    the previous level (mean for bands and probabilities, window-centre pixel
    for the classification), coordinates take window means, and grid-mapping
    arrays hold zero.
    """
    _, source, output = _written_store(_L2A_PATH, tmp_path)

    def read(group: zarr.Group, path: str) -> np.ndarray:
        member = group[path]
        assert isinstance(member, zarr.Array)
        return np.asarray(member[...])

    # a native array is copied (b01 r60m is small enough to carry the pattern)
    native = read(output, "measurements/reflectance/r60m/b01")
    np.testing.assert_array_equal(native, read(source, "measurements/reflectance/r60m/b01"))

    # an injected band is the masked block mean of its finest input level
    injected = read(output, "measurements/reflectance/r60m/b08")
    reference = _reference_block_mean(read(source, "measurements/reflectance/r10m/b08"), 6, 0)
    np.testing.assert_array_equal(injected, reference)

    # a generated band level is the masked block mean of the previous level
    generated = read(output, "measurements/reflectance/r120m/b01")
    np.testing.assert_array_equal(generated, _reference_block_mean(native, 2, 0))

    # a generated classification level takes each window's centre pixel
    scl_native = read(output, "conditions/mask/l2a_classification/r60m/scl")
    scl_generated = read(output, "conditions/mask/l2a_classification/r120m/scl")
    np.testing.assert_array_equal(scl_generated, scl_native[1::2, 1::2])

    # a generated level's coordinates are the window means of the previous level's
    x_native = read(output, "measurements/reflectance/r60m/x")
    x_generated = read(output, "measurements/reflectance/r120m/x")
    np.testing.assert_array_equal(
        x_generated, x_native[: x_generated.shape[0] * 2].reshape(-1, 2).mean(axis=1)
    )

    # grid-mapping arrays hold zero
    assert read(output, "measurements/reflectance/r120m/spatial_ref") == 0


@pytest.mark.parametrize("factor", [2, 3])
@pytest.mark.parametrize("var_name", ["scl", "b00"])
def test_coarsen_classification_takes_the_window_centre(var_name: str, factor: int) -> None:
    """Classification variables (SCL, L1C cloud bitmask) take the centre pixel
    of each window; for an even factor, the lower-right of the middle pixels."""
    size = 2 * factor
    data = np.arange(size * size, dtype="uint8").reshape(size, size)
    result = _coarsen(data, factor, var_name, fill=0)
    centre = factor // 2
    np.testing.assert_array_equal(result, data[centre::factor, centre::factor])


def test_coarsen_float_keeps_all_nan_windows_nan() -> None:
    """The float block mean skips NaN, and a window holding nothing but NaN
    stays NaN (it is nodata, not zero)."""
    data = np.array(
        [
            [1.0, 2.0, np.nan, np.nan],
            [3.0, np.nan, np.nan, np.nan],
        ],
        dtype="float32",
    )
    result = _coarsen(data, 2, "b02", fill=None)
    np.testing.assert_array_equal(result, np.array([[2.0, np.nan]], dtype="float32"))


def test_coarsen_refuses_quality_masks() -> None:
    """Quality-mask downsampling is not implemented, so it is refused."""
    data = np.zeros((4, 4), dtype="uint8")
    with pytest.raises(ValueError, match="quality_mask"):
        _coarsen(data, 2, "quality_cirrus", fill=0)


def test_write_geozarr_refuses_a_source_missing_the_planned_arrays(
    tmp_path: pathlib.Path,
) -> None:
    """A source store that lacks a planned array's data is refused."""
    input_path = create_zarrv2_group_from_json(s2_example_json_paths[0], tmp_path)
    input_group = zarr.open_group(input_path, mode="r", zarr_format=2)
    root = load_sentinel2(input_group)
    empty_source = zarr.create_group(tmp_path / "empty.zarr", zarr_format=2)
    with pytest.raises(ValueError, match="input source"):
        write_geozarr(root, empty_source, str(tmp_path / "out.zarr"))


def _store_contents(path: pathlib.Path) -> dict[str, object]:
    """Every file of a store: metadata documents as parsed JSON (their key
    order is not semantic), chunk files as content digests."""
    contents: dict[str, object] = {}
    for file in sorted(path.rglob("*")):
        if not file.is_file():
            continue
        relative = str(file.relative_to(path))
        if file.name in ("zarr.json", ".zattrs", ".zgroup", ".zarray", ".zmetadata"):
            contents[relative] = json.loads(file.read_text())
        else:
            contents[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
    return contents


@pytest.mark.parametrize("source_path", s2_example_json_paths, ids=get_stem)
def test_write_geozarr_from_a_datatree_matches_the_store_write(
    source_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """
    A DataTree source produces the same store as the Zarr-store source, chunk
    for chunk: the tree's GroupSpec-shaped view validates into the same input
    model, and the tree reader feeds the writer the same values — whether the
    tree holds the raw packed values (`mask_and_scale=False`) or the decoded
    ones, which the reader re-encodes as xarray's own writer would.
    """
    input_path = create_zarrv2_group_from_json(source_path, tmp_path)
    _seed_input_store(zarr.open_group(input_path, mode="r+", zarr_format=2))
    source = zarr.open_group(input_path, mode="r", zarr_format=2)
    store_output = tmp_path / "from_store.zarr"
    write_geozarr(load_sentinel2(source), source, str(store_output))
    reference = _store_contents(store_output)

    for mask_and_scale in (False, True):
        tree = xr.open_datatree(
            str(input_path),
            engine="zarr",
            chunks={},
            consolidated=False,
            mask_and_scale=mask_and_scale,
            decode_coords="all",
        )
        tree_output = tmp_path / f"from_tree_{mask_and_scale}.zarr"
        write_geozarr(load_sentinel2(tree), tree, str(tree_output))
        assert _store_contents(tree_output) == reference, f"{mask_and_scale=}"
