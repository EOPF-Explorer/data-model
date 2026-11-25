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
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import xarray as xr
import zarr
from pydantic_zarr.v3 import GroupSpec as GroupSpecV3
from zarr.storage import MemoryStore

from eopf_geozarr.data_api.s2_plan import WritePolicy, load_sentinel2
from eopf_geozarr.data_api.s2_write import (
    DataTreeSourceReader,
    ZarrSourceReader,
    _coarsen,
    _fill_carried_array,
    write_geozarr,
)
from eopf_geozarr.zarr_io import read_documents

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
    "quality/probability/r20m/cld",
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

    Coordinates are the pixel centres of each group's own `proj:transform`.
    They are recreated as float64 so that the centres made up for groups
    without a transform are stored exactly; the int64 coordinates the
    fixtures declare are covered by the plan's converter comparison.
    Latitude/longitude form a regular geographic grid, and selected arrays
    get deterministic values.
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
            elif name in ("latitude", "longitude") and array.ndim == 1:
                # a regular geographic grid, as the CAMS/ECMWF meteorology
                # groups store it (latitude decreasing), from which the plan
                # derives the groups' georeferencing
                start, stop = (48.0, 46.0) if name == "latitude" else (8.0, 10.0)
                array[...] = np.linspace(start, stop, array.shape[0]).astype(array.dtype)
            elif array.ndim == 2 and np.issubdtype(array.dtype, np.integer):
                if array.size <= _PATTERN_MAX_ELEMENTS or path in patterned:
                    array[...] = _pattern(array.shape, array.dtype)


def _written_store(
    source_path: pathlib.Path, tmp_path: pathlib.Path, policy: WritePolicy | None = None
) -> tuple[AnyGroupSpec, zarr.Group, zarr.Group]:
    """Seed the fixture store, write it with `write_geozarr`, open both."""
    input_path = create_zarrv2_group_from_json(source_path, tmp_path)
    _seed_input_store(zarr.open_group(input_path, mode="r+", zarr_format=2))
    source = zarr.open_group(input_path, mode="r", zarr_format=2)
    output_path = tmp_path / "planned.zarr"
    plan = write_geozarr(load_sentinel2(source), source, str(output_path), policy or WritePolicy())
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


def _reference_plain_mean(values: np.ndarray, factor: int) -> np.ndarray:
    """The block mean of every value, rounded: no value is nodata."""
    rows = values.shape[0] // factor * factor
    cols = values.shape[1] // factor * factor
    windows = values[:rows, :cols].reshape(rows // factor, factor, cols // factor, factor)
    return np.round(windows.astype("float64").mean(axis=(1, 3))).astype(values.dtype)


@pytest.mark.parametrize("source_path", s2_example_json_paths, ids=get_stem)
def test_write_geozarr_realizes_the_plan(source_path: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """
    The written store holds exactly the planned hierarchy: reading it back
    gives documents equal to the plan, for every group and array.
    """
    plan, _, output = _written_store(source_path, tmp_path)
    observed = GroupSpecV3.from_stored_metadata_json(read_documents(output)).model_dump(mode="json")
    assert observed == plan.model_dump(mode="json")
    # the meteorology groups are georeferenced from the coordinate values the
    # writer read from the source (#353): the seeded 9-point regular grid
    meteorology = observed["members"]["conditions"]["members"]["meteorology"]["members"]
    for group in ("cams", "ecmwf"):
        attributes = meteorology[group]["attributes"]
        assert attributes["proj:code"] == "EPSG:4326"
        assert attributes["spatial:transform"] == [0.25, 0.0, 8.0, 0.0, -0.25, 48.0]
    # the geometry grid is georeferenced at its nodes from the x/y values the writer
    # read from the source (EOPF-Explorer/data-model#346): the seeded pixel centres
    geometry = observed["members"]["conditions"]["members"]["geometry"]["attributes"]
    res = _TILE_EXTENT / 23
    assert geometry["spatial:registration"] == "node"
    assert geometry["spatial:transform"] == pytest.approx(
        [res, 0.0, _TILE_X_MIN + 0.5 * res, 0.0, -res, _TILE_Y_MAX - 0.5 * res]
    )


def test_write_geozarr_data_semantics(tmp_path: pathlib.Path) -> None:
    """
    The written chunk data follows the writer's semantics, checked on seeded
    values: native arrays are copied, injected bands are the masked block
    mean of the finest input level holding them, generated levels chain from
    the previous level (mean for bands and probabilities, window-centre pixel
    for the classification), coordinates take window means, and no
    grid-mapping array is written. The mean skips only the nodata the input declares: the
    bands declare 0, the probabilities and atmosphere arrays declare none, so
    their zeros (a valid probability) count.
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

    # a generated level of an array that declares no nodata is the plain block
    # mean of the previous level, zeros included
    for path, name, finer, coarser, factor in (
        ("quality/probability", "cld", "r20m", "r60m", 3),
        ("quality/atmosphere", "aot", "r60m", "r120m", 2),
    ):
        previous = read(output, f"{path}/{finer}/{name}")
        assert (previous == 0).any(), "fixture precondition: the seeded values hold zeros"
        plain = _reference_plain_mean(previous, factor)
        assert not np.array_equal(plain, _reference_block_mean(previous, factor, 0))
        np.testing.assert_array_equal(read(output, f"{path}/{coarser}/{name}"), plain)

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

    # no CF grid-mapping array is written: proj:code is the CRS (#352)
    assert "spatial_ref" not in output["measurements/reflectance/r120m"]


def test_write_geozarr_scale_offset_codec_stores_the_esa_integers(tmp_path: pathlib.Path) -> None:
    """
    Under the scale_offset codec every reflectance band, derived levels
    included, packs to the integers the ESA layout stores: the resampling
    runs on the packed integers, and only the stored form differs. Nodata
    reads as NaN.
    """
    _, _, esa = _written_store(_L2A_PATH, tmp_path / "esa")
    _, _, decoded = _written_store(
        _L2A_PATH, tmp_path / "codec", WritePolicy(scale_offset_codec=True)
    )
    reflectance = esa["measurements/reflectance"]
    assert isinstance(reflectance, zarr.Group)
    bands = [
        f"measurements/reflectance/{level}/{band}"
        for level, group in reflectance.groups()
        for band, member in group.arrays()
        if band.startswith("b")
    ]
    assert len(bands) == 63
    for path in bands:
        packed, codec = esa[path], decoded[path]
        assert isinstance(packed, zarr.Array)
        assert isinstance(codec, zarr.Array)
        assert codec.dtype == np.float32
        integers = np.asarray(packed[...]).astype("int64")
        values = np.asarray(codec[...]).astype("float64")
        np.testing.assert_array_equal(np.isnan(values), integers == 0)
        repacked = np.rint((np.nan_to_num(values, nan=-0.1) + 0.1) * 10000).astype("int64")
        np.testing.assert_array_equal(repacked, integers, err_msg=path)


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


def test_write_geozarr_refuses_values_that_do_not_cast_safely(tmp_path: pathlib.Path) -> None:
    """A band whose input data type does not cast safely to the planned one
    (float32 reflectance, planned as uint16) is refused, not cast silently."""
    document = json.loads(_L2A_PATH.read_text())
    reflectance = document["members"]["measurements"]["members"]["reflectance"]
    reflectance["members"]["r10m"]["members"]["b02"]["dtype"] = "<f4"
    document_path = tmp_path / _L2A_PATH.name
    document_path.write_text(json.dumps(document))
    input_path = create_zarrv2_group_from_json(document_path, tmp_path)
    _seed_input_store(zarr.open_group(input_path, mode="r+", zarr_format=2))
    source = zarr.open_group(input_path, mode="r", zarr_format=2)
    with pytest.raises(ValueError, match="does not cast safely"):
        write_geozarr(load_sentinel2(source), source, str(tmp_path / "out.zarr"))


def test_write_geozarr_refuses_an_output_that_differs_from_the_plan(
    tmp_path: pathlib.Path,
) -> None:
    """An output location that already holds a different hierarchy is refused
    with an error that says to remove it."""
    input_path = create_zarrv2_group_from_json(_L2A_PATH, tmp_path)
    _seed_input_store(zarr.open_group(input_path, mode="r+", zarr_format=2))
    source = zarr.open_group(input_path, mode="r", zarr_format=2)
    output_path = tmp_path / "out.zarr"
    zarr.create_group(output_path, attributes={"written": "before"})
    with pytest.raises(FileExistsError, match="remove it before converting"):
        write_geozarr(load_sentinel2(source), source, str(output_path))


def test_write_geozarr_rewrites_an_output_that_holds_the_plan(tmp_path: pathlib.Path) -> None:
    """An output location that already holds exactly the planned hierarchy (a
    rerun into the same output) is reused, and the store it ends with equals a
    fresh write."""
    _written_store(_L2A_PATH, tmp_path)
    output_path = tmp_path / "planned.zarr"
    reference = _store_contents(output_path)
    source = zarr.open_group(tmp_path / (_L2A_PATH.stem + ".zarr"), mode="r", zarr_format=2)
    write_geozarr(load_sentinel2(source), source, str(output_path))
    assert _store_contents(output_path) == reference


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


class _LabelSource:
    """A `SourceReader` holding one bytes label array, as native products do."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def values(self, path: str) -> np.ndarray | None:
        return self._values if path == "conditions/geometry/band" else None

    def nodata(self, path: str) -> int | None:
        return None

    def is_group(self, path: str) -> bool:
        return False

    def group_names(self, path: str) -> tuple[str, ...]:
        return ()


def test_fill_decodes_bytes_labels_into_a_string_array() -> None:
    """
    A bytes (`S`) label array, which the plan stores as the `string` data type
    (#356), is written decoded as ASCII: it reads back as the label strings.
    """
    output = zarr.open_group(MemoryStore(), mode="w", zarr_format=3)
    output.require_group("conditions/geometry").create_array(
        "band", shape=(3,), dtype=np.dtypes.StringDType(), fill_value=""
    )
    labels = np.array([b"b02", b"b03", b"b8a"], dtype="S3")
    _fill_carried_array(_LabelSource(labels), output, "conditions/geometry/band")
    band = output["conditions/geometry/band"]
    assert isinstance(band, zarr.Array)
    assert np.asarray(band[:]).tolist() == ["b02", "b03", "b8a"]


@pytest.mark.parametrize(
    ("fill_value", "attributes", "expected"),
    [
        (0, {}, 0),
        (None, {"_FillValue": 255}, 255),
        (None, {"fill_value": 0}, 0),
        (7, {"_FillValue": 255, "fill_value": 0}, 7),
        (None, {"_FillValue": 255, "fill_value": 0}, 255),
        (None, {}, None),
        (None, {"fill_value": "NaN"}, None),
    ],
    ids=[
        "zarr-fill-value",
        "cf-attribute",
        "eopf-attribute",
        "zarr-fill-value-first",
        "cf-before-eopf",
        "undeclared",
        "not-a-number",
    ],
)
def test_source_readers_report_the_declared_nodata(
    fill_value: int | None, attributes: dict[str, Any], expected: int | None
) -> None:
    """
    Both source readers report the nodata an input array declares, in the
    converter's order: the CF `_FillValue` (a Zarr V2 array's `fill_value`,
    xarray's encoding, or the attribute), then the EOPF `fill_value`
    attribute; an array that declares none has none.
    """
    group = zarr.open_group({}, mode="w", zarr_format=2)
    group.create_array(
        "level/var", shape=(2, 2), dtype="uint8", fill_value=fill_value
    ).attrs.update(attributes)
    assert ZarrSourceReader(group).nodata("level/var") == expected

    variable = xr.DataArray(np.zeros((2, 2), dtype="uint8"), dims=["y", "x"], attrs=attributes)
    if fill_value is not None:
        variable.encoding["_FillValue"] = fill_value
    tree = xr.DataTree.from_dict({"level": xr.Dataset({"var": variable})})
    assert DataTreeSourceReader(tree).nodata("level/var") == expected
