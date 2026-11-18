"""Integration tests for updated_convert_olci_optimized."""

from __future__ import annotations

import json
import os
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
import zarr
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import GroupSpec
from zarr.core.metadata import ArrayV3Metadata

# from eopf_geozarr.s3_olci_optimization.olci_converter import (
#     _sanitize_olci_array_attrs_keep_fill,
# )

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable

# from eopf_geozarr.s3_olci_optimization.olci_converter import updated_convert_olci_optimized
from eopf_geozarr.s3_olci_optimization.olci_converter import updated_convert_olci_optimized

# A 8x8 nodata block in r0 covers the coarsest pixel [0, 0] of both pyramids:
# EFR r3 (factor 8) and ERR r2 (factor 4).
_OLCI_NODATA_BLOCK = 8
_OLCI_SCALE_FACTOR = 0.013634907081723213  # oa01_radiance
_OLCI_ADD_OFFSET = 0.0
_OLCI_FILL_VALUE = 65535
_OLCI_TIE_POINT_COLUMNS = 4


def _make_minimal_olci_datatree(
    *, product_type: str, rows: int, cols: int, orphans: bool, raw: bool
) -> xr.DataTree:
    """Shared builder for `_make_minimal_olci_efr_datatree` / `_make_minimal_olci_err_datatree`."""
    rng = np.random.default_rng(0)
    swath = ("rows", "columns")

    def _radiance(shape: tuple[int, int], dims: tuple[str, str]) -> xr.DataArray:
        packed = rng.integers(0, 6000, size=shape).astype("uint16")
        if dims == swath:
            packed[:_OLCI_NODATA_BLOCK, :_OLCI_NODATA_BLOCK] = _OLCI_FILL_VALUE
        cf = {
            "scale_factor": _OLCI_SCALE_FACTOR,
            "add_offset": _OLCI_ADD_OFFSET,
            "_FillValue": _OLCI_FILL_VALUE,
        }
        attrs = {"standard_name": "toa_upwelling_spectral_radiance"}
        if raw:
            return xr.DataArray(packed, dims=dims, attrs={**attrs, **cf})
        decoded = np.where(
            packed == _OLCI_FILL_VALUE, np.nan, packed * _OLCI_SCALE_FACTOR + _OLCI_ADD_OFFSET
        )
        da = xr.DataArray(decoded, dims=dims, attrs=attrs)
        da.encoding = {**cf, "dtype": np.dtype("uint16")}
        return da

    def _latlon(n_cols: int) -> dict[str, tuple[tuple[str, str], np.ndarray, dict[str, str]]]:
        lat, lon = np.meshgrid(
            45.0 + 0.003 * np.arange(rows), 10.0 + 0.003 * np.linspace(0, cols - 1, n_cols),
            indexing="ij",
        )  # fmt: skip
        return {
            "latitude": (swath, lat, {"standard_name": "latitude"}),
            "longitude": (swath, lon, {"standard_name": "longitude"}),
        }

    measurements = xr.Dataset(
        {"oa01_radiance": _radiance((rows, cols), swath)}, coords=_latlon(cols)
    )
    quality = xr.Dataset(
        {
            "quality_flags": xr.DataArray(
                rng.integers(0, 2**31, size=(rows, cols), dtype="uint32"), dims=swath
            )
        }
    )
    # Tie-point grid: same rows, far fewer columns than the image grid.
    geometry = xr.Dataset(
        {"sza": (swath, rng.uniform(0.0, 90.0, (rows, _OLCI_TIE_POINT_COLUMNS)))},
        coords=_latlon(_OLCI_TIE_POINT_COLUMNS),
    )

    dt = xr.DataTree()
    dt.attrs = {"stac_discovery": {"properties": {"product:type": product_type}}}
    dt["measurements"] = xr.DataTree(measurements)
    dt["quality"] = xr.DataTree(quality)
    dt["conditions/geometry"] = xr.DataTree(geometry)
    if orphans:
        # Pixels removed from the image grid, per row (EFR only).
        dt["measurements/orphans"] = xr.DataTree(
            xr.Dataset(
                {"oa01_radiance": _radiance((rows, 2), ("rows", "removed_pixels"))},
                coords={"removed_pixels": np.arange(2)},
            )
        )
    return dt


def _make_minimal_olci_efr_datatree(*, raw: bool) -> xr.DataTree:
    """Build a tiny OLCI L1 EFR (full resolution) DataTree for behavior tests.

    Product type `S03OLCEFR` (FR pyramid r0..r3). A 40x48 swath, wider than
    tall like the real 4091x4865 product, with `measurements` (+ `orphans`),
    `quality` and `conditions/geometry` on a tie-point grid. One radiance band,
    packed like ESA OLCI L1 (uint16, `scale_factor`, `_FillValue` 65535), with
    a `_OLCI_NODATA_BLOCK` fill block in the top-left corner.

    `raw=True` gives the packed integers with the CF values in `.attrs`, as the
    CLI opens products (`mask_and_scale=False`). `raw=False` gives decoded
    floats with the CF values in `.encoding`.
    """
    return _make_minimal_olci_datatree(
        product_type="S03OLCEFR", rows=40, cols=48, orphans=True, raw=raw
    )


def _make_minimal_olci_err_datatree(*, raw: bool) -> xr.DataTree:
    """Build a tiny OLCI L1 ERR (reduced resolution) DataTree for behavior tests.

    Product type `S03OLCERR` (RR pyramid r0..r2). A 64x16 swath, a long narrow
    strip like the real 15105x1217 product, and no `orphans` group. Otherwise
    as `_make_minimal_olci_efr_datatree`.
    """
    return _make_minimal_olci_datatree(
        product_type="S03OLCERR", rows=64, cols=16, orphans=False, raw=raw
    )


def _assert_levels(out: str, expected_shapes: dict[str, tuple[int, int]]) -> None:
    """`measurements` holds exactly the expected levels, each a packed uint16 swath."""
    meas = zarr.open_group(out, mode="r")["measurements"]
    assert isinstance(meas, zarr.Group)
    assert {k for k in meas.group_keys() if k != "orphans"} == set(expected_shapes)
    for level, shape in expected_shapes.items():
        for name in ("oa01_radiance", "latitude", "longitude"):
            arr = meas[f"{level}/{name}"]
            assert isinstance(arr, zarr.Array)
            assert arr.shape == shape, f"{level}/{name}"
            assert arr.metadata.dimension_names == ("rows", "columns")  # type: ignore[union-attr]
            if name == "oa01_radiance":
                assert arr.dtype == np.dtype("uint16"), level


def _assert_copied(dt_in: xr.DataTree, out: str, path: str) -> None:
    """Every variable of the input group `path` is copied unchanged.

    Same dims, shape and values (decoded), and stored with the input's on-disk
    dtype (`encoding["dtype"]` for packed variables).
    """
    ds_in = dt_in[path].to_dataset()
    ds_out = xr.open_dataset(out, engine="zarr", group=path, consolidated=False)
    group = zarr.open_group(out, mode="r")[path]
    assert isinstance(group, zarr.Group)
    assert set(ds_out.variables) == set(ds_in.variables), path
    for name, var in ds_in.variables.items():
        observed = ds_out[name]
        assert observed.dims == var.dims, f"{path}/{name}"
        assert observed.dtype == var.dtype, f"{path}/{name}"
        np.testing.assert_array_equal(observed.values, var.values, err_msg=f"{path}/{name}")
        stored = group[str(name)]
        assert isinstance(stored, zarr.Array)
        assert stored.dtype == var.encoding.get("dtype", var.dtype), f"{path}/{name}"
    ds_out.close()


def test_convert_olci_efr(tmp_path: pathlib.Path) -> None:
    """EFR builds the 4-level FR pyramid and copies every other group through unchanged."""
    dt = _make_minimal_olci_efr_datatree(raw=False)
    out = str(tmp_path / "olci_geozarr.zarr")
    result = updated_convert_olci_optimized(dt, output_path=out)

    assert isinstance(result, xr.DataTree)
    assert {"/measurements/orphans", "/conditions/geometry", "/quality"} <= set(result.groups)

    _assert_levels(out, {"r0": (40, 48), "r1": (20, 24), "r2": (10, 12), "r3": (5, 6)})
    # Tie-point grid keeps its own (narrower) `columns`, orphans their `removed_pixels`.
    for path in ("measurements/orphans", "quality", "conditions/geometry"):
        _assert_copied(dt, out, path)


def test_convert_olci_err(tmp_path: pathlib.Path) -> None:
    """ERR builds the 3-level RR pyramid, copies every other group, and has no orphans."""
    dt = _make_minimal_olci_err_datatree(raw=False)
    out = str(tmp_path / "olci_geozarr.zarr")
    result = updated_convert_olci_optimized(dt, output_path=out)

    assert isinstance(result, xr.DataTree)
    assert {"/conditions/geometry", "/quality"} <= set(result.groups)
    assert "/measurements/orphans" not in result.groups

    _assert_levels(out, {"r0": (64, 16), "r1": (32, 8), "r2": (16, 4)})
    for path in ("quality", "conditions/geometry"):
        _assert_copied(dt, out, path)


def test_convert_olci_native_output_opens_as_datatree(tmp_path: object) -> None:
    """Acceptance for the default (native) mode: swath geometry, no CRS.

    The store keeps instrument geometry: r0 holds raw bands with 2-D
    lat/lon over (rows, columns) at exact input size, overview siblings
    halve it, the whole store opens with xr.open_datatree, and nothing
    fabricates a CRS (no spatial_ref, no grid_mapping, no proj: attrs).
    """
    dt = _make_minimal_olci_datatree(
        product_type="S03OLCEFR", rows=512, cols=512, orphans=False, raw=False
    )
    out = str(tmp_path / "olci_native.zarr")  # type: ignore[operator]
    result = updated_convert_olci_optimized(dt, output_path=out)

    opened = xr.open_datatree(out, engine="zarr", consolidated=False, chunks={})

    r0 = opened["/measurements/r0"].to_dataset()
    assert dict(r0.sizes) == {"rows": 512, "columns": 512}  # exact: no warp
    for level, size in (("r0", 512), ("r1", 256), ("r2", 128), ("r3", 64)):
        ds = opened[f"/measurements/{level}"].to_dataset()
        assert dict(ds.sizes) == {"rows": size, "columns": size}
        assert ds["latitude"].dims == ("rows", "columns")
        assert ds["longitude"].dims == ("rows", "columns")
        assert "spatial_ref" not in ds.variables
        assert "grid_mapping" not in ds["oa01_radiance"].attrs
        assert ds.rio.crs is None

    meas_group = zarr.open_group(out, mode="r")["measurements"]
    assert isinstance(meas_group, zarr.Group)
    meas_attrs = dict(meas_group.attrs)

    # excluding spatial: convention
    # assert meas_attrs["spatial:dimensions"] == ["rows", "columns"]

    assert not any(k.startswith("proj:") for k in meas_attrs)
    multiscales = meas_attrs["multiscales"]
    assert isinstance(multiscales, dict)
    layout = multiscales["layout"]
    assert isinstance(layout, list)

    # excluding spatial: convention
    # assert layout[0] == {"asset": "r0", "spatial:shape": [512, 512]}

    assert layout[0] == {"asset": "r0"}
    assert layout[1] == {
        "asset": "r1",
        "derived_from": "r0",
        "transform": {"scale": [2, 2], "translation": [0.0, 0.0]},
    }

    r0_group = meas_group["r0"]
    assert isinstance(r0_group, zarr.Group)
    r0_attrs = dict(r0_group.attrs)

    # excluding spatial: convention
    # assert r0_attrs["spatial:dimensions"] == ["rows", "columns"]

    assert not any(k.startswith("proj:") for k in r0_attrs)

    assert "/measurements/r0" in result.groups
    assert "/measurements/r1" in result.groups


def test_convert_olci_invalid_output_grid_raises(tmp_path: object) -> None:
    dt = _make_minimal_olci_err_datatree(raw=False)
    out = str(tmp_path / "bad.zarr")  # type: ignore[operator]
    with pytest.raises(ValueError, match="need native output_grid"):
        updated_convert_olci_optimized(dt, output_path=out, output_grid="4326")


@pytest.mark.parametrize(
    ("make_datatree", "levels", "orphans"),
    [
        (_make_minimal_olci_efr_datatree, ["r0", "r1", "r2", "r3"], True),
        (_make_minimal_olci_err_datatree, ["r0", "r1", "r2"], False),
    ],
    ids=("efr", "err"),
)
def test_convert_olci_layout_and_consolidation(
    tmp_path: pathlib.Path,
    make_datatree: Callable[..., xr.DataTree],
    levels: list[str],
    orphans: bool,
) -> None:
    """The multiscales layout chains the levels, and `measurements` is consolidated.

    Each overview is derived from the level before it by a 2x reduction, and
    each halves the previous level's shape (floor). `measurements` and the root each carry a
    consolidated block, so a reader can open `measurements` without listing it
    (#291, #303).
    """
    out = tmp_path / "olci_geozarr.zarr"
    updated_convert_olci_optimized(make_datatree(raw=False), output_path=str(out))

    def meta(path: str) -> Any:
        return json.loads((out / path / "zarr.json").read_text())

    multiscales = meta("measurements")["attributes"]["multiscales"]
    assert multiscales["resampling_method"] == "average"
    layout = multiscales["layout"]
    assert [entry["asset"] for entry in layout] == levels
    assert layout[0] == {"asset": "r0"}
    for parent, entry in pairwise(layout):
        assert entry == {
            "asset": entry["asset"],
            "derived_from": parent["asset"],
            "transform": {"scale": [2, 2], "translation": [0.0, 0.0]},
        }

    shapes = [meta(f"measurements/{level}/oa01_radiance")["shape"] for level in levels]
    for previous, current in pairwise(shapes):
        assert current == [previous[0] // 2, previous[1] // 2]

    for path in ("", "measurements"):
        assert meta(path)["consolidated_metadata"] is not None, f"/{path}: no consolidated block"
    opened = xr.open_datatree(
        str(out / "measurements"), engine="zarr", consolidated=True, chunks={}
    )
    assert set(opened.children) == set(levels) | ({"orphans"} if orphans else set())


def _assert_radiance_dtype_and_attrs(
    group: zarr.Group, band_name: str, *, level_label: str
) -> None:
    """Assert that *band_name* in *group* is uint16 with CF packing and no stale attrs.

    This is a regression guard: the converter must preserve raw integer storage,
    CF scale/offset and the 65535 fill value, and must strip the source-only
    attrs `_eopf_attrs` and `dtype`. `valid_min`/`valid_max` are kept: the
    output stays packed, so they are still valid in the raw domain.
    """
    band = group[band_name]
    assert isinstance(band, zarr.Array), f"{level_label}/{band_name} is not a zarr.Array"
    assert band.dtype == np.dtype("uint16"), (
        f"{level_label}/{band_name}: expected uint16, got {band.dtype}"
    )
    attrs = dict(band.attrs)
    assert "scale_factor" in attrs, (
        f"{level_label}/{band_name}: scale_factor missing from attrs (got {list(attrs)})"
    )
    assert attrs.get("_FillValue") == _OLCI_FILL_VALUE, f"{level_label}/{band_name}"
    assert band.fill_value == _OLCI_FILL_VALUE, f"{level_label}/{band_name}"
    for stale_key in ("_eopf_attrs", "dtype"):
        assert stale_key not in attrs, (
            f"{level_label}/{band_name}: stale attr '{stale_key}' present in output attrs"
        )


def _assert_json_string_similarity_geozarrversion(
    observed: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    # Compare as JSON: a NaN fill value never equals itself as a float.
    def canonical(node: object) -> str:
        return json.dumps(node.model_dump(), sort_keys=True)  # type: ignore[attr-defined]

    # catch differences in root stac_discovery such as EOPF_GEOZARR version which is commit-dependant
    different_keys = [
        key for key in observed if canonical(observed[key]) != canonical(expected[key])
    ]
    filtered_different_keys = []

    for key in different_keys:
        cobs, cexp = canonical(observed[key]), canonical(expected[key])
        if key == "" and "EOPF_GEOZARR" in cobs and "EOPF_GEOZARR" in cexp:
            import re

            recobs = re.sub(
                r'"EOPF_GEOZARR"\s*:\s*"[^"]*"\s*,\s*',
                "",
                cobs,
            )
            recexp = re.sub(
                r'"EOPF_GEOZARR"\s*:\s*"[^"]*"\s*,\s*',
                "",
                cexp,
            )

            if recobs != recexp:
                filtered_different_keys.append(key)

    assert filtered_different_keys == []


def _assert_olci_conversion_matches_snapshot(
    group_path: pathlib.Path, tmp_path: pathlib.Path, expected_levels: list[str]
) -> None:
    """Convert the OLCI store at `group_path` and compare it with its golden file.

    We open the whole DataTree raw, as the CLI does, so that conditions,
    quality, and measurements/orphans subgroups are included in the conversion.
    The golden file is `<stem>-olc-native.json`; `<stem>-native.json` and
    `<stem>-epsg4326.json` belong to the old converter
    (`test_old_olci_integration.py`).

    Any change to array metadata (dtype, fill value, chunks, codecs, attributes)
    or to the set of arrays shows up here. When a change is intended, regenerate
    the snapshot with `REGENERATE_SNAPSHOTS=1 pytest -k snapshot` and review
    the JSON diff.
    """
    dt_in = xr.open_datatree(
        str(group_path),
        engine="zarr",
        consolidated=False,
        chunks={},
        mask_and_scale=False,
    )
    out = str(tmp_path / "out.zarr")
    updated_convert_olci_optimized(dt_in, output_path=out, enable_sharding=True)

    # Each consolidated block is a snapshot, and GroupSpec.from_zarr below reads the
    # levels through the `measurements` block, so a stale block would hide a late
    # write. Re-consolidating a current store rewrites the same bytes (#303).
    for path in ("measurements", ""):
        meta = Path(out, path, "zarr.json")
        before = meta.read_bytes()
        assert json.loads(before).get("consolidated_metadata"), f"/{path}: no block"
        zarr.consolidate_metadata(out, path=path or None, zarr_format=3)
        assert meta.read_bytes() == before, f"/{path}: consolidated block is stale"

    observed_group = zarr.open_group(out, use_consolidated=False)
    observed_structure_json = GroupSpec.from_zarr(observed_group).model_dump()

    expected_path = Path("tests/_test_data/optimized_olci_examples") / (
        f"{group_path.stem}-olc-native.json"
    )

    if os.environ.get("REGENERATE_SNAPSHOTS"):
        expected_path.write_text(
            json.dumps(observed_structure_json, indent=2, sort_keys=True) + "\n"
        )

    observed = GroupSpec(**tuplify_json(observed_structure_json)).to_flat()
    expected = GroupSpec(**tuplify_json(json.loads(expected_path.read_text()))).to_flat()
    assert set(observed) == set(expected)

    _assert_json_string_similarity_geozarrversion(observed, expected)

    # Dtype/attrs regression guard on every level: radiance must be stored as
    # uint16 with CF packing preserved and stale source attrs absent.
    meas_g = observed_group["measurements"]
    assert isinstance(meas_g, zarr.Group)
    levels = sorted(k for k in meas_g.group_keys() if k != "orphans")
    assert levels == expected_levels
    for level in levels:
        level_g = meas_g[level]
        assert isinstance(level_g, zarr.Group)
        _assert_radiance_dtype_and_attrs(level_g, "oa01_radiance", level_label=level)


def test_olci_efr_conversion_matches_snapshot(
    s3_olci_efr_v3_group_example: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Snapshot test: converted OLCI EFR structure must match committed golden file.

    The fixture is a Zarr v3 store built from a real CPM OLCI L1 EFR product
    (`product:type` S03OLCEFR), shrunk to test sizes, so the 16x16 measurements
    grid gets the FR pyramid r0..r3 (16, 8, 4, 2). The fixture uses the real
    product's dimension names: tie-point grids reuse 'columns' (at size 4) while
    measurement grids also use 'columns' (at size 16); orphan arrays use
    removed_pixels=4.
    """
    _assert_olci_conversion_matches_snapshot(
        s3_olci_efr_v3_group_example, tmp_path, ["r0", "r1", "r2", "r3"]
    )


def test_olci_err_conversion_matches_snapshot(
    s3_olci_err_v3_group_example: pathlib.Path,
    tmp_path: pathlib.Path,
) -> None:
    """Snapshot test: converted OLCI ERR structure must match committed golden file.

    The fixture is a Zarr v3 store built from a real CPM OLCI L1 ERR product
    (`product:type` S03OLCERR), shrunk to test sizes, so the 16x16 measurements
    grid gets the RR pyramid r0..r2 (16, 8, 4). Unlike EFR, ERR has no orphans.
    Tie-point grids reuse 'columns' (at size 4) as in EFR.
    """
    _assert_olci_conversion_matches_snapshot(
        s3_olci_err_v3_group_example, tmp_path, ["r0", "r1", "r2"]
    )


def test_convert_olci_odd_dims_overview_no_conflicting_sizes(tmp_path: pathlib.Path) -> None:
    """Overview groups written from an odd-dimensioned swath must open without errors.

    Regression test for the off-by-one bug in reduce_swath where coordinate
    decimation via [::factor] produced ceil(N/factor) elements while
    coarsen(boundary="trim") produced floor(N/factor) for the radiance data.
    On an odd-column real OLCI product (4865 cols) this caused xr.open_dataset
    to raise ``ValueError: conflicting sizes for dimension 'columns'``.

    47x55 is odd in both dims at every level that gets reduced (r0, r1, r2), so
    each of the three FR reductions has to trim.
    """
    dt = _make_minimal_olci_datatree(
        product_type="S03OLCEFR", rows=47, cols=55, orphans=False, raw=False
    )
    out = str(tmp_path / "odd_olci.zarr")
    updated_convert_olci_optimized(dt, output_path=out)

    expected = {"r0": (47, 55), "r1": (23, 27), "r2": (11, 13), "r3": (5, 6)}
    for level, shape in expected.items():
        # Must not raise a conflicting-sizes error.
        with xr.open_dataset(
            out, engine="zarr", group=f"measurements/{level}", consolidated=False
        ) as ds:
            assert (ds.sizes["rows"], ds.sizes["columns"]) == shape, level
            for name in ("oa01_radiance", "latitude", "longitude"):
                assert ds[name].shape == shape, f"{level}/{name}"


_per_product = pytest.mark.parametrize(
    "make_datatree",
    [_make_minimal_olci_efr_datatree, _make_minimal_olci_err_datatree],
    ids=("efr", "err"),
)


def _arrays(out: str) -> dict[str, zarr.Array]:
    """Every array in the store at `out`, by path."""
    root = zarr.open_group(out, mode="r", use_consolidated=False)
    return {
        path: node for path, node in root.members(max_depth=None) if isinstance(node, zarr.Array)
    }


def _structure(out: str) -> dict[str, str]:
    """Flattened store metadata as canonical JSON (a NaN fill value never equals itself)."""
    flat = GroupSpec.from_zarr(zarr.open_group(out, mode="r", use_consolidated=False)).to_flat()
    return {path: json.dumps(node.model_dump(), sort_keys=True) for path, node in flat.items()}


def _assert_same_values(out_a: str, out_b: str) -> None:
    """Both stores hold the same arrays with identical stored values."""
    arrays_a, arrays_b = _arrays(out_a), _arrays(out_b)
    assert set(arrays_a) == set(arrays_b)
    for path, arr in arrays_a.items():
        np.testing.assert_array_equal(arr[:], arrays_b[path][:], err_msg=path)


def _codec_names(arr: zarr.Array) -> list[str]:
    """Codec class names, including those nested in a sharding codec."""
    assert isinstance(arr.metadata, ArrayV3Metadata)
    names: list[str] = []
    for codec in arr.metadata.codecs:
        names.append(type(codec).__name__)
        names.extend(type(inner).__name__ for inner in getattr(codec, "codecs", ()))
    return names


@_per_product
def test_convert_olci_raw_and_decoded_inputs_match(
    tmp_path: pathlib.Path, make_datatree: Callable[..., xr.DataTree]
) -> None:
    """Raw (`mask_and_scale=False`, as the CLI opens) and decoded input give the same store.

    The raw form carries `scale_factor`/`add_offset`/`_FillValue` in `.attrs`,
    the decoded form in `.encoding` with NaN for nodata. Both must end up as the
    same packed uint16 arrays with the same metadata, at every level and in
    every copied group.
    """
    raw_out = str(tmp_path / "raw.zarr")
    decoded_out = str(tmp_path / "decoded.zarr")
    updated_convert_olci_optimized(make_datatree(raw=True), output_path=raw_out)
    updated_convert_olci_optimized(make_datatree(raw=False), output_path=decoded_out)

    raw_structure, decoded_structure = _structure(raw_out), _structure(decoded_out)
    assert set(raw_structure) == set(decoded_structure)
    assert [k for k in raw_structure if raw_structure[k] != decoded_structure[k]] == []
    _assert_same_values(raw_out, decoded_out)


@_per_product
def test_convert_olci_swath_has_no_spatial_or_proj_metadata(
    tmp_path: pathlib.Path, make_datatree: Callable[..., xr.DataTree]
) -> None:
    """A swath has no affine transform or CRS; it is geolocated by its 2-D lat/lon arrays.

    So no group or array under `measurements` carries `spatial:` or `proj:`
    attrs, multiscales is the only convention declared on `measurements`, and
    the level groups carry no attrs at all. (The root is not checked: real
    products bring their own root `proj:code`/`spatial:bbox`.)
    """
    out = str(tmp_path / "olci_geozarr.zarr")
    updated_convert_olci_optimized(make_datatree(raw=False), output_path=out)

    meas = zarr.open_group(out, path="measurements", mode="r")
    nodes = [("measurements", meas)] + [
        (f"measurements/{path}", node) for path, node in meas.members(max_depth=None)
    ]
    offenders = [
        f"{path}: {key}"
        for path, node in nodes
        for key in node.attrs
        if key.startswith(("spatial:", "proj:"))
    ]
    assert offenders == []

    conventions = meas.attrs["zarr_conventions"]
    assert isinstance(conventions, list)
    assert len(conventions) == 1
    assert "multiscales" in conventions[0]["schema_url"]  # type: ignore[index,call-overload]

    levels = [k for k in meas.group_keys() if k != "orphans"]
    assert levels
    for level in levels:
        assert dict(meas[level].attrs) == {}, level


@pytest.mark.parametrize("spatial_chunk", [16, None], ids=("chunk16", "default"))
def test_convert_olci_spatial_chunk_applied(
    tmp_path: pathlib.Path, spatial_chunk: int | None
) -> None:
    """Every array is chunked by `spatial_chunk` per dim, capped at the dim size, unsharded.

    Covers the pyramid levels (r3 at 5x6 is smaller than the chunk), the
    orphans (`removed_pixels`=2) and the tie-point grid (4 columns). The
    default (1024) exceeds the mock, so every array is a single chunk.
    """
    out = str(tmp_path / "olci_geozarr.zarr")
    kwargs = (
        {}
        if spatial_chunk is None
        else {"spatial_chunk": spatial_chunk, "enable_sharding": False, "scale_offset_codec": False}
    )
    updated_convert_olci_optimized(
        _make_minimal_olci_efr_datatree(raw=False), output_path=out, **kwargs
    )

    chunk = spatial_chunk or 1024
    arrays = _arrays(out)
    assert "measurements/r3/oa01_radiance" in arrays
    for path, arr in arrays.items():
        assert arr.chunks == tuple(min(size, chunk) for size in arr.shape), path
        assert arr.shards is None, path


def test_convert_olci_enable_sharding(tmp_path: pathlib.Path) -> None:
    """With `enable_sharding`, every array is sharded around `spatial_chunk` inner chunks.

    Inner chunks are `spatial_chunk` capped at the dim size; shards cover the
    array rounded up to whole inner chunks. That includes the 2-D lat/lon of
    every pyramid level and copied group; only 1-D dimension coordinates
    (e.g. `removed_pixels`) stay unsharded. Sharding only changes the layout:
    the stored values equal those of the unsharded conversion.
    """
    sharded = str(tmp_path / "sharded.zarr")
    plain = str(tmp_path / "plain.zarr")
    dt = _make_minimal_olci_efr_datatree(raw=False)
    updated_convert_olci_optimized(dt, output_path=sharded, spatial_chunk=16, enable_sharding=True)
    updated_convert_olci_optimized(dt, output_path=plain, spatial_chunk=16)

    unsharded: list[str] = []
    for path, arr in sorted(_arrays(sharded).items()):
        inner = tuple(min(size, 16) for size in arr.shape)
        assert arr.chunks == inner, path
        name = path.rsplit("/", 1)[-1]
        if arr.metadata.dimension_names == (name,):  # type: ignore[union-attr]
            continue
        if "ShardingCodec" not in _codec_names(arr):
            unsharded.append(path)
            continue
        assert arr.shards == tuple(
            -(-size // c) * c for size, c in zip(arr.shape, inner, strict=True)
        ), path
    assert unsharded == []
    _assert_same_values(sharded, plain)


def test_no_acceptance_of_scale_offset_codec(tmp_path: object) -> None:
    dt = _make_minimal_olci_err_datatree(raw=False)
    out = str(tmp_path / "bad_scale_offset_codec.zarr")  # type: ignore[operator]
    with pytest.raises(NotImplementedError):
        updated_convert_olci_optimized(dt, output_path=out, scale_offset_codec=True)
    assert not Path(out).exists()
