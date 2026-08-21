"""Tests for the GeoZarr minispec store validator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import zarr
from zarr_cm import geo_proj as geo_proj_cm
from zarr_cm import multiscales as multiscales_cm
from zarr_cm import spatial as spatial_cm

from eopf_geozarr.data_api.geozarr.validation import validate_store

if TYPE_CHECKING:
    import pathlib

SPATIAL_CMO = dict(spatial_cm.CMO)
PROJ_CMO = dict(geo_proj_cm.CMO)
MULTISCALES_CMO = dict(multiscales_cm.CMO)


def _level_attrs(shape: tuple[int, int], transform: list[float]) -> dict[str, Any]:
    return {
        "zarr_conventions": [SPATIAL_CMO, PROJ_CMO],
        "proj:code": "EPSG:32632",
        "spatial:dimensions": ["y", "x"],
        "spatial:shape": list(shape),
        "spatial:transform": transform,
        "spatial:bbox": [600000.0, 5090000.0, 600000.0 + 10 * shape[1], 5090000.0 + 10 * shape[0]],
        "spatial:registration": "pixel",
    }


def build_compliant_store(path: pathlib.Path) -> str:
    """Create a minimal store satisfying every minispec requirement."""
    store_path = str(path / "compliant.zarr")
    root = zarr.open_group(store_path, mode="w", zarr_format=3)
    root_attrs: dict[str, Any] = {
        "zarr_conventions": [SPATIAL_CMO, PROJ_CMO],
        "spatial:bbox": [6.0, 44.0, 8.0, 46.0],
        "proj:code": "EPSG:4326",
    }
    root.attrs.update(root_attrs)
    ms = root.create_group("measurements")
    ms_attrs: dict[str, Any] = {
        "zarr_conventions": [MULTISCALES_CMO, SPATIAL_CMO, PROJ_CMO],
        "proj:code": "EPSG:32632",
        "spatial:dimensions": ["y", "x"],
        "spatial:bbox": [600000.0, 5090000.0, 600160.0, 5090160.0],
        "spatial:registration": "pixel",
        "multiscales": {
            "layout": [
                {
                    "asset": "r10m",
                    "spatial:shape": [16, 16],
                    "spatial:transform": [10.0, 0.0, 600000.0, 0.0, -10.0, 5090160.0],
                },
                {
                    "asset": "r20m",
                    "derived_from": "r10m",
                    "transform": {"scale": [2.0, 2.0], "translation": [0.0, 0.0]},
                    "spatial:shape": [8, 8],
                    "spatial:transform": [20.0, 0.0, 600000.0, 0.0, -20.0, 5090160.0],
                },
            ],
            "resampling_method": "average",
        },
    }
    ms.attrs.update(ms_attrs)
    for name, n, res in (("r10m", 16, 10.0), ("r20m", 8, 20.0)):
        level = ms.create_group(name)
        level.attrs.update(_level_attrs((n, n), [res, 0.0, 600000.0, 0.0, -res, 5090160.0]))
        arr = level.create_array("b02", shape=(n, n), dtype="uint16", dimension_names=("y", "x"))
        arr[:] = np.zeros((n, n), dtype="uint16")
        for coord in ("y", "x"):
            coord_arr = level.create_array(
                coord, shape=(n,), dtype="float64", dimension_names=(coord,)
            )
            coord_arr[:] = np.arange(n, dtype="float64")
    return store_path


@pytest.fixture
def compliant_store(tmp_path: pathlib.Path) -> str:
    return build_compliant_store(tmp_path)


def _root_attrs(store_path: str) -> zarr.Group:
    return zarr.open_group(store_path, mode="r+")


def test_compliant_store_passes(compliant_store: str) -> None:
    """A store meeting every minispec requirement validates without issues."""
    report = validate_store(compliant_store)
    assert report.compliant, [str(i) for i in report.issues]
    assert report.roles["/"] == ["store-root"]
    assert "multiscale-dataset" in report.roles["/measurements"]
    # level groups are datasets
    assert "dataset" in report.roles["/measurements/r10m"]
    assert "dataset" in report.roles["/measurements/r20m"]


def test_missing_root_conventions(compliant_store: str) -> None:
    root = _root_attrs(compliant_store)
    del root.attrs["zarr_conventions"]
    report = validate_store(compliant_store)
    assert any("zarr_conventions" in i.message and i.path == "/" for i in report.issues)


def test_missing_root_bbox(compliant_store: str) -> None:
    root = _root_attrs(compliant_store)
    del root.attrs["spatial:bbox"]
    report = validate_store(compliant_store)
    assert any("spatial:bbox" in i.message and i.path == "/" for i in report.issues)


def test_missing_root_crs(compliant_store: str) -> None:
    root = _root_attrs(compliant_store)
    del root.attrs["proj:code"]
    report = validate_store(compliant_store)
    assert any("CRS" in i.message and i.path == "/" for i in report.issues)


def test_multiple_root_crs_allowed(compliant_store: str) -> None:
    """The minispec requires at least one CRS key; redundant encodings are fine."""
    root = _root_attrs(compliant_store)
    root.attrs["proj:wkt2"] = "GEOGCRS[...]"
    report = validate_store(compliant_store)
    assert report.compliant, [str(i) for i in report.issues]


def test_malformed_root_bbox(compliant_store: str) -> None:
    root = _root_attrs(compliant_store)
    root.attrs["spatial:bbox"] = [8.0, 44.0, 6.0, 46.0]  # xmax before xmin
    report = validate_store(compliant_store)
    assert any("xmin" in i.message and i.path == "/" for i in report.issues)


def test_multiscale_missing_bbox(compliant_store: str) -> None:
    ms = zarr.open_group(compliant_store, mode="r+")["measurements"]
    del ms.attrs["spatial:bbox"]
    report = validate_store(compliant_store)
    assert any("spatial:bbox" in i.message and i.path == "/measurements" for i in report.issues)


def test_multiscale_layout_entry_missing_transform(compliant_store: str) -> None:
    ms = zarr.open_group(compliant_store, mode="r+")["measurements"]
    multiscales = dict(cast("dict[str, Any]", ms.attrs["multiscales"]))
    layout = [dict(entry) for entry in cast("list[dict[str, Any]]", multiscales["layout"])]
    del layout[0]["spatial:transform"]
    multiscales["layout"] = layout
    ms.attrs["multiscales"] = multiscales
    report = validate_store(compliant_store)
    assert any(
        "spatial:transform" in i.message and i.path == "/measurements" for i in report.issues
    )


def test_multiscale_layout_derived_from_without_transform(compliant_store: str) -> None:
    ms = zarr.open_group(compliant_store, mode="r+")["measurements"]
    multiscales = dict(cast("dict[str, Any]", ms.attrs["multiscales"]))
    layout = [dict(entry) for entry in cast("list[dict[str, Any]]", multiscales["layout"])]
    del layout[1]["transform"]
    multiscales["layout"] = layout
    ms.attrs["multiscales"] = multiscales
    report = validate_store(compliant_store)
    assert any(
        "'transform' is required" in i.message and i.path == "/measurements" for i in report.issues
    )


def test_multiscale_layout_asset_unresolvable(compliant_store: str) -> None:
    ms = zarr.open_group(compliant_store, mode="r+")["measurements"]
    multiscales = dict(cast("dict[str, Any]", ms.attrs["multiscales"]))
    layout = [dict(entry) for entry in cast("list[dict[str, Any]]", multiscales["layout"])]
    layout[1]["asset"] = "r40m"  # no such member
    multiscales["layout"] = layout
    ms.attrs["multiscales"] = multiscales
    report = validate_store(compliant_store)
    assert any("does not resolve" in i.message and i.path == "/measurements" for i in report.issues)


def test_multiscale_level_not_a_dataset(compliant_store: str) -> None:
    level = zarr.open_group(compliant_store, mode="r+")["measurements/r20m"]
    for key in list(level.attrs):
        del level.attrs[key]
    report = validate_store(compliant_store)
    assert any(i.path == "/measurements/r20m" for i in report.issues)


def test_empty_store_has_no_datasets(tmp_path: pathlib.Path) -> None:
    store_path = str(tmp_path / "empty.zarr")
    root = zarr.open_group(store_path, mode="w", zarr_format=3)
    empty_attrs: dict[str, Any] = {
        "zarr_conventions": [SPATIAL_CMO, PROJ_CMO],
        "spatial:bbox": [6.0, 44.0, 8.0, 46.0],
        "proj:code": "EPSG:4326",
    }
    root.attrs.update(empty_attrs)
    report = validate_store(store_path)
    assert any("nothing qualifies" in i.message for i in report.issues)


def test_zarr_v2_store_rejected(tmp_path: pathlib.Path) -> None:
    store_path = str(tmp_path / "v2.zarr")
    zarr.open_group(store_path, mode="w", zarr_format=2)
    report = validate_store(store_path)
    assert any("Zarr V2" in i.message for i in report.issues)


def test_array_convention_use_without_declaration(compliant_store: str) -> None:
    """An array using proj:code outside any declaring group is flagged."""
    root = zarr.open_group(compliant_store, mode="r+")
    orphan = root.create_group("orphan")
    arr = orphan.create_array("x", shape=(4,), dtype="float64", dimension_names=("x",))
    arr.attrs["proj:code"] = "EPSG:32632"
    report = validate_store(compliant_store)
    assert any(
        "does not declare the geo-proj convention" in i.message and i.path == "/orphan/x"
        for i in report.issues
    )


def test_array_convention_use_with_inherited_declaration(compliant_store: str) -> None:
    """A proj:code override on an array under a declaring group is fine."""
    level = zarr.open_group(compliant_store, mode="r+")["measurements/r10m"]
    assert isinstance(level, zarr.Group)
    arr = level["b02"]
    assert isinstance(arr, zarr.Array)
    arr.attrs["proj:code"] = "EPSG:32632"
    report = validate_store(compliant_store)
    assert report.compliant, [str(i) for i in report.issues]


def test_malformed_zarr_conventions_reports_issue(compliant_store: str) -> None:
    """A malformed zarr_conventions value is diagnosed, not crashed on."""
    root = zarr.open_group(compliant_store, mode="r+")
    orphan = root.create_group("weird")
    orphan.attrs["zarr_conventions"] = 5
    orphan.attrs["spatial:dimensions"] = ["y", "x"]
    report = validate_store(compliant_store)
    assert any(
        "zarr_conventions must be an array" in i.message and i.path == "/weird"
        for i in report.issues
    )


def test_malformed_zarr_conventions_entry_reports_issue(compliant_store: str) -> None:
    root = _root_attrs(compliant_store)
    conventions = cast("list[Any]", root.attrs["zarr_conventions"])
    root.attrs["zarr_conventions"] = [*conventions, 42]
    report = validate_store(compliant_store)
    assert any("zarr_conventions[2]" in i.message and i.path == "/" for i in report.issues)


def test_empty_multiscales_layout_rejected(compliant_store: str) -> None:
    ms = zarr.open_group(compliant_store, mode="r+")["measurements"]
    multiscales = dict(cast("dict[str, Any]", ms.attrs["multiscales"]))
    multiscales["layout"] = []
    ms.attrs["multiscales"] = multiscales
    report = validate_store(compliant_store)
    assert any("layout" in i.message and i.path == "/measurements" for i in report.issues)


def test_subgroup_does_not_inherit_declarations(compliant_store: str) -> None:
    """Convention declarations are inherited by direct child arrays only."""
    root = zarr.open_group(compliant_store, mode="r+")
    parent = root.create_group("parent")
    parent.attrs.update(
        cast(
            "dict[str, Any]",
            {
                "zarr_conventions": [SPATIAL_CMO, PROJ_CMO],
                "proj:code": "EPSG:32632",
                "spatial:dimensions": ["y", "x"],
            },
        )
    )
    sub = parent.create_group("child_group")
    sub.attrs["spatial:dimensions"] = ["y", "x"]
    report = validate_store(compliant_store)
    assert any(
        "does not declare the spatial convention" in i.message and i.path == "/parent/child_group"
        for i in report.issues
    )


def test_scalar_array_in_dataset_rejected(compliant_store: str) -> None:
    level = zarr.open_group(compliant_store, mode="r+")["measurements/r10m"]
    assert isinstance(level, zarr.Group)
    level.create_array("scalar", shape=(), dtype="int64")
    report = validate_store(compliant_store)
    assert any(
        "scalar arrays are not allowed" in i.message and i.path == "/measurements/r10m/scalar"
        for i in report.issues
    )


def test_grid_mapping_container_array_tolerated(compliant_store: str) -> None:
    """A 0-D array referenced via grid_mapping is CF metadata, not a DataArray."""
    level = zarr.open_group(compliant_store, mode="r+")["measurements/r10m"]
    assert isinstance(level, zarr.Group)
    level.create_array("spatial_ref", shape=(), dtype="int64")
    arr = level["b02"]
    assert isinstance(arr, zarr.Array)
    arr.attrs["grid_mapping"] = "spatial_ref"
    report = validate_store(compliant_store)
    assert report.compliant, [str(i) for i in report.issues]


def test_data_variable_without_coordinate_rejected(compliant_store: str) -> None:
    level = zarr.open_group(compliant_store, mode="r+")["measurements/r10m"]
    assert isinstance(level, zarr.Group)
    level.create_array("extra", shape=(4, 4), dtype="uint16", dimension_names=("a", "b"))
    report = validate_store(compliant_store)
    assert any(
        "no matching 1-D coordinate array" in i.message and i.path == "/measurements/r10m/extra"
        for i in report.issues
    )


def test_array_without_dimension_names_rejected(compliant_store: str) -> None:
    level = zarr.open_group(compliant_store, mode="r+")["measurements/r20m"]
    assert isinstance(level, zarr.Group)
    level.create_array("nameless", shape=(4, 4), dtype="uint16")
    report = validate_store(compliant_store)
    assert any(
        "dimension_names must be set" in i.message and i.path == "/measurements/r20m/nameless"
        for i in report.issues
    )
