"""Tests for the spec-level Sentinel-2 GeoZarr plan (`eopf_geozarr.data_api.s2_plan`)."""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import pytest
import zarr
from pydantic import ValidationError
from pydantic_zarr.v2 import GroupSpec as GroupSpecV2
from pydantic_zarr.v3 import GroupSpec as GroupSpecV3
from pyproj import CRS

from eopf_geozarr.data_api.s2 import Sentinel2Root
from eopf_geozarr.data_api.s2_plan import (
    REFLECTANCE_SCALE_LEVELS,
    load_sentinel2,
    plan_geozarr,
    plan_reflectance,
    product_level,
)

_INPUT_DIR = pathlib.Path("tests/_test_data/s2_examples")
_SNAPSHOT_DIR = pathlib.Path("tests/_test_data/optimized_geozarr_examples")
_INPUT_PATHS = sorted(_INPUT_DIR.glob("*.json"))


def _read(path: pathlib.Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    return data


def _planned_root(path: pathlib.Path) -> dict[str, Any]:
    root = Sentinel2Root.model_validate(_read(path))
    return plan_geozarr(root).model_dump(mode="json")


def _snapshot_root(path: pathlib.Path) -> dict[str, Any]:
    snapshot = GroupSpecV3.model_validate(_read(_SNAPSHOT_DIR / path.name))
    return snapshot.model_dump(mode="json")


def _planned_reflectance(path: pathlib.Path) -> dict[str, Any]:
    return _planned_root(path)["members"]["measurements"]["members"]["reflectance"]


@pytest.mark.parametrize("path", _INPUT_PATHS, ids=lambda p: p.stem[:26])
def test_plan_matches_snapshot(path: pathlib.Path) -> None:
    """
    The planned tree reproduces the stored snapshot whole — the member tree,
    every array document (shapes, dtypes, shards, codecs, dimension names)
    and every attribute, georeferencing included, across measurements,
    conditions and quality. The plan is the snapshots' source of truth
    (REGENERATE_SNAPSHOTS=1 rewrites them), and the self-consistency test
    below pins the georeferencing to the product's own metadata.
    """
    planned = _planned_root(path)
    if os.environ.get("REGENERATE_SNAPSHOTS"):
        snapshot_path = _SNAPSHOT_DIR / path.name
        snapshot_path.write_text(json.dumps(planned, indent=2, sort_keys=True) + "\n")
    assert planned == _snapshot_root(path)


@pytest.mark.parametrize("path", _INPUT_PATHS, ids=lambda p: p.stem[:26])
def test_plan_georeferencing_is_self_consistent(path: pathlib.Path) -> None:
    """
    Every planned level's spatial attributes derive from the product's own
    band attributes: the transform is the level resolution over the finest
    band's origin, the bbox is the transform applied to the level shape, and
    the grid-mapping array holds the band's own CRS.
    """
    planned = _planned_reflectance(path)
    finest = planned["members"][f"r{REFLECTANCE_SCALE_LEVELS[0]}m"]
    band = next(v for k, v in finest["members"].items() if k.startswith("b"))
    transform = band["attributes"]["proj:transform"]
    x0, y0 = transform[2], transform[5]
    epsg = band["attributes"]["proj:epsg"]

    for name, level in planned["members"].items():
        meters = float(name[1:-1])
        attributes = level["attributes"]
        assert attributes["proj:code"] == f"EPSG:{epsg}"
        assert attributes["spatial:transform"] == [meters, 0.0, x0, 0.0, -meters, y0]
        rows, cols = level["members"]["x"]["shape"][0], level["members"]["y"]["shape"][0]
        rows, cols = cols, rows  # y is rows, x is cols
        assert attributes["spatial:bbox"] == [
            x0,
            y0 - rows * meters,
            x0 + cols * meters,
            y0,
        ]
        crs = CRS.from_wkt(level["members"]["spatial_ref"]["attributes"]["crs_wkt"])
        assert crs.to_epsg() == epsg


def test_load_sentinel2_keeps_the_model(tmp_path: pathlib.Path) -> None:
    """
    `load_sentinel2` reads a Zarr store into the validated input model, and
    the plan built from it equals the plan built from the source document.
    """
    path = _INPUT_PATHS[0]
    document = _read(path)
    store_path = tmp_path / "s2.zarr"
    GroupSpecV2.model_validate(document).to_zarr(store_path, path="")  # type: ignore[arg-type]

    root = load_sentinel2(zarr.open_group(store_path, mode="r", zarr_format=2))
    assert isinstance(root, Sentinel2Root)
    assert plan_geozarr(root) == plan_geozarr(Sentinel2Root.model_validate(document))


def test_load_sentinel2_refuses_non_s2(tmp_path: pathlib.Path) -> None:
    """A store that is not a Sentinel-2 product is refused with a ValidationError."""
    s1_path = next(pathlib.Path("tests/_test_data/s1_examples").glob("*.json"))
    store_path = tmp_path / "s1.zarr"
    GroupSpecV2.model_validate(_read(s1_path)).to_zarr(store_path, path="")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        load_sentinel2(zarr.open_group(store_path, mode="r", zarr_format=2))


def test_product_level_requires_stac_discovery_evidence() -> None:
    """A product whose stac_discovery names no L1C/L2A level is refused."""
    path = _INPUT_PATHS[0]
    document = _read(path)
    document["attributes"]["stac_discovery"] = {"id": "mystery", "properties": {}}
    root = Sentinel2Root.model_validate(document)
    with pytest.raises(ValueError, match="product level"):
        product_level(root)


def test_plan_carries_a_mixed_groups_own_arrays() -> None:
    """A conditions/quality group holding arrays next to its subgroups keeps
    those arrays in the plan instead of dropping them silently."""
    path = _INPUT_PATHS[0]
    document = _read(path)
    mixed = document["members"]["conditions"]["members"]["mask"]["members"]["detector_footprint"]
    assert any("members" in member for member in mixed["members"].values()), (
        "fixture precondition: detector_footprint holds subgroups"
    )
    mixed["members"]["extra"] = {
        "zarr_format": 2,
        "shape": [4, 4],
        "chunks": [4, 4],
        "dtype": "<u2",
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "compressor": None,
        "dimension_separator": "/",
        "attributes": {"_ARRAY_DIMENSIONS": ["y", "x"]},
    }
    root = Sentinel2Root.model_validate(document)
    planned = plan_geozarr(root).model_dump(mode="json")
    mask_members = planned["members"]["conditions"]["members"]["mask"]["members"]
    planned_mixed = mask_members["detector_footprint"]["members"]
    assert "extra" in planned_mixed
    assert planned_mixed["extra"]["shape"] == [4, 4]
    # the subgroups survive alongside the carried array
    assert any("members" in member for member in planned_mixed.values())


def test_plan_reflectance_requires_product_georeferencing() -> None:
    """A product whose bands carry no proj attributes and whose stac_discovery
    declares none is refused with a ValueError, not a KeyError."""
    path = _INPUT_PATHS[0]
    document = _read(path)
    reflectance = document["members"]["measurements"]["members"]["reflectance"]
    for level in reflectance["members"].values():
        for member in level["members"].values():
            for key in ("proj:transform", "proj:epsg", "proj:wkt2"):
                member["attributes"].pop(key, None)
    document["attributes"]["stac_discovery"].get("properties", {}).pop("proj:bbox", None)
    root = Sentinel2Root.model_validate(document)
    with pytest.raises(ValueError, match="proj"):
        plan_reflectance(root)


def test_plan_reflectance_requires_a_resolution_dataset() -> None:
    """A reflectance group with none of the known resolution datasets is refused."""
    path = _INPUT_PATHS[0]
    document = _read(path)
    document["members"]["measurements"]["members"]["reflectance"]["members"] = {}
    root = Sentinel2Root.model_validate(document)
    with pytest.raises(ValueError, match="resolution datasets"):
        plan_reflectance(root)
