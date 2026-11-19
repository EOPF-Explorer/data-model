"""Tests for the spec-level Sentinel-2 GeoZarr plan (`eopf_geozarr.data_api.s2_plan`)."""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import numpy as np
import pytest
import zarr
from pydantic import ValidationError
from pydantic_zarr.v2 import GroupSpec as GroupSpecV2
from pyproj import CRS
from zarr.storage import LocalStore

from eopf_geozarr.conversion.open_source import open_source_datatree
from eopf_geozarr.data_api.s2 import Sentinel2Root
from eopf_geozarr.data_api.s2_plan import (
    AUX_MULTISCALE_LEVELS,
    REFLECTANCE_SCALE_LEVELS,
    AuxiliaryLevel,
    AuxiliaryLevelMembers,
    AuxiliaryPyramid,
    BandMaskGroup,
    CarriedDataset,
    ConditionsGroup,
    ConditionsMaskGroup,
    GridMappingArray,
    QualityGroup,
    ReflectanceLevel,
    ReflectancePyramid,
    Sentinel2GeoZarr,
    load_sentinel2,
    plan_geozarr,
    plan_reflectance,
    product_level,
)
from eopf_geozarr.s2_optimization.s2_converter import convert_s2_optimized
from eopf_geozarr.zarr_io import write_hierarchy

_INPUT_DIR = pathlib.Path("tests/_test_data/s2_examples")
_INPUT_PATHS = sorted(_INPUT_DIR.glob("*.json"))
_L2A_PATH = next(path for path in _INPUT_PATHS if "MSIL2A" in path.stem)

#: The tile-sized dimensions, shrunk tenfold so that a conversion takes seconds.
_SHRUNK_SIZES = {10980: 1098, 5490: 549, 1830: 183}

#: The spacing of the geometry angle grids, in metres.
_GEOMETRY_SPACING = 5000.0


def _read(path: pathlib.Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    assert isinstance(data, dict)
    return data


def _planned_root(path: pathlib.Path) -> dict[str, Any]:
    root = Sentinel2Root.model_validate(_read(path))
    return plan_geozarr(root).model_dump(mode="json")


def _planned_reflectance(path: pathlib.Path) -> dict[str, Any]:
    return _planned_root(path)["members"]["measurements"]["members"]["reflectance"]


def _shrink(node: dict[str, Any], coordinate_dtype: str | None, name: str = "") -> None:
    """Shrink the tile-sized dimensions of every array under `node`, in place."""
    if "shape" in node:
        node["shape"] = [_SHRUNK_SIZES.get(size, size) for size in node["shape"]]
        node["chunks"] = [
            max(1, min(chunk, size))
            for chunk, size in zip(node["chunks"], node["shape"], strict=True)
        ]
        attributes = node.get("attributes") or {}
        if "proj:shape" in attributes:
            attributes["proj:shape"] = [
                _SHRUNK_SIZES.get(size, size) for size in attributes["proj:shape"]
            ]
        if coordinate_dtype is not None and name in ("x", "y") and len(node["shape"]) == 1:
            node["dtype"] = np.dtype(coordinate_dtype).str
            node["fill_value"] = "NaN" if np.dtype(coordinate_dtype).kind == "f" else None
    for member_name, member in (node.get("members") or {}).items():
        _shrink(member, coordinate_dtype, member_name)


def _fill(group: zarr.Group, origin: tuple[float, float], rng: np.random.Generator) -> None:
    """Give every array under `group` values: coordinates on the tile's grid
    (pixel centres per resolution level, nodes for the geometry grids) and
    random data with nodata patches."""
    x0, y0 = origin
    for path, array in group.members(max_depth=None):
        if not isinstance(array, zarr.Array) or array.ndim == 0:
            continue
        name = path.rsplit("/", 1)[-1]
        kind = array.dtype.kind
        if name in ("x", "y") and array.attrs.get("_ARRAY_DIMENSIONS") == [name]:
            level = re.search(r"/r(\d+)m/", f"/{path}")
            step = float(level.group(1)) if level else _GEOMETRY_SPACING
            offsets = (np.arange(array.shape[0]) + (0.5 if level else 0.0)) * step
            array[...] = x0 + offsets if name == "x" else y0 - offsets
        elif kind == "f":
            values = rng.random(array.shape).astype(array.dtype) * 100
            values[rng.random(array.shape) < 0.05] = np.nan
            array[...] = values
        elif kind in "ui":
            high = 12 if name == "scl" else min(int(np.iinfo(array.dtype).max), 12000) + 1
            values = rng.integers(0, high, size=array.shape).astype(array.dtype)
            values[rng.random(array.shape) < 0.05] = 0
            array[...] = values
        elif kind == "M":
            array[...] = np.arange(array.size).reshape(array.shape).astype(array.dtype)


def _product_with_data(
    path: pathlib.Path, out_dir: pathlib.Path, coordinate_dtype: str | None
) -> pathlib.Path:
    """
    Write the fixture product at `path` as a Zarr V2 store that holds data.

    The fixtures are metadata documents of real products; here their
    tile-sized dimensions are shrunk tenfold and every array gets values. The
    x/y arrays keep the data type the product declares (int64) unless
    `coordinate_dtype` names another.
    """
    document = _read(path)
    _shrink(document, coordinate_dtype)
    store_path = out_dir / "product.zarr"
    write_hierarchy(GroupSpecV2.model_validate(document), LocalStore(store_path))
    group = zarr.open_group(store_path, mode="r+", zarr_format=2)
    band = group["measurements/reflectance/r10m/b02"]
    transform = band.attrs["proj:transform"]
    assert isinstance(transform, list)
    _fill(group, (float(transform[2]), float(transform[5])), np.random.default_rng(0))  # type: ignore[arg-type]
    zarr.consolidate_metadata(group.store, zarr_format=2)
    return store_path


def _canonical(document: object) -> str:
    """A document as sorted JSON text: equal texts are equal documents (a NaN
    never equals itself as a float)."""
    return json.dumps(document, sort_keys=True)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    ("path", "coordinate_dtype"),
    [*((path, None) for path in _INPUT_PATHS), (_L2A_PATH, "float64")],
    ids=lambda value: value.stem[:26] if isinstance(value, pathlib.Path) else str(value),
)
def test_plan_matches_the_converter_output(
    path: pathlib.Path, coordinate_dtype: str | None, tmp_path: pathlib.Path
) -> None:
    """
    The plan describes the store `convert_s2_optimized` writes. For a product
    that holds data, every group and array document the converter stores
    below the root equals the planned one — the member tree, shapes, data
    types, fill values, chunk grids, codecs, dimension names and attributes,
    georeferencing included — across measurements, conditions and quality,
    for L1C and L2A, and for x/y coordinates stored as int64 or as float64.

    The root's own attributes are left out: the converter writes them (the
    product metadata and the STAC assets) in its consolidation step, and the
    plan leaves them empty.
    """
    source_path = _product_with_data(path, tmp_path, coordinate_dtype)
    output_path = tmp_path / "output.zarr"
    convert_s2_optimized(
        dt_input=open_source_datatree(str(source_path)),
        output_path=str(output_path),
        enable_sharding=True,
        spatial_chunk=1024,
        compression_level=3,
        validate_output=False,
    )
    written = {}
    for document_path in output_path.rglob("zarr.json"):
        document = json.loads(document_path.read_text())
        # the converter consolidates metadata; the plan holds node documents only
        document.pop("consolidated_metadata", None)
        written[str(document_path.relative_to(output_path))] = document

    source = zarr.open_group(source_path, mode="r", zarr_format=2)

    def coordinate_values(array_path: str) -> np.ndarray | None:
        member = source.get(array_path)
        return np.asarray(member[...]) if isinstance(member, zarr.Array) else None

    plan = plan_geozarr(load_sentinel2(source), coordinate_values=coordinate_values)
    planned = json.loads(json.dumps(plan.to_stored_metadata_json()))

    assert written.keys() == planned.keys()
    # the output model describes what the converter stores, root attributes included
    Sentinel2GeoZarr.from_stored_metadata_json(written)
    assert written.pop("zarr.json")["node_type"] == planned.pop("zarr.json")["node_type"]
    assert [key for key in written if _canonical(written[key]) != _canonical(planned[key])] == []


@pytest.mark.parametrize("path", _INPUT_PATHS, ids=lambda p: p.stem[:26])
def test_plan_is_the_typed_output_model(path: pathlib.Path) -> None:
    """
    The plan is a `Sentinel2GeoZarr`: the reflectance pyramid holds exactly
    the six scale levels, each a `ReflectanceLevel` with `x`, `y`, its bands
    and a `spatial_ref` grid mapping; the auxiliary pyramids hold the levels
    `AUX_MULTISCALE_LEVELS` lists; `conditions` and `quality` are a
    `ConditionsGroup` and a `QualityGroup`; and the model reads its own dump
    back.
    """
    plan = plan_geozarr(Sentinel2Root.model_validate(_read(path)))
    assert isinstance(plan, Sentinel2GeoZarr)

    reflectance = plan.members["measurements"].members["reflectance"]
    assert isinstance(reflectance, ReflectancePyramid)
    assert set(reflectance.members) == {f"r{meters}m" for meters in REFLECTANCE_SCALE_LEVELS}
    for level in (reflectance.members["r10m"], reflectance.members["r720m"]):
        assert isinstance(level, ReflectanceLevel)
        assert isinstance(level.members["spatial_ref"], GridMappingArray)
        assert level.members["x"].dimension_names == ("x",)
        assert level.members["y"].dimension_names == ("y",)
        band = level.members.get("b02")
        assert band is not None
        assert level.attributes["spatial:shape"] == list(band.shape)

    conditions = plan.members.get("conditions")
    quality = plan.members.get("quality")
    assert isinstance(conditions, ConditionsGroup)
    assert isinstance(quality, QualityGroup)
    assert isinstance(conditions.members.get("geometry"), CarriedDataset)
    mask = conditions.members.get("mask")
    assert isinstance(mask, ConditionsMaskGroup)
    assert isinstance(mask.members.get("detector_footprint"), BandMaskGroup)
    assert isinstance(quality.members.get("mask"), BandMaskGroup)

    for group_path, scale_levels in AUX_MULTISCALE_LEVELS.items():
        pyramid = plan.get_path(group_path)
        if pyramid is None:  # the classification of the other product level
            continue
        assert isinstance(pyramid, AuxiliaryPyramid)
        assert set(pyramid.members) == {f"r{meters}m" for meters in scale_levels}

    dumped = plan.model_dump(mode="json")
    assert Sentinel2GeoZarr.model_validate(dumped).model_dump(mode="json") == dumped


def _planned_level(name: str = "r10m") -> dict[str, Any]:
    """The dump of one planned reflectance level of the L2A fixture."""
    level: dict[str, Any] = _planned_reflectance(_L2A_PATH)["members"][name]
    return level


def test_reflectance_level_refuses_an_unknown_member() -> None:
    """A level holds only coordinates, a grid mapping and known bands."""
    level = _planned_level()
    level["members"]["b99"] = level["members"]["b02"]
    with pytest.raises(ValidationError, match="b99"):
        ReflectanceLevel.model_validate(level)


@pytest.mark.parametrize("missing", ["x", "y", "spatial_ref"])
def test_reflectance_level_requires_coordinates_and_grid_mapping(missing: str) -> None:
    """A level without `x`, `y` or `spatial_ref` is refused."""
    level = _planned_level()
    del level["members"][missing]
    with pytest.raises(ValidationError, match=missing):
        ReflectanceLevel.model_validate(level)


def test_reflectance_level_refuses_an_unknown_attribute() -> None:
    """A level's attributes are exactly the spatial and proj convention keys."""
    level = _planned_level()
    level["attributes"]["spatial:extra"] = 1
    with pytest.raises(ValidationError, match="spatial:extra"):
        ReflectanceLevel.model_validate(level)


def test_reflectance_pyramid_requires_every_level() -> None:
    """A pyramid missing one of the six scale levels is refused."""
    pyramid = _planned_reflectance(_L2A_PATH)
    del pyramid["members"]["r720m"]
    with pytest.raises(ValidationError, match="r720m"):
        ReflectancePyramid.model_validate(pyramid)


def test_plan_refuses_an_extra_root_member() -> None:
    """The planned root holds only `measurements`, `conditions` and `quality`."""
    root = _planned_root(_L2A_PATH)
    root["members"]["extra"] = root["members"]["measurements"]
    with pytest.raises(ValidationError, match="extra"):
        Sentinel2GeoZarr.model_validate(root)


def test_plan_refuses_an_unknown_conditions_member() -> None:
    """`conditions` holds only `geometry`, `mask` and `meteorology`."""
    root = _planned_root(_L2A_PATH)
    conditions = root["members"]["conditions"]
    conditions["members"]["extra"] = conditions["members"]["geometry"]
    with pytest.raises(ValidationError, match="extra"):
        Sentinel2GeoZarr.model_validate(root)


def test_output_model_refuses_a_pyramid_level_without_a_coordinate() -> None:
    """The root model checks the pyramids under `quality`: a level of
    `quality/atmosphere` without its `x` array is refused."""
    root = _planned_root(_L2A_PATH)
    del root["members"]["quality"]["members"]["atmosphere"]["members"]["r10m"]["members"]["x"]
    with pytest.raises(ValidationError, match="atmosphere"):
        Sentinel2GeoZarr.model_validate(root)


def _planned_atmosphere() -> dict[str, Any]:
    """The dump of the planned `quality/atmosphere` pyramid of the L2A fixture."""
    pyramid: dict[str, Any] = _planned_root(_L2A_PATH)["members"]["quality"]["members"][
        "atmosphere"
    ]
    return pyramid


@pytest.mark.parametrize("missing", sorted(AuxiliaryLevelMembers.__required_keys__))
def test_auxiliary_level_requires_its_fixed_members(missing: str) -> None:
    """An auxiliary pyramid level without one of its fixed members is refused."""
    level = _planned_atmosphere()["members"]["r10m"]
    del level["members"][missing]
    with pytest.raises(ValidationError, match=missing):
        AuxiliaryLevel.model_validate(level)


def test_auxiliary_pyramid_refuses_a_member_that_is_not_a_level() -> None:
    """Every member of an auxiliary pyramid is a level group."""
    pyramid = _planned_atmosphere()
    pyramid["members"]["r10m"] = pyramid["members"]["r10m"]["members"]["x"]
    with pytest.raises(ValidationError, match="r10m"):
        AuxiliaryPyramid.model_validate(pyramid)


def test_pyramid_refuses_an_unknown_layout_key() -> None:
    """A multiscales layout entry holds only the keys the converter writes."""
    pyramid = _planned_reflectance(_L2A_PATH)
    pyramid["attributes"]["multiscales"]["layout"][0]["extra"] = 1
    with pytest.raises(ValidationError, match="extra"):
        ReflectancePyramid.model_validate(pyramid)


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
    write_hierarchy(GroupSpecV2.model_validate(document), LocalStore(store_path))

    root = load_sentinel2(zarr.open_group(store_path, mode="r", zarr_format=2))
    assert isinstance(root, Sentinel2Root)
    assert plan_geozarr(root) == plan_geozarr(Sentinel2Root.model_validate(document))


def test_load_sentinel2_refuses_non_s2(tmp_path: pathlib.Path) -> None:
    """A store that is not a Sentinel-2 product is refused with a ValidationError."""
    s1_path = next(pathlib.Path("tests/_test_data/s1_examples").glob("*.json"))
    store_path = tmp_path / "s1.zarr"
    write_hierarchy(GroupSpecV2.model_validate(_read(s1_path)), LocalStore(store_path))
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


_EXTRA_ARRAY: dict[str, Any] = {
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


def test_input_refuses_an_array_beside_the_footprint_levels() -> None:
    """The detector footprints hold resolution datasets only: an array next to
    them is refused when the product is read, not dropped from the plan."""
    document = _read(_INPUT_PATHS[0])
    footprint = document["members"]["conditions"]["members"]["mask"]["members"][
        "detector_footprint"
    ]
    footprint["members"]["extra"] = _EXTRA_ARRAY
    with pytest.raises(ValidationError, match="extra"):
        Sentinel2Root.model_validate(document)


def test_input_refuses_a_group_where_geometry_holds_an_array() -> None:
    """The members of `conditions/geometry` are arrays."""
    document = _read(_INPUT_PATHS[0])
    geometry = document["members"]["conditions"]["members"]["geometry"]
    geometry["members"]["sun_angles"] = {"zarr_format": 2, "attributes": {}, "members": {}}
    with pytest.raises(ValidationError, match="sun_angles"):
        Sentinel2Root.model_validate(document)


def test_plan_carries_a_datetime_array_in_its_v3_form() -> None:
    """A carried V2 datetime array is planned with the Zarr V3 form zarr-python
    writes, `numpy.datetime64` with its unit, which the GeoZarr scope claims."""
    document = _read(_INPUT_PATHS[0])
    cams = document["members"]["conditions"]["members"]["meteorology"]["members"]["cams"]
    cams["members"]["observed"] = {
        **_EXTRA_ARRAY,
        "shape": [3],
        "chunks": [3],
        "dtype": "<M8[ns]",
        "attributes": {"_ARRAY_DIMENSIONS": ["observed"]},
    }
    plan = plan_geozarr(Sentinel2Root.model_validate(document))
    conditions = plan.members.get("conditions")
    assert conditions is not None
    meteorology = conditions.members.get("meteorology")
    assert meteorology is not None
    planned_cams = meteorology.members.get("cams")
    assert planned_cams is not None
    observed = planned_cams.members["observed"].model_dump(mode="json")
    assert observed["data_type"] == {
        "name": "numpy.datetime64",
        "configuration": {"unit": "ns", "scale_factor": 1},
    }
    assert observed["dimension_names"] == ["observed"]


def _tile_epsg(path: pathlib.Path) -> int:
    """The EPSG code of the product's finest reflectance band."""
    finest = _read(path)["members"]["measurements"]["members"]["reflectance"]["members"]["r10m"]
    band = next(v for k, v in finest["members"].items() if k.startswith("b"))
    return int(band["attributes"]["proj:epsg"])


def test_plan_georeferences_the_geometry_grids_at_their_nodes() -> None:
    """
    `conditions/geometry` declares the tile CRS (EOPF-Explorer/data-model#346).
    The angle grids are node-registered: `(x[0], y[0])` is the tile's
    upper-left corner, so the transform starts there and is not moved by a
    half cell. Arrays without the (y, x) dimensions get no grid mapping.
    """
    path = _INPUT_PATHS[0]
    size, step, ulx, uly = 23, 5000.0, 600000.0, 7800000.0
    values = {
        "conditions/geometry/x": ulx + step * np.arange(size),
        "conditions/geometry/y": uly - step * np.arange(size),
    }
    root = Sentinel2Root.model_validate(_read(path))
    planned = plan_geozarr(root, coordinate_values=values.get).model_dump(mode="json")
    geometry = planned["members"]["conditions"]["members"]["geometry"]
    attrs = geometry["attributes"]
    assert attrs["proj:code"] == f"EPSG:{_tile_epsg(path)}"
    assert attrs["spatial:dimensions"] == ["y", "x"]
    assert attrs["spatial:registration"] == "node"
    assert attrs["spatial:shape"] == [size, size]
    assert attrs["spatial:transform"] == [step, 0.0, ulx, 0.0, -step, uly]
    assert attrs["spatial:bbox"] == [ulx, uly - step * (size - 1), ulx + step * (size - 1), uly]
    arrays = geometry["members"]
    assert "spatial_ref" in arrays
    assert arrays["sun_angles"]["attributes"]["grid_mapping"] == "spatial_ref"
    assert arrays["viewing_incidence_angles"]["attributes"]["grid_mapping"] == "spatial_ref"
    assert "grid_mapping" not in arrays["mean_sun_angles"]["attributes"]
    assert "grid_mapping" not in arrays["mean_viewing_incidence_angles"]["attributes"]


def test_plan_carries_an_irregular_geometry_grid_without_georeferencing() -> None:
    """A geometry grid no transform describes is carried as it was, not refused."""
    root = Sentinel2Root.model_validate(_read(_INPUT_PATHS[0]))
    values = {
        "conditions/geometry/x": np.array([0.0, 1.0, 5.0, *range(6, 26)]),
        "conditions/geometry/y": np.arange(23.0),
    }
    planned = plan_geozarr(root, coordinate_values=values.get).model_dump(mode="json")
    geometry = planned["members"]["conditions"]["members"]["geometry"]
    assert "proj:code" not in geometry["attributes"]
    assert "spatial_ref" not in geometry["members"]


def test_plan_georeferences_footprints_without_proj_attributes_from_the_tile() -> None:
    """
    A detector-footprint level whose arrays carry no proj attributes (as in
    native Zarr V3 products) declares the tile CRS, pixel-registered at its
    resolution from the tile origin (EOPF-Explorer/data-model#346).
    """
    path = _INPUT_PATHS[0]
    document = _read(path)
    footprint = document["members"]["conditions"]["members"]["mask"]["members"][
        "detector_footprint"
    ]["members"]["r60m"]
    for array in footprint["members"].values():
        for key in [k for k in array["attributes"] if k.startswith("proj:")]:
            del array["attributes"][key]
    planned = plan_geozarr(Sentinel2Root.model_validate(document)).model_dump(mode="json")
    attrs = planned["members"]["conditions"]["members"]["mask"]["members"]["detector_footprint"][
        "members"
    ]["r60m"]["attributes"]
    reflectance = planned["members"]["measurements"]["members"]["reflectance"]["members"]
    assert attrs["proj:code"] == f"EPSG:{_tile_epsg(path)}"
    assert attrs["spatial:registration"] == "pixel"
    assert attrs["spatial:transform"] == reflectance["r60m"]["attributes"]["spatial:transform"]


@pytest.mark.parametrize(
    "path", [p for p in _INPUT_PATHS if "MSIL2A" in p.name], ids=lambda p: p.stem[:26]
)
def test_plan_builds_the_atmosphere_pyramid(path: pathlib.Path) -> None:
    """
    `quality/atmosphere` (aot, wvp) gets a pyramid like the reflectance bands
    (EOPF-Explorer/data-model#343): native r10m/r20m/r60m levels reused,
    r120m/r360m/r720m coarsened, and all six in the `multiscales` layout.
    """
    planned = _planned_root(path)
    atmosphere = planned["members"]["quality"]["members"]["atmosphere"]
    multiscales = atmosphere["attributes"]["multiscales"]
    levels = ["r10m", "r20m", "r60m", "r120m", "r360m", "r720m"]
    assert [entry["asset"] for entry in multiscales["layout"]] == levels
    assert multiscales["resampling_method"] == "average"
    for level in levels:
        assert {"aot", "wvp"} <= set(atmosphere["members"][level]["members"]), level


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
