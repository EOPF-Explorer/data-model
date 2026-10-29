"""
Unit tests for simplified S2 converter implementation.

Tests the new simplified approach that uses only xarray and proper metadata consolidation.
"""

import json
import uuid
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.core.group import ConsolidatedMetadata

from eopf_geozarr.conversion.utils import (
    add_multiscale_pyramids_to_stac_metadata,
    updated_root_consolidation,
    write_store_geo_metadata,
)
from eopf_geozarr.s2_optimization.s2_converter import (
    _validate_s2_input,
    add_s2_stac_assets,
    convert_s2_optimized,
    initialize_crs_from_dataset,
)


def test_validate_s2_input_skips_trees_without_zarr_backend() -> None:
    """In-memory DataTrees (e.g. from the CPM SAFE reader) skip store validation."""
    tree = xr.DataTree()
    tree["measurements"] = xr.Dataset({"var": (["y", "x"], np.zeros((4, 4)))})
    _validate_s2_input(tree)  # must not raise


def test_validate_s2_input_rejects_non_s2_zarr_backed_tree(tmp_path: Path) -> None:
    """Zarr-v2-backed trees that are not Sentinel-2 products are still rejected."""
    store = tmp_path / "not_s2.zarr"
    source = xr.DataTree(xr.Dataset({"var": (["y", "x"], np.zeros((4, 4)))}))
    source.to_zarr(store, zarr_format=2, consolidated=False)
    backed = xr.open_datatree(store, engine="zarr")
    with pytest.raises(ValueError, match="not a Sentinel-2 product"):
        _validate_s2_input(backed)


def test_validate_s2_input_rejects_non_s2_zarr_v3_backed_tree(tmp_path: Path) -> None:
    """Zarr-v3-backed trees that are not Sentinel-2 products are still rejected."""
    store = tmp_path / "not_s2_v3.zarr"
    source = xr.DataTree(xr.Dataset({"var": (["y", "x"], np.zeros((4, 4)))}))
    source.to_zarr(store, zarr_format=3, consolidated=False)
    backed = xr.open_datatree(store, engine="zarr")
    with pytest.raises(ValueError, match="not a Sentinel-2 product"):
        _validate_s2_input(backed)


@pytest.fixture
def mock_s2_dataset() -> xr.DataTree:
    """Create a mock S2 dataset for testing."""
    # Create test data arrays
    coords = {
        "x": (["x"], np.linspace(0, 1000, 100)),
        "y": (["y"], np.linspace(0, 1000, 100)),
        "time": (["time"], [np.datetime64("2023-01-01")]),
    }

    # Create test variables
    data_vars = {
        "b02": (["time", "y", "x"], np.random.rand(1, 100, 100)),
        "b03": (["time", "y", "x"], np.random.rand(1, 100, 100)),
        "b04": (["time", "y", "x"], np.random.rand(1, 100, 100)),
    }

    ds = xr.Dataset(data_vars, coords=coords)

    # Add rioxarray CRS
    ds = ds.rio.write_crs("EPSG:32632")

    # Create datatree
    dt = xr.DataTree(ds)
    dt.attrs = {"stac_discovery": {"properties": {"mission": "sentinel-2"}}}

    return dt


class TestCRSInitialization:
    """Test CRS initialization functionality."""

    def test_initialize_crs_from_cpm_260_metadata(self) -> None:
        """Test CRS initialization from CPM >= 2.6.0 metadata with integer EPSG."""
        # Create a DataTree with CPM 2.6.0+ style metadata (integer format)
        dt = xr.DataTree()
        dt.attrs = {
            "other_metadata": {
                "horizontal_CRS_code": 32632  # EPSG:32632 (WGS 84 / UTM zone 32N)
            }
        }

        crs = initialize_crs_from_dataset(dt)

        assert crs is not None
        assert crs.to_epsg() == 32632

    def test_initialize_crs_from_cpm_260_metadata_string(self) -> None:
        """Test CRS initialization from CPM >= 2.6.0 metadata with string EPSG."""
        # Create a DataTree with CPM 2.6.0+ style metadata (string format)
        dt = xr.DataTree()
        dt.attrs = {
            "other_metadata": {
                "horizontal_CRS_code": "EPSG:32632"  # String format
            }
        }

        crs = initialize_crs_from_dataset(dt)

        assert crs is not None
        assert crs.to_epsg() == 32632

    def test_initialize_crs_from_cpm_260_metadata_invalid_epsg(self) -> None:
        """Test CRS initialization with invalid EPSG code in CPM 2.6.0 metadata."""
        # Create a DataTree with invalid EPSG code
        dt = xr.DataTree()
        dt.attrs = {
            "other_metadata": {
                "horizontal_CRS_code": 999999  # Invalid EPSG code
            }
        }

        # Should fall through to other methods or return None
        crs = initialize_crs_from_dataset(dt)

        # CRS should be None since there's no other CRS information
        assert crs is None

    def test_initialize_crs_from_rio_accessor(self) -> None:
        """Test CRS initialization from rioxarray accessor."""
        # Create a dataset with rioxarray CRS
        coords = {
            "x": (["x"], np.linspace(0, 1000, 10)),
            "y": (["y"], np.linspace(0, 1000, 10)),
        }
        data_vars = {
            "b02": (["y", "x"], np.random.rand(10, 10)),
        }
        ds = xr.Dataset(data_vars, coords=coords)
        ds = ds.rio.write_crs("EPSG:32633")

        # Create DataTree without CPM 2.6.0 metadata
        dt = xr.DataTree(ds)

        crs = initialize_crs_from_dataset(dt)

        assert crs is not None
        assert crs.to_epsg() == 32633

    def test_initialize_crs_from_proj_epsg_attribute(self) -> None:
        """Test CRS initialization from proj:epsg attribute."""
        # Create a dataset with proj:epsg attribute
        coords = {
            "x": (["x"], np.linspace(0, 1000, 10)),
            "y": (["y"], np.linspace(0, 1000, 10)),
        }
        data_vars = {
            "b02": (["y", "x"], np.random.rand(10, 10)),
        }
        ds = xr.Dataset(data_vars, coords=coords)
        ds["b02"].attrs["proj:epsg"] = 32634

        # Create DataTree
        dt = xr.DataTree(ds)

        crs = initialize_crs_from_dataset(dt)

        assert crs is not None
        assert crs.to_epsg() == 32634

    def test_initialize_crs_no_crs_information(self) -> None:
        """Test CRS initialization when no CRS information is available."""
        # Create a dataset without any CRS information
        coords = {
            "x": (["x"], np.linspace(0, 1000, 10)),
            "y": (["y"], np.linspace(0, 1000, 10)),
        }
        data_vars = {
            "b02": (["y", "x"], np.random.rand(10, 10)),
        }
        ds = xr.Dataset(data_vars, coords=coords)

        # Create DataTree without any CRS metadata
        dt = xr.DataTree(ds)

        crs = initialize_crs_from_dataset(dt)

        assert crs is None

    def test_initialize_crs_priority_cpm_260_over_rio(self) -> None:
        """Test that CPM 2.6.0 metadata takes priority over rio accessor."""
        # Create a dataset with both CPM 2.6.0 metadata and rio CRS
        coords = {
            "x": (["x"], np.linspace(0, 1000, 10)),
            "y": (["y"], np.linspace(0, 1000, 10)),
        }
        data_vars = {
            "b02": (["y", "x"], np.random.rand(10, 10)),
        }
        ds = xr.Dataset(data_vars, coords=coords)
        ds = ds.rio.write_crs("EPSG:32633")  # Different EPSG

        # Create DataTree with CPM 2.6.0 metadata
        dt = xr.DataTree(ds)
        dt.attrs = {
            "other_metadata": {
                "horizontal_CRS_code": 32632  # This should take priority
            }
        }

        crs = initialize_crs_from_dataset(dt)

        assert crs is not None
        # CPM 2.6.0 metadata should take priority
        assert crs.to_epsg() == 32632


@pytest.mark.parametrize("store", ["local", "memory"])
def test_updated_root_consolidation_success(tmp_path: Path, store: str) -> None:
    """
    Test that updated_root_consolidation consolidates the root and each group passed in
    `additional_consolidate_paths` (here the reflectance and an auxiliary pyramid), but
    not other groups. Also runs on an fsspec store, where a `//` in a group path fails.
    """
    # `None` marks the pyramid parent groups, as returned by `generic_multiscales`
    datasets: dict[str, xr.Dataset | None] = {
        "/measurements/reflectance/r10m": xr.Dataset(),
        "/measurements/reflectance": None,
        "/conditions/mask/l2a_classification/r20m": xr.Dataset(),
        "/conditions/mask/l2a_classification": None,
        "/quality/atmosphere": xr.Dataset(),
    }

    # Write every group into one store, as the pipeline does: zarr then creates
    # the root and all parent groups, which `updated_root_consolidation` expects.
    output_path = (
        str(tmp_path / "test.zarr") if store == "local" else f"memory://{uuid.uuid4().hex}.zarr"
    )
    for k, v in datasets.items():
        if v is not None:
            v.to_zarr(output_path, group=k.lstrip("/"), mode="a", zarr_format=3, consolidated=False)
    zarr.open_group(f"{output_path}/measurements/reflectance", mode="r+").attrs["title"] = "refl"

    updated_root_consolidation(
        dt_input=xr.DataTree(),
        output_path=output_path,
        datasets=datasets,
        additional_consolidate_paths=[
            "/measurements/reflectance",
            "/conditions/mask/l2a_classification",
        ],
    )

    def consolidated_metadata(group_path: str) -> ConsolidatedMetadata | None:
        group = zarr.open_group(f"{output_path}/{group_path}".rstrip("/"), mode="r")
        return group.metadata.consolidated_metadata

    root_consolidated = consolidated_metadata("")
    assert root_consolidated is not None
    for group_path, level in [
        ("measurements/reflectance", "r10m"),
        ("conditions/mask/l2a_classification", "r20m"),
    ]:
        group_consolidated = consolidated_metadata(group_path)
        assert group_consolidated is not None, f"{group_path}: not consolidated"
        assert level in group_consolidated.metadata
    assert consolidated_metadata("quality") is None

    # Order guard: the root is consolidated last, so its view of the reflectance
    # group has the same attributes as the group itself.
    reflectance = zarr.open_group(
        f"{output_path}/measurements/reflectance", mode="r", use_consolidated=False
    )
    root_view = root_consolidated.flattened_metadata["measurements/reflectance"]
    assert root_view.attributes == reflectance.metadata.attributes == {"title": "refl"}


@pytest.mark.parametrize(
    "aux_groups",
    [
        (
            "/conditions/mask/l2a_classification",
            "/quality/probability",
            "/conditions/mask/l1c_classification",
        ),
        ("/conditions/mask/l1c_classification",),
    ],
    ids=["all", "l1c_classification_only"],
)
def test_updated_root_consolidation_aux_multiscales(
    tmp_path: Path, aux_groups: tuple[str, ...]
) -> None:
    """Each written auxiliary pyramid gets its own consolidated metadata listing its levels.

    The converter derives the consolidated set from the paths the plan maps to
    None, so a pyramid missing from `datasets` (e.g. an L2A product whose
    processing baseline lacks a group) is never passed on for consolidation.
    """
    output_path = tmp_path / "test.zarr"
    levels = ("r60m", "r120m")
    datasets: dict[str, xr.Dataset | None] = {
        "/measurements/reflectance/r10m": xr.Dataset(),
        "/measurements/reflectance": None,
    }
    for base_path in aux_groups:
        datasets[base_path] = None
        for level in levels:
            datasets[f"{base_path}/{level}"] = xr.Dataset({"b00": (["y", "x"], np.zeros((2, 2)))})

    # Write every group into one store, as the pipeline does.
    for k, v in datasets.items():
        if v is not None:
            v.to_zarr(output_path, group=k.lstrip("/"), mode="a", zarr_format=3, consolidated=False)

    dt_input = xr.DataTree(name="S2A_MSIL2A_20250811T112131")
    updated_root_consolidation(
        dt_input=dt_input,
        output_path=str(output_path),
        datasets=datasets,
        additional_consolidate_paths=[path for path, ds in datasets.items() if ds is None],
    )

    for base_path in aux_groups:
        zmeta = json.loads((output_path / base_path.lstrip("/") / "zarr.json").read_text())
        consolidated = zmeta.get("consolidated_metadata")
        assert isinstance(consolidated, dict), f"{base_path}: not consolidated"
        assert set(consolidated["metadata"]) >= {
            *levels,
            *(f"{level}/b00" for level in levels),
        }, f"{base_path}: {sorted(consolidated['metadata'])}"

    # The reflectance pyramid is consolidated through the same list.
    zmeta = json.loads((output_path / "measurements/reflectance/zarr.json").read_text())
    assert "r10m" in zmeta["consolidated_metadata"]["metadata"]

    if "/quality/probability" not in aux_groups:
        assert not (output_path / "quality/probability").exists()


def test_write_store_geo_metadata_reprojects_utm_to_wgs84(tmp_path: Path) -> None:
    """Store-root bbox unions child-group bboxes and outputs EPSG:4326 coords."""
    import zarr

    store_path = tmp_path / "root.zarr"
    root = zarr.open_group(store_path, mode="w")
    child = root.create_group("measurements/reflectance")
    # UTM zone 32N, ~10x10km block near S2 tile T32TLQ
    child.attrs["spatial:bbox"] = [300000.0, 4890240.0, 409800.0, 5000040.0]
    child.attrs["proj:code"] = "EPSG:32632"

    write_store_geo_metadata(str(store_path))

    root_attrs = dict(zarr.open_group(store_path, mode="r").attrs)
    bbox = root_attrs["spatial:bbox"]
    assert isinstance(bbox, list)
    assert all(isinstance(coord, float) for coord in bbox)
    assert len(bbox) == 4
    xmin, ymin, xmax, ymax = (coord for coord in bbox if isinstance(coord, float))
    # Roughly 7.0-8.0E, 44.1-45.1N after reprojection
    assert 6.0 < xmin < 8.0, bbox
    assert 43.0 < ymin < 45.0, bbox
    assert 7.0 < xmax < 9.0, bbox
    assert 44.0 < ymax < 46.0, bbox
    # CRS is always declared explicitly at the store root
    assert root_attrs["proj:code"] == "EPSG:4326"


def test_write_store_geo_metadata_unions_multiple_children(tmp_path: Path) -> None:
    """Store-root bbox is the union across multiple child-group footprints."""
    import zarr

    store_path = tmp_path / "root.zarr"
    root = zarr.open_group(store_path, mode="w")
    a = root.create_group("a")
    a.attrs["spatial:bbox"] = [0.0, 0.0, 1.0, 1.0]
    a.attrs["proj:code"] = "EPSG:4326"
    b = root.create_group("b")
    b.attrs["spatial:bbox"] = [2.0, 2.0, 3.0, 3.0]
    b.attrs["proj:code"] = "EPSG:4326"

    write_store_geo_metadata(str(store_path))

    root_attrs = dict(zarr.open_group(store_path, mode="r").attrs)
    assert root_attrs["spatial:bbox"] == [0.0, 0.0, 3.0, 3.0]
    assert root_attrs["proj:code"] == "EPSG:4326"


def test_write_store_geo_metadata_densifies_edges(tmp_path: Path) -> None:
    """Reprojection uses transform_bounds, not just the two diagonal corners."""
    import zarr

    store_path = tmp_path / "root.zarr"
    root = zarr.open_group(store_path, mode="w")
    child = root.create_group("m")
    child.attrs["spatial:bbox"] = [699960.0, 4890240.0, 809760.0, 5000040.0]
    child.attrs["proj:code"] = "EPSG:32632"

    write_store_geo_metadata(str(store_path))

    bbox = dict(zarr.open_group(store_path, mode="r").attrs)["spatial:bbox"]
    assert isinstance(bbox, list)
    _xmin, ymin, _xmax, ymax = (float(cast("float", v)) for v in bbox)
    # Corner-only transformation yields ymin ~44.138 / ymax ~45.086; the
    # densified bounds extend past both (ymin ~44.0998, ymax ~45.1255).
    assert ymin < 44.11, bbox
    assert ymax > 45.11, bbox


def test_write_store_geo_metadata_resolves_wkt2_only_crs(tmp_path: Path) -> None:
    """A group carrying only proj:wkt2 is reprojected, not assumed to be degrees."""
    import zarr
    from pyproj import CRS

    store_path = tmp_path / "root.zarr"
    root = zarr.open_group(store_path, mode="w")
    child = root.create_group("m")
    child.attrs["spatial:bbox"] = [300000.0, 4890240.0, 409800.0, 5000040.0]
    child.attrs["proj:wkt2"] = CRS.from_epsg(32632).to_wkt()

    write_store_geo_metadata(str(store_path))

    bbox = dict(zarr.open_group(store_path, mode="r").attrs)["spatial:bbox"]
    assert isinstance(bbox, list)
    assert all(-180.0 <= float(cast("float", v)) <= 180.0 for v in bbox), bbox


def test_write_store_geo_metadata_skips_group_without_crs(tmp_path: Path) -> None:
    """A bbox with no resolvable CRS must not be unioned as if it were degrees."""
    import zarr

    store_path = tmp_path / "root.zarr"
    root = zarr.open_group(store_path, mode="w")
    good = root.create_group("good")
    good.attrs["spatial:bbox"] = [0.0, 0.0, 1.0, 1.0]
    good.attrs["proj:code"] = "EPSG:4326"
    bad = root.create_group("bad")
    bad.attrs["spatial:bbox"] = [300000.0, 4890240.0, 409800.0, 5000040.0]  # metres, no CRS

    write_store_geo_metadata(str(store_path))

    root_attrs = dict(zarr.open_group(store_path, mode="r").attrs)
    assert root_attrs["spatial:bbox"] == [0.0, 0.0, 1.0, 1.0]


def test_write_store_geo_metadata_invalid_code_does_not_raise(tmp_path: Path) -> None:
    """An unresolvable proj:code is skipped with a warning, never raised."""
    import zarr

    store_path = tmp_path / "root.zarr"
    root = zarr.open_group(store_path, mode="w")
    good = root.create_group("good")
    good.attrs["spatial:bbox"] = [0.0, 0.0, 1.0, 1.0]
    good.attrs["proj:code"] = "EPSG:4326"
    bad = root.create_group("bad")
    bad.attrs["spatial:bbox"] = [1.0, 1.0, 2.0, 2.0]
    bad.attrs["proj:code"] = "EPSG:999999"

    write_store_geo_metadata(str(store_path))  # must not raise

    root_attrs = dict(zarr.open_group(store_path, mode="r").attrs)
    assert root_attrs["spatial:bbox"] == [0.0, 0.0, 1.0, 1.0]


def test_write_store_geo_metadata_antimeridian_falls_back_to_full_range(tmp_path: Path) -> None:
    """A footprint crossing the antimeridian widens the root bbox to +/-180."""
    import zarr

    store_path = tmp_path / "root.zarr"
    root = zarr.open_group(store_path, mode="w")
    child = root.create_group("m")
    # UTM zone 1 tile column straddling the antimeridian
    child.attrs["spatial:bbox"] = [299940.0, 6390220.0, 409740.0, 6500020.0]
    child.attrs["proj:code"] = "EPSG:32601"

    write_store_geo_metadata(str(store_path))

    bbox = dict(zarr.open_group(store_path, mode="r").attrs)["spatial:bbox"]
    assert isinstance(bbox, list)
    assert bbox == [-180.0, bbox[1], 180.0, bbox[3]], bbox


@pytest.mark.parametrize(
    "deviation",
    [
        {"enable_sharding": False},
        {"spatial_chunk": 512},
        {"compression_level": 2},
        {"scale_offset_codec": True},
    ],
    ids=lambda d: next(iter(d)),
)
def test_convert_s2_optimized_refuses_non_policy_options(deviation: dict) -> None:
    """The plan-driven writer has a fixed write policy; a deviating layout
    option is refused before any work happens."""
    with pytest.raises(ValueError, match="fixed write policy"):
        convert_s2_optimized(
            xr.DataTree(),
            output_path="/unused",
            validate_output=False,
            **deviation,
        )


def test_add_s2_stac_assets_writes_reflectance_asset_to_root(tmp_path: Path) -> None:
    """The reflectance asset added by `add_s2_stac_assets` ends up in the consolidated root."""
    from pyproj import CRS

    # initialise an empty dataset to mimic the process and write it into the store as
    # the pipeline does, so zarr creates the root and parent groups
    output_path = str(tmp_path / "test.zarr")
    r10m = xr.Dataset(coords={"y": np.arange(10), "x": np.arange(20)})
    r10m.to_zarr(
        output_path,
        group="measurements/reflectance/r10m",
        mode="a",
        zarr_format=3,
        consolidated=False,
    )
    datasets: dict[str, tuple[str, ...] | None] = {"/measurements/reflectance/r10m": ()}

    # adding dummy data to dt_input to mimic input reference dtreee
    dt_input = xr.DataTree(xr.Dataset({"dummy": (["y"], np.zeros(10))}))
    dt_input.attrs = {"stac_discovery": {"properties": {"mission": "sentinel-2"}}}

    # test execution of func: the S2 assets are added before the root metadata is written
    add_s2_stac_assets(dt_input, datasets, crs=CRS.from_epsg(32632), reflectance_shape=(10, 20))
    updated_root_consolidation(
        output_path=output_path,
        datasets=datasets,
        dt_input=dt_input,
    )

    # read in the consolidated dTree root to cross-check with expected outcome
    root_attrs = json.loads((tmp_path / "test.zarr/zarr.json").read_text())["attributes"]
    asset = root_attrs["stac_discovery"]["assets"]["reflectance"]
    assert asset == {
        "href": "/measurements/reflectance",
        "type": "application/vnd.zarr; version=3; profile=multiscales",
        "title": "Surface Reflectance",
        "roles": ["data", "reflectance"],
        "gsd": 10,
        "proj:code": "EPSG:32632",
        "proj:shape": [10, 20],
    }
    # geozarr version is added to the processing history alongside the stac metadata
    assert root_attrs["stac_discovery"]["properties"]["processing:software"]["EOPF_GEOZARR"]


def test_add_multiscale_pyramids_to_stac_metadata_adds_missing_levels() -> None:
    """Pyramid levels absent from existing STAC assets get one asset per data
    variable, whether the level is given as a dataset or as its variable names."""
    existing_asset = {"href": "/measurements/reflectance/r10m/b02", "title": "B02"}
    attrs: dict = {"stac_discovery": {"assets": {"b02": dict(existing_asset)}}}
    datasets: dict[str, xr.Dataset | tuple[str, ...] | None] = {
        # already referenced by an existing asset -> must not be duplicated
        "/measurements/reflectance/r10m": xr.Dataset({"b02": (["y", "x"], np.zeros((4, 4)))}),
        # new pyramid level -> assets added, spatial_ref skipped
        "/measurements/reflectance/r20m": xr.Dataset(
            {
                "b02": (["y", "x"], np.zeros((2, 2))),
                "b05": (["y", "x"], np.zeros((2, 2))),
                "spatial_ref": ((), 0),
            }
        ),
        # the plan-driven converter passes variable names instead of datasets
        "/measurements/reflectance/r60m": ("b01", "spatial_ref"),
        "/measurements/reflectance": None,
    }

    result = add_multiscale_pyramids_to_stac_metadata(datasets, attrs)

    assert isinstance(result, dict)
    assert result["stac_discovery"]["assets"] == {
        "b02": existing_asset,
        "b02_r20m": {"href": "/measurements/reflectance/r20m/b02", "title": "b02_r20m"},
        "b05_r20m": {"href": "/measurements/reflectance/r20m/b05", "title": "b05_r20m"},
        "b01_r60m": {"href": "/measurements/reflectance/r60m/b01", "title": "b01_r60m"},
    }


@pytest.mark.parametrize("attrs", [{}, {"stac_discovery": {}}, {"stac_discovery": {"assets": {}}}])
def test_add_multiscale_pyramids_to_stac_metadata_without_assets_is_noop(attrs: dict) -> None:
    """Missing stac_discovery/assets returns the attributes unchanged instead of raising."""
    datasets = {
        "/measurements/reflectance/r20m": xr.Dataset({"b02": (["y", "x"], np.zeros((2, 2)))})
    }
    expected = json.loads(json.dumps(attrs))

    assert add_multiscale_pyramids_to_stac_metadata(datasets, attrs) == expected


if __name__ == "__main__":
    pytest.main([__file__])
