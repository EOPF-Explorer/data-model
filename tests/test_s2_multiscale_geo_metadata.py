"""
Unit tests for _write_dataset_geo_metadata method in S2MultiscalePyramid.

Tests the geographic metadata writing functionality added to level creation.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from affine import Affine
from zarr_cm import geo_proj
from zarr_cm import spatial as spatial_cm
from zarr_cm.spatial import CMO as SPATIAL_CMO

from eopf_geozarr.conversion.utils import (
    _rechunk_ds,
    create_uniform_encoding,
    stream_write_dataset,
    strip_cf_grid_mapping,
    write_dataset_geo_metadata,
)


@pytest.fixture
def sample_dataset_with_crs() -> xr.Dataset:
    """Create a sample dataset with CRS information."""
    coords = {
        "x": (["x"], np.linspace(0, 1000, 100)),
        "y": (["y"], np.linspace(0, 1000, 100)),
        "time": (["time"], [np.datetime64("2023-01-01")]),
    }

    data_vars = {
        "b02": (["time", "y", "x"], np.random.rand(1, 100, 100)),
        "b03": (["time", "y", "x"], np.random.rand(1, 100, 100)),
        "b04": (["y", "x"], np.random.rand(100, 100)),
    }

    ds = xr.Dataset(data_vars, coords=coords)

    ds["b02"].attrs["proj:epsg"] = 32632
    ds["b03"].attrs["proj:epsg"] = 32632
    ds["b04"].attrs["proj:epsg"] = 32632

    return ds


@pytest.fixture
def sample_dataset_with_epsg_attrs() -> xr.Dataset:
    """Create a sample dataset with EPSG in attributes."""
    coords = {
        "x": (["x"], np.linspace(0, 1000, 50)),
        "y": (["y"], np.linspace(0, 1000, 50)),
    }

    data_vars = {
        "b05": (["y", "x"], np.random.rand(50, 50)),
        "b06": (["y", "x"], np.random.rand(50, 50)),
    }

    ds = xr.Dataset(data_vars, coords=coords)

    # Add EPSG to variable attributes
    ds["b05"].attrs["proj:epsg"] = 32632
    ds["b06"].attrs["proj:epsg"] = 32632

    return ds


@pytest.fixture
def sample_dataset_no_crs() -> xr.Dataset:
    """Create a sample dataset without CRS information."""
    coords = {
        "x": (["x"], np.linspace(0, 1000, 25)),
        "y": (["y"], np.linspace(0, 1000, 25)),
    }

    data_vars = {
        "b11": (["y", "x"], np.random.rand(25, 25)),
        "b12": (["y", "x"], np.random.rand(25, 25)),
    }

    return xr.Dataset(data_vars, coords=coords)


class TestWriteGeoMetadata:
    """Test the _write_dataset_geo_metadata method."""

    def test_write_dataset_geo_metadata_with_rio_crs(
        self, sample_dataset_with_crs: xr.Dataset
    ) -> None:
        """Test _write_dataset_geo_metadata with dataset that has rioxarray CRS."""

        # Call the method
        result = write_dataset_geo_metadata(sample_dataset_with_crs)

        # The CRS is declared with the proj convention only (#352)
        assert result.attrs["proj:code"] == "EPSG:32632"
        assert "spatial_ref" not in result.variables

    def test_write_dataset_geo_metadata_with_epsg_attrs(
        self, sample_dataset_with_epsg_attrs: xr.Dataset
    ) -> None:
        """Test _write_dataset_geo_metadata with dataset that has EPSG in variable attributes."""

        # Verify initial state - no CRS
        assert (
            not hasattr(sample_dataset_with_epsg_attrs, "rio")
            or sample_dataset_with_epsg_attrs.rio.crs is None
        )

        # Call the method
        result = write_dataset_geo_metadata(sample_dataset_with_epsg_attrs)

        # The CRS read from the attributes is declared with the proj convention
        assert result.attrs["proj:code"] == "EPSG:32632"

    def test_write_dataset_geo_metadata_no_crs(self, sample_dataset_no_crs: xr.Dataset) -> None:
        """Test _write_dataset_geo_metadata with dataset that has no CRS information."""

        # Verify initial state - no CRS
        assert not hasattr(sample_dataset_no_crs, "rio") or sample_dataset_no_crs.rio.crs is None

        # Call the method - should not fail but also not add CRS
        result = write_dataset_geo_metadata(sample_dataset_no_crs)

        # Verify no CRS was added (method handles gracefully)
        assert result.rio.crs is None

    def test_write_dataset_geo_metadata_writes_no_cf_grid_mapping(
        self, sample_dataset_with_crs: xr.Dataset
    ) -> None:
        """The proj and spatial conventions are the only georeferencing written: no
        `spatial_ref` variable and no `grid_mapping` attribute (#352)."""
        result = write_dataset_geo_metadata(sample_dataset_with_crs)

        assert "spatial_ref" not in result.variables
        assert "grid_mapping" not in result.attrs
        assert all("grid_mapping" not in result[name].attrs for name in result.data_vars)

    def test_write_dataset_geo_metadata_preserves_existing_data(
        self, sample_dataset_with_crs: xr.Dataset
    ) -> None:
        """Test that _write_dataset_geo_metadata preserves existing data variables and coordinates."""

        # Store original data
        original_vars = list(sample_dataset_with_crs.data_vars.keys())
        original_coords = list(sample_dataset_with_crs.coords.keys())
        original_b02_data = sample_dataset_with_crs["b02"].values.copy()

        # Call the method
        result = write_dataset_geo_metadata(sample_dataset_with_crs)

        # Verify all original data is preserved
        assert list(result.data_vars.keys()) == original_vars
        assert all(coord in result.coords for coord in original_coords)
        assert np.array_equal(result["b02"].values, original_b02_data)

    def test_write_dataset_geo_metadata_empty_dataset(self) -> None:
        """An empty dataset has no data variable to detect a CRS from, so
        `write_dataset_geo_metadata` must no-op rather than raise or write metadata
        for a CRS it never found."""

        # addition of writing geo metadata to empty dataset should be empty

        empty_ds = xr.Dataset({}, coords={})

        result = write_dataset_geo_metadata(empty_ds)

        assert result.attrs == {}

    def test_write_dataset_geo_metadata_does_not_modify_input(
        self, sample_dataset_with_crs: xr.Dataset
    ) -> None:
        """The metadata is added to a returned copy; the input dataset is left unchanged."""
        attrs_before = dict(sample_dataset_with_crs.attrs)
        var_attrs_before = {k: dict(v.attrs) for k, v in sample_dataset_with_crs.data_vars.items()}

        result = write_dataset_geo_metadata(sample_dataset_with_crs)

        assert result is not sample_dataset_with_crs
        assert "zarr_conventions" in result.attrs
        assert dict(sample_dataset_with_crs.attrs) == attrs_before
        assert {
            k: dict(v.attrs) for k, v in sample_dataset_with_crs.data_vars.items()
        } == var_attrs_before

    def test_write_dataset_geo_metadata_crs_from_multiple_sources(self) -> None:
        """Test CRS detection from multiple sources in priority order."""

        # Create dataset with both rio CRS and EPSG attributes
        coords = {
            "x": (["x"], np.linspace(0, 1000, 50)),
            "y": (["y"], np.linspace(0, 1000, 50)),
        }

        data_vars = {"b08": (["y", "x"], np.random.rand(50, 50))}

        ds = xr.Dataset(data_vars, coords=coords)

        # Add both rio CRS and EPSG attribute (rio should take priority)
        ds = ds.rio.write_crs("EPSG:4326")  # Rio CRS
        ds["b08"].attrs["proj:epsg"] = 32632  # EPSG attribute

        # Call the method
        ds = write_dataset_geo_metadata(ds)

        # Verify rio CRS was used (priority over attributes)
        assert ds.rio.crs.to_epsg() == 4326  # Should still be 4326, not 32632

    def test_write_dataset_geo_metadata_integration_with_stream_write(self, tmp_path: Path) -> None:
        """Test that _write_dataset_geo_metadata is properly integrated in _stream_write_dataset."""

        # Create a simple dataset with CRS
        coords = {
            "x": (["x"], np.linspace(0, 1000, 100)),
            "y": (["y"], np.linspace(0, 1000, 100)),
        }

        data_vars = {
            "b02": (["y", "x"], np.random.rand(100, 100)),
        }

        ds = xr.Dataset(data_vars, coords=coords)
        ds = ds.rio.write_crs("EPSG:32632")

        ds = _rechunk_ds(ds, spatial_chunk=1024)

        # Create encoding for the dataset
        encoding = create_uniform_encoding(ds, enable_sharding=True)

        # `add_geo_metadata=True` makes stream_write_dataset call write_dataset_geo_metadata
        dataset_path = "/measurements/reflectance/r10m"
        stream_write_dataset(
            ds,
            path=dataset_path,
            group=zarr.create_group(tmp_path),
            encoding=encoding,
            add_geo_metadata=True,
        )

        # Re-open the written dataset to verify the CRS was persisted
        written_ds = xr.open_dataset(
            tmp_path, engine="zarr", chunks={}, decode_coords="all", group=dataset_path
        )

        # The CRS persists as the proj convention; the CF grid mapping the input
        # carried (rio.write_crs) is not written (#352)
        assert written_ds.attrs["proj:code"] == "EPSG:32632"
        assert "spatial_ref" not in written_ds.variables
        assert "grid_mapping" not in written_ds["b02"].attrs

    def test_write_dataset_geo_metadata_prefers_coordinate_transform_for_inconsistent_rio(
        self,
    ) -> None:
        """Derived datasets should derive spatial:transform from current coordinates."""

        x = 600030.0 + np.arange(3, dtype="float64") * 120.0
        y = 4899990.0 - np.arange(3, dtype="float64") * 120.0
        ds = xr.Dataset(
            {"b01": (["y", "x"], np.ones((3, 3), dtype=np.uint16))},
            coords={"x": x, "y": y},
        ).rio.write_crs("EPSG:32631")

        # A stale GeoTransform (60 m pixels) that no longer matches the 120 m grid.
        ds = ds.rio.write_transform(Affine(60.0, 0.0, 600030.0, 0.0, -60.0, 4899990.0))

        ds = write_dataset_geo_metadata(ds)

        # Origin is the outer pixel edge: half of the 120 m pixel outside the first centre.
        assert ds.attrs["spatial:transform"] == [120.0, 0.0, 599970.0, 0.0, -120.0, 4900050.0]
        assert ds.attrs["spatial:bbox"] == [599970.0, 4899690.0, 600330.0, 4900050.0]


class TestWriteGeoMetadataEdgeCases:
    """Test edge cases for _write_dataset_geo_metadata method."""

    def test_write_dataset_geo_metadata_invalid_crs(
        self,
    ) -> None:
        """Test _write_dataset_geo_metadata with invalid CRS data."""

        coords = {
            "x": (["x"], np.linspace(0, 1000, 10)),
            "y": (["y"], np.linspace(0, 1000, 10)),
        }

        data_vars = {"test_var": (["y", "x"], np.random.rand(10, 10))}

        ds = xr.Dataset(data_vars, coords=coords)

        # Add invalid EPSG code
        ds["test_var"].attrs["proj:epsg"] = "invalid_epsg"

        # Method should raise an exception for invalid CRS (normal behavior)
        from pyproj.exceptions import CRSError

        with pytest.raises(CRSError):
            write_dataset_geo_metadata(ds)

    def test_write_dataset_geo_metadata_mixed_crs_variables(
        self,
    ) -> None:
        """Test _write_dataset_geo_metadata with variables having different CRS information."""

        coords = {
            "x": (["x"], np.linspace(0, 1000, 20)),
            "y": (["y"], np.linspace(0, 1000, 20)),
        }

        data_vars = {
            "var1": (["y", "x"], np.random.rand(20, 20)),
            "var2": (["y", "x"], np.random.rand(20, 20)),
        }

        ds = xr.Dataset(data_vars, coords=coords)

        # Add different EPSG codes to different variables
        ds["var1"].attrs["proj:epsg"] = 32632
        ds["var2"].attrs["proj:epsg"] = 4326

        # Call the method (should use the first CRS found)
        ds = write_dataset_geo_metadata(ds)

        # Verify a CRS was applied (should be the first one found)
        assert hasattr(ds, "rio")

    def test_write_dataset_geo_metadata_maintains_dataset_attrs(
        self, sample_dataset_with_crs: xr.Dataset
    ) -> None:
        """Test that _write_dataset_geo_metadata maintains dataset-level attributes."""

        # Add some dataset attributes
        sample_dataset_with_crs.attrs["pyramid_level"] = 1
        sample_dataset_with_crs.attrs["resolution_meters"] = 20
        sample_dataset_with_crs.attrs["custom_attr"] = "test_value"

        original_attrs = sample_dataset_with_crs.attrs.copy()

        # Call the method
        result = write_dataset_geo_metadata(sample_dataset_with_crs)

        # Verify dataset attributes are preserved
        for key, value in original_attrs.items():
            assert result.attrs[key] == value

    def test_write_dataset_geo_metadata_adds_zarr_conventions(
        self, sample_dataset_with_crs: xr.Dataset
    ) -> None:
        """Test that zarr_conventions are properly added to dataset attributes."""

        # Verify zarr_conventions is not initially present
        assert "zarr_conventions" not in sample_dataset_with_crs.attrs

        # Call the method
        result = write_dataset_geo_metadata(sample_dataset_with_crs)

        # Verify zarr_conventions was added
        assert "zarr_conventions" in result.attrs
        zarr_conventions = result.attrs["zarr_conventions"]

        # Verify it's a list with 2 conventions
        assert isinstance(zarr_conventions, list)
        assert len(zarr_conventions) == 2

        # Verify the conventions contain the expected metadata
        spatial_convention = None
        proj_convention = None

        for convention in zarr_conventions:
            if convention.get("name") == "spatial":
                spatial_convention = convention
            elif convention.get("name") == "proj":
                proj_convention = convention

        # Verify spatial convention
        assert spatial_convention is not None
        expected_spatial = spatial_cm.CMO
        assert spatial_convention == expected_spatial

        # Verify proj convention
        assert proj_convention is not None
        expected_proj = geo_proj.CMO
        assert proj_convention == expected_proj

    def test_write_dataset_geo_metadata_adds_spatial_and_proj_attributes(
        self, sample_dataset_with_crs: xr.Dataset
    ) -> None:
        """Test that spatial and proj attributes are added along with conventions."""

        # Call the method
        result = write_dataset_geo_metadata(sample_dataset_with_crs)

        # Verify spatial attributes
        assert "spatial:dimensions" in result.attrs
        assert result.attrs["spatial:dimensions"] == ["y", "x"]
        assert "spatial:registration" in result.attrs
        assert result.attrs["spatial:registration"] == "pixel"
        assert "spatial:bbox" in result.attrs
        assert "spatial:shape" in result.attrs

        # Verify proj attributes (should have either proj:code or proj:wkt2)
        has_proj_code = "proj:code" in result.attrs
        has_proj_wkt2 = "proj:wkt2" in result.attrs
        assert has_proj_code or has_proj_wkt2, "Should have either proj:code or proj:wkt2"

        # Verify zarr_conventions includes both spatial and proj
        zarr_conventions = result.attrs["zarr_conventions"]
        convention_names = [conv.get("name") for conv in zarr_conventions]
        assert "spatial" in convention_names
        assert "proj" in convention_names

    def test_write_dataset_geo_metadata_no_crs_no_conventions(
        self, sample_dataset_no_crs: xr.Dataset
    ) -> None:
        """Test that zarr_conventions are not added when no CRS is available."""

        # Call the method with no CRS
        result = write_dataset_geo_metadata(sample_dataset_no_crs)

        # Verify zarr_conventions was not added since no CRS was available
        assert "zarr_conventions" not in result.attrs


def _grid_mapped_dataset() -> xr.Dataset:
    """A dataset whose CRS is declared only by a CF grid mapping (rioxarray's)."""
    ds = xr.Dataset(
        {"b02": (["y", "x"], np.zeros((4, 4)))},
        coords={"x": np.arange(4) * 10.0 + 5, "y": 35.0 - np.arange(4) * 10.0},
    )
    out = ds.rio.write_crs("EPSG:32632")
    assert isinstance(out, xr.Dataset)
    return out


@pytest.mark.parametrize(
    ("case", "expected_code", "kept_spatial_ref"),
    [
        ("grid-mapped", "EPSG:32632", False),
        ("grid-mapped-with-proj", "EPSG:4326", False),
        ("unrelated-spatial_ref", None, True),
        ("grid-mapped-with-legacy-epsg", "EPSG:32632", False),
        ("grid-mapped-with-spatial-declared", "EPSG:32632", False),
    ],
)
def test_strip_cf_grid_mapping(
    case: str, expected_code: str | None, kept_spatial_ref: bool
) -> None:
    """
    The CF grid mapping is dropped, its CRS carried over as `proj:code` when the
    dataset declares none (#352); a declared `proj:code` is kept, and a variable
    named `spatial_ref` that is not a grid mapping is not touched.
    """
    if case == "unrelated-spatial_ref":
        ds = xr.Dataset({"spatial_ref": (["n"], np.arange(3))})
    else:
        ds = _grid_mapped_dataset()
        if case == "grid-mapped-with-proj":
            ds.attrs["proj:code"] = "EPSG:4326"
        elif case == "grid-mapped-with-legacy-epsg":
            # a legacy key, not a proj convention declaration
            ds.attrs["proj:epsg"] = 32632
        elif case == "grid-mapped-with-spatial-declared":
            ds.attrs["zarr_conventions"] = [dict(SPATIAL_CMO)]
    result = strip_cf_grid_mapping(ds)
    assert result.attrs.get("proj:code") == expected_code
    if case == "grid-mapped-with-spatial-declared":
        # the carried proj declaration extends the declared ones
        names = [entry["name"] for entry in result.attrs["zarr_conventions"]]
        assert names == ["spatial", "proj"]
    assert ("spatial_ref" in result.variables) is kept_spatial_ref
    assert all("grid_mapping" not in v.attrs for v in result.variables.values())
    assert all("grid_mapping" not in v.encoding for v in result.variables.values())


def test_stream_write_keeps_the_crs_of_a_grid_mapped_input(tmp_path: Path) -> None:
    """
    Without `add_geo_metadata` (the generic rechunker), a dataset whose CRS is
    declared only by a CF grid mapping is written with that CRS as `proj:code`,
    not with no CRS at all, and without the grid mapping (#352).
    """
    ds = _rechunk_ds(_grid_mapped_dataset(), spatial_chunk=1024)
    encoding = create_uniform_encoding(ds, enable_sharding=True)
    stream_write_dataset(ds, path="/data", group=zarr.create_group(tmp_path), encoding=encoding)
    written = xr.open_dataset(tmp_path, engine="zarr", group="/data", decode_coords="all")
    assert written.attrs["proj:code"] == "EPSG:32632"
    assert "spatial_ref" not in written.variables
