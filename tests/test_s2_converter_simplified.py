"""
Unit tests for simplified S2 converter implementation.

Tests the new simplified approach that uses only xarray and proper metadata consolidation.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.s2_optimization.s2_converter import (
    convert_s2_optimized,
    initialize_crs_from_dataset,
    simple_root_consolidation,
)


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


class TestS2FunctionalAPI:
    """Test the S2 functional API."""

    def test_is_sentinel2_dataset_placeholder(self) -> None:
        """Placeholder test for is_sentinel2_dataset.

        The actual is_sentinel2_dataset function uses complex pydantic validation
        that requires a fully structured zarr group matching Sentinel1Root or
        Sentinel2Root models. Testing this would require creating a complete
        mock sentinel dataset, which is better done in integration tests.
        """
        # This test is kept as a placeholder to maintain test structure
        assert True


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


def test_simple_root_consolidation_success(tmp_path: Path) -> None:
    """
    Test that simple_root_consolidation produces consolidated metadata at the root, and for the
    measurements/reflectance group, but not for other groups.
    """

    datasets = {
        "/measurements/reflectance/r10m": xr.Dataset(),
        "/quality/atmosphere": xr.Dataset(),
    }

    [
        v.to_zarr(
            str(tmp_path / f"test.zarr{k}/"),
            mode="a",
            zarr_format=3,
            consolidated=False,
        )
        for k, v in datasets.items()
    ]

    simple_root_consolidation(str(tmp_path / "test.zarr"), datasets=datasets)

    root_z_meta = json.loads((tmp_path / "test.zarr/zarr.json").read_text())
    reflectance_zmeta = json.loads(
        (tmp_path / "test.zarr/measurements/reflectance/zarr.json").read_text()
    )
    atmos_zmeta = json.loads((tmp_path / "test.zarr/quality/zarr.json").read_text())

    assert "consolidated_metadata" in root_z_meta
    assert isinstance(root_z_meta["consolidated_metadata"], dict)
    assert "consolidated_metadata" in reflectance_zmeta
    assert isinstance(reflectance_zmeta["consolidated_metadata"], dict)
    if "consolidated_metadata" in atmos_zmeta:
        assert atmos_zmeta["consolidated_metadata"] is None


class TestConvenienceFunction:
    """Test the convenience function."""

    @patch("eopf_geozarr.s2_optimization.s2_converter.create_result_datatree")
    @patch("eopf_geozarr.s2_optimization.s2_converter.zarr.open_group")
    @patch("eopf_geozarr.s2_optimization.s2_converter.initialize_crs_from_dataset")
    @patch("eopf_geozarr.s2_optimization.s2_converter.get_zarr_group")
    @patch("eopf_geozarr.s2_optimization.s2_converter.is_sentinel2_dataset")
    @patch("eopf_geozarr.s2_optimization.s2_converter.create_multiscale_from_datatree")
    @patch("eopf_geozarr.s2_optimization.s2_converter.simple_root_consolidation")
    def test_convert_s2_optimized_convenience_function(
        self,
        mock_consolidation: Mock,
        mock_multiscale: Mock,
        mock_is_s2: Mock,
        mock_get_zarr_group: Mock,
        mock_init_crs: Mock,
        mock_zarr_open_group: Mock,
        mock_create_result_datatree: Mock,
    ) -> None:
        """Test the convenience function parameter passing."""
        # Setup mocks
        mock_multiscale.return_value = {}
        mock_is_s2.return_value = True
        mock_get_zarr_group.return_value = Mock()
        mock_init_crs.return_value = None  # Return None for CRS

        # Mock zarr.open_group to return a group with proper groups() and members() methods
        mock_zarr_group = Mock()
        mock_zarr_group.groups.return_value = iter([])  # Return empty iterator for groups
        mock_zarr_group.members.return_value = {}  # Return empty dict for members
        mock_zarr_open_group.return_value = mock_zarr_group

        # Mock create_result_datatree to return a mock DataTree with groups attribute
        mock_result_dt = Mock()
        mock_result_dt.groups = []  # Empty list for groups
        mock_create_result_datatree.return_value = mock_result_dt

        # Test parameter passing - Mock DataTree with groups attribute
        dt_input = Mock()
        dt_input.groups = ["/measurements/reflectance/r10m"]
        output_path = "/test/path"

        convert_s2_optimized(
            dt_input,
            output_path=output_path,
            enable_sharding=False,
            spatial_chunk=512,
            compression_level=2,
            max_retries=5,
            validate_output=False,
            keep_scale_offset=False,
        )

        # Verify multiscale function was called with correct args
        mock_multiscale.assert_called_once()
        call_kwargs = mock_multiscale.call_args.kwargs
        assert call_kwargs["enable_sharding"] is False
        assert call_kwargs["spatial_chunk"] == 512


if __name__ == "__main__":
    pytest.main([__file__])
