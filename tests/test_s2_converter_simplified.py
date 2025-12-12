"""
Unit tests for simplified S2 converter implementation.

Tests the new simplified approach that uses only xarray and proper metadata consolidation.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.s2_optimization.s2_converter import (
    convert_s2_optimized,
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

    @pytest.mark.filterwarnings("ignore:.*:zarr.errors.UnstableSpecificationWarning")
    def test_convert_s2_optimized_convenience_function(
        self,
        s2_group_example: Path,
        tmp_path: Path,
    ) -> None:
        """Test the convenience function with real S2 data."""
        # Open the S2 example as a DataTree
        dt_input = xr.open_datatree(
            s2_group_example, engine="zarr", decode_timedelta=True, consolidated=False
        )
        output_path = str(tmp_path / "test_output.zarr")

        # Run the conversion
        result = convert_s2_optimized(
            dt_input,
            output_path=output_path,
            enable_sharding=False,
            spatial_chunk=512,
            compression_level=2,
            max_retries=5,
            validate_output=False,
        )

        # Verify the output was created
        assert Path(output_path).exists()

        # Verify the result is a DataTree
        assert isinstance(result, xr.DataTree)

        # Verify basic structure - output should have multiscale groups
        output_dt = xr.open_datatree(output_path, engine="zarr", decode_timedelta=True)
        assert len(output_dt.groups) > 0


if __name__ == "__main__":
    pytest.main([__file__])
