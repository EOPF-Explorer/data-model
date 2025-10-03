"""Tests for CRS metadata embedding in Zarr attributes."""

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.conversion.geozarr import prepare_dataset_with_crs_info

EPSG_32632 = 32632


@pytest.fixture
def utm_dataset():
    """Minimal UTM dataset."""
    return xr.Dataset(
        {"data": (["y", "x"], np.random.rand(10000, 11000))},
        coords={
            "x": np.arange(0, 110000, 10, dtype=np.float64),
            "y": np.arange(5300000, 5200000, -10, dtype=np.float64),
        },
    )


@pytest.fixture
def zarr_roundtrip(utm_dataset, tmp_path):
    """Dataset after Zarr write/read."""
    ds = prepare_dataset_with_crs_info(utm_dataset, reference_crs=f"EPSG:{EPSG_32632}")
    path = tmp_path / "test.zarr"
    ds.drop_encoding().to_zarr(path, mode="w", zarr_format=3, consolidated=True)
    return xr.open_zarr(path, zarr_format=3, consolidated=True)


def test_crs_attrs_embedded(utm_dataset):
    """CRS metadata written to attrs."""
    ds = prepare_dataset_with_crs_info(utm_dataset, reference_crs=f"EPSG:{EPSG_32632}")
    assert ds.attrs["proj:epsg"] == EPSG_32632
    assert ds.attrs["proj:code"] == f"EPSG:{EPSG_32632}"


def test_crs_persists_through_zarr(zarr_roundtrip):
    """CRS survives Zarr cycle."""
    assert zarr_roundtrip.attrs["proj:epsg"] == EPSG_32632


def test_rioxarray_reads_crs(zarr_roundtrip):
    """rioxarray reads CRS."""
    assert zarr_roundtrip.rio.crs.to_epsg() == EPSG_32632
