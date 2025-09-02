"""Integration tests for Sentinel-1 GeoZarr conversion."""

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.conversion import create_geozarr_dataset

@pytest.fixture
def sample_sentinel1_dataset():
    """Create a sample Sentinel-1 dataset with GCPs."""
    # Create sample dimensions
    nx, ny = 100, 200
    npoints = 4  # 2x2 GCP grid
    
    # Create coordinates
    ds = xr.Dataset(
        coords={
            "x": np.arange(nx),
            "y": np.arange(ny),
        }
    )
    
    # Add measurement data
    ds["measurement"] = xr.DataArray(
        np.random.random((ny, nx)),
        dims=["y", "x"],
        attrs={
            "standard_name": "surface_backwards_scattering_coefficient_of_radar_wave",
            "units": "1",
        }
    )
    
    # Create GCP dataset
    gcp_ds = xr.Dataset()
    points = np.arange(npoints)
    gcp_ds.coords["points"] = points
    
    # Create regular grid of GCPs
    lons = np.array([10.0, 10.1, 10.0, 10.1])  # 2x2 grid
    lats = np.array([45.0, 45.0, 45.1, 45.1])
    heights = np.zeros(npoints)
    
    gcp_ds["longitude"] = xr.DataArray(lons, dims=["points"])
    gcp_ds["latitude"] = xr.DataArray(lats, dims=["points"])
    gcp_ds["height"] = xr.DataArray(heights, dims=["points"])
    
    # Create DataTree with measurements and GCP group
    dt = xr.DataTree()
    dt["measurements"] = ds
    dt["conditions/gcp"] = gcp_ds
    
    # Add Sentinel-1 product type
    dt.attrs["stac_discovery"] = {
        "properties": {
            "product:type": "S01_GRD"
        }
    }
    
    return dt

def test_invalid_gcp_group_raises_error(tmp_path, sample_sentinel1_dataset):
    """Test that specifying a non-existent GCP group raises an error."""
    output_path = tmp_path / "test_s1_invalid_gcp.zarr"
    groups = ["measurements"]
    
    # Try with invalid GCP group
    with pytest.raises(ValueError, match="GCP group 'invalid/gcp' not found"):
        create_geozarr_dataset(
            sample_sentinel1_dataset,
            groups=groups,
            output_path=str(output_path),
            gcp_group="invalid/gcp"
        )

def test_sentinel1_gcp_conversion(tmp_path, sample_sentinel1_dataset):
    """Test conversion of Sentinel-1 data with GCPs."""
    # Prepare test
    output_path = tmp_path / "test_s1_gcp.zarr"
    groups = ["measurements"]
    
    # Execute conversion
    result = create_geozarr_dataset(
        sample_sentinel1_dataset,
        groups=groups,
        output_path=str(output_path),
        gcp_group="conditions/gcp"
    )
    
    # Load the result for validation
    ds = xr.open_zarr(output_path)
    
    # Check basic structure
    assert "measurements" in ds
    assert "spatial_ref" in ds.measurements
    
    # Verify Sentinel-1 specific metadata
    meas = ds.measurements.measurement
    assert meas.attrs["standard_name"] == "surface_backwards_scattering_coefficient_of_radar_wave"
    assert meas.attrs["units"] == "1"
    assert meas.attrs["grid_mapping"] == "spatial_ref"
    
    # Verify GCP handling
    spatial_ref = ds.measurements.spatial_ref
    assert "gcps" in spatial_ref.attrs  # rioxarray adds GCP info here
    assert spatial_ref.attrs["crs"] == "EPSG:4326"  # GCPs are in lat/lon
    
    # Verify GCP storage format and coordinates
    gcps = spatial_ref.attrs["gcps"]
    assert len(gcps) == 4  # We created 4 GCPs
    assert all(hasattr(gcp, "row") for gcp in gcps)  # Check GCP attributes
    assert all(hasattr(gcp, "col") for gcp in gcps)
    assert all(hasattr(gcp, "x") for gcp in gcps)
    assert all(hasattr(gcp, "y") for gcp in gcps)
    assert all(hasattr(gcp, "z") for gcp in gcps)
    assert all(isinstance(gcp.id, str) for gcp in gcps)
    
    # Check first GCP coordinates
    gcp0 = gcps[0]
    np.testing.assert_allclose(gcp0.x, 10.0)  # longitude
    np.testing.assert_allclose(gcp0.y, 45.0)  # latitude
    np.testing.assert_allclose(gcp0.z, 0.0)   # height
    
    # Check that row/col values map to correct grid positions
    assert gcp0.row == 0
    assert gcp0.col == 0
    
    # Check last GCP to verify grid mapping
    gcp3 = gcps[3]
    np.testing.assert_allclose(gcp3.x, 10.1)  # longitude
    np.testing.assert_allclose(gcp3.y, 45.1)  # latitude
    np.testing.assert_allclose(gcp3.col, sample_sentinel1_dataset["measurements"].dims["x"] - 1)
    np.testing.assert_allclose(gcp3.row, sample_sentinel1_dataset["measurements"].dims["y"] - 1)
    
    # Verify no multiscales were created (as per ADR-102)
    measurements_group = xr.open_zarr(output_path).measurements
    assert not any(k.startswith("1") for k in measurements_group.data_vars.keys())
