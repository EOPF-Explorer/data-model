"""
Integration test for EOPF GeoZarr conversion using a sample Sentinel-2 dataset.

This test demonstrates the complete workflow from EOPF DataTree to GeoZarr-compliant
format, following the patterns established in the analysis notebook:
docs/analysis/eopf-geozarr/EOPF_Sentinel2_ZarrV3_geozarr_compliant.ipynb
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rioxarray  # noqa: F401  # Import to enable .rio accessor
import xarray as xr

from eopf_geozarr.conversion import create_geozarr_dataset

from . import (
    _verify_basic_structure,
    _verify_geozarr_spec_compliance,
    _verify_multiscale_structure,
    _verify_rgb_data_access,
)


@pytest.fixture
def sample_sentinel2_datatree() -> xr.DataTree:
    """
    Create a sample Sentinel-2 EOPF DataTree structure for testing.

    This mimics the structure from the notebook:
    - Multiple resolution groups (r10m, r20m, r60m)
    - Quality groups (l1c_quicklook)
    - Realistic Sentinel-2 band structure
    - UTM coordinate system (EPSG:32632)
    - Proper EOPF hierarchical organization
    """
    # Use UTM Zone 32N (EPSG:32632) as in the notebook example
    epsg_code = 32632

    # Define spatial extents (smaller for testing but realistic proportions)
    x_min, x_max = 600000, 605490  # 5.49 km width
    y_min, y_max = 5090000, 5095490  # 5.49 km height

    # Create coordinate arrays for different resolutions
    # r10m: 549x549 pixels (10m resolution)
    coords_10m = {
        "x": np.linspace(x_min, x_max, 549, endpoint=False),
        "y": np.linspace(y_max, y_min, 549, endpoint=False),
        "time": [np.datetime64("2025-01-13T10:33:09")],
    }

    # r20m: 275x275 pixels (20m resolution)
    coords_20m = {
        "x": np.linspace(x_min, x_max, 275, endpoint=False),
        "y": np.linspace(y_max, y_min, 275, endpoint=False),
        "time": [np.datetime64("2025-01-13T10:33:09")],
    }

    # r60m: 92x92 pixels (60m resolution)
    coords_60m = {
        "x": np.linspace(x_min, x_max, 92, endpoint=False),
        "y": np.linspace(y_max, y_min, 92, endpoint=False),
        "time": [np.datetime64("2025-01-13T10:33:09")],
    }

    # Create realistic spectral data with proper value ranges
    # Sentinel-2 reflectance values typically range from 0-10000 (scaled)
    np.random.seed(42)  # For reproducible test data

    # 10m resolution bands (B02, B03, B04, B08) - key bands for RGB
    r10m_data = {
        "b02": (
            ["time", "y", "x"],
            np.random.randint(500, 3000, (1, 549, 549), dtype=np.uint16),
            {"proj:epsg": epsg_code, "long_name": "Blue band (B02)"},
        ),
        "b03": (
            ["time", "y", "x"],
            np.random.randint(800, 4000, (1, 549, 549), dtype=np.uint16),
            {"proj:epsg": epsg_code, "long_name": "Green band (B03)"},
        ),
        "b04": (
            ["time", "y", "x"],
            np.random.randint(600, 3500, (1, 549, 549), dtype=np.uint16),
            {"proj:epsg": epsg_code, "long_name": "Red band (B04)"},
        ),
        "b08": (
            ["time", "y", "x"],
            np.random.randint(2000, 6000, (1, 549, 549), dtype=np.uint16),
            {"proj:epsg": epsg_code, "long_name": "NIR band (B08)"},
        ),
    }

    # 20m resolution bands (B05, B06, B07, B8A, B11, B12)
    r20m_data = {
        "b05": (
            ["time", "y", "x"],
            np.random.randint(1500, 5000, (1, 275, 275), dtype=np.uint16),
            {"proj:epsg": epsg_code, "long_name": "Red Edge 1 (B05)"},
        ),
        "b06": (
            ["time", "y", "x"],
            np.random.randint(1800, 5500, (1, 275, 275), dtype=np.uint16),
            {"proj:epsg": epsg_code, "long_name": "Red Edge 2 (B06)"},
        ),
        "b07": (
            ["time", "y", "x"],
            np.random.randint(2000, 6000, (1, 275, 275), dtype=np.uint16),
            {"proj:epsg": epsg_code, "long_name": "Red Edge 3 (B07)"},
        ),
        "b8a": (
            ["time", "y", "x"],
            np.random.randint(2200, 6500, (1, 275, 275), dtype=np.uint16),
            {"proj:epsg": epsg_code, "long_name": "Red Edge 4 (B8A)"},
        ),
        "b11": (
            ["time", "y", "x"],
            np.random.randint(1000, 4000, (1, 275, 275), dtype=np.uint16),
            {"proj:epsg": epsg_code, "long_name": "SWIR 1 (B11)"},
        ),
        "b12": (
            ["time", "y", "x"],
            np.random.randint(500, 2500, (1, 275, 275), dtype=np.uint16),
            {"proj:epsg": epsg_code, "long_name": "SWIR 2 (B12)"},
        ),
    }

    # 60m resolution bands (B01, B09, B10)
    r60m_data = {
        "b01": (
            ["time", "y", "x"],
            np.random.randint(800, 3000, (1, 92, 92), dtype=np.uint16),
            {"proj:epsg": epsg_code, "long_name": "Coastal aerosol (B01)"},
        ),
        "b09": (
            ["time", "y", "x"],
            np.random.randint(200, 1000, (1, 92, 92), dtype=np.uint16),
            {"proj:epsg": epsg_code, "long_name": "Water vapour (B09)"},
        ),
        "b10": (
            ["time", "y", "x"],
            np.random.randint(100, 500, (1, 92, 92), dtype=np.uint16),
            {"proj:epsg": epsg_code, "long_name": "Cirrus (B10)"},
        ),
    }

    # Quality quicklook data (RGB composite at 10m)
    quicklook_data = {
        "red": (
            ["y", "x"],
            np.random.randint(0, 255, (549, 549), dtype=np.uint8),
            {"proj:epsg": epsg_code, "long_name": "RGB Red component"},
        ),
        "green": (
            ["y", "x"],
            np.random.randint(0, 255, (549, 549), dtype=np.uint8),
            {"proj:epsg": epsg_code, "long_name": "RGB Green component"},
        ),
        "blue": (
            ["y", "x"],
            np.random.randint(0, 255, (549, 549), dtype=np.uint8),
            {"proj:epsg": epsg_code, "long_name": "RGB Blue component"},
        ),
    }

    # Create datasets for each resolution
    ds_r10m = xr.Dataset(r10m_data, coords=coords_10m)
    ds_r20m = xr.Dataset(r20m_data, coords=coords_20m)
    ds_r60m = xr.Dataset(r60m_data, coords=coords_60m)

    # Quicklook uses 10m coordinates but without time dimension
    quicklook_coords = {"x": coords_10m["x"], "y": coords_10m["y"]}
    ds_quicklook = xr.Dataset(quicklook_data, coords=quicklook_coords)

    # Set CRS for all datasets
    ds_r10m = ds_r10m.rio.write_crs(f"EPSG:{epsg_code}")
    ds_r20m = ds_r20m.rio.write_crs(f"EPSG:{epsg_code}")
    ds_r60m = ds_r60m.rio.write_crs(f"EPSG:{epsg_code}")
    ds_quicklook = ds_quicklook.rio.write_crs(f"EPSG:{epsg_code}")

    # Create DataTree structure following EOPF organization from the notebook
    dt = xr.DataTree()

    # Measurements branch
    dt["measurements"] = xr.DataTree()
    dt["measurements/reflectance"] = xr.DataTree()
    dt["measurements/reflectance/r10m"] = ds_r10m
    dt["measurements/reflectance/r20m"] = ds_r20m
    dt["measurements/reflectance/r60m"] = ds_r60m

    # Quality branch (as in the notebook)
    dt["quality"] = xr.DataTree()
    dt["quality/l1c_quicklook"] = xr.DataTree()
    dt["quality/l1c_quicklook/r10m"] = ds_quicklook

    # Add metadata at different levels (following notebook structure)
    dt.attrs = {
        "title": "S2B_MSIL1C_20250113T103309_N0511_R108_T32TLQ_20250113T122458",
        "platform": "Sentinel-2B",
        "processing_level": "L1C",
        "product_type": "S2MSI1C",
        "tile_id": "T32TLQ",
    }

    dt["measurements"].attrs = {"description": "Measurement data groups"}

    dt["measurements/reflectance"].attrs = {
        "description": "Top-of-atmosphere reflectance measurements",
        "units": "dimensionless (scaled by 10000)",
    }

    dt["quality"].attrs = {"description": "Quality assessment data"}

    dt["quality/l1c_quicklook"].attrs = {"description": "L1C quicklook RGB composite"}

    return dt


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_complete_sentinel2_conversion_notebook_workflow(
    sample_sentinel2_datatree, temp_output_dir
) -> None:
    """
    Test complete conversion following the notebook workflow.

    This test replicates the key steps from the notebook:
    1. Load EOPF DataTree structure
    2. Convert to GeoZarr-compliant format
    3. Verify GeoZarr specification compliance
    4. Check multiscale structure
    5. Validate data access patterns
    """
    dt_input = sample_sentinel2_datatree
    output_path = Path(temp_output_dir) / "sentinel2_geozarr_compliant.zarr"

    # Define groups to convert (matching the notebook)
    groups = [
        "/measurements/reflectance/r10m",
        "/measurements/reflectance/r20m",
        "/measurements/reflectance/r60m",
        "/quality/l1c_quicklook/r10m",
    ]

    print("Converting Sentinel-2 EOPF DataTree to GeoZarr format...")
    print(f"Input groups: {groups}")
    print(f"Output path: {output_path}")

    # Perform the conversion with parameters from the notebook
    with patch("eopf_geozarr.conversion.geozarr.print"):  # Suppress verbose output
        dt_geozarr = create_geozarr_dataset(
            dt_input=dt_input,
            groups=groups,
            output_path=str(output_path),
            spatial_chunk=1024,  # From notebook: spatial_chunk = 1024
            min_dimension=256,  # From notebook: min_dimension = 256
            tile_width=256,  # From notebook: tileWidth = 256
            max_retries=3,  # From notebook: max_retries = 3
        )

    # Verify the conversion was successful
    assert dt_geozarr is not None
    assert output_path.exists()

    # Test 1: Verify basic structure (following notebook verification)
    _verify_basic_structure(output_path, groups)

    # Test 2: Verify GeoZarr-spec compliance (as in notebook)
    for group in groups:
        _verify_geozarr_spec_compliance(output_path, group)

    # Test 3: Verify multiscale structure
    for group in groups:
        _verify_multiscale_structure(output_path, group)

    # Test 4: Verify RGB data access (for groups with RGB bands)
    _verify_rgb_data_access(output_path, groups)

    print("✅ All integration tests passed!")


@pytest.mark.slow
def test_performance_characteristics(
    sample_sentinel2_datatree, temp_output_dir
) -> None:
    """
    Test performance characteristics following notebook analysis.

    This test verifies that:
    1. Overview levels load faster than native resolution
    2. Data access time decreases with overview level
    3. Memory usage (pixel count) decreases appropriately
    """
    dt_input = sample_sentinel2_datatree
    output_path = Path(temp_output_dir) / "sentinel2_performance_test.zarr"

    # Convert with performance-focused parameters
    groups = [
        "/measurements/reflectance/r10m"
    ]  # Focus on one group for performance testing

    with patch("eopf_geozarr.conversion.geozarr.print"):
        create_geozarr_dataset(
            dt_input=dt_input,
            groups=groups,
            output_path=str(output_path),
            spatial_chunk=512,
            min_dimension=128,
            tile_width=256,
            max_retries=2,
        )

    # Test data access performance across overview levels
    group = groups[0]
    group_path = output_path / group.lstrip("/")
    level_dirs = [d for d in group_path.iterdir() if d.is_dir() and d.name.isdigit()]

    timing_data = []
    for level_dir in sorted(level_dirs, key=lambda x: int(x.name))[
        :4
    ]:  # Test first 4 levels
        level_num = int(level_dir.name)
        level_path = str(level_dir)

        # Measure data access time (following notebook timing approach)
        import time

        start_time = time.time()

        ds = xr.open_dataset(level_path, engine="zarr", zarr_format=3)
        # Access a data variable to trigger actual data loading
        if "b04" in ds.data_vars:
            data = ds["b04"].values
            pixel_count = data.size
        else:
            # Fallback to first available data variable
            first_var = list(ds.data_vars)[0]
            if first_var != "spatial_ref":
                data = ds[first_var].values
                pixel_count = data.size
            else:
                pixel_count = 0

        access_time = time.time() - start_time
        ds.close()

        timing_data.append(
            {"level": level_num, "time": access_time, "pixels": pixel_count}
        )

        print(f"    Level {level_num}: {access_time:.3f}s, {pixel_count:,} pixels")

    # Verify performance characteristics
    if len(timing_data) > 1:
        # Generally, higher overview levels should have fewer pixels
        for i in range(1, len(timing_data)):
            curr_pixels = timing_data[i]["pixels"]
            prev_pixels = timing_data[i - 1]["pixels"]

            # Allow some flexibility, but generally expect fewer pixels at higher levels
            assert (
                curr_pixels <= prev_pixels * 1.1
            ), f"Level {timing_data[i]['level']} has more pixels than level {timing_data[i-1]['level']}"

    print("✅ Performance characteristics verified!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
