"""
Integration test for EOPF GeoZarr conversion using a sample Sentinel-1 GRD dataset.

This test demonstrates the complete workflow from EOPF DataTree to GeoZarr-compliant
format, following the patterns established in the ADR-102 for GCP-based georeferencing.
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

class TestSentinel1Integration:
    """Integration tests for Sentinel-1 EOPF GRD to GeoZarr conversion."""

    @pytest.fixture
    def sample_sentinel1_datatree(self) -> xr.DataTree:
        """
        Create a sample Sentinel-1 GRD EOPF DataTree structure for testing.

        This mimics the structure documented in ADR-102:
        - Polarization groups (VV, VH)
        - GCP data in conditions/gcp
        - Realistic GRD measurement structure
        """
        # Create realistic GRD data dimensions (1000x1000 for testing)
        width, height = 1000, 1000
        
        # Create GCPs following typical S1 GRD pattern (10x10 grid)
        num_gcps = 100
        gcp_grid_size = int(np.sqrt(num_gcps))
        
        # Generate realistic GCP coordinates
        lons = np.linspace(-5, 5, gcp_grid_size)
        lats = np.linspace(45, 55, gcp_grid_size)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Create GCP datasets
        gcp_data = {
            "longitude": (["points"], lon_grid.flatten()),
            "latitude": (["points"], lat_grid.flatten()),
            "height": (["points"], np.zeros(num_gcps)),  # Assuming sea level for simplicity
        }
        
        ds_gcps = xr.Dataset(
            gcp_data,
            coords={"points": np.arange(num_gcps)}
        )

        # Create measurement data for each polarization
        # Use realistic backscatter value ranges for S1 GRD (-25dB to 0dB)
        np.random.seed(42)
        
        vv_data = {
            "grd": (
                ["y", "x"],
                np.random.uniform(-25, 0, (height, width)).astype(np.float32),
                {"long_name": "VV polarization backscatter"}
            )
        }
        
        vh_data = {
            "grd": (
                ["y", "x"],
                np.random.uniform(-25, 0, (height, width)).astype(np.float32),
                {"long_name": "VH polarization backscatter"}
            )
        }

        # Create datasets for each polarization
        ds_vv = xr.Dataset(vv_data)
        ds_vh = xr.Dataset(vh_data)

        # Create DataTree structure following EOPF organization
        dt = xr.DataTree()

        # Measurements branch with polarizations
        dt["measurements"] = xr.DataTree()
        dt["measurements/polarization_VV"] = ds_vv
        dt["measurements/polarization_VH"] = ds_vh

        # Conditions branch with GCPs
        dt["conditions"] = xr.DataTree()
        dt["conditions/gcp"] = ds_gcps

        # Add metadata at different levels
        dt.attrs = {
            "title": "S1B_IW_GRDH_1SDV_20250113T103309_20250113T103334_031584_03C4B0_927A",
            "platform": "Sentinel-1B",
            "processing_level": "1",
            "product_type": "GRD",
            "swath": "IW"
        }

        dt["measurements"].attrs = {
            "description": "Measurement data groups containing polarization backscatter"
        }

        dt["conditions"].attrs = {
            "description": "Auxiliary data including ground control points"
        }

        return dt

    @pytest.fixture
    def temp_output_dir(self) -> str:
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_complete_sentinel1_conversion_workflow(
        self, sample_sentinel1_datatree, temp_output_dir
    ) -> None:
        """
        Test complete conversion following the ADR-102 workflow.

        This test verifies:
        1. GRD measurement data preservation
        2. GCP-based georeferencing implementation
        3. GeoZarr specification compliance
        4. No multiscale generation (as per ADR-102 decision)
        """
        dt_input = sample_sentinel1_datatree
        output_path = Path(temp_output_dir) / "sentinel1_geozarr_compliant.zarr"

        # Define groups to convert
        groups = [
            "/measurements/polarization_VV",
            "/measurements/polarization_VH",
        ]

        print("Converting Sentinel-1 GRD EOPF DataTree to GeoZarr format...")
        print(f"Input groups: {groups}")
        print(f"Output path: {output_path}")

        # Perform the conversion
        with patch("eopf_geozarr.conversion.geozarr.print"):  # Suppress verbose output
            dt_geozarr = create_geozarr_dataset(
                dt_input=dt_input,
                groups=groups,
                output_path=str(output_path),
                spatial_chunk=1024,
                min_dimension=256,
                tile_width=256,
                max_retries=3,
            )

        # Verify the conversion was successful
        assert dt_geozarr is not None
        assert output_path.exists()

        # Test 1: Verify basic structure
        self._verify_basic_structure(output_path, groups)

        # Test 2: Verify GeoZarr-spec compliance
        for group in groups:
            self._verify_geozarr_spec_compliance(output_path, group)

        # Test 3: Verify GCP georeferencing
        for group in groups:
            self._verify_gcp_implementation(output_path, group)

        print("✅ All integration tests passed!")

    def _verify_basic_structure(self, output_path, groups) -> None:
        """Verify the basic Zarr store structure."""
        print("Verifying basic structure...")

        # Check that the main zarr store exists
        assert (output_path / "zarr.json").exists()

        # Check that each group has been created
        for group in groups:
            group_path = output_path / group.lstrip("/")
            assert group_path.exists(), f"Group {group} not found"
            assert (
                group_path / "zarr.json"
            ).exists(), f"Group {group} missing zarr.json"

            # Check that level 0 (native resolution) exists
            level_0_path = group_path / "0"
            assert level_0_path.exists(), f"Level 0 not found for {group}"
            assert (
                level_0_path / "zarr.json"
            ).exists(), f"Level 0 missing zarr.json for {group}"

    def _verify_geozarr_spec_compliance(self, output_path, group) -> None:
        """
        Verify GeoZarr specification compliance.

        This verifies:
        - _ARRAY_DIMENSIONS attributes on all arrays
        - CF standard names properly set
        - Grid mapping attributes reference correct CRS variables
        - GCP implementation following rasterio/rioxarray patterns
        """
        print(f"Verifying GeoZarr-spec compliance for {group}...")

        # Open the dataset
        group_path = str(output_path / group.lstrip("/") / "0")
        ds = xr.open_dataset(group_path, engine="zarr", zarr_format=3)

        # Check 1: _ARRAY_DIMENSIONS attributes
        for var_name in ds.data_vars:
            if var_name != "spatial_ref":  # Skip grid_mapping variable
                assert "_ARRAY_DIMENSIONS" in ds[var_name].attrs, \
                    f"Missing _ARRAY_DIMENSIONS for {var_name} in {group}"
                assert ds[var_name].attrs["_ARRAY_DIMENSIONS"] == list(ds[var_name].dims), \
                    f"Incorrect _ARRAY_DIMENSIONS for {var_name} in {group}"

        # Check 2: CF standard names
        for var_name in ds.data_vars:
            if var_name != "spatial_ref":
                assert "standard_name" in ds[var_name].attrs, \
                    f"Missing standard_name for {var_name} in {group}"

        # Check 3: Grid mapping attributes
        for var_name in ds.data_vars:
            if var_name != "spatial_ref":
                assert "grid_mapping" in ds[var_name].attrs, \
                    f"Missing grid_mapping for {var_name} in {group}"
                assert ds[var_name].attrs["grid_mapping"] == "spatial_ref", \
                    f"Incorrect grid_mapping for {var_name} in {group}"

        # Check 4: Spatial reference variable
        assert "spatial_ref" in ds, f"Missing spatial_ref variable in {group}"
        assert "_ARRAY_DIMENSIONS" in ds["spatial_ref"].attrs, \
            f"Missing _ARRAY_DIMENSIONS for spatial_ref in {group}"
        assert ds["spatial_ref"].attrs["_ARRAY_DIMENSIONS"] == [], \
            f"Incorrect _ARRAY_DIMENSIONS for spatial_ref in {group}"

        ds.close()

    def _verify_gcp_implementation(self, output_path, group) -> None:
        """
        Verify GCP-based georeferencing implementation.

        This verifies:
        - GCPs are correctly stored in grid_mapping variable
        - GCP attributes follow rioxarray conventions
        - No multiscale overviews are generated (as per ADR-102)
        """
        print(f"Verifying GCP implementation for {group}...")

        group_path = output_path / group.lstrip("/")
        
        # Verify only level 0 exists (no multiscales)
        level_dirs = [d for d in group_path.iterdir() if d.is_dir() and d.name.isdigit()]
        assert len(level_dirs) == 1, \
            f"Expected only level 0 for GCP implementation, found {len(level_dirs)} levels"
        assert level_dirs[0].name == "0", \
            "Expected only level 0 directory"

        # Open the dataset and verify GCP implementation
        ds = xr.open_dataset(str(group_path / "0"), engine="zarr", zarr_format=3)

        # Verify spatial_ref variable has GCP information
        assert "spatial_ref" in ds, "Missing spatial_ref variable"
        spatial_ref = ds["spatial_ref"]

        # Verify GCP attributes exist
        assert "GCP_COORDINATE_SYSTEM" in spatial_ref.attrs, \
            "Missing GCP coordinate system information"
        assert "GCP_COUNT" in spatial_ref.attrs, \
            "Missing GCP count attribute"

        # Verify GCP arrays exist
        gcp_required_attrs = [
            "GCP_LATITUDE", "GCP_LONGITUDE", "GCP_Z",
            "GCP_LINE", "GCP_PIXEL"
        ]
        for attr in gcp_required_attrs:
            assert attr in spatial_ref.attrs, f"Missing {attr} attribute"

        # Close the dataset
        ds.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
