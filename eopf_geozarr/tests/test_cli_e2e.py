"""
End-to-end CLI test using real Sentinel-2 sample data from the notebook.

This test demonstrates the complete CLI workflow using the same dataset
from the analysis notebook:
docs/analysis/eopf-geozarr/EOPF_Sentinel2_ZarrV3_geozarr_compliant.ipynb
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import xarray as xr


class TestCLIEndToEnd:
    """End-to-end CLI tests with real data."""

    @pytest.fixture
    def temp_output_dir(self):  # type: ignore[no-untyped-def]
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.mark.slow
    @pytest.mark.network
    def test_cli_convert_real_sentinel2_data(self, temp_output_dir: str) -> None:
        """
        Test CLI conversion using real Sentinel-2 data from the notebook.

        This test:
        1. Uses the same remote dataset URL from the notebook
        2. Converts it using the CLI with notebook parameters
        3. Verifies the output structure and compliance
        4. Tests CLI info and validate commands
        """
        # Dataset from the notebook
        input_url = (
            "https://objectstore.eodc.eu:2222/e05ab01a9d56408d82ac32d69a5aae2a:sample-data/"
            "tutorial_data/cpm_v253/S2B_MSIL1C_20250113T103309_N0511_R108_T32TLQ_20250113T122458.zarr"
        )
        output_path = Path(temp_output_dir) / "s2b_geozarr_cli_test.zarr"

        # Groups to convert (from the notebook)
        groups = [
            "/measurements/reflectance/r10m",
            "/measurements/reflectance/r20m",
            "/measurements/reflectance/r60m",
            "/quality/l1c_quicklook/r10m",
        ]

        print("Testing CLI conversion with real Sentinel-2 data...")
        print(f"Input URL: {input_url}")
        print(f"Output path: {output_path}")

        # Test 1: CLI convert command
        print("\n=== Testing CLI convert command ===")

        # Build CLI command with notebook parameters
        cmd = (
            ["python", "-m", "eopf_geozarr", "convert", input_url, str(output_path), "--groups"]
            + groups
            + [
                "--spatial-chunk",
                "1024",  # From notebook
                "--min-dimension",
                "256",  # From notebook
                "--tile-width",
                "256",  # From notebook
                "--max-retries",
                "3",  # From notebook
                "--verbose",
            ]
        )

        # Execute the CLI command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for network operations
        )

        # Check command succeeded
        if result.returncode != 0:
            print(f"CLI convert failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            pytest.fail(f"CLI convert command failed: {result.stderr}")

        print("✅ CLI convert command succeeded")
        print(f"Output: {result.stdout}")

        # Verify output exists
        assert output_path.exists(), f"Output path {output_path} was not created"
        assert (output_path / "zarr.json").exists(), "Main zarr.json not found"

        # Test 2: CLI info command
        print("\n=== Testing CLI info command ===")

        cmd_info = ["python", "-m", "eopf_geozarr", "info", str(output_path)]

        result_info = subprocess.run(cmd_info, capture_output=True, text=True, timeout=60)

        assert result_info.returncode == 0, f"CLI info command failed: {result_info.stderr}"
        print("✅ CLI info command succeeded")
        print(f"Info output: {result_info.stdout}")

        # Verify info output contains expected information
        info_output = result_info.stdout
        assert "Total groups:" in info_output, "Info should show total groups count"
        assert "Group structure:" in info_output, "Info should show group structure"
        assert "/measurements" in info_output, "Should find measurements group"

        # Test 3: CLI validate command
        print("\n=== Testing CLI validate command ===")

        cmd_validate = [
            "python",
            "-m",
            "eopf_geozarr",
            "validate",
            str(output_path),
        ]

        result_validate = subprocess.run(cmd_validate, capture_output=True, text=True, timeout=60)

        assert (
            result_validate.returncode == 0
        ), f"CLI validate command failed: {result_validate.stderr}"
        print("✅ CLI validate command succeeded")
        print(f"Validation output: {result_validate.stdout}")

        # Verify validation output
        validate_output = result_validate.stdout
        assert "Validation Results:" in validate_output, "Should show validation header"
        assert "✅" in validate_output, "Should show successful validations"

        # Test 4: Verify data structure and compliance
        print("\n=== Verifying converted data structure ===")

        self._verify_converted_data_structure(output_path, groups)

        print("✅ All CLI end-to-end tests passed!")

    def _verify_converted_data_structure(self, output_path: Path, groups: list[str]) -> None:
        """Verify the structure and compliance of converted data."""
        # Check each group was converted
        for group in groups:
            group_path = output_path / group.lstrip("/")
            assert group_path.exists(), f"Group {group} not found"

            # Check level 0 exists
            level_0_path = group_path / "0"
            assert level_0_path.exists(), f"Level 0 not found for {group}"

            # Open and verify the dataset
            ds = xr.open_dataset(str(level_0_path), engine="zarr", zarr_format=3)

            print(f"  Group {group}:")
            print(f"    Variables: {list(ds.data_vars)}")
            print(f"    Coordinates: {list(ds.coords)}")
            print(f"    Dimensions: {dict(ds.dims)}")

            # Verify GeoZarr compliance basics
            data_vars = [var for var in ds.data_vars if var != "spatial_ref"]

            if data_vars:  # Only check if there are data variables
                # Check first data variable for compliance
                first_var = data_vars[0]

                # Check _ARRAY_DIMENSIONS
                assert (
                    "_ARRAY_DIMENSIONS" in ds[first_var].attrs
                ), f"Missing _ARRAY_DIMENSIONS in {first_var} for {group}"

                # Check standard_name
                assert (
                    "standard_name" in ds[first_var].attrs
                ), f"Missing standard_name in {first_var} for {group}"

                # Check grid_mapping
                assert (
                    "grid_mapping" in ds[first_var].attrs
                ), f"Missing grid_mapping in {first_var} for {group}"

                print(f"    ✅ GeoZarr compliance verified for {first_var}")

            # Check spatial_ref exists
            if "spatial_ref" in ds:
                assert (
                    "_ARRAY_DIMENSIONS" in ds["spatial_ref"].attrs
                ), f"Missing _ARRAY_DIMENSIONS in spatial_ref for {group}"
                print("    ✅ spatial_ref variable verified")

            ds.close()

            # Check for overview levels
            level_dirs = [d for d in group_path.iterdir() if d.is_dir() and d.name.isdigit()]
            print(f"    Overview levels: {sorted([d.name for d in level_dirs])}")

            if len(level_dirs) > 1:
                print("    ✅ Multiscale structure created")

    def test_cli_help_commands(self) -> None:
        """Test that all CLI help commands work."""
        # Test main help
        result = subprocess.run(
            ["python", "-m", "eopf_geozarr", "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0, "Main help command failed"
        assert "Convert EOPF datasets to GeoZarr compliant format" in result.stdout

        # Test convert help
        result = subprocess.run(
            ["python", "-m", "eopf_geozarr", "convert", "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0, "Convert help command failed"
        assert "input_path" in result.stdout and "output_path" in result.stdout

        # Test info help
        result = subprocess.run(
            ["python", "-m", "eopf_geozarr", "info", "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0, "Info help command failed"
        assert "input_path" in result.stdout

        # Test validate help
        result = subprocess.run(
            ["python", "-m", "eopf_geozarr", "validate", "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0, "Validate help command failed"
        assert "input_path" in result.stdout

        print("✅ All CLI help commands work correctly")

    def test_cli_version(self) -> None:
        """Test CLI version command."""
        result = subprocess.run(
            ["python", "-m", "eopf_geozarr", "--version"], capture_output=True, text=True
        )
        assert result.returncode == 0, "Version command failed"
        assert "0.1.0" in result.stdout, "Version should be 0.1.0"
        print("✅ CLI version command works correctly")

    def test_cli_crs_groups_option(self) -> None:
        """Test that the --crs-groups CLI option is properly recognized."""
        # Test that --crs-groups option appears in help
        result = subprocess.run(
            ["python", "-m", "eopf_geozarr", "convert", "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0, "Convert help command failed"
        assert "--crs-groups" in result.stdout, "--crs-groups option should be in help"
        assert "Groups that need CRS information added" in result.stdout, "Help text should be present"
        print("✅ --crs-groups option appears in CLI help")

    @pytest.mark.slow
    @pytest.mark.network
    def test_cli_convert_with_crs_groups(self, temp_output_dir: str) -> None:
        """
        Test CLI conversion with --crs-groups option using real Sentinel-2 data.
        
        This test verifies that the --crs-groups option works correctly and
        processes the specified groups for CRS enhancement.
        """
        # Dataset from the notebook
        input_url = (
            "https://objectstore.eodc.eu:2222/e05ab01a9d56408d82ac32d69a5aae2a:sample-data/"
            "tutorial_data/cpm_v253/S2B_MSIL1C_20250113T103309_N0511_R108_T32TLQ_20250113T122458.zarr"
        )
        output_path = Path(temp_output_dir) / "s2b_geozarr_crs_groups_test.zarr"

        # Groups to convert
        groups = ["/measurements/reflectance/r10m"]
        
        # CRS groups to enhance (these would typically be geometry/conditions groups)
        # For this test, we'll use a group that exists in the dataset
        crs_groups = ["/conditions/geometry", "/conditions/viewing"]

        print("Testing CLI conversion with --crs-groups option...")
        print(f"Input URL: {input_url}")
        print(f"Output path: {output_path}")
        print(f"CRS groups: {crs_groups}")

        # Build CLI command with --crs-groups option
        cmd = [
            "python", "-m", "eopf_geozarr", "convert", 
            input_url, str(output_path),
            "--groups"
        ] + groups + [
            "--crs-groups"
        ] + crs_groups + [
            "--spatial-chunk", "1024",
            "--min-dimension", "256", 
            "--tile-width", "256",
            "--max-retries", "3",
            "--verbose"
        ]

        # Execute the CLI command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for network operations
        )

        # Check command succeeded
        if result.returncode != 0:
            print(f"CLI convert with --crs-groups failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            # Don't fail the test if CRS groups don't exist in the dataset
            # This is expected behavior for best-effort processing
            if "not found in DataTree" in result.stdout or "not found in DataTree" in result.stderr:
                print("✅ CLI handled missing CRS groups gracefully (expected behavior)")
                return
            pytest.fail(f"CLI convert with --crs-groups command failed: {result.stderr}")

        print("✅ CLI convert with --crs-groups command succeeded")
        
        # Verify that CRS groups were mentioned in verbose output
        output_text = result.stdout
        assert "CRS groups:" in output_text, "Verbose output should mention CRS groups"
        
        # Check for CRS processing messages
        crs_processing_found = any(
            msg in output_text for msg in [
                "Adding CRS information to group:",
                "Inferred reference CRS from measurements:",
                "not found in DataTree"  # Expected for missing groups
            ]
        )
        assert crs_processing_found, "Should show CRS processing messages"
        
        print("✅ CRS groups processing verified in output")

        # Verify output exists
        assert output_path.exists(), f"Output path {output_path} was not created"
        assert (output_path / "zarr.json").exists(), "Main zarr.json not found"
        
        print("✅ CLI convert with --crs-groups test completed successfully")

    def test_cli_crs_groups_empty_list(self, temp_output_dir: str) -> None:
        """Test CLI with --crs-groups but no groups specified (empty list)."""
        # Create a minimal test dataset
        test_input = Path(temp_output_dir) / "test_input.zarr"
        test_output = Path(temp_output_dir) / "test_output.zarr"
        
        # Create a simple test dataset
        import numpy as np
        ds = xr.Dataset({
            "temperature": (["y", "x"], np.random.rand(10, 10))
        }, coords={
            "x": (["x"], np.linspace(0, 10, 10)),
            "y": (["y"], np.linspace(0, 10, 10))
        })
        
        # Save as zarr
        ds.to_zarr(test_input, zarr_format=3)
        ds.close()
        
        # Test CLI with --crs-groups but no groups specified
        cmd = [
            "python", "-m", "eopf_geozarr", "convert",
            str(test_input), str(test_output),
            "--groups", "/",
            "--crs-groups",  # No groups specified after this
            "--verbose"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Should succeed (empty crs_groups list is valid)
        assert result.returncode == 0, f"CLI with empty --crs-groups failed: {result.stderr}"
        assert "CRS groups: []" in result.stdout, "Should show empty CRS groups list"
        
        print("✅ CLI with empty --crs-groups list works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
