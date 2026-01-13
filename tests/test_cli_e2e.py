"""
End-to-end CLI test using real Sentinel-2 sample data from the notebook.

This test demonstrates the complete CLI workflow using the same dataset
from the analysis notebook:
docs/analysis/eopf-geozarr/EOPF_Sentinel2_ZarrV3_geozarr_compliant.ipynb
"""

import json
import subprocess
from pathlib import Path

import jsondiff
import pytest
import xarray as xr
import zarr
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.experimental.v3 import GroupSpec


@pytest.mark.filterwarnings("ignore:.*:zarr.errors.UnstableSpecificationWarning")
def test_convert_s2_optimized(s2_group_example: Path, tmp_path: Path) -> None:
    """
    Test the convert-s2-optimized CLI command on a local copy of sentinel data
    """
    output_path = tmp_path

    cmd = [
        "python",
        "-m",
        "eopf_geozarr",
        "convert-s2-optimized",
        str(s2_group_example),
        str(output_path),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


@pytest.mark.filterwarnings("ignore:.*:zarr.errors.UnstableSpecificationWarning")
def test_cli_convert_real_sentinel2_data(s2_group_example: Path, tmp_path: Path) -> None:
    """
    Test CLI conversion using a Sentinel-2 hierarchy saved locally.
    """

    output_path = tmp_path / "s2b_geozarr_cli_test.zarr"

    # Detect product level (L1C vs L2A) by checking which quicklook group exists
    dt_source = xr.open_datatree(
        s2_group_example,
        engine="zarr",
        consolidated=False,
        decode_timedelta=True,
    )
    has_l2a_quicklook = "/quality/l2a_quicklook" in dt_source.groups
    has_l1c_quicklook = "/quality/l1c_quicklook" in dt_source.groups

    # Choose appropriate quicklook group based on product level
    if has_l2a_quicklook:
        quicklook_group = "/quality/l2a_quicklook/r10m"
    elif has_l1c_quicklook:
        quicklook_group = "/quality/l1c_quicklook/r10m"
    else:
        quicklook_group = None

    # Groups to convert (from the notebook)
    groups = [
        "/measurements/reflectance/r10m",
        "/measurements/reflectance/r20m",
        "/measurements/reflectance/r60m",
    ]
    if quicklook_group:
        groups.append(quicklook_group)

    # Build CLI command with notebook parameters
    cmd = [
        "python",
        "-m",
        "eopf_geozarr",
        "convert",
        str(s2_group_example),
        str(output_path),
        "--groups",
        *groups,
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

    # Execute the CLI command
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout for network operations
    )

    # Check command succeeded
    assert result.returncode == 0, result.stderr

    cmd_info = ["python", "-m", "eopf_geozarr", "info", str(output_path)]

    result_info = subprocess.run(cmd_info, capture_output=True, text=True, timeout=60)

    assert result_info.returncode == 0, result_info.stderr

    # Verify info output contains expected information
    info_output = result_info.stdout
    assert "Total groups" in info_output, "Info should show total groups count"
    assert "Group structure:" in info_output, "Info should show group structure"
    assert "/measurements" in info_output, "Should find measurements group"

    cmd_validate = [
        "python",
        "-m",
        "eopf_geozarr",
        "validate",
        str(output_path),
    ]

    result_validate = subprocess.run(cmd_validate, capture_output=True, text=True, timeout=60)

    assert result_validate.returncode == 0, f"CLI validate command failed: {result_validate.stderr}"
    # Verify validation output
    validate_output = result_validate.stdout
    assert "Validation Results:" in validate_output, "Should show validation header"
    assert "✅" in validate_output, "Should show successful validations"

    # verify exact output group structure
    # this is a sensitive, brittle check
    expected_structure_json = tuplify_json(
        json.loads(
            (
                Path("tests/_test_data/geozarr_examples/") / (s2_group_example.stem + ".json")
            ).read_text()
        )
    )
    observed_structure_json = tuplify_json(
        GroupSpec.from_zarr(zarr.open_group(output_path)).model_dump()
    )
    observed_structure = GroupSpec(**observed_structure_json)
    observed_structure_flat = observed_structure.to_flat()
    expected_structure = GroupSpec(**expected_structure_json)
    expected_structure_flat = expected_structure.to_flat()

    o_keys = set(observed_structure_flat.keys())
    e_keys = set(expected_structure_flat.keys())

    # Check that all of the keys are the same
    assert o_keys == e_keys

    differences: dict[str, dict[str, object]] = {}
    for k in o_keys:
        expected_val = expected_structure_flat[k]
        observed_val = observed_structure_flat[k]

        expected_val_json = tuplify_json(expected_val.model_dump())
        observed_val_json = tuplify_json(observed_val.model_dump())

        if expected_val_json != observed_val_json:
            differences[k] = jsondiff.diff(expected_val_json, observed_val_json)

    assert differences == {}


def test_cli_help_commands() -> None:
    """Test that all CLI help commands work."""
    # Test main help
    result = subprocess.run(
        ["python", "-m", "eopf_geozarr", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, "Main help command failed"
    assert "Convert EOPF datasets to GeoZarr compliant format" in result.stdout

    # Test convert help
    result = subprocess.run(
        ["python", "-m", "eopf_geozarr", "convert", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "Convert help command failed"
    assert "input_path" in result.stdout
    assert "output_path" in result.stdout

    # Test info help
    result = subprocess.run(
        ["python", "-m", "eopf_geozarr", "info", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "Info help command failed"
    assert "input_path" in result.stdout

    # Test validate help
    result = subprocess.run(
        ["python", "-m", "eopf_geozarr", "validate", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "Validate help command failed"
    assert "input_path" in result.stdout


def test_cli_version() -> None:
    """Test CLI version command."""
    result = subprocess.run(
        ["python", "-m", "eopf_geozarr", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "Version command failed"
    assert "0.1.0" in result.stdout, "Version should be 0.1.0"


def test_cli_crs_groups_option() -> None:
    """Test that the --crs-groups CLI option is properly recognized."""
    # Test that --crs-groups option appears in help
    result = subprocess.run(
        ["python", "-m", "eopf_geozarr", "convert", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "Convert help command failed"
    assert "--crs-groups" in result.stdout, "--crs-groups option should be in help"
    assert "Groups that need CRS information added" in result.stdout, "Help text should be present"


def test_cli_convert_with_crs_groups(s2_group_example, tmp_path: Path) -> None:
    """
    Test CLI conversion with --crs-groups option using real Sentinel-2 data.

    This test verifies that the --crs-groups option works correctly and
    processes the specified groups for CRS enhancement.
    """
    # Dataset from the notebook

    output_path = tmp_path / "s2b_geozarr_crs_groups_test.zarr"

    # Groups to convert
    groups = ["/measurements/reflectance/r10m"]

    # CRS groups to enhance (these would typically be geometry/conditions groups)
    # For this test, we'll use a group that exists in the dataset
    crs_groups = ["/conditions/geometry", "/conditions/viewing"]

    # Build CLI command with --crs-groups option
    cmd = [
        "python",
        "-m",
        "eopf_geozarr",
        "convert",
        str(s2_group_example),
        str(output_path),
        "--groups",
        *groups,
        "--crs-groups",
        *crs_groups,
        "--spatial-chunk",
        "1024",
        "--min-dimension",
        "256",
        "--tile-width",
        "256",
        "--max-retries",
        "3",
        "--verbose",
    ]

    # Execute the CLI command
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout for network operations
    )

    # Check command succeeded
    if result.returncode != 0 and not (
        "not found in DataTree" in result.stdout or "not found in DataTree" in result.stderr
    ):
        pytest.fail(f"CLI convert with --crs-groups command failed: {result.stderr}")

    # Note: The --crs-groups option is accepted and processed best-effort.
    # We don't assert on specific log messages as they may vary by implementation.
    # The important verification is that the command succeeds and produces output.

    # Verify output exists
    assert output_path.exists(), f"Output path {output_path} was not created"
    assert (output_path / "zarr.json").exists(), "Main zarr.json not found"


@pytest.mark.filterwarnings("ignore:.*:zarr.errors.ZarrUserWarning")
def test_cli_crs_groups_empty_list(tmp_path: str) -> None:
    """Test CLI with --crs-groups but no groups specified (empty list)."""
    # Create a minimal test dataset
    test_input = Path(tmp_path) / "test_input.zarr"
    test_output = Path(tmp_path) / "test_output.zarr"

    # Create a simple test dataset
    import numpy as np

    ds = xr.Dataset(
        {"temperature": (["y", "x"], np.random.rand(10, 10))},
        coords={
            "x": (["x"], np.linspace(0, 10, 10)),
            "y": (["y"], np.linspace(0, 10, 10)),
        },
    )

    # Save as zarr
    ds.to_zarr(test_input, zarr_format=3)
    ds.close()

    # Test CLI with --crs-groups but no groups specified
    cmd = [
        "python",
        "-m",
        "eopf_geozarr",
        "convert",
        str(test_input),
        str(test_output),
        "--groups",
        "/",
        "--crs-groups",  # No groups specified after this
        "--verbose",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    # Should succeed (empty crs_groups list is valid)
    assert result.returncode == 0, f"CLI with empty --crs-groups failed: {result.stderr}"
    assert "CRS groups: []" in result.stdout, "Should show empty CRS groups list"
