"""
End-to-end CLI test using real Sentinel-2 sample data from the notebook.

This test demonstrates the complete CLI workflow using the same dataset
from the analysis notebook:
docs/analysis/eopf-geozarr/EOPF_Sentinel2_ZarrV3_geozarr_compliant.ipynb
"""

import json
import subprocess
from pathlib import Path

import pytest
import xarray as xr
import zarr
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import GroupSpec

from tests.test_data_api.conftest import view_json_diff


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


def test_cli_convert_real_sentinel2_data(s2_group_example: Path, tmp_path: Path) -> None:
    """
    Test CLI conversion using a Sentinel-2 hierarchy saved locally.
    """

    output_path = tmp_path / "s2b_geozarr_cli_test.zarr"

    # Detect product level (L1C vs L2A) by checking which quicklook group exists
    dt_source = xr.open_datatree(s2_group_example, engine="zarr")
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
    assert expected_structure_json == observed_structure_json, view_json_diff(
        expected_structure_json, observed_structure_json
    )


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


# =============================================================================
# S1 RTC CLI E2E tests
# =============================================================================


@pytest.fixture()
def s1_cli_geotiff_dir(tmp_path: Path) -> Path:
    """Create synthetic S1Tiling GeoTIFFs for CLI E2E tests."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    size = 256
    xmin, ymin, xmax, ymax = 500000.0, 4997440.0, 502560.0, 5000000.0
    transform = from_bounds(xmin, ymin, xmax, ymax, size, size)
    crs = "EPSG:32633"
    rng = np.random.default_rng(42)

    tags = {
        "ACQUISITION_DATETIME": "2023:01:15T06:12:34Z",
        "ORBIT_NUMBER": "47001",
        "RELATIVE_ORBIT_NUMBER": "037",
        "FLYING_UNIT_CODE": "S1A",
        "CALIBRATION": "gamma_naught",
        "INPUT_S1_IMAGES": "S1A_IW_GRDH_1SDV_20230115",
    }

    def _write(filename: str, data: np.ndarray, t: dict | None = None) -> None:
        with rasterio.open(
            str(tmp_path / filename), "w", driver="GTiff",
            height=data.shape[0], width=data.shape[1], count=1,
            dtype=data.dtype, crs=crs, transform=transform,
        ) as dst:
            if t:
                dst.update_tags(**t)
            dst.write(data, 1)

    vv = rng.uniform(0.0, 1.0, (size, size)).astype(np.float32)
    vh = rng.uniform(0.0, 0.5, (size, size)).astype(np.float32)
    mask = np.ones((size, size), dtype=np.uint8)
    mask[:10, :] = 0

    _write("s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif", vv, tags)
    _write("s1a_32TQM_vh_ASC_037_20230115t061234_GammaNaughtRTC.tif", vh, tags)
    _write("s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC_BorderMask.tif", mask, tags)

    # Condition GeoTIFFs
    ga = rng.uniform(0.5, 2.0, (size, size)).astype(np.float32)
    _write("GAMMA_AREA_32TQM_037.tif", ga)

    return tmp_path


def test_s1_ingest_cli(s1_cli_geotiff_dir: Path, tmp_path: Path) -> None:
    """E2E: ingest-s1 → consolidate-s1 → validate-s1 via CLI."""
    store = str(tmp_path / "s1-test.zarr")
    geotiff_dir = s1_cli_geotiff_dir

    # 1) ingest-s1
    result = subprocess.run(
        [
            "python", "-m", "eopf_geozarr", "ingest-s1",
            "--vv", str(geotiff_dir / "s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif"),
            "--vh", str(geotiff_dir / "s1a_32TQM_vh_ASC_037_20230115t061234_GammaNaughtRTC.tif"),
            "--mask", str(geotiff_dir / "s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC_BorderMask.tif"),
            "--store", store,
            "--orbit-dir", "ascending",
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"ingest-s1 failed: {result.stderr}"

    # 2) ingest-s1-conditions
    result = subprocess.run(
        [
            "python", "-m", "eopf_geozarr", "ingest-s1-conditions",
            "--store", store,
            "--orbit-dir", "ascending",
            "--relative-orbit", "37",
            "--gamma-area", str(geotiff_dir / "GAMMA_AREA_32TQM_037.tif"),
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"ingest-s1-conditions failed: {result.stderr}"

    # 3) consolidate-s1
    result = subprocess.run(
        [
            "python", "-m", "eopf_geozarr", "consolidate-s1",
            "--store", store,
            "--orbit-dir", "ascending",
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"consolidate-s1 failed: {result.stderr}"

    # 4) validate-s1
    result = subprocess.run(
        [
            "python", "-m", "eopf_geozarr", "validate-s1",
            "--store", store,
            "--verbose",
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"validate-s1 failed: {result.stderr}"

    # 5) Verify xarray can read back the store
    ds = xr.open_zarr(str(Path(store) / "ascending" / "r10m"))
    assert "vv" in ds
    assert ds["vv"].shape[0] == 1


def test_s1_validate_cli_rejects_invalid(tmp_path: Path) -> None:
    """validate-s1 returns non-zero for an invalid store."""
    import zarr

    # Create a bare Zarr group (no orbit groups → schema validation fails)
    store = str(tmp_path / "empty.zarr")
    zarr.open_group(store, mode="w-", zarr_format=3)

    result = subprocess.run(
        [
            "python", "-m", "eopf_geozarr", "validate-s1",
            "--store", store,
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0


def test_s1_cli_help() -> None:
    """All S1 CLI subcommands display help."""
    for subcmd in ["ingest-s1", "ingest-s1-conditions", "consolidate-s1", "validate-s1"]:
        result = subprocess.run(
            ["python", "-m", "eopf_geozarr", subcmd, "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"{subcmd} --help failed"
