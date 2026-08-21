"""Integration tests: converter output must satisfy the GeoZarr minispec validator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import rioxarray  # noqa: F401  # enable .rio accessor
import xarray as xr

from eopf_geozarr.conversion import create_geozarr_dataset
from eopf_geozarr.data_api.geozarr.validation import validate_store

if TYPE_CHECKING:
    import pathlib


@pytest.fixture
def synthetic_s2_tree() -> xr.DataTree:
    """Small S2-like tree; r60m is below the default min_dimension on purpose."""
    epsg = 32632
    x_min, x_max = 600000, 605490
    y_min, y_max = 5090000, 5095490
    sizes = {"r10m": 549, "r20m": 275, "r60m": 92}
    bands = {"r10m": ["b02", "b03"], "r20m": ["b05"], "r60m": ["b01"]}
    rng = np.random.default_rng(42)
    dt = xr.DataTree()
    dt["measurements"] = xr.DataTree()
    dt["measurements/reflectance"] = xr.DataTree()
    for res, n in sizes.items():
        coords = {
            "x": np.linspace(x_min, x_max, n, endpoint=False),
            "y": np.linspace(y_max, y_min, n, endpoint=False),
            "time": [np.datetime64("2025-01-13T10:33:09")],
        }
        data = {
            b: (
                ["time", "y", "x"],
                rng.integers(500, 3000, (1, n, n), dtype=np.uint16),
                {"proj:epsg": epsg},
            )
            for b in bands[res]
        }
        ds = xr.Dataset(data, coords=coords).rio.write_crs(f"EPSG:{epsg}")
        dt[f"measurements/reflectance/{res}"] = ds
    dt.attrs = {"title": "synthetic S2", "product_type": "S2MSI1C"}
    return dt


def test_create_geozarr_dataset_output_is_minispec_compliant(
    synthetic_s2_tree: xr.DataTree, tmp_path: pathlib.Path
) -> None:
    output = str(tmp_path / "geozarr.zarr")
    create_geozarr_dataset(
        dt_input=synthetic_s2_tree,
        groups=[
            "/measurements/reflectance/r10m",
            "/measurements/reflectance/r20m",
            "/measurements/reflectance/r60m",
        ],
        output_path=output,
        spatial_chunk=4096,
        min_dimension=256,
        max_retries=3,
    )
    report = validate_store(output)
    assert report.compliant, "\n".join(str(i) for i in report.issues)


def test_calculate_overview_levels_small_native_keeps_level_zero() -> None:
    """A native grid below min_dimension still yields level 0 (native)."""
    from eopf_geozarr.conversion.geozarr import calculate_overview_levels

    levels = calculate_overview_levels(92, 92, min_dimension=256)
    assert [lvl["level"] for lvl in levels] == [0]
    assert levels[0]["width"] == 92
    assert levels[0]["height"] == 92


def test_small_group_keeps_multiscale_metadata(
    synthetic_s2_tree: xr.DataTree, tmp_path: pathlib.Path
) -> None:
    """The sub-min_dimension r60m group must still carry its conventions.

    Pins the level-0 fix directly: without it the r60m group is written with no
    multiscales/spatial/proj metadata at all and becomes invisible to the
    validator (which only inspects convention-bearing nodes).
    """
    import zarr

    output = str(tmp_path / "geozarr.zarr")
    create_geozarr_dataset(
        dt_input=synthetic_s2_tree,
        groups=["/measurements/reflectance/r60m"],
        output_path=output,
        spatial_chunk=4096,
        min_dimension=256,
        max_retries=3,
    )
    group = zarr.open_group(output, mode="r")["measurements/reflectance/r60m"]
    attrs = dict(group.attrs)
    assert "multiscales" in attrs, sorted(attrs)
    assert "zarr_conventions" in attrs
    assert "spatial:bbox" in attrs
    assert "proj:code" in attrs


def test_cli_validate_exits_nonzero_on_noncompliant_store(tmp_path: pathlib.Path) -> None:
    """The validate command's failure contract: non-compliant store -> exit 1."""
    import subprocess
    import sys

    import zarr

    store = str(tmp_path / "bad.zarr")
    zarr.open_group(store, mode="w", zarr_format=3)  # bare root, no metadata

    result = subprocess.run(
        [sys.executable, "-m", "eopf_geozarr", "validate", store],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "NOT compliant" in result.stdout
