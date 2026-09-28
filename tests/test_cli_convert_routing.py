"""
In-process tests for the convert command's Sentinel-2 auto-detection and routing.

The e2e tests in test_cli_e2e.py drive the CLI through subprocesses; these tests
call ``convert_command`` directly (with the converters stubbed out) so the
routing logic itself is exercised in-process.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr import cli


def _convert_args(input_path: str, output_path: str, **overrides: Any) -> argparse.Namespace:
    """Build a Namespace mirroring the convert subcommand's defaults."""
    ns = argparse.Namespace(
        input_path=input_path,
        output_path=output_path,
        groups=["/measurements/reflectance/r10m"],
        spatial_chunk=4096,
        min_dimension=256,
        max_retries=3,
        crs_groups=None,
        gcp_group=None,
        verbose=False,
        dask_cluster=False,
        enable_sharding=False,
        no_s2_optimized=False,
        no_s3_olci_optimized=False,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


@pytest.fixture
def plain_zarr_input(tmp_path: Path) -> Path:
    """A minimal non-Sentinel-2 zarr store."""
    path = tmp_path / "plain_input.zarr"
    ds = xr.Dataset(
        {"temperature": (["y", "x"], np.zeros((4, 4)))},
        coords={"x": ("x", np.arange(4.0)), "y": ("y", np.arange(4.0))},
    )
    ds.to_zarr(path, zarr_format=3)
    return path


def test_is_sentinel2_input_true(s2_group_example: Path) -> None:
    """A real Sentinel-2 layout is detected."""
    dt = xr.open_datatree(str(s2_group_example), engine="zarr")
    assert cli._is_sentinel2_input(dt) is True


def test_is_sentinel2_input_false(plain_zarr_input: Path) -> None:
    """A plain zarr store is not detected as Sentinel-2 (and never raises)."""
    dt = xr.open_datatree(str(plain_zarr_input), engine="zarr")
    assert cli._is_sentinel2_input(dt) is False


def test_is_sentinel2_input_swallow_errors() -> None:
    """Detection failures (e.g. in-memory datatree with no store) mean 'not S2'."""
    dt = xr.DataTree(xr.Dataset({"a": (["y", "x"], np.zeros((2, 2)))}))
    assert cli._is_sentinel2_input(dt) is False


@pytest.fixture
def converter_spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Stub out both converters, recording which one convert_command dispatches to."""
    calls: dict[str, dict[str, Any]] = {}

    def fake_s2(**kwargs: Any) -> xr.DataTree:
        calls["s2_optimized"] = kwargs
        return xr.DataTree()

    def fake_generic(**kwargs: Any) -> xr.DataTree:
        calls["generic"] = kwargs
        return xr.DataTree()

    def fake_olci(dt_input: xr.DataTree, **kwargs: Any) -> xr.DataTree:
        calls["olci"] = {"dt_input": dt_input, **kwargs}
        return xr.DataTree()

    monkeypatch.setattr(cli, "convert_s2_optimized", fake_s2)
    monkeypatch.setattr(cli, "create_geozarr_dataset", fake_generic)
    monkeypatch.setattr(cli, "convert_olci_optimized", fake_olci)
    return calls


def test_convert_command_routes_s2_to_optimized(
    s2_group_example: Path,
    tmp_path: Path,
    converter_spy: dict[str, dict[str, Any]],
) -> None:
    """Sentinel-2 inputs are auto-routed to the optimized converter."""
    args = _convert_args(str(s2_group_example), str(tmp_path / "out.zarr"))
    cli.convert_command(args)

    assert "s2_optimized" in calls_or_fail(converter_spy)
    assert "generic" not in converter_spy
    assert converter_spy["s2_optimized"]["keep_scale_offset"] is False


def test_convert_command_no_s2_optimized_forces_generic(
    s2_group_example: Path,
    tmp_path: Path,
    converter_spy: dict[str, dict[str, Any]],
) -> None:
    """--no-s2-optimized sends Sentinel-2 inputs down the generic path."""
    args = _convert_args(
        str(s2_group_example),
        str(tmp_path / "out.zarr"),
        no_s2_optimized=True,
        crs_groups=["/conditions/geometry"],
    )
    cli.convert_command(args)

    assert "generic" in calls_or_fail(converter_spy)
    assert "s2_optimized" not in converter_spy
    assert converter_spy["generic"]["crs_groups"] == ["/conditions/geometry"]


def test_convert_command_routes_olci_with_raw_input(
    s3_olci_group_example: Path,
    tmp_path: Path,
    converter_spy: dict[str, dict[str, Any]],
) -> None:
    """Sentinel-3 OLCI inputs are auto-routed to the OLCI converter with raw input.

    convert_olci_optimized requires the source opened with
    ``mask_and_scale=False``: radiance must arrive packed (uint16 with CF
    scale_factor/add_offset in .attrs), not CF-decoded to float.
    """
    args = _convert_args(str(s3_olci_group_example), str(tmp_path / "out.zarr"), min_dimension=128)
    cli.convert_command(args)

    assert "olci" in calls_or_fail(converter_spy)
    assert "s2_optimized" not in converter_spy
    assert "generic" not in converter_spy
    assert converter_spy["olci"]["min_dimension"] == 128

    dt_received = converter_spy["olci"]["dt_input"]
    radiance = next(
        var
        for name, var in dt_received["/measurements"].data_vars.items()
        if str(name).endswith("_radiance")
    )
    assert not np.issubdtype(radiance.dtype, np.floating), (
        "OLCI converter received CF-decoded (mask_and_scale) input; expected raw packed radiance"
    )
    assert "scale_factor" in radiance.attrs


def test_convert_command_no_s3_olci_optimized_forces_generic(
    s3_olci_group_example: Path,
    tmp_path: Path,
    converter_spy: dict[str, dict[str, Any]],
) -> None:
    """--no-s3-olci-optimized sends Sentinel-3 OLCI inputs down the generic path."""
    args = _convert_args(
        str(s3_olci_group_example),
        str(tmp_path / "out.zarr"),
        no_s3_olci_optimized=True,
    )
    cli.convert_command(args)

    assert "generic" in calls_or_fail(converter_spy)
    assert "olci" not in converter_spy


def test_convert_command_routes_non_s2_to_generic(
    plain_zarr_input: Path,
    tmp_path: Path,
    converter_spy: dict[str, dict[str, Any]],
) -> None:
    """Non-Sentinel-2 inputs use the generic converter."""
    args = _convert_args(str(plain_zarr_input), str(tmp_path / "out.zarr"), groups=["/"])
    cli.convert_command(args)

    assert "generic" in calls_or_fail(converter_spy)
    assert "s2_optimized" not in converter_spy


def calls_or_fail(calls: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fail loudly if convert_command dispatched to neither converter."""
    if not calls:
        pytest.fail("convert_command did not call any converter (it likely errored early)")
    return calls
