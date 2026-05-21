"""Guardrails on array/group attributes in converter output.

These tests walk the snapshotted ``GroupSpec`` JSON fixtures produced by
``convert-s2-optimized`` and assert invariants that must hold for the
geozarr layout we ship (issue #171 and related cleanup):

- no source-only ``_eopf_attrs`` blob leaks through;
- no leftover TMS markers (``tile_matrix*``);
- decoded float arrays do not advertise ``units: digital_counts``;
- every float array under ``/measurements/`` carries an explicit
  ``_FillValue`` so xarray's CF NaN-masking round-trips.
"""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


def _walk(node: dict, path: str = "") -> Iterator[tuple[str, dict]]:
    """Yield ``(path, node)`` for every group/array in a GroupSpec tree."""
    yield path or "/", node
    for name, child in (node.get("members") or {}).items():
        child_path = f"{path}/{name}"
        yield from _walk(child, child_path)


def _is_float_array(node: dict) -> bool:
    if node.get("node_type") != "array":
        return False
    dt = node.get("data_type")
    if isinstance(dt, str):
        return dt.startswith("float")
    return False


_SNAPSHOT_DIR = pathlib.Path("tests/_test_data/optimized_geozarr_examples")
_SNAPSHOTS = sorted(_SNAPSHOT_DIR.glob("*.json"))


@pytest.fixture(params=_SNAPSHOTS, ids=lambda p: p.stem)
def snapshot(request: pytest.FixtureRequest) -> dict:
    return json.loads(request.param.read_text())


def test_no_eopf_attrs(snapshot: dict) -> None:
    offenders = [
        path for path, node in _walk(snapshot) if "_eopf_attrs" in (node.get("attributes") or {})
    ]
    assert not offenders, f"`_eopf_attrs` leaked at: {offenders}"


def test_no_tile_matrix_markers(snapshot: dict) -> None:
    offenders = [
        (path, key)
        for path, node in _walk(snapshot)
        for key in (node.get("attributes") or {})
        if key.startswith("tile_matrix")
    ]
    assert not offenders, f"TMS markers leaked: {offenders}"


def test_no_digital_counts_units(snapshot: dict) -> None:
    """Decoded floats must not advertise the source 'digital_counts' unit."""
    offenders = [
        path
        for path, node in _walk(snapshot)
        if _is_float_array(node) and (node.get("attributes") or {}).get("units") == "digital_counts"
    ]
    assert not offenders, f"`units: digital_counts` on floats at: {offenders}"


def test_float_measurements_have_fill_value(snapshot: dict) -> None:
    """Every float array under /measurements/ must declare ``_FillValue``.

    Without this, xarray cannot round-trip NaN masking via
    ``use_zarr_fill_value_as_mask=True`` (xarray issue #11345). The CF
    encoder produces ``"AAAAAAAA+H8="`` (base64 of LE float64 NaN bits)
    when ``_FillValue`` is set to ``np.nan``.
    """
    offenders = []
    for path, node in _walk(snapshot):
        if not _is_float_array(node):
            continue
        if "/measurements/" not in path + "/":
            continue
        # Skip auxiliary coord-like arrays (spatial_ref grid mapping, x/y).
        name = path.rsplit("/", 1)[-1]
        if name in {"spatial_ref", "x", "y"}:
            continue
        attrs = node.get("attributes") or {}
        if "_FillValue" not in attrs:
            offenders.append(path)
    assert not offenders, f"float measurements missing `_FillValue`: {offenders}"


def test_fill_value_masking_roundtrip(tmp_path: pathlib.Path) -> None:
    """End-to-end converter check: float arrays produced by
    ``create_geozarr_dataset`` must round-trip through xarray's
    ``use_zarr_fill_value_as_mask=True`` so NaN cells come back masked.

    Mirrors the rio-tiler reader expectation. Builds a minimal float
    DataTree, runs the real converter, then reopens the output and
    asserts masking semantics on a measurement band.
    """
    import numpy as np
    import xarray as xr

    from eopf_geozarr.conversion import create_geozarr_dataset

    epsg_code = 32632
    x_min, x_max = 600000, 605120  # 5.12 km
    y_min, y_max = 5090000, 5095120
    nx = ny = 512

    # Float reflectance band with a nodata patch (NaN) — what the converter
    # would normally see after CF decoding of scale_factor/add_offset.
    data = np.random.default_rng(42).uniform(0.0, 1.0, size=(ny, nx)).astype("float32")
    data[0:32, 0:32] = np.nan

    ds = xr.Dataset(
        {"b04": (["y", "x"], data, {"long_name": "Red band (B04)"})},
        coords={
            "x": np.linspace(x_min, x_max, nx, endpoint=False),
            "y": np.linspace(y_max, y_min, ny, endpoint=False),
        },
    ).rio.write_crs(f"EPSG:{epsg_code}")

    dt = xr.DataTree()
    dt["measurements"] = xr.DataTree()
    dt["measurements/reflectance"] = xr.DataTree()
    dt["measurements/reflectance/r10m"] = ds

    output_path = tmp_path / "fill_value_roundtrip.zarr"
    create_geozarr_dataset(
        dt_input=dt,
        groups=["/measurements/reflectance/r10m"],
        output_path=str(output_path),
        spatial_chunk=256,
        min_dimension=128,
        max_retries=1,
    )

    band_group = output_path / "measurements" / "reflectance" / "r10m"
    reopened = xr.open_dataset(
        band_group,
        engine="zarr",
        zarr_format=3,
        consolidated=True,
        decode_times=False,
        decode_coords=False,
        use_zarr_fill_value_as_mask=True,
    )
    try:
        masked = reopened["b04"].to_masked_array()
        assert np.ma.is_masked(masked), (
            "NaN cells in converter output should be masked when opened with "
            "use_zarr_fill_value_as_mask=True"
        )
        assert masked.mask[0, 0], "nodata corner cell must be masked"
        assert not masked.mask[-1, -1], "valid cell must not be masked"
    finally:
        reopened.close()
