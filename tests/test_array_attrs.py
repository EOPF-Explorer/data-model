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
