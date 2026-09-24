"""Tests for eopf_geozarr.cpm.routing (importable without eopf-cpm)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.cpm.routing import (
    looks_like_sentinel2,
    looks_like_sentinel3_olci,
    product_type_of,
    select_pipeline,
)


def make_tree(
    product_type: str | None = None,
    with_s2_structure: bool = False,
    with_olci_structure: bool = False,
) -> xr.DataTree:
    """Build a minimal DataTree with optional CPM attrs and mission group structure."""
    dt = xr.DataTree()
    if product_type is not None:
        dt.attrs = {"stac_discovery": {"properties": {"product:type": product_type}}}
    if with_s2_structure:
        ds = xr.Dataset({"b01": (["y", "x"], np.zeros((2, 2)))})
        for resolution in ("r10m", "r20m", "r60m"):
            dt[f"measurements/reflectance/{resolution}"] = ds
    if with_olci_structure:
        # oa01_radiance is a data variable of the measurements group (a zarr
        # array member), not a child node -- unlike the S2 r{N}m subgroups.
        ds = xr.Dataset({"oa01_radiance": (["rows", "columns"], np.zeros((2, 2)))})
        dt["measurements"] = ds
    return dt


@pytest.mark.parametrize(
    ("product_type", "with_s2_structure", "with_olci_structure", "force", "expected"),
    [
        # Declared product type drives auto-detection.
        ("S02MSIL1C", False, False, None, "s2-optimized"),
        ("S02MSIL2A", False, False, None, "s2-optimized"),
        ("S01SIWGRH", False, False, None, "generic"),
        ("S02MSIL0_", False, False, None, "generic"),  # MSI but not L1C/L2A imagery
        ("S02MSIRAW", False, False, None, "generic"),
        ("S03OLCEFR", False, False, None, "s3-olci-optimized"),
        ("S03OLCERR", False, False, None, "s3-olci-optimized"),
        ("S03SLSRBT", False, False, None, "generic"),  # SLSTR not (yet) supported
        # Declared type wins over structure, in both directions.
        ("S03OLCERR", True, False, None, "s3-olci-optimized"),
        ("S02MSIL1C", False, True, None, "s2-optimized"),
        # Structural fallback when no product type is declared.
        (None, True, False, None, "s2-optimized"),
        (None, False, True, None, "s3-olci-optimized"),
        (None, False, False, None, "generic"),
        # Explicit forcing wins over everything.
        ("S02MSIL1C", True, False, "generic", "generic"),
        ("S03OLCERR", False, False, "s2-optimized", "s2-optimized"),
        ("S02MSIL1C", False, False, "s3-olci-optimized", "s3-olci-optimized"),
    ],
)
def test_select_pipeline(
    product_type: str | None,
    with_s2_structure: bool,
    with_olci_structure: bool,
    force: str | None,
    expected: str,
) -> None:
    """select_pipeline routes on declared type, then structure, unless forced."""
    tree = make_tree(product_type, with_s2_structure, with_olci_structure)
    assert select_pipeline(tree, force=force) == expected  # type: ignore[arg-type]


def test_product_type_of_missing_attrs() -> None:
    """product_type_of returns None for absent or malformed attribute chains."""
    assert product_type_of(xr.DataTree()) is None
    malformed = xr.DataTree()
    malformed.attrs = {"stac_discovery": "not-a-mapping"}
    assert product_type_of(malformed) is None
    no_type = xr.DataTree()
    no_type.attrs = {"stac_discovery": {"properties": {"product:type": 42}}}
    assert product_type_of(no_type) is None


def test_looks_like_sentinel2_requires_all_native_resolutions() -> None:
    """Structural detection requires all of r10m/r20m/r60m under reflectance."""
    partial = xr.DataTree()
    partial["measurements/reflectance/r10m"] = xr.Dataset({"b01": (["y", "x"], np.zeros((2, 2)))})
    assert not looks_like_sentinel2(partial)


def test_looks_like_sentinel3_olci_structural_detection() -> None:
    """Structural detection looks for oa01_radiance as a measurements data variable."""
    tree = make_tree(with_olci_structure=True)
    assert looks_like_sentinel3_olci(tree)
    assert not looks_like_sentinel2(tree)


def test_looks_like_sentinel3_olci_no_measurements_group() -> None:
    """A tree with no measurements group at all is not OLCI."""
    assert not looks_like_sentinel3_olci(xr.DataTree())


def test_looks_like_sentinel3_olci_ignores_child_groups() -> None:
    """oa01_radiance as a child node (not a data variable) does not count.

    This mirrors the real CPM layout, where measurements/oa01_radiance is a
    zarr array (a data variable once opened as a DataTree), not a subgroup.
    A tree that happens to have a child node of that name should not be
    misdetected as OLCI.
    """
    tree = xr.DataTree()
    tree["measurements/oa01_radiance"] = xr.DataTree()
    assert not looks_like_sentinel3_olci(tree)
