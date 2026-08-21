"""Cross-product output contract.

Reprojection to a regular grid with a declared CRS is a hard requirement for
every product that is converted to a regular grid (see
docs/superpowers/specs/2026-07-27-s3-olci-reprojection-design.md). One
parametrized test walks each converter's output and asserts the contract at
every multiscale level. Products whose converter has not yet migrated off the
legacy nested layout carry an xfail on the whole-store DataTree check so the
requirement stays on record. OLCI's native (default) mode intentionally keeps
instrument swath geolocation with no CRS; that mode is covered separately by
the swath-geolocation test below rather than by the regular-grid contract.

Per-level CRS detection requires opening each level with
``decode_coords="all"`` (the generic converter's own read path uses the same
flag, see geozarr.py) so the CF `grid_mapping`/`spatial_ref` reference gets
promoted into a coordinate; the default `xr.open_dataset` decode never does
this. Separately, S1/S2 overview sub-groups (r2, r4, ...) still lack the
zarr-cm proj:/spatial: convention attrs that OLCI stamps on every level — a
real design-consistency gap, but not one that defeats CRS detection here, so
it is tracked outside this test rather than xfailed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
import zarr
from pyproj import CRS as ProjCRS

from eopf_geozarr.conversion import create_geozarr_dataset
from eopf_geozarr.s3_olci_optimization.olci_converter import convert_olci_optimized

from .test_integration_sentinel1 import MockSentinel1L1GRDBuilder
from .test_integration_sentinel2 import build_sample_sentinel2_datatree
from .test_olci_integration import build_synthetic_olci

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Callable


def _convert_olci(tmp: pathlib.Path) -> pathlib.Path:
    out = tmp / "olci.zarr"
    convert_olci_optimized(
        build_synthetic_olci(rows=256, cols=256),
        output_path=str(out),
        min_dimension=64,
        output_grid="EPSG:4326",
    )
    return out


def _convert_s1(tmp: pathlib.Path) -> pathlib.Path:
    out = tmp / "s1.zarr"
    dt = MockSentinel1L1GRDBuilder("20170508T164830_0025_A094_8604_01B54C").build()
    with patch("eopf_geozarr.conversion.geozarr.print"):
        create_geozarr_dataset(
            dt,
            groups=["measurements"],
            output_path=str(out),
            gcp_group="conditions/gcp",
        )
    return out


def _convert_s2(tmp: pathlib.Path) -> pathlib.Path:
    out = tmp / "s2.zarr"
    with patch("eopf_geozarr.conversion.geozarr.print"):
        create_geozarr_dataset(
            build_sample_sentinel2_datatree(),
            groups=["/measurements/reflectance/r10m"],
            output_path=str(out),
        )
    return out


PRODUCT_CONVERTERS: dict[str, Callable[[pathlib.Path], pathlib.Path]] = {
    "olci": _convert_olci,
    "s1": _convert_s1,
    "s2": _convert_s2,
}

#: Converters that still write the legacy layout (native at group root with
#: asset "."), which xr.open_datatree rejects. The xfail keeps the hard
#: requirement on record until the generic converter migrates (cf. PR #212).
DATATREE_XFAIL: dict[str, str] = {
    "s1": "generic converter still writes asset='.' nested overview layout",
    "s2": "generic converter still writes asset='.' nested overview layout",
}


@pytest.fixture(scope="module", params=sorted(PRODUCT_CONVERTERS), ids=str)
def converted_store(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[pathlib.Path, str]:
    name = str(request.param)
    store = PRODUCT_CONVERTERS[name](tmp_path_factory.mktemp(name))
    return store, name


def _multiscale_level_paths(store: pathlib.Path) -> list[str]:
    """Every pyramid-level group path, discovered via multiscales layout attrs."""
    root = zarr.open_group(str(store), mode="r")
    found: list[str] = []

    def walk(group: zarr.Group, path: str) -> None:
        attrs = dict(group.attrs)
        multiscales = attrs.get("multiscales")
        if isinstance(multiscales, dict):
            layout = multiscales.get("layout")
            assert isinstance(layout, list)
            for entry in layout:
                assert isinstance(entry, dict)
                asset = entry["asset"]
                assert isinstance(asset, str)
                found.append(path if asset == "." else f"{path}/{asset}" if path else asset)
        for key in group.group_keys():
            child = group[key]
            assert isinstance(child, zarr.Group)
            walk(child, f"{path}/{key}" if path else key)

    walk(root, "")
    return found


def test_every_level_is_regular_grid_with_crs(
    converted_store: tuple[pathlib.Path, str],
) -> None:
    """Hard requirement: each pyramid level is a regular grid with a real CRS."""
    store, name = converted_store
    levels = _multiscale_level_paths(store)
    assert levels, f"{name}: no multiscale groups found in {store}"
    for level in levels:
        ds = xr.open_dataset(
            str(store), group=level, engine="zarr", consolidated=False, decode_coords="all"
        )
        crs = ds.rio.crs
        assert crs is not None, f"{name}:{level}: no CRS declared"
        # CRS round-trips through pyproj
        ProjCRS.from_user_input(crs.to_wkt())
        for dim in ("y", "x"):
            assert dim in ds.sizes, f"{name}:{level}: missing spatial dim {dim}"
            coord = ds[dim]
            assert coord.ndim == 1, f"{name}:{level}: {dim} coordinate is not 1-D"
            steps = np.diff(coord.values)
            assert np.allclose(steps, steps[0]), (
                f"{name}:{level}: {dim} coordinate spacing is not regular"
            )
        ds.close()


def test_store_opens_as_datatree(converted_store: tuple[pathlib.Path, str]) -> None:
    """Hard requirement: the whole store opens with xr.open_datatree."""
    store, name = converted_store
    if name in DATATREE_XFAIL:
        pytest.xfail(DATATREE_XFAIL[name])
    xr.open_datatree(str(store), engine="zarr", consolidated=False, chunks={})


@pytest.fixture(scope="module")
def native_olci_store(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    out = tmp_path_factory.mktemp("olci_native") / "olci_native.zarr"
    convert_olci_optimized(
        build_synthetic_olci(rows=256, cols=256), output_path=str(out), min_dimension=64
    )
    return out


def test_olci_native_store_opens_with_swath_geolocation(
    native_olci_store: pathlib.Path,
) -> None:
    """Native mode keeps instrument geometry: datatree-openable, 2-D lat/lon,
    and no fabricated CRS anywhere under measurements."""
    opened = xr.open_datatree(str(native_olci_store), engine="zarr", consolidated=False, chunks={})
    levels = [k for k in opened["/measurements"].children if str(k).startswith("r")]
    assert levels, "no pyramid levels found"
    for level in levels:
        ds = opened[f"/measurements/{level}"].to_dataset()
        assert ds["latitude"].dims == ("rows", "columns")
        assert ds["longitude"].dims == ("rows", "columns")
        assert "spatial_ref" not in ds.variables
        assert ds.rio.crs is None
