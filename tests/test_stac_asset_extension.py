"""Unit tests of the product-type detection and the `stac_discovery` asset extension.

These work on in-memory datasets and attribute dicts only, so they need no store. The
end-to-end behaviour (assets in the written root and subroot metadata) is tested in
`tests/test_generic_rechunker.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr
from structlog.testing import capture_logs

from eopf_geozarr.conversion.utils import SType, extend_stac_discovery_metadata

BURST_1 = "S01SIWSLC_X_IW1_258175"
BURST_2 = "S01SIWSLC_X_IW3_258176"


def dataset(*names: str) -> xr.Dataset:
    """A tiny dataset with one 1-D data variable per name."""
    return xr.Dataset({name: ("x", np.zeros(2)) for name in names})


def stac_attrs(assets: dict[str, Any] | None) -> dict[Any, Any]:
    """Group attributes with the given `stac_discovery` assets (None: no `assets` key)."""
    stac: dict[str, Any] = {"properties": {}}
    if assets is not None:
        stac["assets"] = assets
    return {"stac_discovery": stac}


def tree_with(product_type: str | None = None, name: str | None = None) -> xr.DataTree:
    tree = xr.DataTree(name=name)
    if product_type is not None:
        tree.attrs["stac_discovery"] = {"properties": {"product:type": product_type}}
    return tree


# --- SType -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("S01SIWSLC", SType.SLC),
        ("S1C_IW_SLC__1SDV_20260731T234807", SType.SLC),
        ("S01SIWGRD", SType.GRD),
        ("S1A_IW_GRDH_1SDV_20260613T201359", SType.GRD),
        ("S02MSIL1C", SType.L1C),
        ("S2B_MSIL2A_20250101T000000", SType.L2A),
        ("S01SIWOCN", SType.OCN),
        ("S03OLCEFR", None),
        ("", None),
        (None, None),
    ],
)
def test_stype_from_filename(filename: str | None, expected: SType | None) -> None:
    assert SType.from_filename(filename) is expected


def test_stype_from_datatree_prefers_stac_product_type() -> None:
    """`product:type` wins over the tree name."""
    tree = tree_with("S01SIWSLC", name="S1A_IW_GRDH_1SDV")
    assert SType.from_datatree(tree) is SType.SLC


def test_stype_from_datatree_falls_back_to_tree_name() -> None:
    """Without a recognised `product:type`, the tree name decides."""
    assert SType.from_datatree(tree_with(name="S1A_IW_GRDH_1SDV")) is SType.GRD
    assert SType.from_datatree(tree_with("S03OLCEFR", name="S2B_MSIL2A")) is SType.L2A


def test_stype_from_datatree_unknown_product() -> None:
    assert SType.from_datatree(xr.DataTree()) is None
    assert SType.from_datatree(tree_with("S03OLCEFR")) is None


# --- extend_stac_discovery_metadata: when nothing is added -----------------------------


def test_unknown_product_type_leaves_assets_unchanged_and_warns() -> None:
    attrs = stac_attrs({})
    with capture_logs() as logs:
        result = extend_stac_discovery_metadata({"/measurements": dataset("a")}, attrs, None)

    assert result is attrs
    assert attrs["stac_discovery"]["assets"] == {}
    assert [entry["log_level"] for entry in logs] == ["warning"]


@pytest.mark.parametrize("attrs", [{}, stac_attrs(None)], ids=["no_stac", "no_assets"])
def test_missing_assets_are_not_created(attrs: dict[Any, Any]) -> None:
    """Only existing `assets` are extended; a group without them is left alone."""
    before = {key: dict(value) for key, value in attrs.items()}
    extend_stac_discovery_metadata({"/measurements": dataset("a")}, attrs, SType.GRD)
    assert attrs == before


def test_unsupported_product_type_raises() -> None:
    """Every `SType` has a key rule; the catch-all guards against a new one without a rule."""
    with pytest.raises(ValueError, match="supported"):
        extend_stac_discovery_metadata(
            {"/measurements": dataset("a")},
            stac_attrs({}),
            "XYZ",  # type: ignore[arg-type]
        )


def test_skips_spatial_ref_none_and_non_datasets() -> None:
    attrs = stac_attrs({})
    datasets: dict[str, object] = {
        "/measurements": dataset("spatial_ref"),
        "/conditions": None,
        "/quality": xr.DataArray(np.zeros(2)),
    }
    with capture_logs() as logs:
        extend_stac_discovery_metadata(datasets, attrs, SType.GRD)

    assert attrs["stac_discovery"]["assets"] == {}
    assert [entry["event"] for entry in logs] == ["Found non-dataset object in datasets!"]


def test_existing_hrefs_are_not_duplicated() -> None:
    """An array already referenced (with or without a leading `/`) gets no second asset."""
    existing = {"grd_vv": {"href": "measurements/grd", "title": "grd_vv"}}
    attrs = stac_attrs(dict(existing))
    extend_stac_discovery_metadata({"/measurements": dataset("grd")}, attrs, SType.GRD)
    assert attrs["stac_discovery"]["assets"] == existing


def test_url_hrefs_do_not_count_as_existing() -> None:
    """A remote asset (e.g. a product URL) does not block an asset for a store array."""
    attrs = stac_attrs({"product": {"href": "s3://bucket/measurements/grd"}})
    extend_stac_discovery_metadata({"/measurements": dataset("grd")}, attrs, SType.GRD)
    assert set(attrs["stac_discovery"]["assets"]) == {"product", "grd"}


# --- extend_stac_discovery_metadata: asset keys per product type -----------------------


@pytest.mark.parametrize("product_type", [SType.L2A, SType.L1C])
def test_s2_keys_carry_the_resolution(product_type: SType) -> None:
    attrs = stac_attrs({})
    datasets = {
        "/measurements/reflectance/r10m": dataset("b02"),
        "/measurements/reflectance/r120m": dataset("b02", "b08"),
    }
    extend_stac_discovery_metadata(datasets, attrs, product_type)

    assert attrs["stac_discovery"]["assets"] == {
        "b02_r10m": {"href": "/measurements/reflectance/r10m/b02", "title": "b02_r10m"},
        "b02_r120m": {"href": "/measurements/reflectance/r120m/b02", "title": "b02_r120m"},
        "b08_r120m": {"href": "/measurements/reflectance/r120m/b08", "title": "b08_r120m"},
    }


def test_grd_keys_are_the_variable_name() -> None:
    attrs = stac_attrs({})
    datasets = {"/measurements": dataset("grd"), "/conditions/gcp": dataset("latitude")}
    extend_stac_discovery_metadata(datasets, attrs, SType.GRD)

    assert attrs["stac_discovery"]["assets"] == {
        "grd": {"href": "/measurements/grd", "title": "grd"},
        "latitude": {"href": "/conditions/gcp/latitude", "title": "latitude"},
    }


def test_slc_keys_carry_swath_and_burst() -> None:
    attrs = stac_attrs({})
    datasets = {
        f"/{BURST_1}/measurements": dataset("slc"),
        f"/{BURST_2}/measurements": dataset("slc"),
        f"/{BURST_2}/conditions/gcp": dataset("azimuth_time_gcp"),
    }
    extend_stac_discovery_metadata(datasets, attrs, SType.SLC)

    assert set(attrs["stac_discovery"]["assets"]) == {
        "slc_IW1_258175",
        "slc_IW3_258176",
        "azimuth_time_gcp_IW3_258176",
    }


def test_new_assets_title_equals_key() -> None:
    attrs = stac_attrs({})
    extend_stac_discovery_metadata({f"/{BURST_1}/measurements": dataset("slc")}, attrs, SType.SLC)
    assert all(asset["title"] == key for key, asset in attrs["stac_discovery"]["assets"].items())


# --- extend_stac_discovery_metadata: key collisions ------------------------------------


def test_colliding_key_gets_the_parent_group_appended() -> None:
    """S1 has the same variable in several groups of a burst (e.g. calibration and noise)."""
    attrs = stac_attrs({})
    datasets = {
        f"/{BURST_1}/quality/calibration": dataset("beta_nought"),
        f"/{BURST_1}/quality/noise": dataset("beta_nought"),
        f"/{BURST_1}/quality/other": dataset("beta_nought"),
    }
    with capture_logs():
        extend_stac_discovery_metadata(datasets, attrs, SType.SLC)

    assets = attrs["stac_discovery"]["assets"]
    assert set(assets) == {
        "beta_nought_IW1_258175",
        "beta_nought_IW1_258175_noise",
        "beta_nought_IW1_258175_other",
    }
    assert assets["beta_nought_IW1_258175_noise"]["href"] == (
        f"/{BURST_1}/quality/noise/beta_nought"
    )


def test_existing_key_is_never_replaced() -> None:
    """A key the input already uses for another array keeps its href."""
    existing = {"href": "/other/grd", "title": "grd"}
    attrs = stac_attrs({"grd": dict(existing)})
    with capture_logs():
        extend_stac_discovery_metadata({"/measurements": dataset("grd")}, attrs, SType.GRD)

    assets = attrs["stac_discovery"]["assets"]
    assert assets["grd"] == existing
    assert assets["grd_measurements"]["href"] == "/measurements/grd"


def test_unresolvable_collision_skips_the_asset_with_a_warning() -> None:
    """Same variable and same parent group name in two places: the second gets no asset.

    This is a known gap of the collision handling; the array is then missing from STAC.
    """
    attrs = stac_attrs({})
    datasets = {
        "/a/gcp": dataset("latitude"),
        "/b/gcp": dataset("latitude"),
        "/c/gcp": dataset("latitude"),
    }
    with capture_logs() as logs:
        extend_stac_discovery_metadata(datasets, attrs, SType.GRD)

    assert set(attrs["stac_discovery"]["assets"]) == {"latitude", "latitude_gcp"}
    warnings = [entry for entry in logs if entry["log_level"] == "warning"]
    assert [(entry["asset"], entry["href"]) for entry in warnings] == [
        ("latitude_gcp", "/c/gcp/latitude")
    ]
