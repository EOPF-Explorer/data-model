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

from eopf_geozarr.conversion.utils import (
    SType,
    _resolve_asset_keys,
    extend_stac_discovery_metadata,
)

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


#: Names of the sample `.SAFE`/`.SEN3` products (`cpm_v300rc4a/safe_products`) and the
#: product type each one has to be detected as.
SAFE_PRODUCT_NAMES: list[tuple[str, SType]] = [
    # S1 GRD (IW GRDH dual/single pol, EW GRDM)
    ("S1A_IW_GRDH_1SDV_20240201T164915_20240201T164940_052368_065517_750E.SAFE", SType.GRD),
    ("S1A_IW_GRDH_1SDV_20260613T201359_20260613T201410_064955_082F9B_E0E9.SAFE", SType.GRD),
    ("S1A_IW_GRDH_1SDV_20260629T004507_20260629T004519_065177_08375F_0312.SAFE", SType.GRD),
    ("S1A_IW_GRDH_1SSH_20260330T015222_20260330T015251_063850_0807AD_5928.SAFE", SType.GRD),
    ("S1A_IW_GRDH_1SSH_20260628T222909_20260628T222942_065175_08374E_2757.SAFE", SType.GRD),
    ("S1B_EW_GRDM_1SDV_20210528T082042_20210528T082142_027102_033CCF_D55D.SAFE", SType.GRD),
    # S1 SLC (IW, EW, WV)
    ("S1A_IW_SLC__1SDV_20250819T045501_20250819T045531_060600_0789BF_B9C4.SAFE", SType.SLC),
    ("S1C_EW_SLC__1SDH_20261119T075227_20261119T075330_010403_014CA1_C059.SAFE", SType.SLC),
    ("S1C_IW_SLC__1SDV_20260731T234807_20260731T234821_008794_0116E8_C079.SAFE", SType.SLC),
    ("S1D_IW_SLC__1SDH_20261119T103125_20261119T103140_005535_00A97F_110E.SAFE", SType.SLC),
    ("S1D_IW_SLC__1SSH_20261119T110506_20261119T110537_005535_00A983_3DC3.SAFE", SType.SLC),
    ("S1D_WV_SLC__1SSV_20261124T072000_20261124T072116_005606_00ABEB_91B8.SAFE", SType.SLC),
    # S1 OCN (IW, EW, WV)
    ("S1C_EW_OCN__2SDH_20261126T075032_20261126T075112_010505_015001_20C2.SAFE", SType.OCN),
    ("S1C_IW_OCN__2SDV_20260823T051821_20260823T051846_009118_0121A3_CC3B.SAFE", SType.OCN),
    ("S1C_IW_OCN__2SDV_20261126T090507_20261126T090537_010506_01500A_88B4.SAFE", SType.OCN),
    ("S1D_WV_OCN__2SSV_20261126T084107_20261126T085239_005636_00AD00_69BF.SAFE", SType.OCN),
    # S2
    ("S2A_MSIL1C_20260721T102041_N0512_R065_T33UUQ_20260721T153621.SAFE", SType.L1C),
    ("S2C_MSIL1C_20260909T124301_N0512_R095_T27WXN_20260909T143930.SAFE", SType.L1C),
    ("S2B_MSIL2A_20260721T100559_N0512_R022_T33UWQ_20260721T143508.SAFE", SType.L2A),
    ("S2C_MSIL2A_20260716T100601_N0512_R022_T32UQU_20260716T152417.SAFE", SType.L2A),
    # S3 OLCI
    (
        "S3A_OL_1_EFR____20260728T061313_20260728T061613_20260728T082446_0179_142_134_1440_PS1_O_NR_004.SEN3",
        SType.EFR,
    ),
    (
        "S3A_OL_1_EFR____20260825T074300_20260825T074600_20260825T094230_0179_143_149_2340_PS1_O_NR_004.SEN3",
        SType.EFR,
    ),
    (
        "S3B_OL_1_EFR____20260923T093057_20260923T093357_20260923T113901_0179_125_036_2160_ESA_O_NR_004.SEN3",
        SType.EFR,
    ),
    (
        "S3B_OL_1_ERR____20260728T071230_20260728T075648_20260728T095240_2658_122_377______ESA_O_NR_004.SEN3",
        SType.ERR,
    ),
    (
        "S3A_OL_2_LFR____20260916T095057_20260916T095357_20260916T120650_0179_144_079_2160_PS1_O_NR_003.SEN3",
        SType.LFR,
    ),
    (
        "S3A_OL_2_LRR____20260915T100712_20260915T105125_20260915T122949_2653_144_065______PS1_O_NR_003.SEN3",
        SType.LRR,
    ),
    # S3 SLSTR
    (
        "S3A_SL_1_RBT____20241224T203525_20241224T203825_20241226T064306_0179_120_328_0900_PS1_O_NT_004.SEN3",
        SType.RBT,
    ),
    (
        "S3A_SL_2_FRP____20260917T092446_20260917T092746_20260917T112506_0180_144_093______MAR_O_NR_003.SEN3",
        SType.FRP,
    ),
    (
        "S3A_SL_2_LST____20240915T072403_20240915T072703_20240915T094328_0179_117_049_2340_PS1_O_NR_004.SEN3",
        SType.LST,
    ),
]

#: Root `stac_discovery` `product:type` of the CPM 3.0.0 conversions of those products
#: (`cpm_v300rc4a/*/*.zarr/zarr.json`).
CPM_PRODUCT_TYPES: list[tuple[str, SType]] = [
    ("S01SIWGRD", SType.GRD),
    ("S01SEWGRD", SType.GRD),
    ("S01SIWSLC", SType.SLC),
    ("S01SEWSLC", SType.SLC),
    ("S01SWVSLC", SType.SLC),
    ("S01SIWOCN", SType.OCN),
    ("S01SEWOCN", SType.OCN),
    ("S01SWVOCN", SType.OCN),
    ("S02MSIL1C", SType.L1C),
    ("S02MSIL2A", SType.L2A),
    ("S03OLCEFR", SType.EFR),
    ("S03OLCERR", SType.ERR),
    ("S03OLCLFR", SType.LFR),
    ("S03OLCLRR", SType.LRR),
    ("S03SLSRBT", SType.RBT),
    ("S03SLSLST", SType.LST),
    # no CPM 3.0.0 conversion of the FRP sample exists (CPM fails on it); spelling as the others
    ("S03SLSFRP", SType.FRP),
]

#: S1 subroot (EOPF product) names of those conversions: the type is in front, the
#: swath/burst or polarisation suffix must not change it.
CPM_SUBROOT_NAMES: list[tuple[str, SType]] = [
    ("S01SIWGRD_20240201T164915_0025_A299_750E_065517", SType.GRD),
    ("S01SEWGRD_20210528T082042_0060_B155_D55D_033CCF", SType.GRD),
    ("S01SIWSLC_20260731T234807_0014_C050_C079_0116E8_IW3_258173", SType.SLC),
    ("S01SEWSLC_20261119T075227_0063_C060_C059_014CA1_EW5_300827", SType.SLC),
    ("S01SWVSLC_20261124T072000_0076_D032_91B8_00ABEB_WV1_001", SType.SLC),
    ("S01SIWOCN_20260823T051821_0025_C052_CC3B_0121A3_VV", SType.OCN),
    ("S01SWVOCN_20261126T084107_0692_D032_69BF_00AD00_VV_WV2_034", SType.OCN),
]


def _id(param: tuple[str, object]) -> str:
    return param[0][:32]


@pytest.mark.parametrize(
    ("filename", "expected"), SAFE_PRODUCT_NAMES, ids=map(_id, SAFE_PRODUCT_NAMES)
)
def test_stype_from_safe_product_name(filename: str, expected: SType) -> None:
    assert SType.from_filename(filename) is expected


@pytest.mark.parametrize(
    ("filename", "expected"), SAFE_PRODUCT_NAMES, ids=map(_id, SAFE_PRODUCT_NAMES)
)
def test_stype_from_safe_product_path(filename: str, expected: SType) -> None:
    """A directory in front of the name (also one with a trailing `/`) does not matter."""
    assert SType.from_filename(f"/data/S1A_IW_SLC__/{filename}/") is expected


@pytest.mark.parametrize(
    ("product_type", "expected"), CPM_PRODUCT_TYPES, ids=[p for p, _ in CPM_PRODUCT_TYPES]
)
def test_stype_from_cpm_product_type(product_type: str, expected: SType) -> None:
    """A tree opened from a CPM store has no name: `product:type` alone decides."""
    assert SType.from_datatree(tree_with(product_type)) is expected


@pytest.mark.parametrize(("name", "expected"), CPM_SUBROOT_NAMES, ids=map(_id, CPM_SUBROOT_NAMES))
def test_stype_from_cpm_subroot_name(name: str, expected: SType) -> None:
    assert SType.from_filename(name) is expected


@pytest.mark.parametrize("filename", ["", None])
def test_stype_from_filename_empty(filename: str | None) -> None:
    assert SType.from_filename(filename) is None


def test_stype_from_datatree_prefers_stac_product_type() -> None:
    """`product:type` wins over the tree name."""
    tree = tree_with("S01SIWSLC", name="S1A_IW_GRDH_1SDV")
    assert SType.from_datatree(tree) is SType.SLC


def test_stype_from_datatree_falls_back_to_tree_name() -> None:
    """Without a recognised `product:type`, the tree name decides."""
    assert SType.from_datatree(tree_with(name="S1A_IW_GRDH_1SDV")) is SType.GRD


def test_stype_from_datatree_without_product_type_or_name() -> None:
    assert SType.from_datatree(xr.DataTree()) is None


# --- extend_stac_discovery_metadata: when nothing is added -----------------------------


def test_unknown_product_type_leaves_assets_unchanged_and_warns() -> None:
    attrs = stac_attrs({})
    with capture_logs() as logs:
        result = extend_stac_discovery_metadata({"/measurements": dataset("a")}, attrs, None)

    assert result is attrs
    assert attrs["stac_discovery"]["assets"] == {}
    assert [entry["log_level"] for entry in logs] == ["warning"]


@pytest.mark.parametrize(
    "product_type",
    [SType.EFR, SType.ERR, SType.LFR, SType.LRR, SType.RBT, SType.FRP, SType.LST],
)
def test_s3_products_are_not_extended_yet(product_type: SType) -> None:
    """S3 has no key rule yet: its assets stay as they are, with a warning instead of the catch-all error."""
    attrs = stac_attrs({})
    with capture_logs() as logs:
        result = extend_stac_discovery_metadata(
            {"/measurements": dataset("oa01_radiance")}, attrs, product_type
        )

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


# --- extend_stac_discovery_metadata: key collisions ------------------------------------


def test_all_colliding_keys_get_their_parent_group() -> None:
    """S1 has the same variable in several groups of a burst (e.g. calibration and noise).

    None of them keeps the bare key, so no key silently stands for one of the arrays.
    """
    attrs = stac_attrs({})
    datasets = {
        f"/{BURST_1}/quality/calibration": dataset("beta_nought"),
        f"/{BURST_1}/quality/noise": dataset("beta_nought"),
        f"/{BURST_1}/quality/other": dataset("beta_nought"),
    }
    with capture_logs():
        extend_stac_discovery_metadata(datasets, attrs, SType.SLC)

    assets = attrs["stac_discovery"]["assets"]
    assert {key: asset["href"] for key, asset in assets.items()} == {
        "beta_nought_IW1_258175_calibration": f"/{BURST_1}/quality/calibration/beta_nought",
        "beta_nought_IW1_258175_noise": f"/{BURST_1}/quality/noise/beta_nought",
        "beta_nought_IW1_258175_other": f"/{BURST_1}/quality/other/beta_nought",
    }


def test_keys_do_not_depend_on_the_group_order() -> None:
    """A tree opened from Zarr lists its groups in store order, which varies between runs."""
    groups = {
        f"/{BURST_1}/conditions/gcp": dataset("elevation_angle", "incidence_angle"),
        f"/{BURST_1}/conditions/antenna_pattern": dataset("elevation_angle", "incidence_angle"),
        f"/{BURST_1}/measurements": dataset("slc"),
    }
    results = []
    for ordered in (list(groups.items()), list(reversed(groups.items()))):
        attrs = stac_attrs({})
        with capture_logs():
            extend_stac_discovery_metadata(dict(ordered), attrs, SType.SLC)
        results.append(attrs["stac_discovery"]["assets"])

    assert results[0] == results[1]
    assert set(results[0]) == {
        "slc_IW1_258175",
        "elevation_angle_IW1_258175_antenna_pattern",
        "elevation_angle_IW1_258175_gcp",
        "incidence_angle_IW1_258175_antenna_pattern",
        "incidence_angle_IW1_258175_gcp",
    }


def test_existing_key_is_never_replaced() -> None:
    """A key the input already uses for another array keeps its href."""
    existing = {"href": "/other/grd", "title": "grd"}
    attrs = stac_attrs({"grd": dict(existing)})
    with capture_logs():
        extend_stac_discovery_metadata({"/measurements": dataset("grd")}, attrs, SType.GRD)

    assets = attrs["stac_discovery"]["assets"]
    assert assets["grd"] == existing
    assert assets["grd_measurements"]["href"] == "/measurements/grd"


# --- _resolve_asset_keys ---------------------------------------------------------------


def test_resolve_qualifies_only_as_deep_as_needed() -> None:
    candidates = {
        "/s/conditions/gcp/lat": "lat",
        "/s/quality/gcp/lat": "lat",
        "/s/conditions/orbit/lat": "lat",
        "/s/measurements/slc": "slc",
    }
    with capture_logs():
        resolved = _resolve_asset_keys(candidates, set())

    assert resolved == {
        "/s/conditions/gcp/lat": "lat_conditions_gcp",
        "/s/quality/gcp/lat": "lat_quality_gcp",
        "/s/conditions/orbit/lat": "lat_conditions_orbit",
        "/s/measurements/slc": "slc",
    }


def test_resolve_never_reuses_a_qualified_name() -> None:
    """A qualified key that equals another array's base key pushes that one further."""
    candidates = {"/g1/x": "k", "/g2/x": "k", "/h/k_g1": "k_g1"}
    with capture_logs():
        resolved = _resolve_asset_keys(candidates, set())

    assert resolved == {"/g1/x": "k_g1", "/g2/x": "k_g2", "/h/k_g1": "k_g1_h"}


def test_resolve_falls_back_to_a_counter() -> None:
    """When every group-qualified name is taken, a counter that skips used keys is appended."""
    with capture_logs():
        resolved = _resolve_asset_keys({"/a/x": "x"}, {"x", "x_a", "x_1"})

    assert resolved == {"/a/x": "x_2"}


def test_resolve_result_is_unique_and_avoids_taken_keys() -> None:
    taken = {"lat", "lat_gcp"}
    candidates = {f"/{group}/gcp/lat": "lat" for group in ("a", "b", "c")} | {"/d/lat": "lat"}
    with capture_logs():
        resolved = _resolve_asset_keys(candidates, taken)

    assert set(resolved) == set(candidates)
    assert len(set(resolved.values())) == len(candidates)
    assert not set(resolved.values()) & taken
