"""Tests for `update_generic_stac_assets` (hotfix for incomplete CPM `stac_discovery` assets).

These build small in-memory trees and call the function directly; nothing is written.
The end-to-end behaviour on real S1 product layouts is covered by the S1 snapshot tests
in `tests/test_generic_rechunker.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import xarray as xr
from structlog.testing import capture_logs

from eopf_geozarr.generic_optimization.generic_converter import update_generic_stac_assets

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

BURST_1 = "S01SIWSLC_20260731T234807_0014_C050_C079_0116E8_IW1_258175"
BURST_2 = "S01SIWSLC_20260731T234807_0014_C050_C079_0116E8_IW3_258176"
GRD = "S01SIWGRD_20240201T164915_0025_A299_750E_065517"


def build_tree(groups: Mapping[str, Iterable[str]]) -> xr.DataTree:
    """A tree with one small float variable per name in each group."""
    return xr.DataTree.from_dict(
        {
            group: xr.Dataset({name: ("x", np.zeros(2, dtype="float32")) for name in names})
            for group, names in groups.items()
        }
    )


def written_datasets(tree: xr.DataTree) -> dict[str, xr.Dataset | None]:
    """The `datasets` the rechunker passes: parents without data as `None`, groups with data."""
    datasets: dict[str, xr.Dataset | None] = {}
    for path in tree.groups:
        node = tree[path]
        if path == "/":
            continue
        if node.children and not node.data_vars:
            datasets[path] = None
        elif node.data_vars:
            datasets[path] = node.to_dataset()
    return datasets


def set_assets(tree: xr.DataTree, assets: dict[str, Any], group: str = "/") -> None:
    tree[group].attrs["stac_discovery"] = {"assets": assets}


def get_assets(tree: xr.DataTree, group: str = "/") -> dict[str, Any]:
    return tree[group].attrs["stac_discovery"]["assets"]


def run(tree: xr.DataTree) -> Sequence[Mapping[str, Any]]:
    """Update the assets of `tree`; return the captured log entries."""
    with capture_logs() as logs:
        update_generic_stac_assets(tree, written_datasets(tree))
    return logs


def hrefs(assets: dict[str, Any]) -> list[str]:
    return [asset["href"] for asset in assets.values()]


# --- key rule ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "name", "expected_key"),
    [
        (f"/{BURST_1}/measurements", "slc", "slc_measurements_IW1_258175"),
        (f"/{BURST_1}/conditions/gcp", "height", "height_gcp_IW1_258175"),
        # the group name is not doubled when the variable name already ends with it
        (f"/{BURST_1}/conditions/gcp", "azimuth_time_gcp", "azimuth_time_gcp_IW1_258175"),
        (f"/{BURST_1}/conditions/attitude", "roll", "roll_attitude_IW1_258175"),
        (f"/{BURST_1}/conditions/antenna_pattern", "roll", "roll_antenna_pattern_IW1_258175"),
        # a subroot not named after a burst (S1 GRD) adds no suffix
        (f"/{GRD}/conditions/gcp", "height", "height_gcp"),
        # no subroot at all
        ("/measurements/inadir", "s7_bt", "s7_bt_inadir"),
    ],
)
def test_added_asset_key_depends_only_on_the_path(group: str, name: str, expected_key: str) -> None:
    tree = build_tree({group: [name]})
    set_assets(tree, {})

    run(tree)

    assert get_assets(tree) == {expected_key: {"href": f"{group}/{name}", "title": expected_key}}


def test_same_variable_name_in_two_groups_gets_two_keys() -> None:
    tree = build_tree(
        {
            f"/{BURST_1}/conditions/attitude": ["roll"],
            f"/{BURST_1}/conditions/antenna_pattern": ["roll"],
        }
    )
    set_assets(tree, {})

    run(tree)

    assert set(get_assets(tree)) == {
        "roll_attitude_IW1_258175",
        "roll_antenna_pattern_IW1_258175",
    }


# --- multi-burst (S1 SLC) ------------------------------------------------------------------


def build_slc_tree() -> xr.DataTree:
    """Two bursts; CPM's root assets are keyed by bare names and point at the last burst."""
    groups: dict[str, list[str]] = {}
    for burst in (BURST_1, BURST_2):
        groups[f"/{burst}/measurements"] = ["slc"]
        groups[f"/{burst}/conditions/attitude"] = ["roll", "pitch"]
        groups[f"/{burst}/conditions/antenna_pattern"] = ["roll"]
        groups[f"/{burst}/conditions/gcp"] = ["latitude"]
    tree = build_tree(groups)
    set_assets(
        tree,
        {
            "slc": {
                "href": f"/{BURST_2}/measurements/slc",
                "title": "slc",
                "roles": ["data"],
            },
            "roll": {"href": f"/{BURST_2}/conditions/antenna_pattern/roll", "title": "roll"},
            "pitch": {"href": f"/{BURST_2}/conditions/attitude/pitch", "title": "pitch"},
        },
    )
    for burst in (BURST_1, BURST_2):
        set_assets(tree, {}, f"/{burst}")
    return tree


def test_slc_every_array_of_every_burst_gets_one_asset() -> None:
    tree = build_slc_tree()
    expected = {
        f"{group}/{name}"
        for group, ds in written_datasets(tree).items()
        if ds is not None
        for name in ds.data_vars
    }

    run(tree)

    root_hrefs = hrefs(get_assets(tree))
    assert sorted(root_hrefs) == sorted(expected)
    assert len(root_hrefs) == len(set(root_hrefs))


def test_slc_bare_cpm_keys_are_rekeyed_and_keep_their_fields() -> None:
    tree = build_slc_tree()

    run(tree)

    assets = get_assets(tree)
    assert not {"slc", "roll", "pitch"} & set(assets)
    assert assets["slc_measurements_IW3_258176"] == {
        "href": f"/{BURST_2}/measurements/slc",
        "title": "slc_measurements_IW3_258176",
        "roles": ["data"],
    }
    assert assets["roll_antenna_pattern_IW3_258176"]["href"] == (
        f"/{BURST_2}/conditions/antenna_pattern/roll"
    )
    # the other burst follows the same scheme
    assert assets["slc_measurements_IW1_258175"]["href"] == f"/{BURST_1}/measurements/slc"


def test_slc_rekeyed_title_is_kept_when_it_is_not_the_key() -> None:
    tree = build_slc_tree()
    get_assets(tree)["slc"]["title"] = "Single Look Complex"

    run(tree)

    assert get_assets(tree)["slc_measurements_IW3_258176"]["title"] == "Single Look Complex"


def test_slc_each_subroot_gets_the_root_assets_inside_it() -> None:
    tree = build_slc_tree()

    run(tree)

    root = get_assets(tree)
    for burst in (BURST_1, BURST_2):
        expected = {k: v for k, v in root.items() if v["href"].startswith(f"/{burst}/")}
        assert get_assets(tree, f"/{burst}") == expected
        assert len(expected) == 5


# --- single subroot (S1 GRD) ---------------------------------------------------------------


def test_grd_cpm_keys_are_kept_and_missing_arrays_added() -> None:
    tree = build_tree(
        {
            f"/{GRD}/measurements": ["grd"],
            f"/{GRD}/conditions/coordinate_conversion": ["sr0", "slant_range_time"],
            f"/{GRD}/conditions/gcp": ["height"],
        }
    )
    cpm_assets = {
        "grd": {"href": f"/{GRD}/measurements/grd", "title": "grd"},
        # CPM keys `sr0` by another variable's name
        "slant_range_time": {
            "href": f"/{GRD}/conditions/coordinate_conversion/sr0",
            "title": "slant_range_time",
        },
    }
    set_assets(tree, {k: dict(v) for k, v in cpm_assets.items()})
    set_assets(tree, {k: dict(v) for k, v in cpm_assets.items()}, f"/{GRD}")

    run(tree)

    assets = get_assets(tree)
    assert {k: assets[k] for k in cpm_assets} == cpm_assets
    assert set(assets) - set(cpm_assets) == {
        "slant_range_time_coordinate_conversion",
        "height_gcp",
    }
    assert get_assets(tree, f"/{GRD}") == assets


# --- removal and kept assets ---------------------------------------------------------------


def test_asset_of_a_not_written_array_is_removed() -> None:
    tree = build_tree({"/measurements": ["a"]})
    set_assets(tree, {"a": {"href": "/measurements/a"}, "gone": {"href": "/measurements/gone"}})

    logs = run(tree)

    assert set(get_assets(tree)) == {"a"}
    assert any(log["event"] == "Removing STAC asset of a not written array" for log in logs)


def test_asset_of_a_group_above_a_written_array_is_kept() -> None:
    tree = build_tree({"/measurements/reflectance/r10m": ["b02"]})
    group_asset = {"href": "/measurements/reflectance", "title": "Reflectance"}
    set_assets(tree, {"reflectance": dict(group_asset)})

    run(tree)

    assert get_assets(tree)["reflectance"] == group_asset


def test_assets_without_store_href_are_kept() -> None:
    tree = build_tree({"/measurements": ["a"]})
    url_asset = {"href": "https://example.org/doc.pdf", "title": "doc"}
    set_assets(tree, {"doc": dict(url_asset), "thumbnail": {"title": "no href"}})

    run(tree)

    assets = get_assets(tree)
    assert assets["doc"] == url_asset
    assert assets["thumbnail"] == {"title": "no href"}


def test_added_key_taken_by_a_cpm_asset_is_skipped_with_a_warning() -> None:
    """Known limitation: the fixed key of an added array can collide with a CPM key."""
    tree = build_tree({"/conditions/gcp": ["height", "other"]})
    set_assets(tree, {"height_gcp": {"href": "/conditions/gcp/other"}})

    logs = run(tree)

    assert get_assets(tree) == {"height_gcp": {"href": "/conditions/gcp/other"}}
    assert any(log["event"].startswith("STAC asset key already in use") for log in logs)


def test_spatial_ref_and_parent_groups_get_no_asset() -> None:
    tree = build_tree({"/measurements/inadir": ["s7_bt", "spatial_ref"]})
    set_assets(tree, {})

    run(tree)

    assert hrefs(get_assets(tree)) == ["/measurements/inadir/s7_bt"]


# --- subroot merge -------------------------------------------------------------------------


def test_subroot_assets_are_merged_not_replaced() -> None:
    tree = build_slc_tree()
    url_asset = {"href": "https://example.org/doc.pdf"}
    # a URL asset and a bare duplicate of an asset the root re-keys
    set_assets(
        tree,
        {"doc": dict(url_asset), "slc": {"href": f"/{BURST_1}/measurements/slc"}},
        f"/{BURST_1}",
    )

    run(tree)

    sub = get_assets(tree, f"/{BURST_1}")
    assert sub["doc"] == url_asset
    assert "slc" not in sub
    assert sub["slc_measurements_IW1_258175"]["href"] == f"/{BURST_1}/measurements/slc"
    assert hrefs(sub).count(f"/{BURST_1}/measurements/slc") == 1


@pytest.mark.parametrize("sub_stac", [{}, {"assets": None}], ids=["no-assets", "assets-none"])
def test_subroot_without_assets_gets_them(sub_stac: dict[str, Any]) -> None:
    tree = build_slc_tree()
    tree[f"/{BURST_1}"].attrs["stac_discovery"] = sub_stac

    run(tree)

    assert len(get_assets(tree, f"/{BURST_1}")) == 5


def test_subroot_without_stac_discovery_is_left_alone() -> None:
    tree = build_slc_tree()
    del tree[f"/{BURST_1}"].attrs["stac_discovery"]

    run(tree)

    assert "stac_discovery" not in tree[f"/{BURST_1}"].attrs


# --- no-ops --------------------------------------------------------------------------------


def test_no_stac_discovery_is_a_no_op() -> None:
    tree = build_tree({"/measurements": ["a"]})

    run(tree)

    assert "stac_discovery" not in tree.attrs


def test_nothing_written_leaves_assets_unchanged() -> None:
    tree = build_tree({"/measurements": ["a"]})
    set_assets(tree, {"a": {"href": "/measurements/a"}})

    with capture_logs():
        update_generic_stac_assets(tree, {"/measurements": None})

    assert get_assets(tree) == {"a": {"href": "/measurements/a"}}
