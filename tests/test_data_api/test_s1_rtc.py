"""
Round-trip and validation tests for Sentinel-1 GRD RTC pydantic-zarr models.

These tests verify that S1 RTC GeoZarr V3 store metadata can be:
1. Loaded from example JSON data using direct instantiation
2. Validated through Pydantic models
3. Round-tripped without data loss
4. Rejects invalid structures

The JSON-driven tests read ``tests/_test_data/s1_rtc_examples/*.json`` through the
``s1_rtc_json_example`` fixture. Those files are NOT hand-maintained: regenerate them with
``tests/_test_data/s1_rtc_examples/regenerate.py``, which dumps a store built by the real
ingester. A hand-edited fixture is how this model came to reject every store the writer
produced while the suite stayed green — see ``TestWriterRoundTrip`` below, which validates
against a freshly ingested store and is the check that cannot go stale.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

from eopf_geozarr.conversion.s1_ingest import (
    consolidate_s1_store,
    ingest_s1tiling_acquisition,
    ingest_s1tiling_conditions,
)
from eopf_geozarr.data_api.s1_rtc import CONDITION_ARRAY_PREFIXES as _CONDITION_PREFIXES
from eopf_geozarr.data_api.s1_rtc import S1RtcRoot
from tests.test_s1_rtc_ingest import ACQ1_TAGS, SIZE, _create_synthetic_geotiff

if TYPE_CHECKING:
    from pathlib import Path


def _nav(node: object, *keys: str) -> dict[str, object]:
    """Walk a path of string keys through a nested JSON dict (test helper).

    Narrows each level to ``dict`` so mutating the raw fixture stays type-safe.
    """
    for key in keys:
        assert isinstance(node, dict), f"expected a dict at {key!r}"
        node = node[key]
    assert isinstance(node, dict), "expected the leaf to be a dict"
    return node


def test_s1_rtc_roundtrip(s1_rtc_json_example: dict[str, object]) -> None:
    """Test that we can round-trip JSON data without loss."""
    model1 = S1RtcRoot.model_validate(s1_rtc_json_example)
    dumped = model1.model_dump()
    model2 = S1RtcRoot.model_validate(dumped)
    assert model1.model_dump() == model2.model_dump()


def test_s1_rtc_descending_present(s1_rtc_json_example: dict[str, object]) -> None:
    """Test that the fixture has a descending orbit group."""
    model = S1RtcRoot.model_validate(s1_rtc_json_example)
    assert model.descending is not None
    assert model.ascending is None


def test_s1_rtc_r10m_has_data_arrays(s1_rtc_json_example: dict[str, object]) -> None:
    """Test that r10m contains vv, vh, border_mask and coordinate arrays."""
    model = S1RtcRoot.model_validate(s1_rtc_json_example)
    assert model.descending is not None
    r10m = model.descending.r10m
    assert r10m.vv is not None
    assert r10m.vh is not None
    assert r10m.border_mask is not None
    assert "time" in r10m.members
    assert "absolute_orbit" in r10m.members
    assert "relative_orbit" in r10m.members
    assert "platform" in r10m.members


def test_s1_rtc_overview_levels(s1_rtc_json_example: dict[str, object]) -> None:
    """Test that overview levels r20m-r720m exist and have vv/vh/border_mask."""
    model = S1RtcRoot.model_validate(s1_rtc_json_example)
    orbit = model.descending
    assert orbit is not None
    for level in ("r20m", "r60m", "r120m", "r360m", "r720m"):
        group = orbit.get_resolution(level)
        assert group is not None, f"Missing overview level {level}"
        assert "vv" in group.members
        assert "vh" in group.members
        assert "border_mask" in group.members
        # Overview levels DO carry the coordinate arrays. `time` is replicated at every level
        # with identical values (data-model #192): TiTiler renders previews off a coarse level,
        # and a level without `time` cannot be selected by datetime. This assertion used to
        # demand the opposite -- it was written against a fixture that predated #192, and it is
        # the reason a writer/model mismatch survived in the suite.
        for coord in ("time", "x", "y", "spatial_ref"):
            assert coord in group.members, f"{level} is missing the {coord!r} coordinate"


def test_s1_rtc_conditions(s1_rtc_json_example: dict[str, object]) -> None:
    """Test that conditions group has gamma_area per-orbit arrays."""
    model = S1RtcRoot.model_validate(s1_rtc_json_example)
    assert model.descending is not None
    conditions = model.descending.conditions
    assert conditions is not None
    gamma_keys = [k for k in conditions.members if k.startswith("gamma_area_")]
    assert len(gamma_keys) >= 1


def test_s1_rtc_orbit_attrs(s1_rtc_json_example: dict[str, object]) -> None:
    """Test that orbit group attributes contain required conventions and metadata."""
    model = S1RtcRoot.model_validate(s1_rtc_json_example)
    assert model.descending is not None
    attrs = model.descending.attributes
    assert len(attrs.zarr_conventions) == 3
    assert attrs.proj_code.startswith("EPSG:")
    assert attrs.spatial_dimensions == ["y", "x"]
    assert len(attrs.spatial_bbox) == 4
    assert len(attrs.multiscales.layout) == 6
    assert attrs.multiscales.layout[0].asset == "r10m"


def test_s1_rtc_rejects_no_orbit(s1_rtc_json_example: dict[str, object]) -> None:
    """Reject a store with no orbit groups."""
    data = copy.deepcopy(s1_rtc_json_example)
    data["members"] = {}
    with pytest.raises(Exception, match="at least one orbit"):
        S1RtcRoot.model_validate(data)


def test_s1_rtc_rejects_missing_r10m(s1_rtc_json_example: dict[str, object]) -> None:
    """Reject an orbit group that lacks r10m."""
    data = copy.deepcopy(s1_rtc_json_example)
    del _nav(data, "members", "descending", "members")["r10m"]
    with pytest.raises(Exception, match="r10m"):
        S1RtcRoot.model_validate(data)


def test_s1_rtc_rejects_missing_convention_uuid(s1_rtc_json_example: dict[str, object]) -> None:
    """Reject orbit attrs with missing convention UUIDs."""
    data = copy.deepcopy(s1_rtc_json_example)
    _nav(data, "members", "descending", "attributes")["zarr_conventions"] = [
        {"uuid": "fake-uuid", "name": "fake"}
    ]
    with pytest.raises(Exception, match="Missing required zarr_conventions"):
        S1RtcRoot.model_validate(data)


def test_s1_rtc_rejects_bad_spatial_dimensions(s1_rtc_json_example: dict[str, object]) -> None:
    """Reject orbit attrs with wrong spatial:dimensions."""
    data = copy.deepcopy(s1_rtc_json_example)
    _nav(data, "members", "descending", "attributes")["spatial:dimensions"] = ["lat", "lon"]
    with pytest.raises(Exception, match="spatial:dimensions"):
        S1RtcRoot.model_validate(data)


def test_s1_rtc_rejects_conditions_without_condition_arrays(
    s1_rtc_json_example: dict[str, object],
) -> None:
    """Reject a conditions group holding no condition array at all.

    Keeps the coordinate arrays so this exercises the condition-array check specifically —
    stripping them too would trip `validate_coordinate_arrays` first and the test would pass
    for the wrong reason.
    """
    data = copy.deepcopy(s1_rtc_json_example)
    cond_members = _nav(data, "members", "descending", "members", "conditions", "members")
    kept = {name: spec for name, spec in cond_members.items() if name in ("x", "y", "spatial_ref")}
    assert set(kept) == {"x", "y", "spatial_ref"}, "fixture lost its conditions coordinates"
    kept["some_other"] = next(iter(cond_members.values()))
    _nav(data, "members", "descending", "members", "conditions")["members"] = kept
    with pytest.raises(Exception, match="at least one condition array"):
        S1RtcRoot.model_validate(data)


def test_s1_rtc_accepts_lia_only_conditions(s1_rtc_json_example: dict[str, object]) -> None:
    """Accept a conditions group with `lia_*` but no `gamma_area_*`.

    `ingest_s1tiling_conditions` requires only that *one* of its three condition paths be
    given, and the CLI's three condition flags all default to None, so a lia-only conditions
    group is writable today. Demanding `gamma_area_*` specifically made such a store
    unopenable by this model.
    """
    data = copy.deepcopy(s1_rtc_json_example)
    conditions = _nav(data, "members", "descending", "members", "conditions")
    cond_members = _nav(data, "members", "descending", "members", "conditions", "members")
    conditions["members"] = {
        name.replace("gamma_area_", "lia_"): spec for name, spec in cond_members.items()
    }
    model = S1RtcRoot.model_validate(data)
    assert model.descending is not None
    assert model.descending.conditions is not None


# =============================================================================
# Writer ↔ model round-trip
# =============================================================================


def _condition_geotiff(tmp_path: Path, name: str) -> Path:
    """Write a synthetic (Y, X) condition raster on the ingest tests' reference grid."""
    path = tmp_path / f"{name}.tif"
    rng = np.random.default_rng(len(name))
    _create_synthetic_geotiff(path, rng.uniform(0.5, 2.0, (SIZE, SIZE)).astype(np.float32))
    return path


def _ingest_store(tmp_path: Path, condition_kwargs: list[str]) -> Path:
    """Build a real S1 RTC store with the production ingester and return its path."""
    rng = np.random.default_rng(216)
    store_path = tmp_path / "s1-grd-rtc-roundtrip.zarr"

    vv = tmp_path / "acq_vv.tif"
    vh = tmp_path / "acq_vh.tif"
    mask = tmp_path / "acq_mask.tif"
    for path, data in (
        (vv, rng.uniform(0.0, 1.0, (SIZE, SIZE)).astype(np.float32)),
        (vh, rng.uniform(0.0, 0.5, (SIZE, SIZE)).astype(np.float32)),
        (mask, np.ones((SIZE, SIZE), dtype=np.uint8)),
    ):
        _create_synthetic_geotiff(path, data, tags=ACQ1_TAGS)
    ingest_s1tiling_acquisition(vv, vh, mask, store_path, "ascending")

    ingest_s1tiling_conditions(
        store_path=store_path,
        orbit_direction="ascending",
        relative_orbit=37,
        **{
            kwarg: _condition_geotiff(tmp_path, kwarg.removesuffix("_path"))
            for kwarg in condition_kwargs
        },
    )
    consolidate_s1_store(store_path, "ascending")
    return store_path


def _open_store(store_path: Path) -> zarr.Group:
    return zarr.open_group(str(store_path), mode="r", zarr_format=3)


class TestWriterRoundTrip:
    """Validate the model against stores this repo's own writer produces.

    Every other test in this module runs off a checked-in JSON fixture, which is why the model
    could reject 23 arrays that the ingester has always written (`x`, `y` and `spatial_ref` at
    every level, plus per-level `time` since data-model #192) without a single test noticing.
    These tests close the loop in both directions: a freshly ingested store must validate, and
    a store with a coordinate removed must not.
    """

    @pytest.mark.parametrize(
        "condition_kwargs",
        [
            pytest.param(["gamma_area_path"], id="gamma-only"),
            # lia-only is reachable from the CLI (all three condition flags default to None),
            # so the model has to open it -- see test_s1_rtc_accepts_lia_only_conditions.
            pytest.param(["lia_path"], id="lia-only"),
            pytest.param(
                ["gamma_area_path", "lia_path", "incidence_angle_path"], id="all-conditions"
            ),
        ],
    )
    def test_ingested_store_validates(self, tmp_path: Path, condition_kwargs: list[str]) -> None:
        """A store built by the ingester opens through the model."""
        store_path = _ingest_store(tmp_path, condition_kwargs)

        model = S1RtcRoot.from_zarr(_open_store(store_path))

        assert model.ascending is not None
        assert model.ascending.conditions is not None
        for level in ("r10m", "r20m", "r60m", "r120m", "r360m", "r720m"):
            group = model.ascending.get_resolution(level)
            assert group is not None, f"missing level {level}"
            for coord in ("time", "x", "y", "spatial_ref"):
                assert coord in group.members, f"{level} lost {coord!r}"

    @pytest.mark.parametrize("coordinate", ["time", "x", "y", "spatial_ref"])
    def test_overview_missing_coordinate_is_rejected(self, tmp_path: Path, coordinate: str) -> None:
        """Deleting a coordinate from one overview level must fail validation.

        This is the regression the model exists to catch: `time` was absent from the overview
        levels before data-model #192, which broke datetime `.sel` on exactly the coarse levels
        TiTiler renders previews from. Listing the keys in the member TypedDicts would NOT catch
        it -- they are `total=False`, so every key is NotRequired and a *missing* one is legal --
        which is why both dataset classes carry an explicit presence validator.
        """
        store_path = _ingest_store(tmp_path, ["gamma_area_path"])
        # Sanity: the untouched store validates, so a failure below is the deletion talking.
        S1RtcRoot.from_zarr(_open_store(store_path))

        orbit = zarr.open_group(
            str(store_path / "ascending"), mode="r+", zarr_format=3, use_consolidated=False
        )
        level = orbit["r60m"]
        assert isinstance(level, zarr.Group)
        del level[coordinate]
        consolidate_s1_store(store_path, "ascending")

        with pytest.raises(Exception, match="coordinate arrays"):
            S1RtcRoot.from_zarr(_open_store(store_path))

    def test_fixture_still_matches_writer_output(
        self, tmp_path: Path, s1_rtc_json_example: dict[str, object]
    ) -> None:
        """The checked-in fixture must keep describing what the writer actually emits.

        The fixture is a writer snapshot produced by
        ``tests/_test_data/s1_rtc_examples/regenerate.py``. Nothing else compares the two, so a
        hand-edit (or a writer change made without regenerating) would silently reopen the drift
        this module exists to close -- one level down from the original bug.

        Compares member *names* per group, which is what the closed TypedDicts and the coordinate
        validators actually read. Condition array names are orbit-suffixed and differ between the
        fixture's tile and this synthetic store, so they are compared by prefix, not verbatim.
        """
        store_path = _ingest_store(tmp_path, ["gamma_area_path"])
        ingested = S1RtcRoot.from_zarr(_open_store(store_path))

        fixture_orbit = _nav(s1_rtc_json_example, "members", "descending")
        ingested_root = ingested.model_dump(by_alias=True, mode="json")
        ingested_orbit = _nav(ingested_root, "members", "ascending")

        fixture_levels = _nav(fixture_orbit, "members")
        ingested_levels = _nav(ingested_orbit, "members")
        assert set(fixture_levels) == set(ingested_levels), (
            "the fixture and the writer disagree on the orbit group's members; "
            "re-run tests/_test_data/s1_rtc_examples/regenerate.py"
        )

        for level in ("r10m", "r20m", "r60m", "r120m", "r360m", "r720m"):
            fixture_members = set(_nav(fixture_levels, level, "members"))
            ingested_members = set(_nav(ingested_levels, level, "members"))
            assert fixture_members == ingested_members, (
                f"{level} members drifted from writer output; "
                "re-run tests/_test_data/s1_rtc_examples/regenerate.py"
            )

        # Conditions: coordinates verbatim, condition rasters by prefix (orbit suffixes differ).
        def _split(members: set[str]) -> tuple[set[str], set[str]]:
            conditions = {m.rsplit("_", 1)[0] for m in members if m.startswith(_CONDITION_PREFIXES)}
            return members - {m for m in members if m.startswith(_CONDITION_PREFIXES)}, conditions

        fixture_coords, fixture_conditions = _split(
            set(_nav(fixture_levels, "conditions", "members"))
        )
        ingested_coords, ingested_conditions = _split(
            set(_nav(ingested_levels, "conditions", "members"))
        )
        assert fixture_coords == ingested_coords, (
            "conditions-group coordinates drifted from writer output; "
            "re-run tests/_test_data/s1_rtc_examples/regenerate.py"
        )
        assert ingested_conditions <= fixture_conditions


@pytest.mark.parametrize("coordinate", ["time", "x", "y", "spatial_ref"])
def test_native_missing_coordinate_is_rejected(
    s1_rtc_json_example: dict[str, object], coordinate: str
) -> None:
    """The native level is validated by the same rule as the overviews.

    `test_overview_missing_coordinate_is_rejected` only deletes from an overview level, so
    without this the native validator could be removed and the suite would stay green.
    """
    data = copy.deepcopy(s1_rtc_json_example)
    del _nav(data, "members", "descending", "members", "r10m", "members")[coordinate]
    with pytest.raises(Exception, match="coordinate arrays"):
        S1RtcRoot.model_validate(data)


@pytest.mark.parametrize("coordinate", ["x", "y", "spatial_ref"])
def test_conditions_missing_coordinate_is_rejected(
    s1_rtc_json_example: dict[str, object], coordinate: str
) -> None:
    """A conditions group without 1-D coordinates is not georeferenced at all.

    It opens with "dimensions without coordinates" and rioxarray infers an identity transform,
    which is the same class of defect the level-coordinate validators exist to catch. The
    conditions group carries no `time` (its rasters are time-invariant).
    """
    data = copy.deepcopy(s1_rtc_json_example)
    del _nav(data, "members", "descending", "members", "conditions", "members")[coordinate]
    with pytest.raises(Exception, match="coordinate arrays"):
        S1RtcRoot.model_validate(data)
