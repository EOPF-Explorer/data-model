"""
Pydantic-zarr integrated models for Sentinel-1 GRD gamma0T RTC GeoZarr stores.

Uses the pyz.v3 GroupSpec/ArraySpec with TypedDict members to enforce strict
structure validation — same pattern as s2.py (which uses pyz.v2 for Zarr V2).

These models validate time-series Zarr V3 stores built from S1Tiling GeoTIFFs
on the Sentinel-2 MGRS grid. This is a *different data product* from the EOPF
L1 GRD models in s1.py — those describe radar-geometry Zarr V2 products.

Store hierarchy::

    s1-grd-rtc-{tile}.zarr/
    ├── zarr.json
    ├── ascending/
    │   ├── zarr.json          # zarr_conventions, multiscales, proj:, spatial:
    │   ├── r10m/              # native resolution dataset
    │   │   ├── vv/            # (time, Y, X) float32
    │   │   ├── vh/            # (time, Y, X) float32
    │   │   ├── border_mask/   # (time, Y, X) uint8
    │   │   ├── time/          # (time,) int64 datetime
    │   │   ├── x/, y/         # (X,) / (Y,) float64
    │   │   ├── spatial_ref/   # CF grid mapping (scalar)
    │   │   ├── absolute_orbit/
    │   │   ├── relative_orbit/
    │   │   └── platform/
    │   ├── r20m/ … r720m/     # overview levels: vv, vh, border_mask +
    │   │                      #   time, x, y, spatial_ref (no orbit metadata)
    │   └── conditions/
    │       ├── x/, y/, spatial_ref/
    │       └── gamma_area_{orbit}/  # (Y, X) float32; also lia_*/incidence_angle_*
    └── descending/
        └── (same structure)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict
from zarr_cm import geo_proj
from zarr_cm import multiscales as multiscales_cm
from zarr_cm import spatial as spatial_cm

from eopf_geozarr.data_api.geozarr.common import DatasetAttrs
from eopf_geozarr.data_api.geozarr.multiscales.zcm import Multiscales
from eopf_geozarr.pyz.v3 import ArraySpec, GroupSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

# ============================================================================
# Constants
# ============================================================================

MULTISCALES_UUID = multiscales_cm.UUID
GEO_PROJ_UUID = geo_proj.UUID
SPATIAL_UUID = spatial_cm.UUID

REQUIRED_CONVENTION_UUIDS = frozenset({MULTISCALES_UUID, GEO_PROJ_UUID, SPATIAL_UUID})

ResolutionLevel = Literal["r10m", "r20m", "r60m", "r120m", "r360m", "r720m"]
OrbitDirection = Literal["ascending", "descending"]
Polarisation = Literal["vv", "vh"]

# ============================================================================
# Attributes models
# ============================================================================


class S1RtcOrbitGroupAttrs(BaseModel):
    """Attributes for an orbit-direction group (ascending or descending).

    Carries the three GeoZarr conventions plus proj:/spatial:/multiscales metadata.
    """

    zarr_conventions: list[dict[str, Any]]
    multiscales: Multiscales
    proj_code: str = Field(alias="proj:code")
    spatial_dimensions: list[str] = Field(alias="spatial:dimensions")
    spatial_bbox: list[float] = Field(alias="spatial:bbox")

    model_config = {"extra": "allow", "populate_by_name": True, "serialize_by_alias": True}

    @model_validator(mode="after")
    def validate_zarr_conventions(self) -> Self:
        """Ensure all three required convention UUIDs are present."""
        present = {c["uuid"] for c in self.zarr_conventions if "uuid" in c}
        missing = REQUIRED_CONVENTION_UUIDS - present
        if missing:
            raise ValueError(f"Missing required zarr_conventions UUIDs: {missing}")
        return self

    @model_validator(mode="after")
    def validate_spatial_dimensions(self) -> Self:
        if self.spatial_dimensions != ["y", "x"]:
            raise ValueError(
                f"spatial:dimensions must be ['y', 'x'], got {self.spatial_dimensions}"
            )
        return self

    @model_validator(mode="after")
    def validate_spatial_bbox(self) -> Self:
        if len(self.spatial_bbox) != 4:
            raise ValueError(f"spatial:bbox must have 4 elements, got {len(self.spatial_bbox)}")
        return self


class S1RtcResolutionAttrs(BaseModel):
    """Attributes for a resolution-level group (r10m, r20m, ...)."""

    spatial_shape: list[int] = Field(alias="spatial:shape")
    spatial_transform: list[float] = Field(alias="spatial:transform")

    model_config = {"extra": "allow", "populate_by_name": True, "serialize_by_alias": True}

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if len(self.spatial_shape) != 2:
            raise ValueError(f"spatial:shape must have 2 elements, got {len(self.spatial_shape)}")
        return self

    @model_validator(mode="after")
    def validate_transform(self) -> Self:
        if len(self.spatial_transform) != 6:
            raise ValueError(
                f"spatial:transform must have 6 elements, got {len(self.spatial_transform)}"
            )
        return self


class S1RtcConditionsAttrs(BaseModel):
    """Attributes for the conditions group."""

    proj_code: str = Field(alias="proj:code")
    spatial_dimensions: list[str] = Field(alias="spatial:dimensions")
    spatial_transform: list[float] = Field(alias="spatial:transform")

    model_config = {"extra": "allow", "populate_by_name": True, "serialize_by_alias": True}


# ============================================================================
# TypedDict members (same pattern as S2 Sentinel2ResolutionMembers)
# ============================================================================


class S1RtcNativeResolutionMembers(TypedDict, closed=True, total=False):
    """Members for the native resolution dataset (r10m).

    Data variables (time, Y, X), the 1-D coordinates (`time`, `y`, `x`), the per-acquisition
    metadata arrays (time,) and the CF `spatial_ref` grid-mapping scalar.

    Every key is optional *in the type* so the members mapping can be built up incrementally;
    presence of the coordinate arrays is enforced separately by `validate_coordinate_arrays`.
    That split is deliberate — `Required[...]` here would break incremental construction, while
    relying on the closed TypedDict alone could never detect a writer that *stops* emitting a
    coordinate, which is exactly the data-model #192 regression class.
    """

    vv: ArraySpec[Any]
    vh: ArraySpec[Any]
    border_mask: ArraySpec[Any]
    time: ArraySpec[Any]
    x: ArraySpec[Any]
    y: ArraySpec[Any]
    spatial_ref: ArraySpec[Any]
    absolute_orbit: ArraySpec[Any]
    relative_orbit: ArraySpec[Any]
    platform: ArraySpec[Any]


class S1RtcOverviewResolutionMembers(TypedDict, closed=True, total=False):
    """Members for overview resolution datasets (r20m … r720m).

    Data variables plus the same coordinate set as the native level: `x`/`y` (georeferencing),
    `spatial_ref` (CF grid mapping) and `time`. `time` is replicated at every level with
    identical values — TiTiler renders previews off a coarse level, and a level without `time`
    cannot be selected by datetime (data-model #192) — so an overview is *not* a data-only group.

    None of the per-acquisition metadata arrays (`absolute_orbit`, `relative_orbit`, `platform`)
    are written at overview levels; they live on r10m only.
    """

    vv: ArraySpec[Any]
    vh: ArraySpec[Any]
    border_mask: ArraySpec[Any]
    time: ArraySpec[Any]
    x: ArraySpec[Any]
    y: ArraySpec[Any]
    spatial_ref: ArraySpec[Any]


# ============================================================================
# Group models (same pattern as S2 Sentinel2ResolutionDataset etc.)
# ============================================================================

# Coordinate arrays every resolution level must carry, native and overview alike. Listing them
# in the member TypedDicts only stops them being rejected as `extra_forbidden`; because both
# TypedDicts are `total=False` (needed for the incremental `.get()` construction pattern), every
# key is NotRequired, so nothing in the type would notice a writer that STOPPED emitting one.
# That is exactly the regression class this model exists to catch: `time` was missing from the
# overview levels until data-model #192, which made datetime `.sel` fail on the coarse levels
# TiTiler renders previews from. `x`/`y` absent means the level has no georeferencing at all
# (rioxarray falls back to an identity transform), and `spatial_ref` absent means its CRS cannot
# be resolved. Hence the explicit presence check below rather than `Required[...]`.
LEVEL_COORDINATE_ARRAYS = ("time", "x", "y", "spatial_ref")

# The conditions group holds time-invariant rasters, so it carries no `time` — but it does need
# the spatial coordinates. Without them the group opens with "dimensions without coordinates" and
# rioxarray infers an identity transform: the georeferencing is absent, not merely undeclared.
CONDITION_COORDINATE_ARRAYS = ("x", "y", "spatial_ref")


def _validate_coordinates(members: Mapping[str, Any], kind: str, required: tuple[str, ...]) -> None:
    """Raise unless every coordinate in ``required`` is present in ``members``."""
    missing = [name for name in required if name not in members]
    if missing:
        raise ValueError(
            f"{kind} must contain coordinate arrays {missing} "
            f"(expected {list(required)}; see data-model #192)"
        )


class S1RtcNativeResolutionDataset(GroupSpec[S1RtcResolutionAttrs, S1RtcNativeResolutionMembers]):
    """The r10m dataset: data variables + coordinate arrays."""

    @model_validator(mode="after")
    def validate_data_variables(self) -> Self:
        """Ensure vv, vh, and border_mask are present."""
        for name in ("vv", "vh", "border_mask"):
            if name not in self.members:
                raise ValueError(f"Native resolution dataset must contain '{name}' array")
        return self

    @model_validator(mode="after")
    def validate_coordinate_arrays(self) -> Self:
        """Ensure time, x, y and spatial_ref are present (see LEVEL_COORDINATE_ARRAYS)."""
        _validate_coordinates(self.members, "Native resolution dataset", LEVEL_COORDINATE_ARRAYS)
        return self

    @property
    def vv(self) -> ArraySpec[Any]:
        # Present post-validation (see validate_data_variables); the key is NotRequired in the
        # TypedDict to allow incremental construction, so narrow it explicitly here.
        vv = self.members.get("vv")
        if vv is None:
            raise KeyError("vv")
        return vv

    @property
    def vh(self) -> ArraySpec[Any]:
        vh = self.members.get("vh")
        if vh is None:
            raise KeyError("vh")
        return vh

    @property
    def border_mask(self) -> ArraySpec[Any]:
        border_mask = self.members.get("border_mask")
        if border_mask is None:
            raise KeyError("border_mask")
        return border_mask


class S1RtcOverviewResolutionDataset(
    GroupSpec[S1RtcResolutionAttrs, S1RtcOverviewResolutionMembers]
):
    """An overview resolution dataset (r20m-r720m): data variables + coordinate arrays."""

    @model_validator(mode="after")
    def validate_coordinate_arrays(self) -> Self:
        """Ensure time, x, y and spatial_ref are present (see LEVEL_COORDINATE_ARRAYS)."""
        _validate_coordinates(self.members, "Overview resolution dataset", LEVEL_COORDINATE_ARRAYS)
        return self


# Arrays the writer can put in a conditions group. `ingest_s1tiling_conditions` takes three
# optional GeoTIFF paths and only requires that *one* of them be given, so a store whose
# conditions group holds nothing but `lia_*` (or `incidence_angle_*`) is writable today — and
# the CLI's three condition flags all default to None, so it is reachable without any special
# invocation. Requiring `gamma_area_*` specifically made those stores unopenable by this model.
CONDITION_ARRAY_PREFIXES = ("gamma_area_", "lia_", "incidence_angle_")


class S1RtcConditionsGroup(GroupSpec[S1RtcConditionsAttrs, dict[str, ArraySpec[Any]]]):
    """Time-invariant condition arrays, keyed by name (e.g. gamma_area_008)."""

    # ORDER MATTERS: this runs before `validate_has_condition_array`, so a group missing both its
    # coordinates and its rasters reports the coordinates. Swapping the two changes which error an
    # operator sees; `test_s1_rtc_rejects_conditions_without_condition_arrays` keeps its
    # coordinates precisely so it exercises the other check.
    @model_validator(mode="after")
    def validate_coordinate_arrays(self) -> Self:
        """Ensure x, y and spatial_ref are present (see CONDITION_COORDINATE_ARRAYS).

        Same reasoning as the resolution levels: without 1-D coordinates the group opens with
        "dimensions without coordinates" and its rasters are not georeferenced at all. The writer
        creates them with the group, so this closes the last member-level hole rather than
        guarding a path it can currently reach.
        """
        _validate_coordinates(self.members, "Conditions group", CONDITION_COORDINATE_ARRAYS)
        return self

    @model_validator(mode="after")
    def validate_has_condition_array(self) -> Self:
        """At least one condition array must be present (see CONDITION_ARRAY_PREFIXES)."""
        if not any(k.startswith(CONDITION_ARRAY_PREFIXES) for k in self.members):
            raise ValueError(
                "Conditions group must contain at least one condition array "
                f"(one of {CONDITION_ARRAY_PREFIXES})"
            )
        return self


class S1RtcOrbitGroupMembers(TypedDict, closed=True, total=False):
    """Members for an orbit-direction group.

    Contains resolution-level datasets and conditions.
    All optional to support incremental store construction.
    """

    r10m: S1RtcNativeResolutionDataset
    r20m: S1RtcOverviewResolutionDataset
    r60m: S1RtcOverviewResolutionDataset
    r120m: S1RtcOverviewResolutionDataset
    r360m: S1RtcOverviewResolutionDataset
    r720m: S1RtcOverviewResolutionDataset
    conditions: S1RtcConditionsGroup


class S1RtcOrbitGroup(GroupSpec[S1RtcOrbitGroupAttrs, S1RtcOrbitGroupMembers]):
    """One orbit direction (ascending or descending) with multiscale layout."""

    @model_validator(mode="after")
    def validate_r10m_present(self) -> Self:
        if "r10m" not in self.members:
            raise ValueError("Orbit group must contain 'r10m' native resolution dataset")
        return self

    @property
    def r10m(self) -> S1RtcNativeResolutionDataset:
        # Present post-validation (see validate_r10m_present); NotRequired in the TypedDict to
        # allow incremental construction, so narrow it explicitly here.
        r10m = self.members.get("r10m")
        if r10m is None:
            raise KeyError("r10m")
        return r10m

    @property
    def conditions(self) -> S1RtcConditionsGroup | None:
        return self.members.get("conditions")

    def get_resolution(self, level: ResolutionLevel) -> GroupSpec[Any, Any] | None:
        """Retrieve a resolution dataset by level name."""
        return self.members.get(level)


# ============================================================================
# Root model (same pattern as S2 Sentinel2Root)
# ============================================================================


class S1RtcRootMembers(TypedDict, closed=True, total=False):
    """Members for the root group. At least one orbit direction must be present."""

    ascending: S1RtcOrbitGroup
    descending: S1RtcOrbitGroup


class S1RtcRoot(GroupSpec[DatasetAttrs, S1RtcRootMembers]):
    """Complete S1 GRD RTC GeoZarr V3 hierarchy.

    The hierarchy follows the implementation plan::

        s1-grd-rtc-{tile}.zarr/
        ├── zarr.json
        ├── ascending/
        │   ├── zarr.json          # zarr_conventions, multiscales, proj:, spatial:
        │   ├── r10m/
        │   │   ├── vv/            # (time, Y, X) float32
        │   │   ├── vh/            # (time, Y, X) float32
        │   │   ├── border_mask/   # (time, Y, X) uint8
        │   │   ├── time/          # (time,) int64
        │   │   ├── x/, y/, spatial_ref/
        │   │   ├── absolute_orbit/
        │   │   ├── relative_orbit/
        │   │   └── platform/
        │   ├── r20m/ … r720m/     # + time, x, y, spatial_ref at every level
        │   └── conditions/
        │       └── gamma_area_{orbit}/
        └── descending/
            └── (same)
    """

    @model_validator(mode="after")
    def validate_at_least_one_orbit(self) -> Self:
        if "ascending" not in self.members and "descending" not in self.members:
            raise ValueError("Store must contain at least one orbit group (ascending/descending)")
        return self

    @property
    def ascending(self) -> S1RtcOrbitGroup | None:
        return self.members.get("ascending")

    @property
    def descending(self) -> S1RtcOrbitGroup | None:
        return self.members.get("descending")
