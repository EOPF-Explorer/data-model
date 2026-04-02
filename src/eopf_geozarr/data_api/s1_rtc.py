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
    │   │   ├── absolute_orbit/
    │   │   ├── relative_orbit/
    │   │   └── platform/
    │   ├── r20m/ … r720m/     # overview levels (vv, vh, border_mask only)
    │   └── conditions/
    │       └── gamma_area_{orbit}/  # (Y, X) float32
    └── descending/
        └── (same structure)
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, Self

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict
from zarr_cm import geo_proj
from zarr_cm import multiscales as multiscales_cm
from zarr_cm import spatial as spatial_cm

from eopf_geozarr.data_api.geozarr.common import DatasetAttrs
from eopf_geozarr.data_api.geozarr.multiscales.zcm import Multiscales
from eopf_geozarr.pyz.v3 import ArraySpec, GroupSpec

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
    spatial_dimensions: tuple[Literal["y"], Literal["x"]] = Field(alias="spatial:dimensions")
    spatial_bbox: tuple[float, float, float, float] = Field(alias="spatial:bbox")

    model_config = {"extra": "allow", "populate_by_name": True, "serialize_by_alias": True}

    @model_validator(mode="after")
    def validate_zarr_conventions(self) -> Self:
        """Ensure all three required convention UUIDs are present."""
        present = {c["uuid"] for c in self.zarr_conventions if "uuid" in c}
        missing = REQUIRED_CONVENTION_UUIDS - present
        if missing:
            raise ValueError(f"Missing required zarr_conventions UUIDs: {missing}")
        return self


class S1RtcResolutionAttrs(BaseModel):
    """Attributes for a resolution-level group (r10m, r20m, ...)."""

    spatial_shape: tuple[int, int] = Field(alias="spatial:shape")
    spatial_transform: tuple[float, float, float, float, float, float] = Field(
        alias="spatial:transform"
    )

    model_config = {"extra": "allow", "populate_by_name": True, "serialize_by_alias": True}


class S1RtcConditionsAttrs(BaseModel):
    """Attributes for the conditions group."""

    proj_code: str = Field(alias="proj:code")
    spatial_dimensions: tuple[Literal["y"], Literal["x"]] = Field(alias="spatial:dimensions")
    spatial_transform: tuple[float, float, float, float, float, float] = Field(
        alias="spatial:transform"
    )

    model_config = {"extra": "allow", "populate_by_name": True, "serialize_by_alias": True}


# ============================================================================
# TypedDict members (same pattern as S2 Sentinel2ResolutionMembers)
# ============================================================================


class S1RtcNativeResolutionMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for the native resolution dataset (r10m).

    Data variables (time, Y, X) plus 1-D spatial and temporal coordinate
    variables.  All fields optional since not all arrays are present during
    incremental construction.
    """

    vv: ArraySpec[Any]
    vh: ArraySpec[Any]
    border_mask: ArraySpec[Any]
    x: ArraySpec[Any]
    y: ArraySpec[Any]
    time: ArraySpec[Any]
    absolute_orbit: ArraySpec[Any]
    relative_orbit: ArraySpec[Any]
    platform: ArraySpec[Any]


class S1RtcOverviewResolutionMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for overview resolution datasets (r20m … r720m).

    Data variables plus 1-D spatial coordinate arrays (x, y).
    """

    vv: ArraySpec[Any]
    vh: ArraySpec[Any]
    border_mask: ArraySpec[Any]
    x: ArraySpec[Any]
    y: ArraySpec[Any]


# ============================================================================
# Group models (same pattern as S2 Sentinel2ResolutionDataset etc.)
# ============================================================================


class S1RtcNativeResolutionDataset(
    GroupSpec[S1RtcResolutionAttrs, S1RtcNativeResolutionMembers]  # type: ignore[type-var]
):
    """The r10m dataset: data variables + coordinate arrays."""

    @model_validator(mode="after")
    def validate_data_variables(self) -> Self:
        """Ensure vv, vh, and border_mask are present."""
        for name in ("vv", "vh", "border_mask"):
            if name not in self.members:
                raise ValueError(f"Native resolution dataset must contain '{name}' array")
        return self

    @property
    def vv(self) -> ArraySpec[Any]:
        return self.members["vv"]

    @property
    def vh(self) -> ArraySpec[Any]:
        return self.members["vh"]

    @property
    def border_mask(self) -> ArraySpec[Any]:
        return self.members["border_mask"]


class S1RtcOverviewResolutionDataset(
    GroupSpec[S1RtcResolutionAttrs, S1RtcOverviewResolutionMembers]  # type: ignore[type-var]
):
    """An overview resolution dataset (r20m-r720m): data variables only."""


class S1RtcConditionsGroup(GroupSpec[S1RtcConditionsAttrs, dict[str, ArraySpec[Any]]]):
    """Time-invariant condition arrays, keyed by name (e.g. gamma_area_008)."""

    @model_validator(mode="after")
    def validate_has_gamma_area(self) -> Self:
        """At least one gamma_area_* array should be present."""
        if not any(k.startswith("gamma_area_") for k in self.members):
            raise ValueError("Conditions group must contain at least one 'gamma_area_*' array")
        return self


class S1RtcOrbitGroupMembers(TypedDict, closed=True):  # type: ignore[call-arg]
    """Members for an orbit-direction group.

    r10m is always required; overview levels and conditions are optional.
    """

    r10m: S1RtcNativeResolutionDataset
    r20m: NotRequired[S1RtcOverviewResolutionDataset]
    r60m: NotRequired[S1RtcOverviewResolutionDataset]
    r120m: NotRequired[S1RtcOverviewResolutionDataset]
    r360m: NotRequired[S1RtcOverviewResolutionDataset]
    r720m: NotRequired[S1RtcOverviewResolutionDataset]
    conditions: NotRequired[S1RtcConditionsGroup]


class S1RtcOrbitGroup(
    GroupSpec[S1RtcOrbitGroupAttrs, S1RtcOrbitGroupMembers]  # type: ignore[type-var]
):
    """One orbit direction (ascending or descending) with multiscale layout."""

    @property
    def r10m(self) -> S1RtcNativeResolutionDataset:
        return self.members["r10m"]

    @property
    def conditions(self) -> S1RtcConditionsGroup | None:
        return self.members.get("conditions")

    def get_resolution(self, level: ResolutionLevel) -> GroupSpec[Any, Any] | None:
        """Retrieve a resolution dataset by level name."""
        return self.members.get(level)

    def resolution_levels(self) -> list[ResolutionLevel]:
        """List available resolution levels in this orbit group."""
        all_levels: tuple[ResolutionLevel, ...] = (
            "r10m",
            "r20m",
            "r60m",
            "r120m",
            "r360m",
            "r720m",
        )
        return [lvl for lvl in all_levels if lvl in self.members]


# ============================================================================
# Root model (same pattern as S2 Sentinel2Root)
# ============================================================================


class S1RtcRootMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for the root group. At least one orbit direction must be present."""

    ascending: S1RtcOrbitGroup
    descending: S1RtcOrbitGroup


class S1RtcRoot(GroupSpec[DatasetAttrs, S1RtcRootMembers]):  # type: ignore[type-var]
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
        │   │   ├── absolute_orbit/
        │   │   ├── relative_orbit/
        │   │   └── platform/
        │   ├── r20m/ … r720m/
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
