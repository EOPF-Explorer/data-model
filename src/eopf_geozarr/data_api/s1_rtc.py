"""
Pydantic-zarr integrated models for Sentinel-1 GRD γ0T RTC GeoZarr stores.

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

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict
from zarr_cm import geo_proj, multiscales as multiscales_cm, spatial as spatial_cm

from eopf_geozarr.data_api.geozarr.common import DatasetAttrs
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


class S1RtcOrbitGroupAttrs(BaseModel, extra="allow"):
    """Attributes for an orbit-direction group (ascending or descending).

    Carries the three GeoZarr conventions plus proj:/spatial:/multiscales metadata.
    """

    zarr_conventions: list[dict[str, Any]]
    multiscales: dict[str, Any]  # validated structurally below
    proj_code: str = Field(alias="proj:code")
    spatial_dimensions: list[str] = Field(alias="spatial:dimensions")
    spatial_bbox: list[float] = Field(alias="spatial:bbox")

    model_config = {"populate_by_name": True, "serialize_by_alias": True}

    @model_validator(mode="after")
    def validate_zarr_conventions(self) -> Self:
        """Ensure all three required convention UUIDs are present."""
        present = {c["uuid"] for c in self.zarr_conventions if "uuid" in c}
        missing = REQUIRED_CONVENTION_UUIDS - present
        if missing:
            raise ValueError(f"Missing required zarr_conventions UUIDs: {missing}")
        return self

    @model_validator(mode="after")
    def validate_multiscales_layout(self) -> Self:
        """Ensure multiscales has a layout array with at least one entry."""
        layout = self.multiscales.get("layout")
        if not layout or not isinstance(layout, (list, tuple)):
            raise ValueError("multiscales must contain a non-empty 'layout' array")
        for entry in layout:
            if "asset" not in entry:
                raise ValueError("Each multiscales layout entry must have an 'asset' key")
        return self

    @model_validator(mode="after")
    def validate_spatial_dimensions(self) -> Self:
        if self.spatial_dimensions != ["Y", "X"]:
            raise ValueError(
                f"spatial:dimensions must be ['Y', 'X'], got {self.spatial_dimensions}"
            )
        return self

    @model_validator(mode="after")
    def validate_spatial_bbox(self) -> Self:
        if len(self.spatial_bbox) != 4:
            raise ValueError(f"spatial:bbox must have 4 elements, got {len(self.spatial_bbox)}")
        return self


class S1RtcResolutionAttrs(BaseModel, extra="allow"):
    """Attributes for a resolution-level group (r10m, r20m, …)."""

    spatial_shape: list[int] = Field(alias="spatial:shape")
    spatial_transform: list[float] = Field(alias="spatial:transform")

    model_config = {"populate_by_name": True, "serialize_by_alias": True}

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


class S1RtcConditionsAttrs(BaseModel, extra="allow"):
    """Attributes for the conditions group."""

    proj_code: str = Field(alias="proj:code")
    spatial_dimensions: list[str] = Field(alias="spatial:dimensions")
    spatial_transform: list[float] = Field(alias="spatial:transform")

    model_config = {"populate_by_name": True, "serialize_by_alias": True}


# ============================================================================
# TypedDict members (same pattern as S2 Sentinel2ResolutionMembers)
# ============================================================================


class S1RtcNativeResolutionMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for the native resolution dataset (r10m).

    Data variables (time, Y, X) plus 1-D coordinate variables (time,).
    All fields optional since not all arrays are present during incremental construction.
    """

    vv: ArraySpec[Any]
    vh: ArraySpec[Any]
    border_mask: ArraySpec[Any]
    time: ArraySpec[Any]
    absolute_orbit: ArraySpec[Any]
    relative_orbit: ArraySpec[Any]
    platform: ArraySpec[Any]


class S1RtcOverviewResolutionMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for overview resolution datasets (r20m … r720m).

    Only data variables, no coordinate arrays.
    """

    vv: ArraySpec[Any]
    vh: ArraySpec[Any]
    border_mask: ArraySpec[Any]


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
    """An overview resolution dataset (r20m–r720m): data variables only."""


class S1RtcConditionsGroup(
    GroupSpec[S1RtcConditionsAttrs, dict[str, ArraySpec[Any]]]  # type: ignore[type-var]
):
    """Time-invariant condition arrays, keyed by name (e.g. gamma_area_008)."""

    @model_validator(mode="after")
    def validate_has_gamma_area(self) -> Self:
        """At least one gamma_area_* array should be present."""
        if not any(k.startswith("gamma_area_") for k in self.members):
            raise ValueError(
                "Conditions group must contain at least one 'gamma_area_*' array"
            )
        return self


class S1RtcOrbitGroupMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
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


class S1RtcOrbitGroup(
    GroupSpec[S1RtcOrbitGroupAttrs, S1RtcOrbitGroupMembers]  # type: ignore[type-var]
):
    """One orbit direction (ascending or descending) with multiscale layout."""

    @model_validator(mode="after")
    def validate_r10m_present(self) -> Self:
        if "r10m" not in self.members:
            raise ValueError("Orbit group must contain 'r10m' native resolution dataset")
        return self

    @property
    def r10m(self) -> S1RtcNativeResolutionDataset:
        return self.members["r10m"]

    @property
    def conditions(self) -> S1RtcConditionsGroup | None:
        return self.members.get("conditions")

    def get_resolution(self, level: ResolutionLevel) -> GroupSpec[Any, Any] | None:
        """Retrieve a resolution dataset by level name."""
        return self.members.get(level)


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
