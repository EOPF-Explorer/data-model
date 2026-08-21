"""GeoZarr store model.

Enforces the store-level GeoZarr mini-spec profile: the store root carries a
mandatory spatial footprint (`spatial:bbox` + a CRS via one of `proj:code`,
`proj:wkt2`, `proj:projjson`), and nested multiscale groups carry mandatory
`spatial:bbox` at the root plus `spatial:transform` + `spatial:shape` on every
layout entry.

Tightens the zarr convention-level models defined in `geozarr.multiscales`,
`geozarr.spatial` and `geozarr.geoproj` without replacing them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.experimental.missing_sentinel import MISSING
from pydantic_zarr.v3 import ArraySpec, GroupSpec
from zarr_cm import ConventionMetadataObject
from zarr_cm import geo_proj as geo_proj_cm
from zarr_cm import multiscales as multiscales_cm
from zarr_cm import spatial as spatial_cm

from eopf_geozarr.data_api.geozarr.common import is_none
from eopf_geozarr.data_api.geozarr.multiscales import MultiscaleMeta
from eopf_geozarr.data_api.geozarr.multiscales.geozarr import MultiscaleGroupAttrs
from eopf_geozarr.data_api.geozarr.multiscales.zcm import ScaleLevel
from eopf_geozarr.data_api.geozarr.projjson import ProjJSON


def declared_convention_uuids(
    zarr_conventions: object,
) -> set[str]:
    """Return the set of convention UUIDs declared in a ``zarr_conventions`` array.

    Tolerates malformed input (wrong container type, non-mapping entries) by
    ignoring it — the value comes from untrusted store metadata and shape
    problems are reported separately by the validator.
    """
    if not isinstance(zarr_conventions, (list, tuple)):
        return set()
    return {str(c["uuid"]) for c in zarr_conventions if isinstance(c, Mapping) and "uuid" in c}


def _require_conventions(
    zarr_conventions: object,
    required: dict[str, str],
) -> None:
    """Raise if any of ``required`` (uuid -> convention name) is not declared."""
    declared = declared_convention_uuids(zarr_conventions)
    missing = [name for uuid, name in required.items() if uuid not in declared]
    if missing:
        raise ValueError(
            f"zarr_conventions must declare the {', '.join(sorted(missing))} "
            "convention(s) used by this node"
        )


class GeoZarrStoreAttrs(BaseModel):
    """Attributes required at the store root (outermost Zarr group).

    Both `spatial:bbox` and a CRS are mandatory. The CRS is encoded by at
    least one of `proj:code`, `proj:wkt2`, or `proj:projjson`; there is no
    implicit default. Use `"EPSG:4326"` when no other CRS is meaningful.
    """

    zarr_conventions: tuple[ConventionMetadataObject, ...]
    bbox: list[float] = Field(alias="spatial:bbox", min_length=4, max_length=4)
    code: str | None = Field(None, alias="proj:code", exclude_if=is_none, pattern="^[A-Z]+:[0-9]+$")
    wkt2: str | None = Field(None, alias="proj:wkt2", exclude_if=is_none)
    projjson: ProjJSON | None = Field(None, alias="proj:projjson", exclude_if=is_none)

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    @model_validator(mode="after")
    def validate_bbox_order(self) -> Self:
        xmin, ymin, xmax, ymax = self.bbox
        if xmin > xmax:
            raise ValueError(
                f"spatial:bbox: xmin ({xmin}) must be <= xmax ({xmax}); "
                "expected [xmin, ymin, xmax, ymax]"
            )
        if ymin > ymax:
            raise ValueError(
                f"spatial:bbox: ymin ({ymin}) must be <= ymax ({ymax}); "
                "expected [xmin, ymin, xmax, ymax]"
            )
        return self

    @model_validator(mode="after")
    def validate_crs(self) -> Self:
        if not any(v is not None for v in (self.code, self.wkt2, self.projjson)):
            raise ValueError(
                "Store root requires a CRS: set at least one of proj:code, proj:wkt2, or proj:projjson"
            )
        return self

    @model_validator(mode="after")
    def validate_conventions_declared(self) -> Self:
        _require_conventions(
            self.zarr_conventions,
            {spatial_cm.UUID: "spatial", geo_proj_cm.UUID: "geo-proj"},
        )
        return self


class GeoZarrScaleLevel(ScaleLevel):
    """Multiscale layout entry with mandatory `spatial:transform` + `spatial:shape`."""

    spatial_shape: list[int] = Field(alias="spatial:shape", min_length=2, max_length=2)
    spatial_transform: list[float] = Field(alias="spatial:transform", min_length=6, max_length=6)

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    @model_validator(mode="after")
    def validate_transform_with_derived_from(self) -> Self:
        if self.derived_from is not MISSING and self.transform is MISSING:
            raise ValueError(
                f"layout entry {self.asset!r}: 'transform' is required when "
                "'derived_from' is present"
            )
        return self


class GeoZarrMultiscaleMeta(MultiscaleMeta):
    """Multiscale metadata where every layout entry is a `GeoZarrScaleLevel`."""

    @model_validator(mode="after")
    def validate_layout_not_empty(self) -> Self:
        if len(self.layout) < 1:
            raise ValueError("multiscales.layout must not be empty")
        return self

    # Intentionally tightens the base ``layout`` field: ``GeoZarrScaleLevel`` is a
    # subclass of ``ScaleLevel`` and the optional ``MISSING`` default is dropped to make
    # the field mandatory in this store-level profile. pyright flags the narrowed,
    # now-required override on a mutable (invariant) field.
    layout: tuple[GeoZarrScaleLevel, ...]  # pyright: ignore[reportGeneralTypeIssues, reportIncompatibleVariableOverride]


class GeoZarrMultiscaleGroupAttrs(MultiscaleGroupAttrs):
    """Multiscale group attributes with a mandatory `spatial:bbox`."""

    # Intentionally tightens the base ``multiscales`` field to the ``GeoZarrMultiscaleMeta``
    # subclass; pyright flags the narrowed override on a mutable (invariant) field.
    multiscales: GeoZarrMultiscaleMeta  # pyright: ignore[reportIncompatibleVariableOverride]
    # The base class allows zarr_conventions to be MISSING; the minispec requires the
    # multiscale group to declare all three conventions, so make the field mandatory.
    zarr_conventions: tuple[ConventionMetadataObject, ...]  # pyright: ignore[reportGeneralTypeIssues, reportIncompatibleVariableOverride]
    spatial_bbox: list[float] = Field(alias="spatial:bbox", min_length=4, max_length=4)
    spatial_dimensions: list[str] = Field(alias="spatial:dimensions", min_length=1)
    code: str | None = Field(None, alias="proj:code", exclude_if=is_none, pattern="^[A-Z]+:[0-9]+$")
    wkt2: str | None = Field(None, alias="proj:wkt2", exclude_if=is_none)
    projjson: ProjJSON | None = Field(None, alias="proj:projjson", exclude_if=is_none)

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        serialize_by_alias=True,
    )

    @model_validator(mode="after")
    def validate_bbox_order(self) -> Self:
        xmin, ymin, xmax, ymax = self.spatial_bbox
        if xmin > xmax or ymin > ymax:
            raise ValueError(
                "spatial:bbox must be ordered as [xmin, ymin, xmax, ymax] with xmin<=xmax and ymin<=ymax"
            )
        return self

    @model_validator(mode="after")
    def validate_crs_present(self) -> Self:
        if not any([self.code, self.wkt2, self.projjson]):
            raise ValueError(
                "Multiscale dataset requires a CRS: set one of proj:code, proj:wkt2, or proj:projjson"
            )
        return self

    @model_validator(mode="after")
    def validate_conventions_declared(self) -> Self:
        _require_conventions(
            self.zarr_conventions,
            {
                multiscales_cm.UUID: "multiscales",
                spatial_cm.UUID: "spatial",
                geo_proj_cm.UUID: "geo-proj",
            },
        )
        return self


GeoZarrMember = GroupSpec[Any, Any] | ArraySpec


class GeoZarr(GroupSpec[GeoZarrStoreAttrs, GeoZarrMember]):
    """A GeoZarr store.

    Pairs the required store-root attributes with arbitrary Zarr children.
    Intended as a reusable building block: downstream models can constrain the
    `members` type further, e.g. require a nested `GeoZarrMultiscaleGroupAttrs`
    group.
    """
