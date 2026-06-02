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

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.experimental.missing_sentinel import MISSING  # noqa: F401  (re-export for mypy)
from pydantic_zarr.v3 import ArraySpec, GroupSpec

from eopf_geozarr.data_api.geozarr.common import is_none
from eopf_geozarr.data_api.geozarr.multiscales import MultiscaleMeta
from eopf_geozarr.data_api.geozarr.multiscales.geozarr import MultiscaleGroupAttrs
from eopf_geozarr.data_api.geozarr.multiscales.zcm import ScaleLevel
from eopf_geozarr.data_api.geozarr.projjson import (
    ProjJSON,  # noqa: TC001  (runtime use by pydantic)
)


class GeoZarrStoreAttrs(BaseModel):
    """Attributes required at the store root (outermost Zarr group).

    Both `spatial:bbox` and a CRS are mandatory. The CRS is encoded by exactly
    one of `proj:code`, `proj:wkt2`, or `proj:projjson`; there is no implicit
    default. Use `"EPSG:4326"` when no other CRS is meaningful.
    """

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
        crs_fields_set = sum(1 for v in (self.code, self.wkt2, self.projjson) if v is not None)
        if crs_fields_set == 0:
            raise ValueError(
                "Store root requires a CRS: set exactly one of proj:code, proj:wkt2, or proj:projjson"
            )
        if crs_fields_set > 1:
            raise ValueError(
                "At most one of proj:code, proj:wkt2, proj:projjson may be set at the store root"
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


class GeoZarrMultiscaleMeta(MultiscaleMeta):
    """Multiscale metadata where every layout entry is a `GeoZarrScaleLevel`."""

    layout: tuple[GeoZarrScaleLevel, ...]


class GeoZarrMultiscaleGroupAttrs(MultiscaleGroupAttrs):
    """Multiscale group attributes with a mandatory `spatial:bbox`."""

    multiscales: GeoZarrMultiscaleMeta
    spatial_bbox: list[float] = Field(alias="spatial:bbox", min_length=4, max_length=4)

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


GeoZarrMember = GroupSpec[Any, Any] | ArraySpec


class GeoZarr(GroupSpec[GeoZarrStoreAttrs, GeoZarrMember]):
    """A GeoZarr store.

    Pairs the required store-root attributes with arbitrary Zarr children.
    Intended as a reusable building block: downstream models can constrain the
    `members` type further, e.g. require a nested `GeoZarrMultiscaleGroupAttrs`
    group.
    """
