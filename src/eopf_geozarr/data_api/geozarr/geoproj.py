"""
Models for the GeoProj Zarr Convention
"""

from __future__ import annotations

from typing import Literal, Self, TypeGuard

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

from eopf_geozarr.data_api.geozarr.projjson import ProjJSON


def is_none(data: object) -> TypeGuard[None]:
    return data is None


GEO_PROJ_UUID: Literal["f17cb550-5864-4468-aeb7-f3180cfb622f"] = (
    "f17cb550-5864-4468-aeb7-f3180cfb622f"
)


class GeoProjConvention(TypedDict):
    version: Literal["0.1.0"]
    schema: Literal[
        "https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v0.1.0/schema.json"
    ]
    name: Literal["geo-proj"]
    description: Literal["Coordinate reference system information for geospatial data"]
    spec: Literal["https://github.com/zarr-experimental/geo-proj/blob/v0.1.0/README.md"]


GeoProjConventions = TypedDict(  # type: ignore[misc]
    "GeoProjConventions", {GEO_PROJ_UUID: GeoProjConvention}, closed=False
)


class GeoProj(BaseModel):
    code: str | None = Field(None, alias="proj:code", exclude_if=is_none)
    wkt2: str | None = Field(None, alias="proj:wkt2", exclude_if=is_none)
    projjson: ProjJSON | None = Field(None, alias="proj:projjson", exclude_if=is_none)
    spatial_dimensions: tuple[str, str] = Field(alias="proj:spatial_dimensions")
    transform: tuple[float, float, float, float, float, float] | None = Field(
        None, alias="proj:transform", exclude_if=is_none
    )
    bbox: tuple[float, float, float, float] | None = Field(
        None, alias="proj:bbox", exclude_if=is_none
    )
    shape: tuple[int, int] | None = Field(None, alias="proj:shape", exclude_if=is_none)

    model_config = {"extra": "allow", "serialize_by_alias": True}

    @model_validator(mode="after")
    def ensure_required_conditional_attributes(self) -> Self:
        if self.code is None and self.wkt2 is None and self.projjson is None:
            raise ValueError("One of 'code', 'wkt2', or 'projjson' must be provided.")
        return self
