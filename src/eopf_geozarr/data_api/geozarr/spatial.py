"""
Models for the Spatial Zarr Convention
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from eopf_geozarr.data_api.geozarr.common import ZarrConventionMetadata, is_none

SPATIAL_UUID: Literal["689b58e2-cf7b-45e0-9fff-9cfc0883d6b4"] = (
    "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4"
)


class SpatialConvention(TypedDict):
    uuid: Literal["689b58e2-cf7b-45e0-9fff-9cfc0883d6b4"]
    name: Literal["spatial:"]
    schema_url: Literal["https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json"]
    spec_url: Literal["https://github.com/zarr-conventions/spatial/blob/v1/README.md"]
    description: Literal["Spatial coordinate and transformation information"]


class SpatialConventionMetadata(ZarrConventionMetadata):
    uuid: Literal["689b58e2-cf7b-45e0-9fff-9cfc0883d6b4"] = SPATIAL_UUID
    name: Literal["spatial:"] = "spatial:"
    schema_url: Literal["https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json"] = (
        "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json"
    )
    spec_url: Literal["https://github.com/zarr-conventions/spatial/blob/v1/README.md"] = (
        "https://github.com/zarr-conventions/spatial/blob/v1/README.md"
    )
    description: Literal["Spatial coordinate and transformation information"] = (
        "Spatial coordinate and transformation information"
    )


class Spatial(BaseModel):
    dimensions: list[str] = Field(alias="spatial:dimensions")  # Required field
    bbox: list[float] | None = Field(None, alias="spatial:bbox", exclude_if=is_none)
    transform_type: str = Field("affine", alias="spatial:transform_type")
    transform: list[float] | None = Field(None, alias="spatial:transform", exclude_if=is_none)
    shape: list[int] | None = Field(None, alias="spatial:shape", exclude_if=is_none)
    registration: str = Field("pixel", alias="spatial:registration")

    model_config = {"extra": "allow", "serialize_by_alias": True}