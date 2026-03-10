"""
Models for the Spatial Zarr Convention
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from zarr_cm import spatial as spatial_cm

from eopf_geozarr.data_api.geozarr.common import ZarrConventionMetadata, is_none

SPATIAL_UUID = spatial_cm.UUID

# Re-export the zarr-cm TypedDict for the convention metadata object
SpatialConvention = spatial_cm.SpatialAttrs


class SpatialConventionMetadata(ZarrConventionMetadata):
    uuid: str = spatial_cm.CMO["uuid"]
    name: str = spatial_cm.CMO["name"]
    schema_url: str = spatial_cm.CMO["schema_url"]
    spec_url: str = spatial_cm.CMO["spec_url"]
    description: str = spatial_cm.CMO["description"]


class Spatial(BaseModel):
    dimensions: list[str] = Field(alias="spatial:dimensions")  # Required field
    bbox: list[float] | None = Field(None, alias="spatial:bbox", exclude_if=is_none)
    transform_type: str = Field("affine", alias="spatial:transform_type")
    transform: list[float] | None = Field(None, alias="spatial:transform", exclude_if=is_none)
    shape: list[int] | None = Field(None, alias="spatial:shape", exclude_if=is_none)
    registration: str = Field("pixel", alias="spatial:registration")

    model_config = {"extra": "allow", "serialize_by_alias": True}

    @model_validator(mode="after")
    def validate_dimensions_not_empty(self) -> Spatial:
        """Validate that dimensions list is not empty."""
        if not self.dimensions:
            raise ValueError("spatial:dimensions must contain at least one dimension")
        return self
