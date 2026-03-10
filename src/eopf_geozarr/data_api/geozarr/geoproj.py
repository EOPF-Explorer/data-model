"""
Models for the Proj Zarr Convention (v1.0)
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from zarr_cm import geo_proj

from eopf_geozarr.data_api.geozarr.common import ZarrConventionMetadata, is_none
from eopf_geozarr.data_api.geozarr.projjson import ProjJSON  # noqa: TC001

PROJ_UUID = geo_proj.UUID

# Re-export the zarr-cm TypedDict for the convention metadata object
ProjConvention = geo_proj.GeoProjAttrs


class ProjConventionMetadata(ZarrConventionMetadata):
    uuid: str = geo_proj.CMO["uuid"]
    name: str = geo_proj.CMO["name"]
    schema_url: str = geo_proj.CMO["schema_url"]
    spec_url: str = geo_proj.CMO["spec_url"]
    description: str = geo_proj.CMO["description"]


class Proj(BaseModel):
    # At least one of code, wkt2, or projjson must be provided
    code: str | None = Field(None, alias="proj:code", exclude_if=is_none)
    wkt2: str | None = Field(None, alias="proj:wkt2", exclude_if=is_none)
    projjson: ProjJSON | None = Field(None, alias="proj:projjson", exclude_if=is_none)

    model_config = {"extra": "allow", "serialize_by_alias": True}

    @model_validator(mode="after")
    def validate_at_least_one_crs(self) -> Proj:
        """Validate that at least one CRS field is provided"""
        if not any([self.code, self.wkt2, self.projjson]):
            raise ValueError(
                "At least one of proj:code, proj:wkt2, or proj:projjson must be provided"
            )
        return self


# Backwards compatibility alias
GeoProj = Proj
