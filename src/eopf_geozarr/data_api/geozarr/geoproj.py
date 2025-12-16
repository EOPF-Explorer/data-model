"""
Models for the Proj Zarr Convention (v1.0)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

from eopf_geozarr.data_api.geozarr.common import ZarrConventionMetadata, is_none
from eopf_geozarr.data_api.geozarr.projjson import ProjJSON  # noqa: TC001

PROJ_UUID: Literal["f17cb550-5864-4468-aeb7-f3180cfb622f"] = "f17cb550-5864-4468-aeb7-f3180cfb622f"


class ProjConvention(TypedDict):
    uuid: Literal["f17cb550-5864-4468-aeb7-f3180cfb622f"]
    name: Literal["proj:"]
    schema_url: Literal[
        "https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v1/schema.json"
    ]
    spec_url: Literal["https://github.com/zarr-experimental/geo-proj/blob/v1/README.md"]
    description: Literal["Coordinate reference system information for geospatial data"]


class ProjConventionMetadata(ZarrConventionMetadata):
    uuid: Literal["f17cb550-5864-4468-aeb7-f3180cfb622f"] = PROJ_UUID
    name: Literal["proj:"] = "proj:"
    schema_url: Literal[
        "https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v1/schema.json"
    ] = "https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v1/schema.json"
    spec_url: Literal["https://github.com/zarr-experimental/geo-proj/blob/v1/README.md"] = (
        "https://github.com/zarr-experimental/geo-proj/blob/v1/README.md"
    )
    description: Literal["Coordinate reference system information for geospatial data"] = (
        "Coordinate reference system information for geospatial data"
    )


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
