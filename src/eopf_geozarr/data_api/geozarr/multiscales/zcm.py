from __future__ import annotations

from typing import Final

from pydantic import BaseModel, field_validator
from pydantic.experimental.missing_sentinel import MISSING
from zarr_cm import ConventionMetadataObject
from zarr_cm import multiscales as multiscales_cm

from eopf_geozarr.data_api.geozarr.common import ZarrConventionMetadata

# Convention constants from zarr-cm
CONVENTION_ID = multiscales_cm.UUID
CONVENTION_SCHEMA_URL = multiscales_cm.SCHEMA_URL
CONVENTION_SPEC_URL = multiscales_cm.SPEC_URL
CONVENTION_NAME = multiscales_cm.CMO["name"]
CONVENTION_DESCRIPTION = multiscales_cm.CMO["description"]

# Re-export zarr-cm TypedDicts
TransformJSON = multiscales_cm.Transform
ScaleLevelJSON = multiscales_cm.LayoutObject
MultiscalesJSON = multiscales_cm.MultiscalesAttrs
MultiscalesConventionAttrsJSON = multiscales_cm.MultiscalesConventionAttrs

# A final dict representation of the Multiscales convention metadata
MULTISCALE_CONVENTION_METADATA: Final[ConventionMetadataObject] = multiscales_cm.CMO


class MultiscaleConventionMetadata(ZarrConventionMetadata):
    uuid: str = multiscales_cm.CMO["uuid"]
    schema_url: str = multiscales_cm.CMO["schema_url"]
    name: str = multiscales_cm.CMO["name"]
    description: str = multiscales_cm.CMO["description"]
    spec_url: str = multiscales_cm.CMO["spec_url"]


class ZarrConventionAttrs(BaseModel):
    zarr_conventions: tuple[ZarrConventionMetadata, ...]

    model_config = {"extra": "allow"}


class Transform(BaseModel):
    scale: tuple[float, ...] | MISSING = MISSING
    translation: tuple[float, ...] | MISSING = MISSING


class ScaleLevel(BaseModel):
    asset: str
    derived_from: str | MISSING = MISSING
    transform: Transform | MISSING = MISSING
    resampling_method: str | MISSING = MISSING

    model_config = {"extra": "allow"}


class Multiscales(BaseModel):
    layout: tuple[ScaleLevel, ...]
    resampling_method: str | MISSING = MISSING

    model_config = {"extra": "allow"}


class MultiscalesAttrs(ZarrConventionAttrs):
    multiscales: Multiscales
    model_config = {"extra": "allow"}

    @field_validator("zarr_conventions", mode="after")
    @classmethod
    def ensure_multiscales_convention(
        cls, value: tuple[ZarrConventionMetadata, ...]
    ) -> tuple[ZarrConventionMetadata, ...]:
        """
        Iterate over the elements of zarr_conventions and check that at least one of them is
        multiscales
        """
        success: bool = False
        errors: dict[int, ValueError] = {}
        for idx, convention_meta in enumerate(value):
            try:
                MultiscaleConventionMetadata(**convention_meta.model_dump())
                success = True
            except ValueError as e:
                errors[idx] = e
        if not success:
            raise ValueError("Multiscales convention not found. Errors: " + str(errors))
        return value
