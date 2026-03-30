from __future__ import annotations

from pydantic import BaseModel, field_validator, model_serializer
from pydantic.experimental.missing_sentinel import MISSING
from zarr_cm import ConventionMetadataObject
from zarr_cm import multiscales as multiscales_cm

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


class ZarrConventionAttrs(BaseModel):
    zarr_conventions: tuple[ConventionMetadataObject, ...]

    model_config = {"extra": "allow"}


class Transform(BaseModel):
    scale: tuple[float, ...] | MISSING = MISSING  # type: ignore[valid-type]
    translation: tuple[float, ...] | MISSING = MISSING  # type: ignore[valid-type]

    @model_serializer
    def serialize_model(self) -> dict[str, tuple[float, ...]]:
        result: dict[str, tuple[float, ...]] = {}
        if self.scale is not MISSING:  # type: ignore[comparison-overlap]
            result["scale"] = self.scale
        if self.translation is not MISSING:  # type: ignore[comparison-overlap]
            result["translation"] = self.translation
        return result


class ScaleLevel(BaseModel):
    asset: str
    derived_from: str | MISSING = MISSING  # type: ignore[valid-type]
    transform: Transform | MISSING = MISSING  # type: ignore[valid-type]
    resampling_method: str | MISSING = MISSING  # type: ignore[valid-type]

    model_config = {"extra": "allow"}


class Multiscales(BaseModel):
    layout: tuple[ScaleLevel, ...]
    resampling_method: str | MISSING = MISSING  # type: ignore[valid-type]

    model_config = {"extra": "allow"}


class MultiscalesAttrs(ZarrConventionAttrs):
    multiscales: Multiscales
    model_config = {"extra": "allow"}

    @field_validator("zarr_conventions", mode="after")
    @classmethod
    def ensure_multiscales_convention(
        cls, value: tuple[ConventionMetadataObject, ...]
    ) -> tuple[ConventionMetadataObject, ...]:
        """
        Iterate over the elements of zarr_conventions and check that at least one of them is
        multiscales
        """
        expected_uuid = multiscales_cm.CMO["uuid"]
        if not any(c["uuid"] == expected_uuid for c in value):
            raise ValueError(
                f"Multiscales convention (uuid={expected_uuid}) not found in zarr_conventions"
            )
        return value
