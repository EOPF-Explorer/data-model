from __future__ import annotations

from typing import Final, Literal, NotRequired

from pydantic import BaseModel, field_validator, model_serializer
from pydantic.experimental.missing_sentinel import MISSING
from typing_extensions import TypedDict

from eopf_geozarr.data_api.geozarr.common import (
    ZarrConventionMetadata,
    ZarrConventionMetadataJSON,
)

ConventionID = Literal["d35379db-88df-4056-af3a-620245f8e347"]
CONVENTION_ID: Final[ConventionID] = "d35379db-88df-4056-af3a-620245f8e347"

ConventionSchemaURL = Literal[
    "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json"
]
CONVENTION_SCHEMA_URL: Final[ConventionSchemaURL] = (
    "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json"
)

ConventionSpecURL = Literal["https://github.com/zarr-conventions/multiscales/blob/v1/README.md"]
CONVENTION_SPEC_URL: Final[ConventionSpecURL] = (
    "https://github.com/zarr-conventions/multiscales/blob/v1/README.md"
)

ConventionDescription = Literal["Multiscale layout of zarr datasets"]
CONVENTION_DESCRIPTION: Final[ConventionDescription] = "Multiscale layout of zarr datasets"

ConventionName = Literal["multiscales"]
CONVENTION_NAME: Final[ConventionName] = "multiscales"


class MultiscaleConventionMetadata(ZarrConventionMetadata):
    uuid: ConventionID = CONVENTION_ID
    schema_url: ConventionSchemaURL = CONVENTION_SCHEMA_URL
    name: ConventionName = CONVENTION_NAME
    description: ConventionDescription = CONVENTION_DESCRIPTION
    spec_url: ConventionSpecURL = CONVENTION_SPEC_URL


class MultiscaleConventionMetadataJSON(TypedDict):
    """
    A TypedDict representation of the Multiscales convention metadata
    """

    uuid: NotRequired[ConventionID]
    schema_url: NotRequired[ConventionSchemaURL]
    name: NotRequired[ConventionName]
    description: NotRequired[ConventionDescription]
    spec_url: NotRequired[ConventionSpecURL]


# A final dict representation of the Multiscales convention metadata
MULTISCALE_CONVENTION_METADATA: Final[MultiscaleConventionMetadataJSON] = {
    "uuid": CONVENTION_ID,
    "schema_url": CONVENTION_SCHEMA_URL,
    "name": CONVENTION_NAME,
    "description": CONVENTION_DESCRIPTION,
    "spec_url": CONVENTION_SPEC_URL,
}


class ZarrConventionAttrs(BaseModel):
    zarr_conventions: tuple[ZarrConventionMetadata, ...]

    model_config = {"extra": "allow"}


class Transform(BaseModel):
    scale: tuple[float, ...] | MISSING = MISSING
    translation: tuple[float, ...] | MISSING = MISSING

    @model_serializer
    def serialize_model(self) -> dict[str, tuple[float, ...]]:
        result: dict[str, tuple[float, ...]] = {}
        if self.scale is not MISSING:
            result["scale"] = self.scale
        if self.translation is not MISSING:
            result["translation"] = self.translation
        return result


class TransformJSON(TypedDict):
    scale: NotRequired[tuple[float, ...]]
    translation: NotRequired[tuple[float, ...]]


class ScaleLevel(BaseModel):
    asset: str
    derived_from: str | MISSING = MISSING
    transform: Transform | MISSING = MISSING
    resampling_method: str | MISSING = MISSING

    model_config = {"extra": "allow"}


class ScaleLevelJSON(TypedDict):
    asset: str
    derived_from: NotRequired[str]
    transform: TransformJSON
    resampling_method: NotRequired[str]


class Multiscales(BaseModel):
    layout: tuple[ScaleLevel, ...]
    resampling_method: str | MISSING = MISSING

    model_config = {"extra": "allow"}


class MultiscalesJSON(TypedDict):
    version: NotRequired[str]
    layout: tuple[ScaleLevelJSON, ...]
    resampling_method: NotRequired[str]


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


class MultiscalesAttrsJSON(TypedDict):
    zarr_conventions: tuple[ZarrConventionMetadataJSON, ...]
    multiscales: MultiscalesJSON
