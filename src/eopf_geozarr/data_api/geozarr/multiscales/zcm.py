from __future__ import annotations

from typing import Literal, NotRequired

from pydantic import BaseModel
from pydantic.experimental.missing_sentinel import MISSING
from typing_extensions import TypedDict

ConventionID = Literal["d35379db-88df-4056-af3a-620245f8e347"]


class MultiscaleConvention(TypedDict):
    version: Literal["0.1.0"]
    schema: Literal[
        "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1.0/schema.json"
    ]
    name: Literal["multiscales"]
    description: Literal["Multiscale layout of zarr datasets"]
    spec: Literal[
        "https://github.com/zarr-conventions/multiscales/blob/v0.1.0/README.md"
    ]


MultiscaleConventions = TypedDict(  # type: ignore[misc]
    "MultiscaleConventions",
    {"d35379db-88df-4056-af3a-620245f8e347": MultiscaleConvention},
    closed=False,
)

MULTISCALE_CONVENTION: MultiscaleConventions = {  # type: ignore[typeddict-unknown-key]
    "d35379db-88df-4056-af3a-620245f8e347": {
        "version": "0.1.0",
        "schema": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v0.1.0/schema.json",
        "name": "multiscales",
        "description": "Multiscale layout of zarr datasets",
        "spec": "https://github.com/zarr-conventions/multiscales/blob/v0.1.0/README.md",
    }
}


class ConventionAttributes(BaseModel):
    zarr_conventions_version: Literal["0.1.0"]
    zarr_conventions: MultiscaleConventions

    model_config = {"extra": "allow"}


class TransformJSON(TypedDict):
    scale: NotRequired[tuple[float, ...]]
    translation: NotRequired[tuple[float, ...]]


class Transform(BaseModel):
    scale: tuple[float, ...] | MISSING = MISSING
    translation: tuple[float, ...] | MISSING = MISSING


class ScaleLevelJSON(TypedDict):
    asset: str
    derived_from: NotRequired[str]
    transform: TransformJSON
    resampling_method: NotRequired[str]


class ScaleLevel(BaseModel):
    asset: str
    derived_from: str | MISSING = MISSING
    transform: Transform | MISSING = MISSING
    resampling_method: str | MISSING = MISSING

    model_config = {"extra": "allow"}


class MultiscalesJSON(TypedDict):
    layout: tuple[ScaleLevelJSON, ...]
    resampling_method: NotRequired[str]


class Multiscales(BaseModel):
    layout: tuple[ScaleLevel, ...]
    resampling_method: str | MISSING = MISSING

    model_config = {"extra": "allow"}

class MultiscalesAttrsJSON(TypedDict):
    zarr_conventions_version: Literal["0.1.0"]
    zarr_conventions: MultiscaleConventions
    multiscales: Multiscales


class MultiscalesAttrs(ConventionAttributes):
    multiscales: Multiscales
    model_config = {"extra": "allow"}
