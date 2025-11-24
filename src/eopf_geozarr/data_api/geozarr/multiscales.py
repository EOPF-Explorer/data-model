from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, model_validator
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
    spec: Literal["https://github.com/zarr-conventions/geo-proj/blob/v0.1.0/README.md"]


MultiscaleConventions = TypedDict(  # type: ignore[misc]
    "MultiscaleConventions",
    {"d35379db-88df-4056-af3a-620245f8e347": MultiscaleConvention},
    closed=False,
)


class ConventionAttributes(BaseModel):
    zarr_conventions_version: Literal["0.1.0"]
    zarr_conventions: MultiscaleConventions


class ScaleLevel(BaseModel):
    group: str
    from_group: str | MISSING = MISSING
    translation: tuple[float, ...] | MISSING = MISSING
    factors: tuple[float, ...] | MISSING = MISSING
    scale: tuple[float, ...] | MISSING = MISSING
    resampling_method: str | MISSING = MISSING

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def check_model(self: Self) -> Self:
        if self.from_group is not MISSING and self.scale is MISSING:
            raise ValueError(
                f"from_group was set to {self.from_group}, but scale was unset. This is an error."
            )
        return self


class Multiscales(BaseModel):
    layout: tuple[ScaleLevel, ...]
    resampling_method: str | MISSING = MISSING
    model_config = {"extra": "allow"}


class MultiscalesAttributes(ConventionAttributes):
    multiscales: Multiscales
