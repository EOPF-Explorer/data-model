"""Common utilities for GeoZarr data API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Protocol,
    Self,
    TypeGuard,
    TypeVar,
)

from pydantic import BaseModel, Field, model_validator
from pydantic.experimental.missing_sentinel import MISSING
from typing_extensions import runtime_checkable

from eopf_geozarr.data_api.geozarr.projjson import ProjJSON

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class UNSET_TYPE:
    """
    Sentinel value to indicate that a value is not set.
    """


GEO_PROJ_VERSION: Final = "0.1"


class ProjAttrs(BaseModel, extra="allow"):
    """
    Zarr attributes for coordinate reference system (CRS) encoding.

    Attributes
    version: str
        The version of the metadata.
    code: str | None
        Authority:Code identifier.
    wkt2 : str | None
        WKT2 (ISO 19162) representation of the CRS.
    projjson: ProjJson | None
        PROJJSON representation of the CRS.
    bbox:
    """

    version: Literal["0.1"] = "0.1"
    code: str | None = Field(None, pattern="^[A-Z]+:[0-9]+$")
    wkt2: str | None = None
    projjson: ProjJSON | None = None
    bbox: tuple[float, float, float, float] | None = None
    transform: tuple[float, float, float, float, float, float] | None = None
    # TODO: enclosing object must validate these properties against the arrays
    spatial_dimensions: tuple[str, str] | None = None

    @model_validator(mode="after")
    def encure_crs(self) -> Self:
        if self.code is None and self.wkt2 is None and self.projjson is None:
            raise ValueError("One of 'code', 'wkt2', or 'projjson' must be provided.")
        return self


class BaseDataArrayAttrs(BaseModel, extra="allow"):
    """
    Base attributes for a  GeoZarr DataArray.

    Attributes
    ----------
    """

    grid_mapping: str | MISSING = MISSING


class GridMappingAttrs(BaseModel, extra="allow"):
    """
    Grid mapping attributes for a GeoZarr grid mapping variable.

    Attributes
    ----------
    grid_mapping_name : str
        The name of the grid mapping.

    Extra fields are permitted.

    Additional attributes might be present depending on the type of grid mapping.

    References
    ----------
    https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#grid-mappings-and-projections
    """

    grid_mapping_name: str


@runtime_checkable
class GroupLike(Protocol):
    members: Mapping[str, Any] | None
    attributes: Any


TGroupLike = TypeVar("TGroupLike", bound=GroupLike)


def check_valid_coordinates[TGroupLike: GroupLike](model: TGroupLike) -> TGroupLike:
    """
    Check if the coordinates of the DataArrayLike objects listed in GroupLike objects are valid.

    For each DataArrayLike in the model, we check the dimensions associated with the DataArrayLike.
    For each dimension associated with a data variable, a DataArrayLike with the name of that data
    variable must be present in the members of the group.

    Parameters
    ----------
    model : GroupLike
        An object that implements the GroupLike protocol.

    Returns
    -------
    GroupLike
        A GroupLike object with referentially valid coordinates.
    """
    if model.members is None:
        raise ValueError("Model members cannot be None")

    arrays: dict[str, DataArrayLike] = {
        k: v for k, v in model.members.items() if isinstance(v, DataArrayLike)
    }
    for key, array in arrays.items():
        for idx, dim in enumerate(array.array_dimensions):
            if dim not in model.members:
                raise ValueError(
                    f"Dimension '{dim}' for array '{key}' is not defined in the model members."
                )
            member = model.members[dim]
            if isinstance(member, GroupLike):
                raise TypeError(
                    f"Dimension '{dim}' for array '{key}' should be a group. Found an array instead."
                )
            if member.shape[0] != array.shape[idx]:
                raise ValueError(
                    f"Dimension '{dim}' for array '{key}' has a shape mismatch: "
                    f"{member.shape[0]} != {array.shape[idx]}."
                )
    return model


@runtime_checkable
class DataArrayLike(Protocol):
    """
    This is a protocol that models the relevant properties of Zarr V2 and Zarr V3 DataArrays.
    """

    @property
    def array_dimensions(self) -> tuple[str, ...]: ...

    shape: tuple[int, ...]
    attributes: BaseDataArrayAttrs


class DatasetAttrs(BaseModel, extra="allow"):
    """
    Attributes for a GeoZarr dataset.

    A dataset is a collection of DataArrays. This class models the attributes of a dataset
    """


@runtime_checkable
class DatasetLike(Protocol):
    members: Mapping[str, DataArrayLike] | None


TDataSetLike = TypeVar("TDataSetLike", bound=DatasetLike)


def check_grid_mapping[TDataSetLike: DatasetLike](model: TDataSetLike) -> TDataSetLike:
    """
    Ensure that a grid mapping variable is present, and that it refers to a member of the model.
    """
    if model.members is not None:
        for name, member in model.members.items():
            if (
                hasattr(member.attributes, "grid_mapping")
                and isinstance(member.attributes.grid_mapping, str)
                and member.attributes.grid_mapping not in model.members
            ):
                msg = f"Grid mapping variable '{member.attributes.grid_mapping}' declared by {name} was not found in dataset members"
                raise ValueError(msg)
    return model


def is_none(data: object) -> TypeGuard[None]:
    return data is None
