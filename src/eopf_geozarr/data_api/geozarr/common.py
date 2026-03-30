"""Common utilities for GeoZarr data API."""

from __future__ import annotations

import io
import urllib
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    Literal,
    Protocol,
    Self,
    TypeGuard,
    TypeVar,
)
from urllib.error import URLError

from cf_xarray.utils import parse_cf_standard_name_table
from pydantic import AfterValidator, BaseModel, Field, model_validator
from pydantic.experimental.missing_sentinel import MISSING
from typing_extensions import runtime_checkable

from eopf_geozarr.data_api.geozarr.projjson import ProjJSON  # noqa: TC001
from eopf_geozarr.data_api.geozarr.types import (
    CF_SCALE_OFFSET_KEYS,
    CFScaleOffset,
    EmptyDict,
)

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

    grid_mapping: str | MISSING = MISSING  # type: ignore[valid-type]


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


def get_cf_standard_names(url: str) -> tuple[str, ...]:
    """Retrieve the set of CF standard names and return them as a tuple."""

    headers = {"User-Agent": "eopf_geozarr"}

    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as response:
        content = response.read()  # Read the entire response body into memory
        content_fobj = io.BytesIO(content)

    _info, table, _aliases = parse_cf_standard_name_table(source=content_fobj)
    return tuple(table.keys())


# This is a URL to the CF standard names table.
CF_STANDARD_NAME_URL = (
    "https://raw.githubusercontent.com/cf-convention/cf-convention.github.io/"
    "master/Data/cf-standard-names/current/src/cf-standard-name-table.xml"
)

# this does IO against github. consider locally storing this data instead if fetching every time
# is problematic.

try:
    CF_STANDARD_NAMES = get_cf_standard_names(url=CF_STANDARD_NAME_URL)
    DO_CF_NAME_VALIDATION = True
except URLError:
    CF_STANDARD_NAMES = ()
    DO_CF_NAME_VALIDATION = False


def check_standard_name(name: str) -> str:
    """
    Check if the standard name is valid according to the CF conventions.

    Parameters
    ----------
    name : str
        The standard name to check.

    Returns
    -------
    str
        The validated standard name.

    Raises
    ------
    ValueError
        If the standard name is not valid.
    """
    if DO_CF_NAME_VALIDATION:
        if name in CF_STANDARD_NAMES:
            return name
        raise ValueError(
            f"Invalid standard name: {name}. This name was not found in the list of CF standard names."
        )
    return name


CFStandardName = Annotated[str, AfterValidator(check_standard_name)]


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


def extract_scale_offset(
    data: Mapping[str, object],
) -> tuple[dict[str, object], CFScaleOffset | EmptyDict]:
    """
    Extract scale/offset information from a mapping, returning the remaining data and the scale/offset info.

    Parameters
    ----------
    data : Mapping[[str, object]]
        The input mapping from which to extract scale/offset information.

    Returns
    -------
    tuple[Mapping[str, object], CFScaleOffset]
        A tuple containing the remaining data (with scale/offset keys removed) and the extracted scale/offset info.
    """
    scale_offset: CFScaleOffset = {}  # type: ignore[typeddict-item]
    remaining_data: dict[str, object] = {}

    if set(data.keys()).isdisjoint(CF_SCALE_OFFSET_KEYS):
        return dict(data), {}

    if set(data.keys()).issuperset(CF_SCALE_OFFSET_KEYS):
        for key, value in data.items():
            if key in CF_SCALE_OFFSET_KEYS:
                scale_offset[key] = value  # type: ignore[literal-required]
            else:
                remaining_data[key] = value
        return remaining_data, scale_offset

    raise ValueError(
        "Incomplete scale/offset information: all of 'scale_factor', 'add_offset', must be present."
    )
