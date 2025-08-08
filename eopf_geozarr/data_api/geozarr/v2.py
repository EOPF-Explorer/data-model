"""GeoZarr data API for Zarr V2."""
from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_serializer, model_validator
from pydantic_zarr.v2 import (
    AnyArraySpec,
    AnyGroupSpec,
    ArraySpec,
    GroupSpec,
    TAttr,
    TItem,
    from_flat_group,
)

from eopf_geozarr.data_api.geozarr.common import MultiscaleAttrs, check_standard_name

CFStandardName = Annotated[str, AfterValidator(check_standard_name)]


class CoordArrayAttrs(BaseModel):
    """
    Attributes for a GeoZarr coordinate array.

    Attributes
    ----------
    array_dimensions : tuple[str, ...]
        The dimensions of the array.
    standard_name : str
        The CF standard name of the variable.
    grid_mapping : object
        The grid mapping of the variable, which is a reference to a grid mapping variable that
        describes the spatial reference of the variable.
    grid_mapping_name : str
        The name of the grid mapping, which is a string that describes the type of grid mapping
        used for the variable.
    """
    # model_config is necessary to ensure that this model dictifies with the key 
    # `"_ARRAY_DIMENSIONS"` instead of "array_dimensions"
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    array_dimensions: tuple[str] = Field(alias="_ARRAY_DIMENSIONS")
    standard_name: CFStandardName
    long_name: str | None = None
    units: str
    axis: str


class CoordArray(ArraySpec[CoordArrayAttrs]):
    """
    A GeoZarr coordinate array variable. 
    
    It must be 1-dimensional and have a single element in its array_dimensions attribute.
    """

    shape: tuple[int]


class DataArrayAttrs(BaseModel, populate_by_name=True):
    """
    Attributes for a GeoZarr DataArray.

    Attributes
    ----------
    array_dimensions : tuple[str, ...]
        Alias for the _ARRAY_DIMENSIONS attribute, which lists the dimension names for this array.
    standard_name : str
        The CF standard name of the variable.
    grid_mapping : object
        The grid mapping of the variable, which is a reference to a grid mapping variable that
        describes the spatial reference of the variable.
    grid_mapping_name : str
        The name of the grid mapping, which is a string that describes the type of grid mapping
        used for the variable.
    """

    # todo: validate that this names listed here are the names of zarr arrays
    # unless the variable is an auxiliary variable
    # see https://github.com/zarr-developers/geozarr-spec/blob/main/geozarr-spec.md#geozarr-coordinates
    array_dimensions: tuple[str, ...] = Field(alias="_ARRAY_DIMENSIONS")
    standard_name: CFStandardName
    grid_mapping: object
    grid_mapping_name: str

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

class DataArray(ArraySpec[DataArrayAttrs]):
    """
    A GeoZarr DataArray variable.

    References
    ----------
    https://github.com/zarr-developers/geozarr-spec/blob/main/geozarr-spec.md#geozarr-dataarray
    """


def check_valid_coordinates(model: GroupSpec[Any, DataArray | CoordArray]) -> Dataset:
    """
    Check if the coordinates of the DataArrays listed in a GeoZarr DataSet are valid.

    For each DataArray in the model, we check the dimensions associated with the DataArray.
    For each dimension associated with a data variable, an array with the name of that data variable
    must be present in the members of the group.

    Parameters
    ----------
    model : GroupSpec[Any, Any]
        The GeoZarr DataArray model to check.

    Returns
    -------
    GroupSpec[Any, Any]
        The validated GeoZarr DataArray model.
    """
    if model.members is None:
        raise ValueError("Model members cannot be None")

    arrays: dict[str, DataArray] = {
        k: v for k, v in model.members.items() if isinstance(v, DataArray)
    }
    for key, array in arrays.items():
        for idx, dim in enumerate(array.attributes.array_dimensions):
            if dim not in model.members:
                raise ValueError(
                    f"Dimension '{dim}' for array '{key}' is not defined in the model members."
                )
            member = model.members[dim]
            if isinstance(member, GroupSpec):
                raise ValueError(
                    f"Dimension '{dim}' for array '{key}' should be a group. Found an array instead."
                )
            if member.shape[0] != array.shape[idx]:
                raise ValueError(
                    f"Dimension '{dim}' for array '{key}' has a shape mismatch: "
                    f"{member.shape[0]} != {array.shape[idx]}."
                )
    return model


class DatasetAttrs(BaseModel):
    """
    Attributes for a GeoZarr dataset.

    Attributes
    ----------
    multiscales: MultiscaleAttrs
    """

    multiscales: MultiscaleAttrs


class Dataset(GroupSpec[DatasetAttrs, GroupSpec[Any, Any] | DataArray]):
    """
    A GeoZarr Dataset.
    """

    @model_validator(mode="after")
    def check_valid_coordinates(self) -> Self:
        """
        Validate the coordinates of the GeoZarr DataSet.

        This method checks that all DataArrays in the dataset have valid coordinates
        according to the GeoZarr specification.

        Returns
        -------
        GroupSpec[Any, Any]
            The validated GeoZarr DataSet.
        """
        return check_valid_coordinates(self)
