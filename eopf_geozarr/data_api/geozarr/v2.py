"""GeoZarr data API for Zarr V2."""
from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import AfterValidator, BaseModel, Field, model_serializer, model_validator
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


class MyGroupSpec(GroupSpec[TAttr, TItem]):
    """
    A custom GroupSpec
    
    We override the from_flat method to ensure that we can pass by_alias=True
    when creating a GroupSpec from a flat hierarchy.
    """
    @classmethod
    def from_flat(
        cls, data: dict[str, AnyArraySpec | AnyGroupSpec], *, by_alias: bool = False
    ) -> Self:
        """
        Create a `GroupSpec` from a flat hierarchy representation. The flattened hierarchy is a
        `dict` with the following constraints: keys must be valid paths; values must
        be `ArraySpec` or `GroupSpec` instances.

        Parameters
        ----------
        data : Dict[str, ArraySpec | GroupSpec]
            A flattened representation of a Zarr hierarchy.

        Returns
        -------
        GroupSpec
            A `GroupSpec` representation of the hierarchy.

        Examples
        --------
        >>> from pydantic_zarr.v2 import GroupSpec, ArraySpec
        >>> import numpy as np
        >>> flat = {'': GroupSpec(attributes={'foo': 10}, members=None)}
        >>> GroupSpec.from_flat(flat)
        GroupSpec(zarr_format=2, attributes={'foo': 10}, members={})
        >>> flat = {
            '': GroupSpec(attributes={'foo': 10}, members=None),
            '/a': ArraySpec.from_array(np.arange(10))}
        >>> GroupSpec.from_flat(flat)
        GroupSpec(zarr_format=2, attributes={'foo': 10}, members={'a': ArraySpec(zarr_format=2, attributes={}, shape=(10,), chunks=(10,), dtype='<i8', fill_value=0, order='C', filters=None, dimension_separator='/', compressor=None)})
        """
        from_flated = from_flat_group(data)
        return cls(**from_flated.model_dump(by_alias=by_alias))


class CoordArrayAttrs(BaseModel, populate_by_name=True):
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


class DataArray(ArraySpec[DataArrayAttrs]):
    """
    A GeoZarr DataArray variable.

    References
    ----------
    https://github.com/zarr-developers/geozarr-spec/blob/main/geozarr-spec.md#geozarr-dataarray
    """


def check_valid_coordinates(model: GroupSpec[Any, Any]) -> Dataset:
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
