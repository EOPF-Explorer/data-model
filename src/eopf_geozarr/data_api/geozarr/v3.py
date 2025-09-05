"""Zarr V3 Models for the GeoZarr Zarr Hierarchy."""

from __future__ import annotations

from typing import Any, Self, TypeVar

from pydantic import model_validator
from pydantic.experimental.missing_sentinel import MISSING
from pydantic_zarr.v3 import ArraySpec, GroupSpec

from eopf_geozarr.data_api.geozarr.common import (
    BaseDataArrayAttrs,
    DatasetAttrs,
    GridMappingAttrs,
    MultiscaleAttrs,
)


class DataArray(ArraySpec[BaseDataArrayAttrs]):
    """
    A Zarr array that represents as GeoZarr DataArray variable.

    The attributes of this array are defined in `BaseDataArrayAttrs`.

    This array has an additional constraint: the dimension_names field must be a tuple of strings.

    References
    ----------
    https://github.com/zarr-developers/geozarr-spec/blob/main/geozarr-spec.md#geozarr-dataarray
    """

    # The dimension names must be a tuple of strings
    dimension_names: tuple[str, ...]

    @property
    def array_dimensions(self) -> tuple[str, ...]:
        return self.dimension_names


class GridMappingVariable(ArraySpec[GridMappingAttrs]):
    """
    A Zarr array that represents a GeoZarr grid mapping variable.

    The attributes of this array are defined in `GridMappingAttrs`.

    References
    ----------
    https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#grid-mappings-and-projections
    """

    ...


T = TypeVar("T", bound=GroupSpec[Any, Any])


def check_valid_coordinates(model: T) -> T:
    """
    Check if the coordinates of the DataArrays listed in a GeoZarr DataSet are valid.

    For each DataArray in the model, we check the dimensions associated with the DataArray.
    For each dimension associated with a data variable, an array with the name of that data variable
    must be present in the members of the group, and the shape of that array must align with the
    DataArray shape.


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
        for idx, dim in enumerate(array.array_dimensions):
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


class Dataset(GroupSpec[DatasetAttrs, DataArray | GridMappingVariable]):
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

    @model_validator(mode="after")
    def validate_grid_mapping(self) -> Self:
        if self.members is not None:
            for key, val in self.members.items():
                if (
                    hasattr(val.attributes, "grid_mapping")
                    and val.attributes.grid_mapping is not MISSING
                ):
                    grid_mapping_var: str = val.attributes.grid_mapping
                    missing_key = grid_mapping_var not in self.members
                    if missing_key:
                        msg = f"Grid mapping variable {grid_mapping_var} declared by {key} was not found in dataset members."
                        raise ValueError(msg)
                    if not (
                        isinstance(self.members[grid_mapping_var], GridMappingVariable)
                    ):
                        raise ValueError(
                            f"Grid mapping variable '{grid_mapping_var}' is not of type GridMappingVariable. "
                            f"Found {type(self.members[grid_mapping_var])} instead."
                        )
        return self


class MultiscaleGroup(GroupSpec[MultiscaleAttrs, Dataset]):
    """
    A GeoZarr Multiscale group.

    Attributes
    ----------
    attributes: MultiscaleAttrs
        Attributes for a multiscale GeoZarr group.
    members: Mapping[str, Dataset]
        A mapping of dataset names to GeoZarr Datasets.
    ----------
    """

    # todo: define a validation routine that ensures the referential integrity between
    # multiscale attributes and the actual datasets
    ...
