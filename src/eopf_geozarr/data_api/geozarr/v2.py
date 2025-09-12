"""Zarr V2 Models for the GeoZarr Zarr Hierarchy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable, Literal, Self, TypeVar

from pydantic import ConfigDict, Field, model_validator
from pydantic_zarr.v2 import ArraySpec, GroupSpec, auto_attributes

from eopf_geozarr.data_api.geozarr.common import (
    XARRAY_DIMS_KEY,
    BaseDataArrayAttrs,
    DatasetAttrs,
    GridMappingAttrs,
    MultiscaleAttrs,
)


class DataArrayAttrs(BaseDataArrayAttrs):
    """
    Attributes for a GeoZarr DataArray.

    Attributes
    ----------
    array_dimensions : tuple[str, ...]
        Alias for the _ARRAY_DIMENSIONS attribute, which lists the dimension names for this array.
    """

    # todo: validate that this names listed here are the names of zarr arrays
    # unless the variable is an auxiliary variable
    # see https://github.com/zarr-developers/geozarr-spec/blob/main/geozarr-spec.md#geozarr-coordinates
    array_dimensions: tuple[str, ...] = Field(alias="_ARRAY_DIMENSIONS")

    # this is necessary to serialize the `array_dimensions` attribute as `_ARRAY_DIMENSIONS`
    model_config = ConfigDict(serialize_by_alias=True)


class DataArray(ArraySpec[DataArrayAttrs]):
    """
    A GeoZarr DataArray variable. It must have attributes that contain an `"_ARRAY_DIMENSIONS"`
    key, with a length that matches the dimensionality of the array.

    References
    ----------
    https://github.com/zarr-developers/geozarr-spec/blob/main/geozarr-spec.md#geozarr-dataarray
    """

    @classmethod
    def from_array(
        cls,
        array: Any,
        chunks: tuple[int, ...] | Literal["auto"] = "auto",
        attributes: Mapping[str, object] | Literal["auto"] = "auto",
        fill_value: object | Literal["auto"] = "auto",
        order: Literal["C", "F"] | Literal["auto"] = "auto",
        filters: tuple[Any, ...] | Literal["auto"] = "auto",
        dimension_separator: Literal[".", "/"] | Literal["auto"] = "auto",
        compressor: Any | Literal["auto"] = "auto",
        dimension_names: Iterable[str] | Literal["auto"] = "auto",
    ) -> Self:
        """
        Override the default from_array method to include a dimension_names parameter.
        """
        if attributes == "auto":
            auto_attrs = dict(auto_attributes(array))
        else:
            auto_attrs = dict(attributes)
        if dimension_names != "auto":
            auto_attrs = auto_attrs | {XARRAY_DIMS_KEY: tuple(dimension_names)}
        model = super().from_array(
            array=array,
            chunks=chunks,
            attributes=auto_attrs,
            fill_value=fill_value,
            order=order,
            filters=filters,
            dimension_separator=dimension_separator,
            compressor=compressor,
        )
        return model  # type: ignore[no-any-return]

    @model_validator(mode="after")
    def check_array_dimensions(self) -> Self:
        if (len_dim := len(self.attributes.array_dimensions)) != (
            ndim := len(self.shape)
        ):
            msg = (
                f"The {XARRAY_DIMS_KEY} attribute has length {len_dim}, which does not "
                f"match the number of dimensions for this array (got {ndim})."
            )
            raise ValueError(msg)
        return self

    @property
    def array_dimensions(self) -> tuple[str, ...]:
        return self.attributes.array_dimensions  # type: ignore[no-any-return]


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
                    and val.attributes.grid_mapping is not None
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
