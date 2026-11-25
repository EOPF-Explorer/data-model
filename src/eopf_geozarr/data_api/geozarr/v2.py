"""Zarr V2 Models for the GeoZarr Zarr Hierarchy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from pydantic import ConfigDict, Field, model_validator
from pydantic_zarr.v2 import ArraySpec, GroupSpec, auto_attributes

from eopf_geozarr.data_api.geozarr.common import (
    BaseDataArrayAttrs,
    DatasetAttrs,
    DatasetLike,
    GridMappingAttrs,
    GroupLike,
    check_grid_mapping,
    check_valid_coordinates,
)
from eopf_geozarr.data_api.geozarr.multiscales import MultiscaleGroupAttrs
from eopf_geozarr.data_api.geozarr.types import XARRAY_DIMS_KEY

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pydantic.experimental.missing_sentinel import MISSING as MISSING


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

    # The override intentionally widens the accepted argument types (e.g. plain mappings for
    # ``attributes``) and adds a ``dimension_names`` parameter, so the signature is not a strict
    # subtype of the parent's. This is by design and does not change runtime behavior.
    @classmethod
    def from_array(  # type: ignore[override]
        cls,
        array: Any,
        chunks: tuple[int, ...] | Literal["auto"] = "auto",
        attributes: Mapping[str, object] | Literal["auto"] = "auto",
        fill_value: object | Literal["auto"] = "auto",
        order: Literal["C", "F", "auto"] = "auto",
        filters: tuple[Any, ...] | Literal["auto"] = "auto",
        dimension_separator: Literal[".", "/", "auto"] = "auto",
        compressor: Any | Literal["auto"] = "auto",
        dimension_names: Iterable[str] | Literal["auto"] = "auto",
    ) -> Self:
        """
        Override the default from_array method to include a dimension_names parameter.
        """
        auto_attrs = dict(auto_attributes(array)) if attributes == "auto" else dict(attributes)
        if dimension_names != "auto":
            auto_attrs = auto_attrs | {XARRAY_DIMS_KEY: tuple(dimension_names)}
        # ``auto_attrs``/``fill_value``/``filters`` are validated/coerced by pydantic at
        # construction time; cast to the parent's declared types to satisfy the static checker.
        return super().from_array(
            array=array,
            chunks=chunks,
            attributes=cast("Literal['auto'] | DataArrayAttrs", auto_attrs),
            fill_value=cast("Literal['auto'] | float | None", fill_value),
            order=order,
            filters=cast("Literal['auto'] | list[dict[str, Any]] | None", filters),
            dimension_separator=dimension_separator,
            compressor=compressor,
        )

    @model_validator(mode="after")
    def check_array_dimensions(self) -> Self:
        if (len_dim := len(self.attributes.array_dimensions)) != (ndim := len(self.shape)):
            msg = (
                f"The {XARRAY_DIMS_KEY} attribute has length {len_dim}, which does not "
                f"match the number of dimensions for this array (got {ndim})."
            )
            raise ValueError(msg)
        return self

    @property
    def array_dimensions(self) -> tuple[str, ...]:
        return self.attributes.array_dimensions


class GridMappingVariable(ArraySpec[GridMappingAttrs]):
    """
    A Zarr array that represents a GeoZarr grid mapping variable.

    The attributes of this array are defined in `GridMappingAttrs`.

    References
    ----------
    https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#grid-mappings-and-projections
    """


class Dataset(GroupSpec[DatasetAttrs, Mapping[str, DataArray | GridMappingVariable]]):
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
        # ``self`` structurally satisfies the ``GroupLike`` protocol, but mypy cannot bind the
        # helper's TypeVar to ``Self``; cast through the protocol and back to ``Self`` (the helper
        # returns the same object).
        check_valid_coordinates(cast("GroupLike", self))
        return self

    @model_validator(mode="after")
    def check_grid_mapping(self) -> Self:
        # See note above: ``self`` satisfies ``DatasetLike`` but the TypeVar can't bind to ``Self``.
        check_grid_mapping(cast("DatasetLike", self))
        return self


class MultiscaleGroup(
    GroupSpec[MultiscaleGroupAttrs, Mapping[str, DataArray | GroupSpec[Any, Any]]]
):
    """
    A GeoZarr Multiscale Group.
    """
