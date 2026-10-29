"""Zarr V3 Models for the GeoZarr Zarr Hierarchy."""

from __future__ import annotations

from typing import Self, cast

from pydantic import model_validator
from pydantic_zarr.v3 import AnyGroupSpec, ArraySpec, GroupSpec

from eopf_geozarr.data_api.geozarr.common import (
    BaseDataArrayAttrs,
    DatasetAttrs,
    DatasetLike,
    GroupLike,
    check_grid_mapping,
    check_valid_coordinates,
)
from eopf_geozarr.data_api.geozarr.multiscales import MultiscaleGroupAttrs
from eopf_geozarr.data_api.geozarr.scope import GEOZARR_SCOPE


class DataArray(ArraySpec[BaseDataArrayAttrs]):
    """
    A Zarr array that represents as GeoZarr DataArray variable.

    The attributes of this array are defined in `BaseDataArrayAttrs`.

    This array has an additional constraint: the dimension_names field must be a tuple of strings.

    References
    ----------
    https://github.com/zarr-developers/geozarr-spec/blob/main/geozarr-spec.md#geozarr-dataarray
    """

    # GeoZarr requires dimension names, so tighten the parent's optional
    # `tuple[str | None, ...] | None` field to a required tuple of strings.
    dimension_names: tuple[str, ...]  # pyright: ignore[reportGeneralTypeIssues, reportIncompatibleVariableOverride]

    # Every extension-point name (data type, chunk grid, chunk key encoding,
    # codecs, storage transformers) must be claimed by a definition in the
    # validation scope, instead of passing unjudged: a GeoZarr DataArray is a
    # conformance contract, so a typo'd or unknown codec or data type name is
    # refused at validation. The scope is the default plus the data types
    # zarr-python writes that it lacks (fixed_length_utf32 band/angle
    # coordinates); an explicit pydantic validation context still overrides.
    zarr_scope = GEOZARR_SCOPE
    zarr_scope_strict = True

    @property
    def array_dimensions(self) -> tuple[str, ...]:
        return self.dimension_names


class Dataset(GroupSpec[DatasetAttrs, AnyGroupSpec | DataArray]):
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
        AnyGroupSpec
            The validated GeoZarr DataSet.
        """
        # ``self`` structurally satisfies the ``GroupLike`` protocol, but mypy cannot bind the
        # helper's TypeVar to ``Self``; cast through the protocol and back to ``Self`` (the helper
        # returns the same object).
        check_valid_coordinates(cast("GroupLike", self))
        return self

    @model_validator(mode="after")
    def validate_grid_mapping(self) -> Self:
        # See note above: ``self`` satisfies ``DatasetLike`` but the TypeVar can't bind to ``Self``.
        check_grid_mapping(cast("DatasetLike", self))
        return self


class MultiscaleGroup(GroupSpec[MultiscaleGroupAttrs, DataArray | AnyGroupSpec]):
    """
    A GeoZarr Multiscale Group.
    """
