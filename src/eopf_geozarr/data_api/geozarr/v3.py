from __future__ import annotations

from typing import Any, Self, TypeVar

from pydantic import model_validator
from pydantic_zarr.v3 import ArraySpec, GroupSpec

from eopf_geozarr.data_api.geozarr.common import BaseDataArrayAttrs, DatasetAttrs


class DataArray(ArraySpec[BaseDataArrayAttrs]):
    """
    A Zarr array that represents as GeoZarr DataArray variable.

    The attributes of this array are defined in `BaseDataArrayAttrs`.

    This array has an additional constraint: the dimension_names field must be a tuple of strings.

    References
    ----------
    https://github.com/zarr-developers/geozarr-spec/blob/main/geozarr-spec.md#geozarr-dataarray
    """

    dimension_names: tuple[str, ...]

    @model_validator(mode="after")
    def check_coordinates_dimensionality(self) -> Self:
        # split coordinates on whitespace
        coords_split = self.attributes.coordinates.split()
        missing_dims = set(coords_split) - set(self.dimension_names)
        if len(missing_dims) > 0:
            msg = (
                f"The coordinates {coords_split} are inconsistent with the dimension names "
                f"{self.dimension_names}. The following dimensions are missing: {missing_dims}"
            )
            raise ValueError(msg)
        return self

    @property
    def array_dimensions(self) -> tuple[str, ...]:
        return self.dimension_names


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
