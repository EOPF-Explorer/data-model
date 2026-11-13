from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from pydantic import ValidationError
from pydantic_zarr.v2 import GroupSpec

from eopf_geozarr.data_api.geozarr.common import DatasetAttrs
from eopf_geozarr.data_api.geozarr.v2 import (
    DataArray,
    Dataset,
    check_valid_coordinates,
)

if TYPE_CHECKING:
    from eopf_geozarr.data_api.geozarr.common import GroupLike


def test_invalid_dimension_names() -> None:
    msg = r"The _ARRAY_DIMENSIONS attribute has length 3, which does not match the number of dimensions for this array \(got 2\)"
    with pytest.raises(ValidationError, match=msg):
        DataArray.from_array(np.zeros((10, 10)), dimension_names=["x", "y", "z"])


class TestCheckValidCoordinates:
    @staticmethod
    @pytest.mark.parametrize("data_shape", [(10,), (10, 12)])
    def test_valid(data_shape: tuple[int, ...]) -> None:
        """
        Test the check_valid_coordinates function to ensure it validates coordinates correctly.
        """

        base_array = DataArray.from_array(
            np.zeros((data_shape), dtype="uint8"),
            dimension_names=[f"dim_{s}" for s in range(len(data_shape))],
        )
        coords_arrays = {
            f"dim_{idx}": DataArray.from_array(np.arange(s), dimension_names=(f"dim_{idx}",))
            for idx, s in enumerate(data_shape)
        }
        group = GroupSpec[Mapping[str, object], DataArray](
            attributes={}, members={"base": base_array, **coords_arrays}
        )
        # ``group`` structurally satisfies ``GroupLike``, but mypy cannot bind the
        # invariant ``Mapping`` value type, mirroring the cast used in the source.
        group_like = cast("GroupLike", group)
        assert check_valid_coordinates(group_like) == group_like

    @staticmethod
    @pytest.mark.parametrize("data_shape", [(10,), (10, 12)])
    def test_invalid_coordinates(
        data_shape: tuple[int, ...],
    ) -> None:
        """
        Test the check_valid_coordinates function to ensure it validates coordinates correctly.

        This test checks that the function raises a ValueError when the dimensions of the data variable
        do not match the dimensions of the coordinate arrays.
        """
        base_array = DataArray.from_array(
            np.zeros((data_shape), dtype="uint8"),
            dimension_names=[f"dim_{s}" for s in range(len(data_shape))],
        )
        coords_arrays = {
            f"dim_{idx}": DataArray.from_array(np.arange(s + 1), dimension_names=(f"dim_{idx}",))
            for idx, s in enumerate(data_shape)
        }
        group = GroupSpec[Mapping[str, object], DataArray](
            attributes={}, members={"base": base_array, **coords_arrays}
        )
        msg = "Dimension .* for array 'base' has a shape mismatch:"
        with pytest.raises(ValueError, match=msg):
            check_valid_coordinates(cast("GroupLike", group))


class TestDataset:
    @staticmethod
    def _members() -> dict[str, DataArray]:
        base_array = DataArray.from_array(
            np.zeros((4, 4), dtype="uint8"), dimension_names=["y", "x"]
        )
        coords_arrays = {
            name: DataArray.from_array(np.arange(4), dimension_names=(name,)) for name in ("y", "x")
        }
        return {"band": base_array, **coords_arrays}

    def test_valid(self) -> None:
        """A dataset with consistent coordinates and no grid mapping validates."""
        ds = Dataset(attributes=DatasetAttrs(), members=self._members())
        assert isinstance(ds, Dataset)

    def test_missing_grid_mapping_variable(self) -> None:
        """A member declaring a grid_mapping that is not in the dataset fails validation."""
        members = self._members()
        band = members["band"]
        members["band"] = band.replace(
            attributes=band.attributes.model_copy(update={"grid_mapping": "nope"})
        )
        with pytest.raises(ValidationError, match="Grid mapping variable 'nope'"):
            Dataset(attributes=DatasetAttrs(), members=members)
