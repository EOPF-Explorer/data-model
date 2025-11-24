from typing import Any

import numpy as np
import pytest
import zarr
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import ArraySpec, GroupSpec

from eopf_geozarr.data_api.geozarr.v3 import (
    DataArray,
    MultiscaleGroup,
    check_valid_coordinates,
)


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
            f"dim_{idx}": DataArray.from_array(
                np.arange(s), dimension_names=(f"dim_{idx}",)
            )
            for idx, s in enumerate(data_shape)
        }
        group = GroupSpec[Any, DataArray](members={"base": base_array, **coords_arrays})
        assert check_valid_coordinates(group) == group

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
            f"dim_{idx}": DataArray.from_array(
                np.arange(s + 1), dimension_names=(f"dim_{idx}",)
            )
            for idx, s in enumerate(data_shape)
        }
        group = GroupSpec[Any, DataArray](members={"base": base_array, **coords_arrays})
        msg = "Dimension .* for array 'base' has a shape mismatch:"
        with pytest.raises(ValueError, match=msg):
            check_valid_coordinates(group)


def test_dataarray_round_trip(example_group: Any) -> None:
    """
    Ensure that we can round-trip dataarray attributes through the `Multiscales` model.
    """
    source_untyped = GroupSpec.from_zarr(example_group)
    flat = source_untyped.to_flat()
    for key, val in flat.items():
        if isinstance(val, ArraySpec) and val.dimension_names is not None:
            model_json = val.model_dump()
            assert DataArray(**model_json).model_dump() == model_json


def test_multiscale_attrs_round_trip(example_group: Any) -> None:
    """
    Test that multiscale datasets round-trip through the `Multiscales` model
    """
    source_group_members = dict(example_group.members(max_depth=None))
    for key, val in source_group_members.items():
        if isinstance(val, zarr.Group):
            if "multiscales" in val.attrs.asdict():
                model_json = MultiscaleGroup.from_zarr(val).model_dump()
                assert MultiscaleGroup(**model_json).model_dump() == tuplify_json(
                    model_json
                )
