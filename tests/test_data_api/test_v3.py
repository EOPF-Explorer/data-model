from typing import Any

import pytest
import zarr
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import ArraySpec, GroupSpec

from eopf_geozarr.data_api.geozarr.v3 import DataArray, Dataset, check_valid_coordinates

from .conftest import example_group


class TestCheckValidCoordinates:
    @pytest.mark.parametrize("data_shape", [(10,), (10, 12)])
    def test_valid(data_shape: tuple[int, ...]) -> None:
        """
        Test the check_valid_coordinates function to ensure it validates coordinates correctly.
        """

        # create a group containing "coordinated" arrays, with 1 n-dimensional data array, and
        # N 1-dimensional coordinate arrays

        group = GroupSpec.from_flat(example)
        assert check_valid_coordinates(group) == group

    @pytest.mark.parametrize(
        "example",
        [
            {
                "": GroupSpec(attributes={}, members=None),
                "/data_var": DataArray(
                    shape=(9, 10),
                    data_type="uint8",
                    chunks=(10, 11),
                    attributes=DataArrayAttrs(
                        _ARRAY_DIMENSIONS=["time", "lat"],
                        _FILL_VALUE="AAAAAAAA+H8=",
                        standard_name="air_temperature",
                        grid_mapping=None,
                        grid_mapping_name="latitude_longitude",
                    ),
                ),
                "/time": DataArray(
                    shape=(10,),
                    data_type="uint8",
                    chunks=(10,),
                    attributes=DataArrayAttrs(
                        _ARRAY_DIMENSIONS=["time"],
                        standard_name="time",
                        units="s",
                        axis="T",
                    ),
                ),
                "/lat": DataArray(
                    shape=(11,),
                    data_type="uint8",
                    chunks=(11,),
                    attributes=DataArrayAttrs(
                        _ARRAY_DIMENSIONS=["lat"],
                        standard_name="latitude",
                        units="m",
                        axis="Y",
                    ),
                ),
            },
        ],
    )
    @staticmethod
    def test_invalid_coordinates(
        example: dict[str, ArraySpec[Any] | GroupSpec[Any, Any]],
    ) -> None:
        """
        Test the check_valid_coordinates function to ensure it validates coordinates correctly.

        This test checks that the function raises a ValueError when the dimensions of the data variable
        do not match the dimensions of the coordinate arrays.
        """
        group = GroupSpec[Any, DataArray].from_flat(example)
        with pytest.raises(ValueError):
            check_valid_coordinates(group)


def test_dataarray_round_trip() -> None:
    """
    Ensure that we can round-trip dataarray attributes through the `Multiscales` model.
    """
    source_untyped = GroupSpec.from_zarr(example_group)
    flat = source_untyped.to_flat()
    for key, val in flat.items():
        if isinstance(val, ArraySpec):
            model_json = val.model_dump()
            assert DataArray(**model_json).model_dump() == model_json


def test_multiscale_attrs_round_trip() -> None:
    """
    Test that multiscale datasets round-trip through the `Multiscales` model
    """
    source_group_members = dict(example_group.members(max_depth=None))
    for key, val in source_group_members.items():
        if isinstance(val, zarr.Group):
            if "multiscales" in val.attrs.asdict():
                model_json = GroupSpec.from_zarr(val).model_dump()
                assert Dataset(**model_json).model_dump() == tuplify_json(model_json)
