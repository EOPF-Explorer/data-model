from __future__ import annotations

from typing import Any

import pytest
from pydantic_zarr.v2 import ArraySpec, GroupSpec

from eopf_geozarr.data_api.geozarr.v2 import (
    DataArray,
    DataArrayAttrs,
    check_valid_coordinates,
)

from .conftest import example_group


class TestCheckValidCoordinates:
    @pytest.mark.parametrize(
        "example",
        [
            {
                "": GroupSpec(attributes={}, members=None),
                "/data_var": DataArray(
                    shape=(10, 11),
                    dtype="|u1",
                    chunks=(10, 11),
                    attributes=DataArrayAttrs(
                        _FILL_VALUE="AAAAAAAA+H8=",
                        _ARRAY_DIMENSIONS=["time", "lat"],
                        standard_name="air_temperature",
                        grid_mapping=None,
                        grid_mapping_name="latitude_longitude",
                    ),
                ),
                "/time": DataArray(
                    shape=(10,),
                    dtype="|u1",
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
                    dtype="|u1",
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
    def test_valid(example: dict[str, ArraySpec[Any] | GroupSpec[Any, Any]]) -> None:
        """
        Test the check_valid_coordinates function to ensure it validates coordinates correctly.
        """
        group = GroupSpec.from_flat(example)
        assert check_valid_coordinates(group) == group

    @pytest.mark.parametrize(
        "example",
        [
            {
                "": GroupSpec(attributes={}, members=None),
                "/data_var": DataArray(
                    shape=(9, 10),
                    dtype="|u1",
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
                    dtype="|u1",
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
                    dtype="|u1",
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


@pytest.mark.skip(reason="We don't have a v2 example group yet")
def test_dataarray_attrs_round_trip() -> None:
    """
    Ensure that we can round-trip dataarray attributes through the `Multiscales` model.
    """
    source_untyped = GroupSpec.from_zarr(example_group)
    flat = source_untyped.to_flat()
    for key, val in flat.items():
        if isinstance(val, ArraySpec):
            model_json = val.model_dump()["attributes"]
            assert DataArrayAttrs(**model_json).model_dump() == model_json
