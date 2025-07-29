from __future__ import annotations

import re
from typing import Any

import pytest
from pydantic_zarr.v2 import ArraySpec, GroupSpec

from eopf_geozarr.data_api.geozarr.common import (
    CF_STANDARD_NAME_URL,
    check_standard_name,
    get_cf_standard_names,
)
from eopf_geozarr.data_api.geozarr.v2 import (
    CoordArray,
    CoordArrayAttrs,
    DataArray,
    DataArrayAttrs,
    MyGroupSpec,
    check_valid_coordinates,
)


def test_get_cf_standard_names() -> None:
    """
    Test the get_cf_standard_names function to ensure it retrieves the CF standard names correctly.
    """
    standard_names = get_cf_standard_names(CF_STANDARD_NAME_URL)
    assert isinstance(standard_names, tuple)
    assert len(standard_names) > 0
    assert all(isinstance(name, str) for name in standard_names)


@pytest.mark.parametrize(
    "name", ["air_temperature", "sea_surface_temperature", "precipitation_flux"]
)
def test_check_standard_name_valid(name: str) -> None:
    """
    Test the check_standard_name function with valid standard names.
    """
    assert check_standard_name(name) == name


def test_check_standard_name_invalid() -> None:
    """
    Test the check_standard_name function with an invalid standard name.
    """
    with pytest.raises(ValueError):
        check_standard_name("invalid_standard_name")


def test_coord_array_attrs_dimensions_length() -> None:
    """
    Test that the array_dimensions attribute must have length 1.
    """
    msg = (
        "1 validation error for CoordArrayAttrs\n_ARRAY_DIMENSIONS\n "
        " Tuple should have at most 1 item after validation, not 2"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        CoordArrayAttrs(
            _ARRAY_DIMENSIONS=("time", "lat"),
            standard_name="air_temperature",
            units="mm",
            axis="Y",
        )


def test_coord_array_dimensionality() -> None:
    """
    Test that only 1-dimensional arrays are allowed.
    """
    msg = (
        "1 validation error for CoordArray\nshape\n  "
        "Tuple should have at most 1 item after validation, not 2"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        CoordArray(
            shape=(10, 11),
            dtype="|u1",
            chunks=(10, 11),
            attributes=CoordArrayAttrs(
                _ARRAY_DIMENSIONS=("time",),
                standard_name="air_temperature",
                units="s",
                axis="Y",
            ),
        )


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
                        array_dimensions=["time", "lat"],
                        standard_name="air_temperature",
                        grid_mapping=None,
                        grid_mapping_name="latitude_longitude",
                    ),
                ),
                "/time": CoordArray(
                    shape=(10,),
                    dtype="|u1",
                    chunks=(10,),
                    attributes=CoordArrayAttrs(
                        array_dimensions=["time"], standard_name="time", units="s", axis="T"
                    ),
                ),
                "/lat": CoordArray(
                    shape=(11,),
                    dtype="|u1",
                    chunks=(11,),
                    attributes=CoordArrayAttrs(
                        array_dimensions=["lat"], standard_name="latitude", units="m", axis="Y"
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
        group = MyGroupSpec.from_flat(example, by_alias=True)
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
                        standard_name="air_temperature",
                        grid_mapping=None,
                        grid_mapping_name="latitude_longitude",
                    ),
                ),
                "/time": ArraySpec(
                    shape=(10,),
                    dtype="|u1",
                    chunks=(10,),
                    attributes=CoordArrayAttrs(
                        _ARRAY_DIMENSIONS=["time"], standard_name="time", units="s", axis="T"
                    ),
                ),
                "/lat": ArraySpec(
                    shape=(11,),
                    dtype="|u1",
                    chunks=(11,),
                    attributes=CoordArrayAttrs(
                        _ARRAY_DIMENSIONS=["lat"], standard_name="latitude", units="m", axis="Y"
                    ),
                ),
            },
        ],
    )
    @staticmethod
    def test_invalid(example: dict[str, ArraySpec[Any] | GroupSpec[Any, Any]]) -> None:
        """
        Test the check_valid_coordinates function to ensure it validates coordinates correctly.

        This test checks that the function raises a ValueError when the dimensions of the data variable
        do not match the dimensions of the coordinate arrays.
        """
        group = MyGroupSpec[Any, DataArray | CoordArray].from_flat(example, by_alias=True)
        with pytest.raises(ValueError):
            check_valid_coordinates(group)
