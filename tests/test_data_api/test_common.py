from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v2 import GroupSpec as GroupSpec_V2
from pydantic_zarr.v3 import GroupSpec as GroupSpec_V3

from eopf_geozarr.data_api.geozarr.common import (
    CF_STANDARD_NAME_URL,
    DataArrayLike,
    GroupLike,
    check_standard_name,
    get_cf_standard_names,
)
from eopf_geozarr.data_api.geozarr.v2 import DataArray as DataArray_V2
from eopf_geozarr.data_api.geozarr.v2 import DataArray as DataArray_V3


@pytest.mark.parametrize(
    "obj",
    [
        DataArray_V2.from_array(
            np.arange(10), attributes={"_ARRAY_DIMENSIONS": ("time",)}
        ),
        DataArray_V3.from_array(np.arange(10), dimension_names=("time",)),
    ],
)
def test_datarraylike(obj: DataArray_V2 | DataArray_V3) -> None:
    """
    Test that the DataArrayLike protocol works correctly
    """
    assert isinstance(obj, DataArrayLike)


@pytest.mark.parametrize("obj", [GroupSpec_V2(), GroupSpec_V3()])
def test_grouplike(obj: GroupSpec_V3[Any, Any] | GroupSpec_V2[Any, Any]) -> None:
    """
    Test that the GroupLike protocol works correctly
    """
    assert isinstance(obj, GroupLike)


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


def test_multiscales_round_trip(example_group) -> None:
    """
    Ensure that we can round-trip multiscale metadata through the `Multiscales` model.
    """
    from eopf_geozarr.data_api.geozarr.common import Multiscales

    source_untyped = GroupSpec_V3.from_zarr(example_group)
    flat = source_untyped.to_flat()
    meta = flat["/measurements/reflectance/r60m"].attributes["multiscales"]
    assert Multiscales(**meta).model_dump() == tuplify_json(meta)
