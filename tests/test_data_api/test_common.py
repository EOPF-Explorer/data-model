from __future__ import annotations

import pytest
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import GroupSpec as GroupSpec_V3

from eopf_geozarr.data_api.geozarr.common import (
    CF_STANDARD_NAME_URL,
    check_standard_name,
    get_cf_standard_names,
)

from .conftest import example_group


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


def test_multiscales_round_trip() -> None:
    """
    Ensure that we can round-trip multiscale metadata through the `Multiscales` model.
    """
    from eopf_geozarr.data_api.geozarr.common import Multiscales

    source_untyped = GroupSpec_V3.from_zarr(example_group)
    flat = source_untyped.to_flat()
    meta = flat["/measurements/reflectance/r60m"].attributes["multiscales"]
    assert Multiscales(**meta).model_dump() == tuplify_json(meta)
