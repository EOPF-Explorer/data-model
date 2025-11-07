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
    ProjAttrs,
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


def test_projattrs_crs_required() -> None:
    """
    Test that the ProjAttrs model raises a ValueError if none of the CRS fields are specified.
    """
    with pytest.raises(
        ValueError, match="One of 'code', 'wkt2', or 'projjson' must be provided."
    ):
        ProjAttrs()


def test_projattrs_json_examples(
    proj_attrs_examples: dict[tuple[int, int], dict[str, Any]],
) -> None:
    """
    Test that proj attributes in the JSON examples of the proj extension README are valid.
    """
    proj_examples_found: int = 0

    for _, json_block in proj_attrs_examples.items():
        # Check if this JSON block contains geo.proj attributes
        if "attributes" in json_block and isinstance(json_block["attributes"], dict):
            geo: Any = json_block["attributes"].get("geo")
            if geo and isinstance(geo, dict) and "proj" in geo:
                proj_examples_found += 1
                proj_data: dict[str, Any] = geo["proj"]

                # Validate that ProjAttrs can parse this data
                proj_attrs: ProjAttrs = ProjAttrs(**proj_data)

                # Verify that all fields from the original data are present in the model
                for key, value in proj_data.items():
                    if value is not None:
                        model_value: Any = getattr(proj_attrs, key)
                        # Handle tuple/list comparison for transform and bbox fields
                        if isinstance(value, list) and isinstance(model_value, tuple):
                            assert tuple(value) == model_value, f"Field {key} mismatch"
                        else:
                            assert model_value == value, f"Field {key} mismatch"

                # Verify that the model satisfies the CRS requirement
                assert (
                    proj_attrs.code is not None
                    or proj_attrs.wkt2 is not None
                    or proj_attrs.projjson is not None
                ), "At least one CRS field must be present"

    # Ensure we found and tested at least some examples
    assert proj_examples_found >= 4, (
        f"Expected at least 4 proj examples in README, found {proj_examples_found}"
    )
