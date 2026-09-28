from __future__ import annotations

import numpy as np
import pytest
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v2 import AnyGroupSpec as AnyGroupSpec_V2
from pydantic_zarr.v2 import GroupSpec as GroupSpec_V2
from pydantic_zarr.v3 import AnyGroupSpec as AnyGroupSpec_V3
from pydantic_zarr.v3 import GroupSpec as GroupSpec_V3

from eopf_geozarr.data_api.geozarr.common import (
    CF_STANDARD_NAME_URL,
    DataArrayLike,
    GroupLike,
    ProjAttrs,
    check_standard_name,
    get_cf_standard_names,
)
from eopf_geozarr.data_api.geozarr.multiscales.zcm import (
    Multiscales as ZCMMultiscales,
)
from eopf_geozarr.data_api.geozarr.v2 import DataArray as DataArray_V2
from eopf_geozarr.data_api.geozarr.v2 import DataArray as DataArray_V3


@pytest.mark.parametrize(
    "obj",
    [
        DataArray_V2.from_array(np.arange(10), attributes={"_ARRAY_DIMENSIONS": ("time",)}),
        DataArray_V3.from_array(np.arange(10), dimension_names=("time",)),
    ],
)
def test_datarraylike(obj: object) -> None:
    """
    Test that the DataArrayLike protocol works correctly
    """
    assert isinstance(obj, DataArrayLike)


@pytest.mark.parametrize("obj", [GroupSpec_V2(attributes={}), GroupSpec_V3(attributes={})])
def test_grouplike(obj: AnyGroupSpec_V3 | AnyGroupSpec_V2) -> None:
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
    with pytest.raises(ValueError, match=r"Invalid standard name.*not found in the list"):
        check_standard_name("invalid_standard_name")


def test_multiscales_round_trip() -> None:
    """
    removing dependcy from GroupSpec_V3 model

    Ensure that we can round-trip multiscale metadata through the `Multiscales` model.

    Round-trip fidelity is a property of the model's field definitions, not
    of any particular converted product, so a minimal hand-built `layout`
    (one native level, one derived overview) is enough to exercise it —
    `ScaleLevel` only requires `asset`, and `Multiscales` only requires
    `layout`.
    """
    meta: dict[str, object] = {
        "layout": (
            {"asset": "."},
            {"asset": "r2", "derived_from": ".", "resampling_method": "average"},
        ),
        "resampling_method": "average",
    }
    # pull out the multiscales keys, ignore extra
    submodel = tuplify_json({k: meta[k] for k in ZCMMultiscales.model_fields if k in meta})
    assert ZCMMultiscales(**submodel).model_dump() == submodel


def test_projattrs_crs_required() -> None:
    """
    Test that the ProjAttrs model raises a ValueError if none of the CRS fields are specified.
    """
    with pytest.raises(
        ValueError, match=r"One of 'code', 'wkt2', or 'projjson' must be provided\."
    ):
        ProjAttrs()  # pyright: ignore[reportCallIssue]  # no-args construction tests the validation error


def test_projattrs_json_examples(
    proj_attrs_examples: dict[tuple[int, int], dict[str, object]],
) -> None:
    """
    Test that proj attributes in the JSON examples of the proj extension README are valid.
    """
    proj_examples_found: int = 0

    for json_block in proj_attrs_examples.values():
        # Check if this JSON block contains geo.proj attributes
        attributes = json_block.get("attributes")
        if isinstance(attributes, dict):
            geo: object = attributes.get("geo")
            if geo and isinstance(geo, dict) and "proj" in geo:
                proj_examples_found += 1
                proj_data_obj: object = geo["proj"]
                assert isinstance(proj_data_obj, dict)
                proj_data: dict[str, object] = proj_data_obj

                # Validate that ProjAttrs can parse this data
                proj_attrs: ProjAttrs = ProjAttrs.model_validate(proj_data)

                # Verify that all fields from the original data are present in the model
                for key, value in proj_data.items():
                    if value is not None:
                        model_value: object = getattr(proj_attrs, key)
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
