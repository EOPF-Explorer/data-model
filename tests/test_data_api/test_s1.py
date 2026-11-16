"""
Round-trip tests for Sentinel-1 pydantic-zarr integrated models.

These tests verify that Sentinel-1 data can be:
1. Loaded from example JSON data using direct instantiation
2. Validated through Pydantic models
3. Round-tripped without data loss

Note: Documentation code examples are tested separately via pytest-examples
from the markdown files in docs/models/sentinel1.md
"""

import inspect
from typing import Any

import pytest
from pydantic_zarr.v2 import GroupSpec

import eopf_geozarr.data_api.s1 as s1_module
from eopf_geozarr.data_api.s1 import Sentinel1PolarizationGroup, Sentinel1Root


def test_sentinel1_roundtrip(s1_json_example: dict[str, object]) -> None:
    """Test that we can round-trip JSON data without loss"""
    model1 = Sentinel1Root.model_validate(s1_json_example)
    dumped = model1.model_dump()
    model2 = Sentinel1Root.model_validate(dumped)
    assert model1.model_dump() == model2.model_dump()


def _member_accessor_cases() -> list[Any]:
    """Collect every generated member-accessor property on the S1 group models.

    The S1 group classes define uniform properties of the shape
    ``self.members.get(key)`` + ``raise KeyError(key)`` when absent; identify
    them by their source so unrelated properties are not swept in.
    """
    cases: list[Any] = []
    for cls_name, cls in inspect.getmembers(s1_module, inspect.isclass):
        if cls.__module__ != s1_module.__name__ or not issubclass(cls, GroupSpec):
            continue
        for prop_name, prop in vars(cls).items():
            if (
                isinstance(prop, property)
                and prop.fget is not None
                and "raise KeyError" in inspect.getsource(prop.fget)
            ):
                cases.append(pytest.param(cls, prop_name, id=f"{cls_name}.{prop_name}"))
    return cases


@pytest.mark.parametrize(("cls", "prop_name"), _member_accessor_cases())
def test_member_accessor(cls: type[GroupSpec[Any, Any]], prop_name: str) -> None:
    """Member accessors return the member when present and raise KeyError when absent."""
    empty = cls.model_construct(members={})
    with pytest.raises(KeyError) as excinfo:
        getattr(empty, prop_name)
    member_key = excinfo.value.args[0]

    sentinel = object()
    populated = cls.model_construct(members={member_key: sentinel})
    assert getattr(populated, prop_name) is sentinel


def test_polarization_group_helpers(s1_json_example: dict[str, object]) -> None:
    """The root model's polarization lookups find the VH and VV groups."""
    model = Sentinel1Root.model_validate(s1_json_example)

    pols = model.get_polarization_groups()
    assert pols
    assert all(isinstance(group, Sentinel1PolarizationGroup) for group in pols.values())

    vh = model.get_vh_group()
    assert vh is not None
    assert any("VH" in name for name in pols)

    vv = model.get_vv_group()
    assert vv is not None
    assert any("VV" in name for name in pols)


def test_polarization_group_helpers_empty() -> None:
    """The polarization lookups return None / empty on a root with no members."""
    empty = Sentinel1Root.model_construct(members={})
    assert empty.get_polarization_groups() == {}
    assert empty.get_vh_group() is None
    assert empty.get_vv_group() is None
