from __future__ import annotations

import jsondiff
import pytest
from pydantic_zarr.core import tuplify_json

from eopf_geozarr.data_api.geozarr.geoproj import GeoProj
from tests.test_data_api.conftest import GEOPROJ_EXAMPLES, view_json_diff


@pytest.mark.parametrize("json_example", GEOPROJ_EXAMPLES.items(), ids=lambda v: v[0])
def test_geoproj_roundtrip(json_example: tuple[str, dict[str, object]]) -> None:
    _, value = json_example
    value_tup = tuplify_json(value)
    attrs_json = value_tup["attributes"]
    model = GeoProj(**attrs_json)
    observed = model.model_dump()
    expected = attrs_json
    assert jsondiff.diff(expected, observed) == {}, view_json_diff(expected, observed)
