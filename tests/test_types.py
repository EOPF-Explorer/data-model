"""Tests for the validating constructors in eopf_geozarr.types.

``make_crs_code`` / ``make_bounding_box`` replace unchecked ``cast()`` reads of
zarr store attributes, so corrupt attrs must fail loudly with a clear message.
"""

from __future__ import annotations

import pytest

from eopf_geozarr.types import make_bounding_box, make_crs_code, make_epsg_code

# =============================================================================
# make_crs_code
# =============================================================================


def test_crs_code_accepts_epsg_string() -> None:
    assert make_crs_code("EPSG:32631") == "EPSG:32631"


@pytest.mark.parametrize("bad", [32631, None, ["EPSG:32631"], b"EPSG:32631"])
def test_crs_code_rejects_non_string(bad: object) -> None:
    with pytest.raises(TypeError, match="CRS code"):
        make_crs_code(bad)


@pytest.mark.parametrize("bad", ["", "   "])
def test_crs_code_rejects_empty_or_whitespace(bad: str) -> None:
    with pytest.raises(TypeError, match="non-empty"):
        make_crs_code(bad)


# =============================================================================
# make_epsg_code
# =============================================================================


def test_epsg_code_accepts_int() -> None:
    assert make_epsg_code(32631) == 32631


@pytest.mark.parametrize("value", ["32631", "EPSG:32631", "  EPSG:32631  "])
def test_epsg_code_accepts_string_forms(value: str) -> None:
    # Stored attrs and CPM metadata carry EPSG codes as "32631" or "EPSG:32631".
    code = make_epsg_code(value)
    assert code == 32631
    assert isinstance(code, int)


@pytest.mark.parametrize("bad", [None, True, 32631.0, ["EPSG:32631"], "EPSG:", "EPSG:abc", ""])
def test_epsg_code_rejects_non_codes(bad: object) -> None:
    with pytest.raises(TypeError, match="EPSG code"):
        make_epsg_code(bad)


# =============================================================================
# make_bounding_box
# =============================================================================


def test_bounding_box_from_list() -> None:
    bbox = make_bounding_box([300000.0, 4900000.0, 400000.0, 5000000.0])
    assert bbox == (300000.0, 4900000.0, 400000.0, 5000000.0)


def test_bounding_box_from_tuple_and_ints() -> None:
    # zarr attrs serialize tuples to JSON arrays and read back lists; ints coerce to float.
    bbox = make_bounding_box((300000, 4900000, 400000.5, 5000000))
    assert bbox == (300000.0, 4900000.0, 400000.5, 5000000.0)
    assert all(isinstance(v, float) for v in bbox)


@pytest.mark.parametrize("bad", [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0], []])
def test_bounding_box_rejects_wrong_length(bad: list[float]) -> None:
    with pytest.raises(ValueError, match="exactly 4"):
        make_bounding_box(bad)


@pytest.mark.parametrize("bad", ["EPSG:4326", None, 4.0, {"xmin": 0.0}])
def test_bounding_box_rejects_non_sequence(bad: object) -> None:
    with pytest.raises(TypeError, match="sequence"):
        make_bounding_box(bad)


@pytest.mark.parametrize("bad_element", ["300000", None, True])
def test_bounding_box_rejects_non_numeric_elements(bad_element: object) -> None:
    with pytest.raises(TypeError, match="numbers"):
        make_bounding_box([bad_element, 4900000.0, 400000.0, 5000000.0])
