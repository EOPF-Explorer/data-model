from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from zarr import open_group
from zarr.core.buffer import default_buffer_prototype

if TYPE_CHECKING:
    from typing import Any

    from zarr import Group


@pytest.fixture
def example_group() -> Group:
    meta = _load_geozarr_file("sentinel_2.json")
    example_group = open_group(
        store={
            "zarr.json": default_buffer_prototype().buffer.from_bytes(
                json.dumps(meta).encode("utf-8")
            )
        },
        mode="r",
    )
    return example_group


def _load_geozarr_file(filename: str) -> dict[str, Any]:
    """Load an example Geozarr group metadata file from the geozarr_examples directory."""
    examples_dir = Path(__file__).parent / "geozarr_examples"
    file_path = examples_dir / filename
    with open(file_path, "r") as f:
        return json.load(f)


def _load_projjson_file(filename: str) -> dict[str, Any]:
    """Load a PROJ JSON file from the projjson_examples directory."""
    examples_dir = Path(__file__).parent / "projjson_examples"
    file_path = examples_dir / filename
    with open(file_path, "r") as f:
        return json.load(f)


@pytest.fixture
def projected_crs_json() -> dict[str, Any]:
    """Load projected CRS example."""
    return _load_projjson_file("projected_crs.json")


@pytest.fixture
def bound_crs_json() -> dict[str, Any]:
    """Load bound CRS example."""
    return _load_projjson_file("bound_crs.json")


@pytest.fixture
def compound_crs_json() -> dict[str, Any]:
    """Load compound CRS example."""
    return _load_projjson_file("compound_crs.json")


@pytest.fixture
def transformation_json() -> dict[str, Any]:
    """Load transformation example."""
    return _load_projjson_file("transformation.json")


@pytest.fixture
def datum_ensemble_json() -> dict[str, Any]:
    """Load datum ensemble example."""
    return _load_projjson_file("datum_ensemble.json")


@pytest.fixture
def explicit_prime_meridian_json() -> dict[str, Any]:
    """Load explicit prime meridian example."""
    return _load_projjson_file("explicit_prime_meridian.json")


@pytest.fixture
def implicit_prime_meridian_json() -> dict[str, Any]:
    """Load implicit prime meridian example."""
    return _load_projjson_file("implicit_prime_meridian.json")


@pytest.fixture
def all_projjson_examples() -> list[dict[str, Any]]:
    """Load all PROJ JSON examples."""
    examples_dir = Path(__file__).parent / "projjson_examples"
    examples = []
    for json_file in examples_dir.glob("*.json"):
        with open(json_file, "r") as f:
            examples.append(json.load(f))
    return examples
