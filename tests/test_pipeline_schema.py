from __future__ import annotations

import importlib

import pytest
from jsonschema import ValidationError

pipeline_module = importlib.import_module("eopf_geozarr.pipeline")
PAYLOAD_JSON_SCHEMA = pipeline_module.PAYLOAD_JSON_SCHEMA
get_payload_schema = pipeline_module.get_payload_schema
load_example_payload = pipeline_module.load_example_payload
validate_payload = pipeline_module.validate_payload


def test_validate_payload_accepts_bundled_example() -> None:
    payload = load_example_payload("minimal")
    validate_payload(payload)


def test_validate_payload_rejects_missing_required() -> None:
    payload = {"output_zarr": "s3://bucket/out.zarr"}
    with pytest.raises(ValidationError):
        validate_payload(payload)


def test_get_payload_schema_returns_deep_copy() -> None:
    schema = get_payload_schema()
    schema["title"] = "Mutated"
    assert PAYLOAD_JSON_SCHEMA["title"] == "GeoZarrPayload"  # noqa: S101 - pytest assertion


def test_load_example_payload_unknown_name() -> None:
    with pytest.raises(FileNotFoundError):
        load_example_payload("unknown")
