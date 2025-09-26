# ruff: noqa: S101

from __future__ import annotations

import importlib

import pytest

pipeline_module = importlib.import_module("eopf_geozarr.pipeline")
GeoZarrPayload = pipeline_module.GeoZarrPayload


def test_from_payload_normalizes_groups_and_aliases() -> None:
    payload = {
        "src_item": "https://example.com/items/foo",
        "output_zarr": "s3://bucket/out.zarr",
        "groups": "measurements/r10m,measurements/r20m",
        "crs_groups": "/conditions/geometry",
        "register_collection": "sentinel-2-l2a",
        "register_mode": "replace",
        "dask_cluster": True,
        "verbose": False,
    }

    model = GeoZarrPayload.from_payload(payload)

    assert model.groups == (
        "/measurements/reflectance/r10m",
        "/measurements/reflectance/r20m",
    )
    assert model.crs_groups == ("/conditions/geometry",)
    assert model.register_mode == "replace"
    assert model.dask_cluster is True
    assert model.verbose is False

    serialized = model.to_payload()
    assert serialized["groups"] == (
        "/measurements/reflectance/r10m,/measurements/reflectance/r20m"
    )
    assert serialized["crs_groups"] == "/conditions/geometry"
    # Alias keeps backwards compatible key for routing
    assert serialized["register_collection"] == "sentinel-2-l2a"
    assert serialized["collection"] == "sentinel-2-l2a"
    # False is dropped to keep payload lean
    assert "verbose" not in serialized


def test_to_payload_includes_numeric_and_boolean_flags() -> None:
    model = GeoZarrPayload.from_cli(
        src_item="https://example.com/items/bar",
        output_zarr="s3://bucket/out.zarr",
        groups="/measurements/reflectance/r60m",
        spatial_chunk=2048,
        max_retries=5,
        dask_cluster=True,
        verbose=True,
    )

    serialized = model.to_payload()

    assert serialized["groups"] == "/measurements/reflectance/r60m"
    assert serialized["spatial_chunk"] == 2048
    assert serialized["max_retries"] == 5
    assert serialized["dask_cluster"] is True
    assert serialized["verbose"] is True


def test_scaling_fields_roundtrip() -> None:
    payload = {
        "src_item": "https://example.com/items/scaling",
        "output_zarr": "s3://bucket/scaling.zarr",
        "scaling_strategy": "v1-queue",
        "worker_replicas_min": 4,
        "worker_replicas_max": 16,
        "queue_depth_target": 8,
    }

    model = GeoZarrPayload.from_payload(payload)

    assert model.scaling_strategy == "v1-queue"
    assert model.worker_replicas_min == 4
    assert model.worker_replicas_max == 16
    assert model.queue_depth_target == 8

    serialized = model.to_payload()
    assert serialized["scaling_strategy"] == "v1-queue"
    assert serialized["worker_replicas_min"] == 4
    assert serialized["worker_replicas_max"] == 16
    assert serialized["queue_depth_target"] == 8


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
