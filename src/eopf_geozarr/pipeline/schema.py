"""Shared RabbitMQ payload schema and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from importlib import resources
from json import dumps, load, loads
from typing import Any, cast

from jsonschema import Draft202012Validator

PAYLOAD_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "GeoZarrPayload",
    "type": "object",
    "additionalProperties": False,
    "required": ["src_item", "output_zarr"],
    "properties": {
        "src_item": {
            "type": "string",
            "format": "uri",
            "minLength": 1,
            "description": "Upstream STAC item URL that seeded the conversion workflow.",
        },
        "output_zarr": {
            "type": "string",
            "minLength": 1,
            "description": "Destination GeoZarr store (local path or s3:// URI).",
        },
        "groups": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Comma-separated measurement groups to convert (e.g. "
                "measurements/reflectance/r10m)."
            ),
            "default": (
                "/measurements/reflectance/r10m,/measurements/reflectance/r20m,"
                "/measurements/reflectance/r60m"
            ),
        },
        "crs_groups": {
            "type": "string",
            "description": (
                "Optional comma-separated groups that should receive CRS metadata "
                "augmentation."
            ),
        },
        "register_url": {
            "type": "string",
            "format": "uri",
            "description": "STAC Transactions endpoint for registration.",
        },
        "register_collection": {
            "type": "string",
            "description": "STAC collection identifier used during registration.",
        },
        "collection": {
            "type": "string",
            "description": (
                "Backwards-compatible alias for register_collection used by legacy payloads."
            ),
        },
        "register_bearer_token": {
            "type": "string",
            "description": "Bearer token applied when posting to the STAC Transactions API.",
        },
        "register_href": {
            "type": "string",
            "format": "uri",
            "description": "Optional href that should be added to the registered STAC item.",
        },
        "register_mode": {
            "type": "string",
            "description": "Registration strategy overriding the default create-or-skip behaviour.",
            "default": "create-or-skip",
        },
        "id_policy": {
            "type": "string",
            "description": "Identifier policy for derived STAC items (for example src or hash).",
            "default": "src",
        },
        "s3_endpoint": {
            "type": "string",
            "format": "uri",
            "description": "Override for the S3-compatible endpoint (OVH, MinIO, etc.).",
        },
        "s3_region": {
            "type": "string",
            "description": "Region passed to S3-compatible clients (defaults to us-east-1).",
            "default": "us-east-1",
        },
        "aws_addressing_style": {
            "type": "string",
            "description": "Addressing style forwarded to boto3 (path or virtual).",
            "default": "path",
        },
        "max_retries": {
            "type": "integer",
            "minimum": 0,
            "description": "Maximum retries for network operations inside the converter.",
            "default": 3,
        },
        "overwrite": {
            "type": "string",
            "description": (
                "Behaviour when the destination Zarr already exists (replace, skip, etc.)."
            ),
            "default": "replace",
        },
        "metrics_out": {
            "type": "string",
            "description": "Optional path where conversion metrics should be written.",
        },
        "spatial_chunk": {
            "type": "integer",
            "minimum": 64,
            "description": "Spatial chunk size used during conversion.",
            "default": 4096,
        },
        "min_dimension": {
            "type": "integer",
            "minimum": 64,
            "description": "Smallest dimension retained when deriving overview levels.",
            "default": 256,
        },
        "tile_width": {
            "type": "integer",
            "minimum": 64,
            "description": "Tile width for generated multiscale pyramids.",
            "default": 256,
        },
        "dask_cluster": {
            "type": "boolean",
            "description": "Whether the workflow should launch a transient Dask cluster.",
            "default": False,
        },
        "verbose": {
            "type": "boolean",
            "description": "Enable verbose logging inside the converter.",
            "default": True,
        },
        "aws_session_token": {
            "type": "string",
            "description": (
                "Session token forwarded to AWS-compatible clients (temporary "
                "credentials)."
            ),
        },
        "collection_thumbnail": {
            "type": "string",
            "format": "uri",
            "description": "Thumbnail URL injected into the registered STAC collection or item.",
        },
        "owner": {
            "type": "string",
            "description": "Optional owner metadata stored for observability.",
        },
        "service_account": {
            "type": "string",
            "description": "Downstream service account identifier for audit logs.",
        },
        "scaling_strategy": {
            "type": "string",
            "description": "Scaling phase requested by the workflow (v0-hpa or v1-queue).",
            "enum": ["v0-hpa", "v1-queue"],
            "default": "v0-hpa",
        },
        "worker_replicas_min": {
            "type": "integer",
            "minimum": 1,
            "description": "Minimum number of converter workers to maintain when scaling.",
            "default": 2,
        },
        "worker_replicas_max": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum number of converter workers allowed by scaling policies.",
            "default": 10,
        },
        "queue_depth_target": {
            "type": "integer",
            "minimum": 1,
            "description": "Desired messages-per-worker threshold for the queue-driven scaler.",
            "default": 5,
        },
    },
}


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(PAYLOAD_JSON_SCHEMA)


def get_payload_schema() -> dict[str, Any]:
    """Return a deep copy of the payload JSON schema."""
    return cast(dict[str, Any], loads(dumps(PAYLOAD_JSON_SCHEMA)))


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Validate ``payload`` against :data:`PAYLOAD_JSON_SCHEMA`.

    Raises ``jsonschema.ValidationError`` on failure.
    """
    _validator().validate(dict(payload))


def load_example_payload(name: str = "minimal") -> dict[str, Any]:
    """Load an example RabbitMQ payload bundled with the package."""

    filename = f"payload-{name}.json"
    resource = resources.files("eopf_geozarr.pipeline.resources").joinpath(filename)
    if not resource.is_file():
        raise FileNotFoundError(f"No bundled payload fixture named '{name}'")
    with resource.open("r", encoding="utf-8") as handle:
        return cast(dict[str, Any], load(handle))
