# Pipeline Integration Overview

The `eopf-geozarr` package provides the conversion logic that powers the GeoZarr workflow in
`data-model-pipeline`. This guide highlights the contract between both repositories so that developer
workflows, Argo templates, and RabbitMQ messages remain synchronized.

## Shared Responsibilities

| Concern | Provided by `data-model` | Provided by `data-model-pipeline` |
| --- | --- | --- |
| GeoZarr conversion & validation | `eopf_geozarr` conversion engine, CLI, pipeline runner, payload schema | Invokes library within Argo Workflows, publishes AMQP payloads |
| Payload contract | `GeoZarrPayload` dataclass, JSON schema, bundled fixtures | Sensors and tests consume the shared helpers |
| Observability | `--metrics-out` flag routes metrics to local or S3 destinations | Workflows collect and forward metrics to long-term storage |
| STAC registration helpers | `validate_geozarr_store`, pipeline runner hooks | Orchestrates STAC Transactions based on payload flags |

## Command-Line Surfaces

### `eopf-geozarr`

The original CLI remains the most direct way to convert EOPF datasets. It now accepts
`--metrics-out` so you can persist run summaries alongside converted assets. Metrics targets support
both local paths and S3 URIs.

### `eopf-geozarr-pipeline`

The pipeline-specific entrypoint mirrors the RabbitMQ payload processed by production Argo sensors.
It is ideal for replaying payloads locally or verifying template changes:

```bash
# Validate payload flags before triggering the workflow
$ eopf-geozarr-pipeline run --help

# Replay a bundled example payload
$ eopf-geozarr-pipeline run --payload-file <(python - <<'PY'
from eopf_geozarr.pipeline import load_example_payload
import json
print(json.dumps(load_example_payload("minimal")))
PY
)
```

Both CLIs normalize group names, default to the Sentinel-2 reflectance groups, and respect the shared
payload schema described below.

## Python Helpers

The `eopf_geozarr.pipeline` package exposes helpers that keep repositories aligned:

```python
from eopf_geozarr.pipeline import (
    GeoZarrPayload,
    PAYLOAD_JSON_SCHEMA,
    get_payload_schema,
    load_example_payload,
    run_pipeline,
    validate_payload,
)

payload = GeoZarrPayload.from_payload(load_example_payload("full"))
payload.ensure_required()
validate_payload(payload.to_payload())
print(PAYLOAD_JSON_SCHEMA["required"])  # ["src_item", "output_zarr"]
```

- `GeoZarrPayload` parses CLI arguments or RabbitMQ payloads and produces normalized values.
- `PAYLOAD_JSON_SCHEMA` and `get_payload_schema()` deliver a canonical JSON schema for validation in
  tests or runtime checks.
- `load_example_payload()` exposes fixtures that mirror the messages published by the AMQP tooling.
- `validate_payload()` wraps `jsonschema` with the library-managed schema, ensuring the same rules
  apply everywhere.

## Bundled Fixtures

Two JSON fixtures live under `eopf_geozarr/pipeline/resources`:

- `payload-minimal.json` represents the baseline message with only required fields.
- `payload-full.json` exercises optional knobs such as STAC registration and metrics targets.

Use these fixtures to seed integration tests in `data-model-pipeline` or to document payload
expectations in other repositories. They are also accessible at runtime via `load_example_payload()`.

## Next Steps

- `data-model-pipeline` should import the schema helpers when validating AMQP payloads and update its
  tests to rely on the shared fixtures.
- `platform-deploy` can reference the same schema when templating new WorkflowTemplates or Flux
  overlays, ensuring environment values stay in sync.
- Future payload changes should originate in this repository so all downstream consumers inherit the
  update automatically.
