# EOPF GeoZarr

Turn EOPF datasets into a GeoZarr-style Zarr v3 store. Keep the data values intact and add standard geospatial metadata, multiscale overviews, and per-variable dimensions.

## Quick Start

Install (uv):

```bash
uv sync --frozen
uv run eopf-geozarr --help
```

Or pip:

```bash
pip install -e .
```

## Workflows

For Argo / batch orchestration use: https://github.com/EOPF-Explorer/data-model-pipeline

## Convert

Remote → local:

```bash
uv run eopf-geozarr convert \
  "https://.../S2B_MSIL2A_... .zarr" \
  "/tmp/S2B_MSIL2A_..._geozarr.zarr" \
  --groups /measurements/reflectance \
  --verbose
```

Notes:
- Parent groups auto-expand to leaf datasets.
- Overviews use /2 coarsening; multiscales live on parent groups.
- Defaults: Blosc Zstd, conservative chunking, metadata consolidation after write.

## S3

Env for S3/S3-compatible storage:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=eu-west-1
# Custom endpoint (OVH, MinIO, etc.)
export AWS_ENDPOINT_URL=https://s3.your-endpoint.example
```

Write to S3:

```bash
uv run eopf-geozarr convert \
  "https://.../S2B_MSIL2A_... .zarr" \
  "s3://your-bucket/path/S2B_MSIL2A_..._geozarr.zarr" \
  --groups /measurements/reflectance \
  --verbose
```

## Info & Validate

Summary:

```bash
uv run eopf-geozarr info "/tmp/S2B_MSIL2A_..._geozarr.zarr"
```

HTML report:

```bash
uv run eopf-geozarr info "/tmp/S2B_MSIL2A_..._geozarr.zarr" --html /tmp/summary.html
```

Validate (counts only real data vars, skips `spatial_ref`/`crs`):

```bash
uv run eopf-geozarr validate "/tmp/S2B_MSIL2A_..._geozarr.zarr"
```

## Benchmark (optional)

```bash
uv run eopf-geozarr benchmark "/tmp/..._geozarr.zarr" --samples 8 --window 1024 1024
```

## STAC

```bash
uv run eopf-geozarr stac \
  "/tmp/..._geozarr.zarr" \
  "/tmp/..._collection.json" \
  --bbox "minx miny maxx maxy" \
  --start "YYYY-MM-DDTHH:MM:SSZ" \
  --end "YYYY-MM-DDTHH:MM:SSZ"
```

## Python API

```python
from eopf_geozarr.conversion.geozarr import GeoZarrWriter
from eopf_geozarr.validation.validate import validate_store
from eopf_geozarr.info.summary import summarize

src = "https://.../S2B_MSIL2A_... .zarr"
dst = "/tmp/S2B_MSIL2A_..._geozarr.zarr"

writer = GeoZarrWriter(src, dst, storage_options={})
writer.write(groups=["/measurements/reflectance"], verbose=True)

report = validate_store(dst)
print(report.ok)

tree = summarize(dst)
print(tree["summary"])  # or write HTML via CLI
```

## What it writes

- `_ARRAY_DIMENSIONS` per variable (correct axis order).
- `grid_mapping = "spatial_ref"` per variable; `spatial_ref` holds CRS/georeferencing.
- Multiscales on parent groups; /2 overviews.
- Blosc Zstd compression; conservative chunking; consolidated metadata.
- Overviews keep per-band attributes (grid_mapping reattached across levels).

## Consolidated metadata

Speeds up reads. Some tools note it isn’t in the core Zarr v3 spec yet; data stays valid. You can disable consolidation during writes or remove the index if preferred.

## Troubleshooting

- Parent group shows no data vars: select leaves (CLI auto-expands).
- S3 errors: check env vars and `AWS_ENDPOINT_URL` for custom endpoints.
- HTML path is a directory: a default filename is created inside.

