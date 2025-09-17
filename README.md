# EOPF GeoZarr

GeoZarr compliant data model for EOPF (Earth Observation Processing Framework) datasets.

Turn EOPF datasets into a GeoZarr-style Zarr v3 store while:
- Preserving native CRS (no forced TMS reprojection)
- Adding CF + GeoZarr compliant metadata
- Building /2 multiscale overviews
- Writing robust, retry-aware band data with validation

## Overview

This library converts EOPF datatrees into GeoZarr-spec 0.4 aligned Zarr v3 stores without forcing web-mercator style tiling. It focuses on scientific fidelity (native CRS), robust metadata (CF + GeoZarr), and operational resilience (retry + completeness auditing) while supporting multiscale /2 overviews.

## Key Features

- **GeoZarr Specification Compliance** (0.4 features implemented)
- **Native CRS Preservation** (UTM, polar, arbitrary projections)
- **Multiscale /2 Overviews** (COG-style hierarchy as child groups)
- **CF Conventions** (`standard_name`, `grid_mapping`, `_ARRAY_DIMENSIONS`)
- **Resilient Writing** (band-by-band with retries & auditing)
- **S3 & S3-Compatible Support** (AWS, OVH, MinIO, custom endpoints)
- **Optional Parallel Processing** (local Dask cluster)
- **Automatic Chunk Alignment** (prevents overlapping Dask/Zarr chunks)
- **HTML Summary & Validation Tools**
- **STAC & Benchmark Commands**
- **Consolidated Metadata** (faster open)

## GeoZarr Compliance Features

- `_ARRAY_DIMENSIONS` attributes on all arrays
- CF standard names for all variables
- `grid_mapping` attributes referencing CF grid_mapping variables
- `GeoTransform` attributes in grid_mapping variables
- Proper multiscales metadata structure
- Native CRS tile matrix sets

## Installation

```bash
pip install eopf-geozarr
```

Development (uv):
```bash
uv sync --frozen
uv run eopf-geozarr --help
```

Editable (pip):
```bash
pip install -e .[dev]
```

## Quick Start

### Command Line Interface

After installation, you can use the `eopf-geozarr` command:

```bash
# Convert EOPF dataset to GeoZarr format (local output)
eopf-geozarr convert input.zarr output.zarr

# Convert specific groups (e.g. resolution groups)
eopf-geozarr convert input.zarr output.zarr --groups /measurements/r10m /measurements/r20m

# Convert EOPF dataset to GeoZarr format (S3 output)
eopf-geozarr convert input.zarr s3://my-bucket/path/to/output.zarr

# Convert with parallel processing using dask cluster
eopf-geozarr convert input.zarr output.zarr --dask-cluster

# Convert with dask cluster and verbose output
eopf-geozarr convert input.zarr output.zarr --dask-cluster --verbose

# Generate an HTML summary while inspecting
eopf-geozarr info output.zarr --html report.html

# Get information about a dataset
eopf-geozarr info input.zarr

# Validate GeoZarr compliance
eopf-geozarr validate output.zarr

# Benchmark access patterns (optional)
eopf-geozarr benchmark output.zarr --samples 8 --window 1024 1024

# Produce draft STAC artifacts
eopf-geozarr stac output.zarr stac_collection.json \
  --bbox "minx miny maxx maxy" --start 2025-01-01T00:00:00Z --end 2025-01-31T23:59:59Z

# Get help
eopf-geozarr --help
```

#### Notes
- Parent groups auto-expand to leaf datasets when omitted.
- Multiscale overviews are generated with /2 coarsening and attached as child groups.
- Defaults: Blosc Zstd (level 3), conservative chunking, metadata consolidation enabled.
- Use `--groups` to limit processing or speed up experimentation.

## S3 Support

Environment vars:
```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=eu-west-1
export AWS_ENDPOINT_URL=https://s3.your-endpoint.example   # optional custom endpoint
```

Write directly:
```bash
eopf-geozarr convert input.zarr s3://my-bucket/path/output_geozarr.zarr --groups /measurements/r10m
```

Features:
- Credential validation before write
- Custom endpoints (OVH, MinIO, etc.)
- Retry logic around object writes

## Parallel Processing with Dask

```bash
eopf-geozarr convert input.zarr out.zarr --dask-cluster --verbose
```
Benefits:
- Local cluster auto-start & cleanup
- Chunk alignment to prevent overlapping writes
- Better memory distribution for large scenes

## Python API

High-level dataset conversion:
```python
import xarray as xr
from eopf_geozarr import create_geozarr_dataset

dt = xr.open_datatree("path/to/eopf.zarr", engine="zarr")
out = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m", "/measurements/r20m"],
    output_path="/tmp/out_geozarr.zarr",
    spatial_chunk=4096,
    min_dimension=256,
    tile_width=256,
)
```

Selective writer usage (advanced):
```python
from eopf_geozarr.conversion.geozarr import GeoZarrWriter
writer = GeoZarrWriter(output_path="/tmp/out.zarr", spatial_chunk=4096)
# writer.write_group(...)
```

## API Reference

`create_geozarr_dataset(dt_input, groups, output_path, spatial_chunk=4096, ...) -> xr.DataTree`
: Produce a GeoZarr-compliant hierarchy.

`setup_datatree_metadata_geozarr_spec_compliant(dt, groups) -> dict[str, xr.Dataset]`
: Apply CF + GeoZarr metadata to selected groups.

`downsample_2d_array(source_data, target_h, target_w) -> np.ndarray`
: Block-average /2 overview generation primitive.

`calculate_aligned_chunk_size(dimension_size, target_chunk_size) -> int`
: Returns evenly dividing chunk to avoid overlap.

## Architecture

```
eopf_geozarr/
  commands/        # CLI subcommands (convert, validate, info, stac, benchmark)
  conversion/      # Core geozarr pipeline, helpers, multiscales, encodings
  metrics.py       # Lightweight metrics hooks (optional)
```

## Contributing to GeoZarr Specification

Upstream issue discussions influenced:
- Arbitrary CRS preservation
- Chunking performance & strategies
- Multiscale hierarchy clarity

## Benchmark & STAC Commands

Benchmark:
```bash
eopf-geozarr benchmark /tmp/out_geozarr.zarr --samples 8 --window 1024 1024
```

STAC draft artifacts:
```bash
eopf-geozarr stac /tmp/out_geozarr.zarr /tmp/collection.json \
  --bbox "minx miny maxx maxy" --start 2025-01-01T00:00:00Z --end 2025-01-31T23:59:59Z
```

## What Gets Written

- `_ARRAY_DIMENSIONS` per variable (deterministic axis order)
- Per-variable `grid_mapping` referencing `spatial_ref`
- Multiscales metadata on parent groups; /2 overviews
- Blosc Zstd compression, conservative chunking
- Consolidated metadata index
- Band attribute propagation across levels

## Consolidated Metadata

Improves open performance. Spec discussion ongoing; toggle by disabling consolidation if strict minimalism required.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Parent group empty | Only leaf groups hold arrays | Use `--groups` or rely on auto-expansion |
| Overlapping chunk error | Misaligned dask vs encoding chunks | Allow auto chunk alignment or reduce spatial_chunk |
| S3 auth failure | Missing env vars or endpoint | Export AWS_* vars / set AWS_ENDPOINT_URL |
| HTML path is a directory | Provided path not file | A default filename is created inside |

## Development & Contributing
Preferred (reproducible) workflow uses [uv](https://github.com/astral-sh/uv):

```bash
git clone <repo-url>
cd eopf-geozarr

# Ensure uv is installed (macOS/Linux quick install)
curl -Ls https://astral.sh/uv/install.sh | sh  # or follow official docs

# Create and sync environment with dev extras
uv sync --extra dev

# Run tools through uv (ensures correct virtual env)
uv run pre-commit install
uv run pytest -q
```

Common tasks:
```bash
uv run ruff check .
uv run mypy src
uv run eopf-geozarr --help
```

Fallback (less reproducible) pip editable install:
```bash
pip install -e '.[dev]'
pre-commit install
pytest
```

Quality stack: Ruff (lint + format), isort (via Ruff), Mypy (strict), Pytest, Coverage.

## License & Acknowledgments

Apache 2.0. Built atop xarray, zarr, dask; follows evolving GeoZarr specification.

---
For questions or issues open a GitHub issue.

