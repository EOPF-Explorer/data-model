# Using the GeoZarr Converter

The GeoZarr converter provides tools to transform EOPF datasets into GeoZarr format. This guide explains how to use the converter effectively.

## Command Line Interface

The converter can be accessed via the `eopf-geozarr` command-line tool. Below are some common use cases:

### Basic Conversion

Convert an EOPF dataset to GeoZarr format:

```bash
eopf-geozarr convert input.zarr output.zarr
```

Sentinel-2 inputs are auto-detected and converted with the optimized flat
multiscale layout (see [Sentinel-2 Optimized Conversion](#sentinel-2-optimized-conversion)).
Pass `--no-s2-optimized` to force the generic per-group conversion path.

### S3 Output

Convert and save the output directly to an S3 bucket:

```bash
eopf-geozarr convert input.zarr s3://my-bucket/output.zarr
```

### Parallel Processing

Enable parallel processing for large datasets using a Dask cluster:

```bash
eopf-geozarr convert input.zarr output.zarr --dask-cluster
```

### Validation

Validate a dataset against the [GeoZarr Mini Spec](geozarr-minispec.md):

```bash
eopf-geozarr validate output.zarr
```

The validator checks the store root (conventions declaration, `spatial:bbox`,
CRS), every multiscale group (layout completeness, per-level georeferencing),
every node using the `proj:` / `spatial:` conventions, and the structural
Dataset rules (no scalar arrays, unique string `dimension_names`, matching
1-D coordinate arrays for every data-variable dimension). Each violation is
reported with its Zarr node path, and the command exits non-zero when the
store is not compliant.

> [!Note]
> **Migration**: earlier releases' `validate` command always exited 0 and only
> warned about per-variable CF attributes. Stores converted with
> eopf-geozarr <= 0.10.x predate the store-root `zarr_conventions`
> requirement and will report as non-compliant until reconverted (or until the
> root metadata is added, e.g. via
> `eopf_geozarr.conversion.utils.write_store_root_geo_metadata`). Pipelines
> that scripted against the old output text or exit code need updating.

## Python API

The converter also provides a Python API for programmatic usage:

### Example: Basic Conversion

```python
# test: skip
import xarray as xr
from eopf_geozarr import create_geozarr_dataset

# Load your EOPF DataTree
dt = xr.open_datatree("path/to/eopf/dataset.zarr", engine="zarr")

# Convert to GeoZarr format
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/reflectance/r10m", "/measurements/reflectance/r20m", "/measurements/reflectance/r60m"],
    output_path="path/to/output/geozarr.zarr",
    spatial_chunk=4096,
    min_dimension=256,
    max_retries=3
)
```

### Example: S3 Output

```python
import os
from eopf_geozarr import create_geozarr_dataset

# Configure S3 credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'your_access_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret_key'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# Convert and save to S3
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/reflectance/r10m", "/measurements/reflectance/r20m", "/measurements/reflectance/r60m"],
    output_path="s3://my-bucket/output.zarr",
    spatial_chunk=4096,
    min_dimension=256,
    max_retries=3
)
```

## Advanced Features

### Chunk Alignment

The converter ensures proper chunk alignment to optimize storage and prevent data corruption. It uses the `calculate_aligned_chunk_size` function to determine optimal chunk sizes.

### Multiscale Support

The converter writes the native resolution arrays at the group root and adds
factor-of-two overviews as sibling subgroups named `r{2**level}` (`r2`, `r4`,
`r8`, ...). Each overview is a complete dataset with its own coordinates and
`spatial:` / `proj:` attributes; the parent group's `multiscales` metadata
records each level via `asset`, `derived_from`, and `transform`.

### Native CRS Preservation

The converter maintains the native coordinate reference system (CRS) of the dataset, avoiding reprojection to Web Mercator.

## Sentinel-2 Optimized Conversion

The Sentinel-2 optimized converter (`convert-s2-optimized` / `convert_s2_optimized`)
builds an efficient multiscale pyramid by **reusing the original multi-resolution
data** (r10m, r20m, r60m) without duplication, and adding coarser overview
levels (r120m, r360m, r720m) for visualization at lower resolutions.

The general `convert` command emits the same flat `r{N}` sibling layout for
any input, but only the S2-optimized command takes advantage of S2's native
multi-resolution structure.

### Layout

Both converters produce a flat pyramid where each level is a sibling group:

```
output.zarr/
└── measurements/
    └── reflectance/
        ├── r10m/           # Native 10m data (reused as-is)
        ├── r20m/           # Native 20m data (reused as-is)
        ├── r60m/           # Native 60m data (reused as-is)
        ├── r120m/          # Computed from r60m (2x downsampling)
        ├── r360m/          # Computed from r120m (3x downsampling)
        └── r720m/          # Computed from r360m (2x downsampling)
```

**Why these specific resolution levels?**

The resolution levels are chosen to balance data preservation with storage optimization:

- **Native ESA resolutions (10m, 20m, 60m)**: These are the original resolutions delivered by ESA for Sentinel-2 data and are reused as-is to preserve the source data without any loss
- **Computed overview levels (120m, 360m, 720m)**: These additional levels were specifically chosen because their downsampling factors allow the data to be chunked and sharded in complete pieces, ensuring:
  - **120m** (2x from 60m): Standard doubling for the first computed overview
  - **360m** (3x from 120m): Selected for optimal chunking alignment
  - **720m** (2x from 360m): Final level for global-scale visualization

This approach maintains the integrity of ESA's original multi-resolution data while adding computationally efficient overview levels for performance at coarser scales.

**Benefits:**
- No data duplication — native resolutions are reused directly
- Efficient storage
- Simple, flat hierarchy
- Natural fit for Sentinel-2's multi-resolution data model

### Key Capabilities

- **Smart Resolution Consolidation**: Combines Sentinel-2's native multi-resolution structure (10m, 20m, 60m) into a unified multiscale pyramid
- **Non-Duplicative Downsampling**: Reuses original resolution data instead of recreating it, adding only the coarser levels (120m, 360m, 720m)
- **Variable-Aware Processing**: Applies appropriate resampling methods for different data types (reflectance, classification, quality masks, probabilities)
- **Efficient Testing**: Improved test infrastructure for faster local development

### Usage Example

```python
from eopf_geozarr.s2_optimization.s2_converter import convert_s2_optimized
import xarray as xr

# Load Sentinel-2 DataTree
dt_input = xr.open_datatree("path/to/s2/product.zarr", engine="zarr")

# Convert to optimized multiscale structure
dt_optimized = convert_s2_optimized(
    dt_input=dt_input,
    output_path="path/to/output/optimized.zarr",
    enable_sharding=True,
    spatial_chunk=256,
    compression_level=3,
    validate_output=True
)
```

The result is a space-efficient multiscale pyramid: `/measurements/reflectance/{r10m, r20m, r60m, r120m, r360m, r720m}` where the native resolutions are preserved as-is and only the coarser levels are computed.

## Sentinel-3 OLCI L1 EFR Conversion

Sentinel-3 OLCI (Ocean and Land Colour Instrument) Level-1 EFR (Full Resolution)
products are supported.  OLCI measurements keep **native swath geometry by
default**: a per-pixel 2-D lat/lon grid, with no reprojection.  Passing
`--output-grid <CRS>` opts into a one-time warp of the swath measurements onto
a regular grid (e.g. `EPSG:4326`) using the swath geolocation arrays.  Either
way, the converter then generates /2 fill-aware block-averaged overview
subgroups for multi-resolution access.

### Auto-detection

The generic `convert` command detects OLCI products automatically:

```bash
eopf-geozarr convert S3A_OL_1_EFR.zarr output.zarr
```

### Dedicated command

A dedicated command offers fine-grained control:

```bash
eopf-geozarr convert-s3-olci-optimized S3A_OL_1_EFR.zarr output.zarr \
    --spatial-chunk 1024 \
    --min-dimension 256 \
    --compression-level 3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--spatial-chunk` | 1024 | Target spatial chunk size in pixels |
| `--compression-level` | 3 | Blosc/zstd compression level (1–9) |
| `--min-dimension` | 256 | Minimum spatial dimension for overview levels |
| `--enable-sharding` | off | Accepted but not yet wired into encoding (follow-up task) |
| `--keep-scale-offset` | off | Accepted but not yet wired into encoding (follow-up task) |
| `--output-grid` | native | `native` preserves instrument geometry; any CRS string (e.g. `EPSG:4326`) warps onto a regular grid |

### Output layout

```
output.zarr/
├── measurements/        # Carries multiscales + spatial: convention metadata
│                        # (+ proj: when --output-grid is a CRS)
│   ├── r0/             # Native-resolution bands (instrument grid by default; regular
│   │                   # y/x grid with spatial_ref when --output-grid is a CRS)
│   ├── r2/             # 1/2-resolution overview
│   ├── r4/             # 1/4-resolution overview
│   └── ...
├── conditions/          # Copied through unmodified (tie-point geometry, meteorology)
└── quality/             # Copied through unmodified (quality flags)
```

Each measurement group carries GeoZarr `spatial:` convention metadata (plus
`proj:` metadata when a CRS is selected).
In native mode (`--output-grid native`, the default) bands keep per-pixel 2-D
`latitude`/`longitude`/`altitude` and per-row `time_stamp`, with no projected CRS.
When `--output-grid <CRS>` is given, bands are warped onto a regular grid with 1-D
`y`/`x` dimension coordinates and a `spatial_ref` variable referenced by
`grid_mapping` on every band; per-scan-line `time_stamp` has no representation on
a regular grid and is dropped (it remains in the source product).

> **Note:** OLCI support is initial/measurements-focused (v1).  Tie-point grid
> groups in `conditions/geometry`, `meteorology`, and `instrument` are copied
> through but not converted to GeoZarr convention.  Encoding wiring for
> `--enable-sharding`, `--spatial-chunk`, `--compression-level`, and
> `--keep-scale-offset` is accepted but scheduled as a follow-up task.

## Error Handling

The converter includes robust error handling and retry logic for network operations, ensuring reliable processing even in challenging environments.

For more details, refer to the [API Reference](api-reference.md).
