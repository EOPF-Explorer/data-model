# Using the GeoZarr Converter

The GeoZarr converter provides tools to transform EOPF datasets into GeoZarr-spec 0.4 compliant format. This guide explains how to use the converter effectively.

## Command Line Interface

The converter can be accessed via the `eopf-geozarr` command-line tool. Below are some common use cases:

### Basic Conversion

Convert an EOPF dataset to GeoZarr format:

```bash
eopf-geozarr convert input.zarr output.zarr
```

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

Validate the GeoZarr compliance of a dataset:

```bash
eopf-geozarr validate output.zarr
```

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
    groups=["/measurements/r10m", "/measurements/r20m", "/measurements/r60m"],
    output_path="path/to/output/geozarr.zarr",
    spatial_chunk=4096,
    min_dimension=256,
    tile_width=256,
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
    groups=["/measurements/r10m", "/measurements/r20m", "/measurements/r60m"],
    output_path="s3://my-bucket/output.zarr",
    spatial_chunk=4096,
    min_dimension=256,
    tile_width=256,
    max_retries=3
)
```

## Advanced Features

### Chunk Alignment

The converter ensures proper chunk alignment to optimize storage and prevent data corruption. It uses the `calculate_aligned_chunk_size` function to determine optimal chunk sizes.

### Multiscale Support

The converter supports multiscale datasets, creating overview levels with /2 downsampling logic. Each level is stored as a sibling group (e.g., `/0`, `/1`, `/2`).

### Native CRS Preservation

The converter maintains the native coordinate reference system (CRS) of the dataset, avoiding reprojection to Web Mercator.

## Sentinel-2 Optimized Conversion (V1)

The Sentinel-2 optimized converter (`convert_s2_optimized`) represents the refined approach (V1) that creates an efficient multiscale pyramid by **reusing the original multi-resolution data** (r10m, r20m, r60m) without duplication, and adding coarser overview levels (r120m, r360m, r720m) for efficient visualization at lower resolutions.

### V0 vs V1 Converter: Key Differences

Understanding the structural differences between the old (V0) and new (V1) converter approaches:

#### V0 Approach (Deprecated - `create_geozarr_dataset`)

Creates **pyramids within each resolution group**:

```
output.zarr/
└── measurements/
    └── reflectance/
        ├── r10m/
        │   ├── 0/          # Native 10m data
        │   ├── 1/          # Downsampled to 20m
        │   ├── 2/          # Downsampled to 40m
        │   ├── 3/          # Downsampled to 80m
        │   ├── 4/          # Downsampled to 160m
        │   └── 5/          # Downsampled to 320m
        ├── r20m/
        │   ├── 0/          # Native 20m data
        │   ├── 1/          # Downsampled to 40m
        │   ├── 2/          # Downsampled to 80m
        │   ├── 3/          # Downsampled to 160m
        │   └── 4/          # Downsampled to 320m
        └── r60m/
            ├── 0/          # Native 60m data
            ├── 1/          # Downsampled to 120m
            └── 2/          # Downsampled to 240m
```

**Issues with V0:**
- Creates redundant data at overlapping resolutions (e.g., r10m/1 ≈ r20m/0)
- Inefficient storage due to duplication
- Complex hierarchy with nested levels within each resolution group

#### V1 Approach (Current - `convert_s2_optimized`)

Creates a **consolidated pyramid** by reusing native resolutions and adding coarser levels:

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

**Benefits of V1:**
- No data duplication - native resolutions are reused directly
- More efficient storage
- Simpler, flatter hierarchy
- Natural fit for Sentinel-2's multi-resolution data model

### Key Capabilities

- **Smart Resolution Consolidation**: Combines Sentinel-2's native multi-resolution structure (10m, 20m, 60m) into a unified multiscale pyramid
- **Non-Duplicative Downsampling**: Reuses original resolution data instead of recreating it, adding only the coarser levels (120m, 360m, 720m)
- **Variable-Aware Processing**: Applies appropriate resampling methods for different data types (reflectance, classification, quality masks, probabilities)
- **Efficient Testing**: Improved test infrastructure for faster local development

> **Note:** The V0 converter (`create_geozarr_dataset`) is deprecated and will be removed in future versions. All new projects should use the V1 converter (`convert_s2_optimized`).

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

## Error Handling

The converter includes robust error handling and retry logic for network operations, ensuring reliable processing even in challenging environments.

For more details, refer to the [API Reference](api-reference.md).
