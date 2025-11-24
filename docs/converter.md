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

## Error Handling

The converter includes robust error handling and retry logic for network operations, ensuring reliable processing even in challenging environments.

For more details, refer to the [API Reference](api-reference.md).
