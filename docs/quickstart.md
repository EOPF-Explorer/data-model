# Quick Start

Get up and running with EOPF GeoZarr in minutes. This guide shows you how to convert your first EOPF dataset to GeoZarr format.

## Prerequisites

- EOPF GeoZarr library installed ([Installation Guide](installation.md))
- An EOPF dataset in Zarr format
- Basic familiarity with Python and command-line tools

## Your First Conversion

### Command Line (Simplest)

Convert an EOPF dataset to GeoZarr format:

```bash
eopf-geozarr convert input.zarr output.zarr
```

That's it! The converter will:
- Analyze your EOPF dataset structure
- Apply GeoZarr 0.4 specification compliance
- Create multiscale overviews
- Preserve native CRS and scientific accuracy

### Python API (More Control)

For programmatic usage with custom parameters:

```python
import xarray as xr
from eopf_geozarr import create_geozarr_dataset

# Load your EOPF DataTree
dt = xr.open_datatree("input.zarr", engine="zarr")

# Convert to GeoZarr
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m", "/measurements/r20m", "/measurements/r60m"],
    output_path="output.zarr",
    spatial_chunk=4096,
    min_dimension=256
)

print("Conversion complete!")
```

## Working with Cloud Storage

### S3 Output

Save directly to AWS S3:

```bash
# Set credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Convert to S3
eopf-geozarr convert input.zarr s3://my-bucket/output.zarr
```

### S3 Input and Output

```python
# Both input and output on S3
dt_geozarr = create_geozarr_dataset(
    dt_input=xr.open_datatree("s3://input-bucket/data.zarr", engine="zarr"),
    groups=["/measurements/r10m"],
    output_path="s3://output-bucket/geozarr.zarr"
)
```

## Validation

Verify your GeoZarr dataset meets the specification:

```bash
eopf-geozarr validate output.zarr
```

Or in Python:

```python
from eopf_geozarr.cli import validate_command
import argparse

# Create args object
args = argparse.Namespace()
args.input_path = "output.zarr"
args.verbose = True

validate_command(args)
```

## Inspecting Results

### Dataset Information

Get detailed information about your converted dataset:

```bash
eopf-geozarr info output.zarr
```

### Python Inspection

```python
import xarray as xr

# Open the converted dataset
dt = xr.open_datatree("output.zarr", engine="zarr")

# Explore the structure
print(dt)

# Check multiscales metadata
print(dt.attrs.get('multiscales', 'No multiscales found'))

# Examine a specific resolution level
ds_native = dt["/measurements/r10m/0"].ds  # Native resolution
ds_overview = dt["/measurements/r10m/1"].ds  # First overview

print(f"Native shape: {ds_native.dims}")
print(f"Overview shape: {ds_overview.dims}")
```

## Common Patterns

### Sentinel-2 Data

For Sentinel-2 L2A data with standard resolution groups:

```python
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=[
        "/measurements/r10m",  # 10m bands (B02, B03, B04, B08)
        "/measurements/r20m",  # 20m bands (B05, B06, B07, B8A, B11, B12)
        "/measurements/r60m"   # 60m bands (B01, B09, B10)
    ],
    output_path="s2_geozarr.zarr",
    spatial_chunk=4096
)
```

### Large Datasets with Dask

For processing large datasets efficiently:

```bash
eopf-geozarr convert large_input.zarr output.zarr --dask-cluster
```

Or in Python:

```python
from dask.distributed import Client

# Start Dask client
client = Client('scheduler-address:8786')  # Or Client() for local

# Process with Dask
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m"],
    output_path="output.zarr",
    spatial_chunk=2048  # Smaller chunks for distributed processing
)

client.close()
```

## Key Features Demonstrated

Your converted dataset now includes:

✅ **GeoZarr 0.4 Compliance** - Full specification adherence  
✅ **Native CRS Preservation** - No unnecessary reprojection  
✅ **Multiscale Pyramids** - Efficient overview levels  
✅ **Optimized Chunking** - Aligned chunks for performance  
✅ **CF Conventions** - Standard metadata attributes  
✅ **Cloud-Ready** - S3 and other cloud storage support  

## Next Steps

- **Detailed Usage**: See the [User Guide](converter.md) for advanced options
- **API Reference**: Explore the [API Reference](api-reference.md) for all functions
- **Examples**: Check out [Examples](examples.md) for specific use cases
- **Architecture**: Understand the [Architecture](architecture.md) behind the conversion

## Troubleshooting Quick Fixes

**Memory errors with large datasets?**
```bash
eopf-geozarr convert input.zarr output.zarr --spatial-chunk 2048
```

**S3 permission errors?**
```bash
aws sts get-caller-identity  # Verify credentials
```

**Validation failures?**
```bash
eopf-geozarr validate output.zarr --verbose  # Get detailed error info
```

For more troubleshooting help, see the [FAQ](faq.md).
