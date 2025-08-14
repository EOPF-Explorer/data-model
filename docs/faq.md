# Frequently Asked Questions

Common questions and solutions for using the EOPF GeoZarr library.

## General Questions

### What is EOPF GeoZarr?

EOPF GeoZarr is a Python library that converts EOPF (Earth Observation Processing Framework) datasets to GeoZarr-spec 0.4 compliant format. It maintains scientific accuracy while optimizing for cloud-native workflows and performance.

### What makes this different from standard Zarr?

GeoZarr is a specification that extends Zarr with geospatial metadata standards. Our library specifically:

- Ensures GeoZarr 0.4 specification compliance
- Preserves native coordinate reference systems
- Creates multiscale pyramids for efficient visualization
- Maintains CF conventions for scientific interoperability
- Optimizes chunking for Earth observation data patterns

### Which satellite missions are supported?

Currently, the library is optimized for:

- **Sentinel-2** (L1C and L2A products)
- **Sentinel-1** (planned support)

The architecture is designed to support additional missions with minimal modifications.

## Installation and Setup

### Why do I need Python 3.11 or higher?

The library uses modern Python features and depends on recent versions of scientific libraries (xarray, zarr, dask) that require Python 3.11+.

### Can I use conda instead of pip?

While the library is primarily distributed via PyPI, you can install it in a conda environment:

```bash
conda create -n eopf-geozarr python=3.11
conda activate eopf-geozarr
pip install eopf-geozarr
```

### How do I set up AWS credentials?

Multiple options are available:

```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Option 3: IAM roles (for EC2/ECS)
# No setup needed - automatic detection
```

## Usage Questions

### How do I know which groups to convert?

Inspect your EOPF dataset first:

```python
import xarray as xr

dt = xr.open_datatree("input.zarr", engine="zarr")
print(dt)  # Shows the full structure

# Common Sentinel-2 groups:
groups = [
    "/measurements/r10m",  # 10m bands: B02, B03, B04, B08
    "/measurements/r20m",  # 20m bands: B05, B06, B07, B8A, B11, B12
    "/measurements/r60m"   # 60m bands: B01, B09, B10
]
```

### What chunk size should I use?

The optimal chunk size depends on your data and use case:

```python
from eopf_geozarr.conversion.utils import calculate_aligned_chunk_size

# For typical Sentinel-2 data (10980x10980)
optimal_chunk = calculate_aligned_chunk_size(10980, 4096)
print(optimal_chunk)  # Returns 3660

# General guidelines:
# - 4096: Good default for most cases
# - 2048: Better for memory-constrained environments
# - 8192: For high-memory systems and large datasets
```

### How do I process very large datasets?

Use Dask for distributed processing:

```python
from dask.distributed import Client

# Start Dask cluster
client = Client('scheduler-address:8786')

# Use smaller chunks for distributed processing
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m"],
    output_path="output.zarr",
    spatial_chunk=2048  # Smaller chunks work better with Dask
)
```

### Can I convert only specific bands?

Currently, the library processes all bands within a group. To process specific bands, you would need to create a subset of your input dataset first:

```python
# Create subset with specific bands
dt_subset = dt.copy()
ds_10m = dt_subset["/measurements/r10m"].ds
ds_subset = ds_10m[["b02", "b03", "b04"]]  # Only RGB bands
dt_subset["/measurements/r10m"].ds = ds_subset

# Then convert
dt_geozarr = create_geozarr_dataset(
    dt_input=dt_subset,
    groups=["/measurements/r10m"],
    output_path="rgb_only.zarr"
)
```

## Error Troubleshooting

### "ImportError: No module named 'eopf_geozarr'"

**Cause**: Library not installed or wrong Python environment.

**Solutions**:

```bash
# Verify installation
pip list | grep eopf-geozarr

# Reinstall if missing
pip install eopf-geozarr

# Check Python environment
which python
python --version
```

### "ValueError: Invalid groups specified"

**Cause**: Specified groups don't exist in the input dataset.

**Solution**:

```python
# Check available groups
dt = xr.open_datatree("input.zarr", engine="zarr")
print("Available groups:", list(dt.groups))

# Use correct group paths
groups = [g for g in dt.groups if "measurements" in g]
```

### "MemoryError" during conversion

**Cause**: Dataset too large for available memory.

**Solutions**:

```python
# 1. Use smaller chunks
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m"],
    output_path="output.zarr",
    spatial_chunk=1024  # Smaller chunks
)

# 2. Use Dask for out-of-core processing
from dask.distributed import Client
client = Client()

# 3. Process groups one at a time
for group in ["/measurements/r10m", "/measurements/r20m"]:
    dt_geozarr = create_geozarr_dataset(
        dt_input=dt,
        groups=[group],
        output_path=f"output_{group.split('/')[-1]}.zarr"
    )
```

### "PermissionError" with S3

**Cause**: Insufficient S3 permissions or incorrect credentials.

**Solutions**:

```python
# 1. Verify credentials
from eopf_geozarr.conversion.fs_utils import get_s3_credentials_info
print(get_s3_credentials_info())

# 2. Test S3 access
from eopf_geozarr.conversion.fs_utils import validate_s3_access
is_valid, error = validate_s3_access("s3://your-bucket/test.zarr")
print(f"Valid: {is_valid}, Error: {error}")

# 3. Check IAM permissions (need s3:GetObject, s3:PutObject, s3:ListBucket)
```

### "zarr.errors.ArrayNotFoundError"

**Cause**: Corrupted or incomplete Zarr dataset.

**Solutions**:

```python
# 1. Validate input dataset
try:
    dt = xr.open_datatree("input.zarr", engine="zarr")
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Dataset error: {e}")

# 2. Check for missing arrays
import zarr
store = zarr.open("input.zarr", mode="r")
print("Available arrays:", list(store.array_keys()))

# 3. Consolidate metadata if needed
zarr.consolidate_metadata("input.zarr")
```

### "CRS not found" or projection errors

**Cause**: Missing or invalid coordinate reference system information.

**Solutions**:

```python
# Check CRS information
ds = dt["/measurements/r10m"].ds
print("CRS variables:", [v for v in ds.data_vars if 'crs' in v.lower() or 'spatial_ref' in v])

# Check coordinate attributes
print("X coord attrs:", ds.x.attrs)
print("Y coord attrs:", ds.y.attrs)

# Verify rioxarray can read CRS
import rioxarray
try:
    crs = ds.rio.crs
    print(f"CRS: {crs}")
except Exception as e:
    print(f"CRS error: {e}")
```

## Performance Questions

### Why is conversion slow?

Several factors affect performance:

1. **Chunk size**: Too small = many operations, too large = memory issues
2. **Network**: S3 operations depend on bandwidth and latency
3. **CPU**: Overview generation is CPU-intensive
4. **Memory**: Insufficient RAM causes swapping

**Optimization strategies**:

```python
# 1. Optimal chunking
chunk_size = calculate_aligned_chunk_size(data_width, 4096)

# 2. Use Dask for parallelization
from dask.distributed import Client
client = Client(n_workers=4, threads_per_worker=2)

# 3. Process in batches
for group in groups:
    # Process one group at a time
    pass

# 4. Use SSD storage for temporary files
import tempfile
import os
os.environ['TMPDIR'] = '/path/to/fast/storage'
```

### How can I monitor progress?

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Or use the CLI with verbose flag
# eopf-geozarr convert input.zarr output.zarr --verbose
```

### What's the expected output size?

GeoZarr datasets are typically larger than input due to:

- Multiscale overviews (adds ~33% for 2 overview levels)
- Additional metadata
- Chunk alignment padding

**Estimation**:

```python
# Rough estimate: input_size * 1.4 (with 2 overview levels)
# For Sentinel-2 10m band: ~400MB input → ~560MB GeoZarr
```

## Cloud Storage Questions

### Which cloud providers are supported?

- **AWS S3**: Full support
- **S3-compatible**: MinIO, DigitalOcean Spaces, etc.
- **Google Cloud Storage**: Via S3 compatibility mode
- **Azure Blob Storage**: Via S3 compatibility (limited)

### How do I optimize S3 performance?

```python
# 1. Use appropriate region
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'  # Close to your data

# 2. Configure multipart uploads
s3_config = {
    'config_kwargs': {
        'max_pool_connections': 50,
        'multipart_threshold': 64 * 1024 * 1024,  # 64MB
        'multipart_chunksize': 16 * 1024 * 1024   # 16MB
    }
}

# 3. Use VPC endpoints for EC2 instances
# 4. Consider S3 Transfer Acceleration for global access
```

### Can I use different storage for input and output?

Yes, the library supports mixed storage:

```python
# Local input, S3 output
dt = xr.open_datatree("local_input.zarr", engine="zarr")
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m"],
    output_path="s3://bucket/output.zarr"
)

# S3 input, local output
dt = xr.open_datatree("s3://bucket/input.zarr", engine="zarr")
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m"],
    output_path="local_output.zarr"
)
```

## Validation and Quality

### How do I verify the conversion worked correctly?

```python
# 1. Use built-in validation
from eopf_geozarr.cli import validate_command
import argparse

args = argparse.Namespace()
args.input_path = "output.zarr"
args.verbose = True
validate_command(args)

# 2. Manual checks
dt = xr.open_datatree("output.zarr", engine="zarr")

# Check multiscales metadata
print("Multiscales:", dt.attrs.get('multiscales', 'Missing'))

# Check overview levels
for level in ['0', '1', '2']:
    path = f"/measurements/r10m/{level}"
    if path in dt.groups:
        ds = dt[path].ds
        print(f"Level {level}: {dict(ds.dims)}")

# Check required attributes
ds = dt["/measurements/r10m/0"].ds
for var_name in ds.data_vars:
    var = ds[var_name]
    print(f"{var_name}: grid_mapping={var.attrs.get('grid_mapping', 'Missing')}")
```

### What should I do if validation fails?

1. **Check the error message** - it usually indicates the specific issue
2. **Verify input data** - ensure the EOPF dataset is complete and valid
3. **Check dependencies** - ensure all required libraries are up to date
4. **Try with verbose logging** - get more detailed error information
5. **Report issues** - if it seems like a bug, please report it

### How do I compare input and output data?

```python
# Load both datasets
dt_input = xr.open_datatree("input.zarr", engine="zarr")
dt_output = xr.open_datatree("output.zarr", engine="zarr")

# Compare native resolution data
ds_input = dt_input["/measurements/r10m"].ds
ds_output = dt_output["/measurements/r10m/0"].ds

# Check data values (should be identical)
import numpy as np
for band in ["b02", "b03", "b04"]:
    if band in ds_input and band in ds_output:
        diff = np.abs(ds_input[band].values - ds_output[band].values)
        print(f"{band} max difference: {diff.max()}")
        # Should be 0 or very close to 0
```

## Integration Questions

### Can I use this with STAC?

Yes, you can create STAC items for GeoZarr datasets. See the [Examples](examples.md#stac-integration) section for detailed code.

### How does this work with Jupyter notebooks?

The library works well in Jupyter environments. See [Examples](examples.md#jupyter-notebook-integration) for interactive visualization patterns.

### Can I integrate this into my processing pipeline?

Absolutely! The library is designed for integration:

```python
# Example pipeline integration
def process_sentinel2_scene(input_path: str, output_path: str):
    """Process a single Sentinel-2 scene to GeoZarr."""
    try:
        dt = xr.open_datatree(input_path, engine="zarr")
        dt_geozarr = create_geozarr_dataset(
            dt_input=dt,
            groups=["/measurements/r10m", "/measurements/r20m"],
            output_path=output_path,
            spatial_chunk=4096
        )
        return True, "Success"
    except Exception as e:
        return False, str(e)

# Use in batch processing
results = []
for scene in scene_list:
    success, message = process_sentinel2_scene(scene.input, scene.output)
    results.append((scene.id, success, message))
```

## Getting Help

### Where can I get more help?

1. **Documentation**: Check the [User Guide](converter.md) and [API Reference](api-reference.md)
2. **Examples**: See [Examples](examples.md) for common use cases
3. **GitHub Issues**: Report bugs or request features at the [GitHub repository](https://github.com/eopf-explorer/data-model/issues)
4. **Community**: Join discussions in the GeoZarr community

### How do I report a bug?

When reporting issues, please include:

1. **Version information**:

   ```bash
   eopf-geozarr --version
   python --version
   pip list | grep -E "(xarray|zarr|dask)"
   ```

2. **Error message**: Full traceback if available

3. **Minimal example**: Code that reproduces the issue

4. **Environment**: OS, Python version, installation method

5. **Data information**: Dataset type, size, structure (if shareable)

### How can I contribute?

Contributions are welcome! See the project repository for contribution guidelines. Areas where help is needed:

- Additional satellite mission support
- Performance optimizations
- Documentation improvements
- Test coverage expansion
- Bug fixes and feature enhancements
