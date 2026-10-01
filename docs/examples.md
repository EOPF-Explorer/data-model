---
title: Examples
description: Examples for eopf-geozarr, the GeoZarr driver for EOPF CPM — convert SAFE and EOPF Zarr Sentinel products to GeoZarr, write to S3, process with Dask and add STAC metadata.
---

# Examples

## GeoZarr driver for EOPF CPM

Convert native products with CPM (see the [CPM driver guide](cpm-driver.md)):

```bash
# Sentinel-2 L2A SAFE to GeoZarr, with sharding
eopf convert-geozarr S2B_MSIL2A_….SAFE out.zarr --enable-sharding

# Sentinel-3 OLCI EFR on a regular WGS 84 grid
eopf convert-geozarr S3A_OL_1_EFR____….SEN3 out.zarr --output-grid EPSG:4326

# Sentinel-2 to S3, packed with the Zarr scale-offset codecs
eopf convert-geozarr S2B_MSIL2A_….SAFE s3://bucket/out.zarr \
    --stage-output --scale-offset-codec
```

```python
# test: skip (needs eopf-cpm and source products)
from pathlib import Path

import eopf_geozarr.cpm.writer  # registers the "geozarr" engine
from eopf.store.convert import convert

# Convert every SAFE product in a folder
for safe in sorted(Path("inputs").glob("S2*_MSIL2A_*.SAFE")):
    convert(
        str(safe),
        f"outputs/{safe.stem}.zarr",
        target_store_kwargs={"engine": "geozarr", "enable_sharding": True},
    )
```

## Standalone converter

### Command line

```bash
# Detects Sentinel-2 and Sentinel-3 OLCI and selects the optimized pipeline
eopf-geozarr convert input.zarr output.zarr

# Write to S3
eopf-geozarr convert input.zarr s3://my-bucket/output.zarr

# Validate the result
eopf-geozarr validate output.zarr
```

### Generic pipeline for selected groups

`convert` and `convert_s2_optimized` are the recommended paths for
Sentinel-2. The generic `create_geozarr_dataset` converts only the groups you
list, with factor-of-two overviews:

```python
import xarray as xr
from eopf_geozarr import create_geozarr_dataset

# Load Sentinel-2 L2A dataset
dt = xr.open_datatree("S2A_MSIL2A_20230615T103031_N0509_R108_T32TQM_20230615T170304.zarr", 
                      engine="zarr")

# Convert all resolution groups
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=[
        "/measurements/reflectance/r10m",  # B02, B03, B04, B08
        "/measurements/reflectance/r20m",  # B05, B06, B07, B8A, B11, B12
        "/measurements/reflectance/r60m"   # B01, B09, B10
    ],
    output_path="s2_l2a_geozarr.zarr",
    spatial_chunk=4096,
    min_dimension=256
)

# Inspect the result
print(f"Groups created: {list(dt_geozarr.groups)}")
for group_name in dt_geozarr.groups:
    group = dt_geozarr[group_name]
    if hasattr(group, 'ds') and group.ds is not None:
        print(f"{group_name}: {dict(group.ds.dims)}")
```

### Sentinel-2 Band Analysis

Access bands from the consolidated pyramid structure produced by
`convert_s2_optimized`:

```python
# test: skip (needs a Sentinel-2 product)
import xarray as xr
import matplotlib.pyplot as plt
from eopf_geozarr.s2_optimization.s2_converter import convert_s2_optimized

# Convert using the S2-optimized converter
dt_input = xr.open_datatree("s2_l2a_input.zarr", engine="zarr", chunks={})
dt = convert_s2_optimized(
    dt_input,
    output_path="s2_l2a.zarr",
    enable_sharding=True,
    spatial_chunk=256,
    compression_level=3,
    validate_output=False,
)

# Access data from different resolution levels
ds_10m = dt["/measurements/reflectance/r10m"].ds   # Native 10m
ds_20m = dt["/measurements/reflectance/r20m"].ds   # Native 20m
ds_60m = dt["/measurements/reflectance/r60m"].ds   # Native 60m
ds_120m = dt["/measurements/reflectance/r120m"].ds # Computed 120m

# Extract RGB bands for visualization (10m resolution)
red = ds_10m["b04"]    # Red band
green = ds_10m["b03"]  # Green band  
blue = ds_10m["b02"]   # Blue band

# Create RGB composite
rgb = xr.concat([red, green, blue], dim="band")

# Plot comparison of different resolutions
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 10m resolution
rgb.plot.imshow(ax=axes[0], robust=True)
axes[0].set_title("10m Resolution (Native)")

# 20m resolution (reused native data)
rgb_20m = xr.concat([ds_20m["b04"], ds_20m["b03"], ds_20m["b02"]], dim="band")
rgb_20m.plot.imshow(ax=axes[1], robust=True)
axes[1].set_title("20m Resolution (Native)")

# 60m resolution (reused native data)
rgb_60m = xr.concat([ds_60m["b04"], ds_60m["b03"], ds_60m["b02"]], dim="band")
rgb_60m.plot.imshow(ax=axes[2], robust=True)
axes[2].set_title("60m Resolution (Native)")

plt.tight_layout()
plt.show()
```

## Cloud Storage Examples

### AWS S3 Integration

Complete workflow with S3 input and output:

```python
import os
import xarray as xr
from eopf_geozarr import create_geozarr_dataset
from eopf_geozarr.conversion.fs_utils import validate_s3_access

# Configure AWS credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'your_access_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret_key'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# Define paths
input_path = "s3://sentinel-data/input.zarr"
output_path = "s3://processed-data/output.zarr"

# Validate S3 access
is_valid, error = validate_s3_access(output_path)
if not is_valid:
    raise RuntimeError(f"S3 access validation failed: {error}")

# Load from S3
dt = xr.open_datatree(input_path, engine="zarr")

# Convert and save to S3
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/reflectance/r10m", "/measurements/reflectance/r20m"],
    output_path=output_path,
    spatial_chunk=2048
)

print(f"Successfully converted and saved to {output_path}")
```

### S3 with Custom Credentials

Using custom S3 credentials and endpoint:

```python
from eopf_geozarr import create_geozarr_dataset
from eopf_geozarr.conversion.fs_utils import get_s3_storage_options

# Custom S3 configuration
s3_config = {
    'key': 'custom_access_key',
    'secret': 'custom_secret_key',
    'endpoint_url': 'https://s3.custom-provider.com',
    'region_name': 'eu-west-1'
}

# Get storage options
storage_opts = get_s3_storage_options("s3://custom-bucket/output.zarr", **s3_config)

# Convert with custom S3 settings
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/reflectance/r10m"],
    output_path="s3://custom-bucket/output.zarr",
    **storage_opts
)
```

## Performance Optimization Examples

### Large Dataset Processing with Dask

Process large datasets efficiently using Dask:

```python
import xarray as xr
from dask.distributed import Client, LocalCluster
from eopf_geozarr import create_geozarr_dataset

# Set up Dask cluster
cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='4GB')
client = Client(cluster)

try:
    # Load large dataset
    dt = xr.open_datatree("large_sentinel2.zarr", engine="zarr")
    
    # Process with optimized chunking for Dask
    dt_geozarr = create_geozarr_dataset(
        dt_input=dt,
        groups=["/measurements/reflectance/r10m", "/measurements/reflectance/r20m", "/measurements/reflectance/r60m"],
        output_path="large_geozarr.zarr",
        spatial_chunk=2048,  # Smaller chunks for distributed processing
        max_retries=5
    )
    
    print("Large dataset processing completed!")
    
finally:
    client.close()
    cluster.close()
```

### Memory-Efficient Processing

Process datasets with limited memory:

```python
from eopf_geozarr import create_geozarr_dataset
from eopf_geozarr.conversion.utils import calculate_aligned_chunk_size

# Calculate memory-efficient chunk size
data_width, data_height = 10980, 10980
memory_limit_mb = 512  # 512 MB limit

# Estimate chunk size for memory constraint
# Assuming float32 data (4 bytes per pixel)
pixels_per_mb = (1024 * 1024) // 4
target_chunk = int((pixels_per_mb * memory_limit_mb) ** 0.5)

# Align chunk size with data dimensions
optimal_chunk = calculate_aligned_chunk_size(data_width, target_chunk)

print(f"Using chunk size: {optimal_chunk}")

# Process with memory-efficient settings
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/reflectance/r10m"],
    output_path="memory_efficient.zarr",
    spatial_chunk=optimal_chunk
)
```

## Advanced Use Cases

### Custom Metadata Enhancement

Add custom metadata to the converted dataset:

```python
import xarray as xr

import eopf_geozarr
from eopf_geozarr import create_geozarr_dataset

# Convert dataset
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/reflectance/r10m"],
    output_path="enhanced.zarr"
)

# Add custom metadata
dt_geozarr.attrs.update({
    'processing_date': '2024-01-15',
    'processing_software': f'eopf-geozarr {eopf_geozarr.__version__}',
    'custom_parameter': 'value'
})

# Add group-specific metadata
for group_name in dt_geozarr.groups:
    group = dt_geozarr[group_name]
    if hasattr(group, 'ds') and group.ds is not None:
        group.ds.attrs['processing_level'] = 'L2A_GeoZarr'

# Save enhanced metadata
dt_geozarr.to_zarr("enhanced.zarr", mode="a")
```

### Validation and Quality Control

Comprehensive validation workflow:

```python
import xarray as xr
from eopf_geozarr import create_geozarr_dataset
from eopf_geozarr.conversion.utils import validate_existing_band_data

# Convert dataset
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/reflectance/r10m"],
    output_path="validated.zarr"
)

# Validate the conversion
dt_check = xr.open_datatree("validated.zarr", engine="zarr")

# Check multiscales metadata
multiscales = dt_check.attrs.get('multiscales', [])
print(f"Multiscales levels: {len(multiscales)}")

# Validate each resolution level
for level in ["0", "1", "2"]:
    group_path = f"/measurements/reflectance/r10m/{level}"
    if group_path in dt_check.groups:
        ds = dt_check[group_path].ds
        print(f"Level {level}: {dict(ds.dims)}")
        
        # Check required attributes
        for var_name in ds.data_vars:
            var = ds[var_name]
            has_dims = '_ARRAY_DIMENSIONS' in var.attrs
            has_grid_mapping = 'grid_mapping' in var.attrs
            print(f"  {var_name}: dims={has_dims}, grid_mapping={has_grid_mapping}")

# Validate CRS information
for group_name in dt_check.groups:
    group = dt_check[group_name]
    if hasattr(group, 'ds') and group.ds is not None:
        crs_vars = [v for v in group.ds.data_vars if 'spatial_ref' in v or 'crs' in v]
        print(f"{group_name} CRS variables: {crs_vars}")
```

### Batch Processing

Process multiple datasets in batch:

```python
import os
from pathlib import Path
from eopf_geozarr import create_geozarr_dataset
import xarray as xr

def batch_convert_datasets(input_dir: str, output_dir: str, groups: list):
    """Convert multiple EOPF datasets to GeoZarr format."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find all .zarr directories
    zarr_files = list(input_path.glob("*.zarr"))
    
    for zarr_file in zarr_files:
        try:
            print(f"Processing {zarr_file.name}...")
            
            # Load dataset
            dt = xr.open_datatree(str(zarr_file), engine="zarr")
            
            # Convert to GeoZarr
            output_file = output_path / f"{zarr_file.stem}_geozarr.zarr"
            dt_geozarr = create_geozarr_dataset(
                dt_input=dt,
                groups=groups,
                output_path=str(output_file),
                spatial_chunk=4096
            )
            
            print(f"✓ Completed {zarr_file.name}")
            
        except Exception as e:
            print(f"✗ Failed {zarr_file.name}: {e}")

# Usage
batch_convert_datasets(
    input_dir="/data/sentinel2/raw",
    output_dir="/data/sentinel2/geozarr",
    groups=["/measurements/reflectance/r10m", "/measurements/reflectance/r20m"]
)
```

## Integration examples

### STAC metadata

For CPM products, the Sentinel-2 pipeline copies the product's
`stac_discovery` metadata to the store root and adds a `reflectance` asset that
points to the multiscale group:

```python
# test: skip (needs a converted store)
import zarr

root = zarr.open_group("out.zarr", mode="r")
asset = root.attrs["stac_discovery"]["assets"]["reflectance"]
print(asset["href"], asset["type"])
#> /measurements/reflectance application/vnd.zarr; version=3; profile=multiscales
```

For Sentinel-1 GRD RTC stores, build a complete STAC item:

```python
# test: skip (needs a consolidated S1 GRD RTC store)
from eopf_geozarr.stac.s1_rtc import build_s1_rtc_stac_item

item = build_s1_rtc_stac_item("s1-rtc.zarr", collection_id="sentinel-1-grd-rtc")
print(item.to_dict()["properties"]["datetime"])
```

### Exploring the pyramid in a notebook

```python
# test: skip (needs a converted store)
import matplotlib.pyplot as plt
import xarray as xr

dt = xr.open_datatree("out.zarr", engine="zarr")
reflectance = dt["measurements/reflectance"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, level in zip(axes, ["r60m", "r120m", "r720m"]):
    reflectance[level].ds["b04"].plot(ax=ax, robust=True, cmap="gray")
    ax.set_title(f"b04 at {level}")
plt.show()
```
