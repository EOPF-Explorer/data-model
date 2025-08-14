# Examples

Practical examples demonstrating common use cases for the EOPF GeoZarr library.

## Basic Examples

### Simple Local Conversion

Convert a local EOPF dataset to GeoZarr format:

```python
import xarray as xr
from eopf_geozarr import create_geozarr_dataset

# Load EOPF dataset
dt = xr.open_datatree("sentinel2_l2a.zarr", engine="zarr")

# Convert to GeoZarr
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m"],
    output_path="sentinel2_geozarr.zarr"
)

print("Conversion completed successfully!")
```

### Command Line Conversion

```bash
# Basic conversion
eopf-geozarr convert input.zarr output.zarr

# With custom chunk size
eopf-geozarr convert input.zarr output.zarr --spatial-chunk 2048

# Validate the result
eopf-geozarr validate output.zarr
```

## Sentinel-2 Examples

### Multi-Resolution Sentinel-2 Processing

Process all resolution groups from a Sentinel-2 L2A dataset:

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
        "/measurements/r10m",  # B02, B03, B04, B08
        "/measurements/r20m",  # B05, B06, B07, B8A, B11, B12
        "/measurements/r60m"   # B01, B09, B10
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

Access and analyze specific bands from the converted dataset:

```python
import xarray as xr
import matplotlib.pyplot as plt

# Open converted GeoZarr dataset
dt = xr.open_datatree("s2_l2a_geozarr.zarr", engine="zarr")

# Access 10m resolution native data
ds_10m = dt["/measurements/r10m/0"].ds

# Extract RGB bands for visualization
red = ds_10m["b04"]    # Red band
green = ds_10m["b03"]  # Green band  
blue = ds_10m["b02"]   # Blue band

# Create RGB composite
rgb = xr.concat([red, green, blue], dim="band")

# Plot the result
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Native resolution
rgb.plot.imshow(ax=ax1, robust=True)
ax1.set_title("Native Resolution (10m)")

# Overview level 1
ds_overview = dt["/measurements/r10m/1"].ds
rgb_overview = xr.concat([ds_overview["b04"], ds_overview["b03"], ds_overview["b02"]], dim="band")
rgb_overview.plot.imshow(ax=ax2, robust=True)
ax2.set_title("Overview Level 1 (20m)")

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
    groups=["/measurements/r10m", "/measurements/r20m"],
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
    groups=["/measurements/r10m"],
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
        groups=["/measurements/r10m", "/measurements/r20m", "/measurements/r60m"],
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
    groups=["/measurements/r10m"],
    output_path="memory_efficient.zarr",
    spatial_chunk=optimal_chunk
)
```

## Advanced Use Cases

### Custom Metadata Enhancement

Add custom metadata to the converted dataset:

```python
import xarray as xr
from eopf_geozarr import create_geozarr_dataset

# Convert dataset
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m"],
    output_path="enhanced.zarr"
)

# Add custom metadata
dt_geozarr.attrs.update({
    'processing_date': '2024-01-15',
    'processing_software': 'eopf-geozarr v0.1.0',
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
    groups=["/measurements/r10m"],
    output_path="validated.zarr"
)

# Validate the conversion
dt_check = xr.open_datatree("validated.zarr", engine="zarr")

# Check multiscales metadata
multiscales = dt_check.attrs.get('multiscales', [])
print(f"Multiscales levels: {len(multiscales)}")

# Validate each resolution level
for level in ["0", "1", "2"]:
    group_path = f"/measurements/r10m/{level}"
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
    groups=["/measurements/r10m", "/measurements/r20m"]
)
```

## Integration Examples

### STAC Integration

Create STAC items for converted GeoZarr datasets:

```python
import json
from datetime import datetime
import xarray as xr
from eopf_geozarr import create_geozarr_dataset

# Convert dataset
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m"],
    output_path="stac_ready.zarr"
)

# Extract metadata for STAC
ds = dt_geozarr["/measurements/r10m/0"].ds
spatial_ref = ds.get('spatial_ref', ds.get('crs', None))

# Create basic STAC item
stac_item = {
    "stac_version": "1.0.0",
    "type": "Feature",
    "id": "sentinel2_geozarr_example",
    "properties": {
        "datetime": datetime.now().isoformat(),
        "platform": "sentinel-2",
        "instruments": ["msi"],
        "processing:level": "L2A",
        "processing:software": "eopf-geozarr"
    },
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            # Extract from dataset bounds
            [float(ds.x.min()), float(ds.y.min())],
            [float(ds.x.max()), float(ds.y.min())],
            [float(ds.x.max()), float(ds.y.max())],
            [float(ds.x.min()), float(ds.y.max())],
            [float(ds.x.min()), float(ds.y.min())]
        ]]
    },
    "assets": {
        "geozarr": {
            "href": "stac_ready.zarr",
            "type": "application/vnd+zarr",
            "roles": ["data"],
            "title": "GeoZarr Dataset"
        }
    }
}

# Save STAC item
with open("stac_item.json", "w") as f:
    json.dump(stac_item, f, indent=2)
```

### Jupyter Notebook Integration

Interactive exploration in Jupyter:

```python
# Cell 1: Setup and conversion
import xarray as xr
import matplotlib.pyplot as plt
from eopf_geozarr import create_geozarr_dataset

dt = xr.open_datatree("input.zarr", engine="zarr")
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m"],
    output_path="notebook_example.zarr"
)

# Cell 2: Interactive visualization
%matplotlib widget
import ipywidgets as widgets

def plot_band(band_name, level):
    ds = dt_geozarr[f"/measurements/r10m/{level}"].ds
    band_data = ds[band_name]
    
    plt.figure(figsize=(10, 8))
    band_data.plot(robust=True, cmap='viridis')
    plt.title(f"{band_name} - Level {level}")
    plt.show()

# Create interactive widgets
band_widget = widgets.Dropdown(
    options=['b02', 'b03', 'b04', 'b08'],
    value='b04',
    description='Band:'
)

level_widget = widgets.Dropdown(
    options=['0', '1', '2'],
    value='0',
    description='Level:'
)

widgets.interact(plot_band, band_name=band_widget, level=level_widget)
```

These examples demonstrate the flexibility and power of the EOPF GeoZarr library across various use cases, from simple conversions to complex cloud-based workflows.
