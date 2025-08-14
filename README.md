# EOPF GeoZarr

GeoZarr compliant data model for EOPF (Earth Observation Processing Framework) datasets.

## Overview

This library provides tools to convert EOPF datasets to GeoZarr-spec 0.4 compliant format while maintaining native projections and using /2 downsampling logic for multiscale support.

## Key Features

- **GeoZarr Specification Compliance**: Full compliance with GeoZarr spec 0.4
- **Native CRS Preservation**: No reprojection to TMS, maintains original coordinate reference systems
- **Multiscale Support**: COG-style /2 downsampling with overview levels as children groups
- **CF Conventions**: Proper CF standard names and grid_mapping attributes
- **Robust Processing**: Band-by-band writing with validation and retry logic
- **S3 Support**: Direct output to Amazon S3 buckets with automatic credential validation
- **Parallel Processing**: Optional dask cluster support for parallel chunk processing
- **Chunk Alignment**: Automatic chunk alignment to prevent data corruption with dask

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

For development:

```bash
git clone <repository-url>
cd eopf-geozarr
pip install -e ".[dev]"
```

## Quick Start

### Command Line Interface

After installation, you can use the `eopf-geozarr` command:

```bash
# Convert EOPF dataset to GeoZarr format (local output)
eopf-geozarr convert input.zarr output.zarr

# Convert EOPF dataset to GeoZarr format (S3 output)
eopf-geozarr convert input.zarr s3://my-bucket/path/to/output.zarr

# Convert with parallel processing using dask cluster
eopf-geozarr convert input.zarr output.zarr --dask-cluster

# Convert with dask cluster and verbose output
eopf-geozarr convert input.zarr output.zarr --dask-cluster --verbose

# Get information about a dataset
eopf-geozarr info input.zarr

# Validate GeoZarr compliance
eopf-geozarr validate output.zarr

# Get help
eopf-geozarr --help
```

### S3 Support

The library supports direct output to S3-compatible storage, including custom providers like OVH Cloud. Simply provide an S3 URL as the output path:

```bash
# Convert to S3
eopf-geozarr convert local_input.zarr s3://my-bucket/geozarr-data/output.zarr --verbose
```

#### S3 Configuration

Before using S3 output, ensure your S3 credentials are configured:

**For AWS S3:**

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

**For OVH Cloud Object Storage:**

```bash
export AWS_ACCESS_KEY_ID=your_ovh_access_key
export AWS_SECRET_ACCESS_KEY=your_ovh_secret_key
export AWS_DEFAULT_REGION=gra  # or other OVH region
export AWS_ENDPOINT_URL=https://s3.gra.cloud.ovh.net  # OVH endpoint
```

**For other S3-compatible providers:**

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=your_region
export AWS_ENDPOINT_URL=https://your-s3-endpoint.com
```

**Alternative: AWS CLI Configuration**

```bash
aws configure
# Note: For custom endpoints, you'll still need to set AWS_ENDPOINT_URL
```

#### S3 Features

- **Custom Endpoints**: Support for any S3-compatible storage (AWS, OVH Cloud, MinIO, etc.)
- **Automatic Validation**: The tool validates S3 access before starting conversion
- **Credential Detection**: Automatically detects and validates S3 credentials
- **Error Handling**: Provides helpful error messages for S3 configuration issues
- **Performance**: Optimized for S3 with proper chunking and retry logic

### Parallel Processing with Dask

The library supports parallel processing using dask clusters for improved performance on large datasets:

```bash
# Enable dask cluster for parallel processing
eopf-geozarr convert input.zarr output.zarr --dask-cluster

# With verbose output to see cluster information
eopf-geozarr convert input.zarr output.zarr --dask-cluster --verbose
```

#### Dask Features

- **Local Cluster**: Automatically starts a local dask cluster with multiple workers
- **Dashboard Access**: Provides access to the dask dashboard for monitoring (shown in verbose mode)
- **Automatic Cleanup**: Properly closes the cluster even if errors occur during processing
- **Chunk Alignment**: Automatically aligns Zarr chunks with dask chunks to prevent data corruption
- **Memory Efficiency**: Better memory management through parallel chunk processing
- **Error Handling**: Graceful handling of dask import errors with helpful installation instructions

#### Chunk Alignment

The library includes advanced chunk alignment logic to prevent the common issue of overlapping chunks when using dask:

- **Smart Detection**: Automatically detects if data is dask-backed and uses existing chunk structure
- **Aligned Calculation**: Uses `calculate_aligned_chunk_size()` to find optimal chunk sizes that divide evenly into data dimensions
- **Proper Rechunking**: Ensures datasets are rechunked to match encoding before writing
- **Fallback Logic**: For non-dask arrays, uses reasonable chunk sizes that don't exceed data dimensions

This prevents errors like:

```
❌ Failed to write tci after 2 attempts: Specified Zarr chunks encoding['chunks']=(1, 3660, 3660)
for variable named 'tci' would overlap multiple Dask chunks
```

#### S3 Python API

```python
import os
import xarray as xr
from eopf_geozarr import create_geozarr_dataset

# Configure for OVH Cloud (example)
os.environ['AWS_ACCESS_KEY_ID'] = 'your_ovh_access_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_ovh_secret_key'
os.environ['AWS_DEFAULT_REGION'] = 'gra'
os.environ['AWS_ENDPOINT_URL'] = 'https://s3.gra.cloud.ovh.net'

# Load your EOPF DataTree
dt = xr.open_datatree("path/to/eopf/dataset.zarr", engine="zarr")

# Convert directly to S3
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m", "/measurements/r20m", "/measurements/r60m"],
    output_path="s3://my-bucket/geozarr-data/output.zarr",
    spatial_chunk=4096,
    min_dimension=256,
    tile_width=256,
    max_retries=3
)
```

### Python API

```python
import xarray as xr
from eopf_geozarr import create_geozarr_dataset

# Load your EOPF DataTree
dt = xr.open_datatree("path/to/eopf/dataset.zarr", engine="zarr")

# Define groups to convert (e.g., resolution groups)
groups = ["/measurements/r10m", "/measurements/r20m", "/measurements/r60m"]

# Convert to GeoZarr compliant format
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=groups,
    output_path="path/to/output/geozarr.zarr",
    spatial_chunk=4096,
    min_dimension=256,
    tile_width=256,
    max_retries=3
)
```

## API Reference

### Main Functions

#### `create_geozarr_dataset`

Create a GeoZarr-spec 0.4 compliant dataset from EOPF data.

**Parameters:**

- `dt_input` (xr.DataTree): Input EOPF DataTree
- `groups` (List[str]): List of group names to process as Geozarr datasets
- `output_path` (str): Output path for the Zarr store
- `spatial_chunk` (int, default=4096): Spatial chunk size for encoding
- `min_dimension` (int, default=256): Minimum dimension for overview levels
- `tile_width` (int, default=256): Tile width for TMS compatibility
- `max_retries` (int, default=3): Maximum number of retries for network operations

**Returns:**

- `xr.DataTree`: DataTree containing the GeoZarr compliant data

#### `setup_datatree_metadata_geozarr_spec_compliant`

Set up GeoZarr-spec compliant CF standard names and CRS information.

**Parameters:**

- `dt` (xr.DataTree): The data tree containing the datasets to process
- `groups` (List[str]): List of group names to process as Geozarr datasets

**Returns:**

- `Dict[str, xr.Dataset]`: Dictionary of datasets with GeoZarr compliance applied

### Utility Functions

#### `downsample_2d_array`

Downsample a 2D array using block averaging.

#### `calculate_aligned_chunk_size`

Calculate a chunk size that divides evenly into the dimension size. This ensures that Zarr chunks align properly with the data dimensions, preventing chunk overlap issues when writing with Dask.

**Parameters:**

- `dimension_size` (int): Size of the dimension to chunk
- `target_chunk_size` (int): Desired chunk size

**Returns:**

- `int`: Aligned chunk size that divides evenly into dimension_size

**Example:**

```python
from eopf_geozarr.conversion.utils import calculate_aligned_chunk_size

# For a dimension of size 5490 with target chunk size 3660
aligned_size = calculate_aligned_chunk_size(5490, 3660)  # Returns 2745
```

#### `is_grid_mapping_variable`

Check if a variable is a grid_mapping variable by looking for references to it.

#### `validate_existing_band_data`

Validate that a specific band exists and is complete in the dataset.

## Architecture

The library is organized into the following modules:

- **`conversion`**: Core conversion tools for EOPF to GeoZarr transformation
  - `geozarr.py`: Main conversion functions and GeoZarr spec compliance
  - `utils.py`: Utility functions for data processing and validation
- **`data_api`**: Data access API (future development with pydantic-zarr)

## GeoZarr Specification Compliance

This library implements the GeoZarr specification 0.4 with the following key requirements:

1. **Array Dimensions**: All arrays must have `_ARRAY_DIMENSIONS` attributes
2. **CF Standard Names**: All variables must have CF-compliant `standard_name` attributes
3. **Grid Mapping**: Data variables must reference CF grid_mapping variables via `grid_mapping` attributes
4. **Multiscales Structure**: Overview levels are stored as children groups with proper tile matrix metadata
5. **Native CRS**: Coordinate reference systems are preserved without reprojection

## Contributing to GeoZarr Specification

Our implementation has contributed valuable feedback to the GeoZarr specification development process. Based on our real-world experience with Earth observation data, we have identified and reported several areas for improvement:

### Key Contributions

- **[Arbitrary Coordinate Systems Support](https://github.com/zarr-developers/geozarr-spec/issues/81)**: Advocating for native CRS preservation instead of web mapping bias
- **[Chunking Performance Optimization](https://github.com/zarr-developers/geozarr-spec/issues/82)**: Proposing flexible chunking strategies for optimal performance
- **[Multiscale Hierarchy Clarification](https://github.com/zarr-developers/geozarr-spec/issues/83)**: Providing clear structure definitions for multiscale implementations

Our implementation demonstrates that scientific accuracy and performance can be maintained while working with arbitrary coordinate systems, not just web mapping projections. This is particularly important for Earth observation data that often comes in UTM zones, polar stereographic, or other scientific projections.

For detailed information about our contributions, see our [GeoZarr Specification Contribution documentation](docs/geozarr-specification-contribution.md).

## Development

### Setting up Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd eopf-geozarr

# Install in development mode with all dependencies
pip install -e ".[dev,docs,all]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
pytest
```

### Code Quality

The project uses:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **pre-commit** for automated checks

### Building Documentation

```bash
cd docs
make html
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure code quality checks pass
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- Built on top of the excellent [xarray](https://xarray.pydata.org/) and [zarr](https://zarr.readthedocs.io/) libraries
- Follows the [GeoZarr specification](https://github.com/zarr-developers/geozarr-spec) for geospatial data in Zarr
- Designed for compatibility with [EOPF](https://eopf.readthedocs.io/) datasets

## Support

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/developmentseed/eopf-geozarr).
