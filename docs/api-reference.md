# API Reference

Complete reference for the EOPF GeoZarr library's Python API.

## Core Functions

### create_geozarr_dataset

The main function for converting EOPF datasets to GeoZarr format.

```python
def create_geozarr_dataset(
    dt_input: xr.DataTree,
    groups: List[str],
    output_path: str,
    spatial_chunk: int = 4096,
    min_dimension: int = 256,
    tile_width: int = 256,
    max_retries: int = 3,
    **storage_kwargs
) -> xr.DataTree
```

**Parameters:**

- `dt_input` (xr.DataTree): Input EOPF DataTree to convert
- `groups` (List[str]): List of group paths to process (e.g., `["/measurements/r10m"]`)
- `output_path` (str): Output path for the GeoZarr dataset (local or S3)
- `spatial_chunk` (int, optional): Target spatial chunk size. Default: 4096
- `min_dimension` (int, optional): Minimum dimension size for processing. Default: 256
- `tile_width` (int, optional): Tile width for multiscale levels. Default: 256
- `max_retries` (int, optional): Maximum retry attempts for operations. Default: 3
- `**storage_kwargs`: Additional storage options (S3 credentials, etc.)

**Returns:**

- `xr.DataTree`: The converted GeoZarr-compliant DataTree

**Example:**

```python
import xarray as xr
from eopf_geozarr import create_geozarr_dataset

dt = xr.open_datatree("input.zarr", engine="zarr")
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m", "/measurements/r20m"],
    output_path="output.zarr",
    spatial_chunk=2048
)
```

## Conversion Functions

### setup_datatree_metadata_geozarr_spec_compliant

Sets up GeoZarr-compliant metadata for a DataTree.

```python
def setup_datatree_metadata_geozarr_spec_compliant(
    dt: xr.DataTree,
    geozarr_groups: Dict[str, xr.Dataset]
) -> None
```

### write_geozarr_group

Writes a single group to GeoZarr format with proper metadata.

```python
def write_geozarr_group(
    group_path: str,
    datasets: Dict[str, xr.Dataset],
    output_path: str,
    spatial_chunk: int = 4096,
    max_retries: int = 3,
    **storage_kwargs
) -> None
```

### create_geozarr_compliant_multiscales

Creates multiscales metadata compliant with GeoZarr specification.

```python
def create_geozarr_compliant_multiscales(
    datasets: Dict[str, xr.Dataset],
    tile_width: int = 256
) -> List[Dict[str, Any]]
```

## Utility Functions

### calculate_aligned_chunk_size

Calculates optimal chunk size that aligns with data dimensions.

```python
def calculate_aligned_chunk_size(
    dimension_size: int,
    target_chunk_size: int
) -> int
```

**Parameters:**

- `dimension_size` (int): Size of the data dimension
- `target_chunk_size` (int): Desired chunk size

**Returns:**

- `int`: Optimal aligned chunk size

**Example:**

```python
from eopf_geozarr.conversion.utils import calculate_aligned_chunk_size

# For a 10980x10980 image with target 4096 chunks
chunk_size = calculate_aligned_chunk_size(10980, 4096)
print(chunk_size)  # Returns 3660 (10980 / 3 = 3660)
```

### downsample_2d_array

Downsamples a 2D array by factor of 2 using mean aggregation.

```python
def downsample_2d_array(
    data: np.ndarray,
    factor: int = 2
) -> np.ndarray
```

### validate_existing_band_data

Validates existing band data against expected specifications.

```python
def validate_existing_band_data(
    dataset: xr.Dataset,
    band_name: str,
    expected_shape: Tuple[int, ...],
    expected_chunks: Tuple[int, ...]
) -> bool
```

## File System Functions

### Storage Path Utilities

```python
# Path normalization and validation
def normalize_path(path: str) -> str
def is_s3_path(path: str) -> bool
def parse_s3_path(s3_path: str) -> tuple[str, str]

# Storage options
def get_storage_options(path: str, **kwargs: Any) -> Optional[Dict[str, Any]]
def get_s3_storage_options(s3_path: str, **s3_kwargs: Any) -> Dict[str, Any]
```

### S3 Operations

```python
# S3 store creation and validation
def create_s3_store(s3_path: str, **s3_kwargs: Any) -> str
def validate_s3_access(s3_path: str, **s3_kwargs: Any) -> tuple[bool, Optional[str]]
def s3_path_exists(s3_path: str, **s3_kwargs: Any) -> bool

# S3 metadata operations
def write_s3_json_metadata(
    s3_path: str,
    metadata: Dict[str, Any],
    **s3_kwargs: Any
) -> None

def read_s3_json_metadata(s3_path: str, **s3_kwargs: Any) -> Dict[str, Any]
```

### Zarr Operations

```python
# Zarr group operations
def open_zarr_group(path: str, mode: str = "r", **kwargs: Any) -> zarr.Group
def open_s3_zarr_group(s3_path: str, mode: str = "r", **s3_kwargs: Any) -> zarr.Group

# Metadata consolidation
def consolidate_metadata(output_path: str, **storage_kwargs) -> None
async def async_consolidate_metadata(output_path: str, **storage_kwargs) -> None
```

## Metadata Functions

### Coordinate Metadata

```python
def _add_coordinate_metadata(ds: xr.Dataset) -> None
```

Adds proper coordinate metadata including:

- `_ARRAY_DIMENSIONS` attributes
- CF standard names
- Coordinate variable attributes

### Grid Mapping

```python
def _setup_grid_mapping(ds: xr.Dataset, grid_mapping_var_name: str) -> None
def _add_geotransform(ds: xr.Dataset, grid_mapping_var: str) -> None
```

### CRS and Tile Matrix

```python
def create_native_crs_tile_matrix_set(
    crs: Any,
    transform: Any,
    width: int,
    height: int,
    tile_width: int = 256
) -> Dict[str, Any]
```

Creates a tile matrix set for native CRS (non-Web Mercator).

## Overview Generation

### calculate_overview_levels

```python
def calculate_overview_levels(
    width: int,
    height: int,
    min_dimension: int = 256
) -> List[int]
```

Calculates appropriate overview levels based on data dimensions.

### create_overview_dataset_all_vars

```python
def create_overview_dataset_all_vars(
    ds: xr.Dataset,
    overview_factor: int
) -> xr.Dataset
```

Creates overview dataset with all variables downsampled.

## Error Handling

### Retry Logic

```python
def write_dataset_band_by_band_with_validation(
    ds: xr.Dataset,
    output_path: str,
    max_retries: int = 3,
    **storage_kwargs
) -> None
```

Writes dataset with robust error handling and retry logic.

## Constants and Enums

### Coordinate Attributes

```python
def _get_x_coord_attrs() -> Dict[str, Any]
def _get_y_coord_attrs() -> Dict[str, Any]
```

Returns standard attributes for X and Y coordinates.

### Grid Mapping Detection

```python
def is_grid_mapping_variable(ds: xr.Dataset, var_name: str) -> bool
```

Determines if a variable is a grid mapping variable.

## Usage Examples

### Basic Conversion

```python
import xarray as xr
from eopf_geozarr import create_geozarr_dataset

# Load and convert
dt = xr.open_datatree("input.zarr", engine="zarr")
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m"],
    output_path="output.zarr"
)
```

### Advanced S3 Usage

```python
from eopf_geozarr.conversion.fs_utils import (
    validate_s3_access,
    get_s3_storage_options
)

# Validate S3 access
s3_path = "s3://my-bucket/data.zarr"
is_valid, error = validate_s3_access(s3_path)

if is_valid:
    # Get storage options
    storage_opts = get_s3_storage_options(s3_path)
    
    # Convert with S3
    dt_geozarr = create_geozarr_dataset(
        dt_input=dt,
        groups=["/measurements/r10m"],
        output_path=s3_path,
        **storage_opts
    )
```

### Custom Chunking

```python
from eopf_geozarr.conversion.utils import calculate_aligned_chunk_size

# Calculate optimal chunks for your data
width, height = 10980, 10980
optimal_chunk = calculate_aligned_chunk_size(width, 4096)

dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m"],
    output_path="output.zarr",
    spatial_chunk=optimal_chunk
)
```

## Type Hints

The library uses comprehensive type hints. Import types as needed:

```python
from typing import Dict, List, Optional, Tuple, Any
import xarray as xr
import numpy as np
```

## Error Types

Common exceptions you may encounter:

- `ValueError`: Invalid parameters or data
- `FileNotFoundError`: Missing input files
- `PermissionError`: Insufficient permissions for S3 or file operations
- `zarr.errors.ArrayNotFoundError`: Missing Zarr arrays
- `xarray.core.common.DataWithCoords`: Data structure issues

For detailed error handling examples, see the [FAQ](faq.md).
