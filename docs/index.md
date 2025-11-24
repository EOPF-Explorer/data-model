# EOPF GeoZarr Documentation

Welcome to the EOPF GeoZarr library documentation. This library provides tools to convert EOPF (Earth Observation Processing Framework) datasets to GeoZarr-spec 0.4 compliant format while maintaining scientific accuracy and optimizing for cloud-native workflows.

## Quick Navigation

### Getting Started

- **[Installation](installation.md)** - Install the library and set up your environment
- **[Quick Start](quickstart.md)** - Convert your first dataset in minutes
- **[User Guide](converter.md)** - Comprehensive usage guide with advanced options

### Reference

- **[API Reference](api-reference.md)** - Complete Python API documentation
- **[Examples](examples.md)** - Practical examples for common use cases
- **[Architecture](architecture.md)** - Technical architecture and design principles
- **[GeoZarr Mini Spec](geozarr-minispec.md)** - Implementation-specific GeoZarr specification

### Support

- **[FAQ](faq.md)** - Frequently asked questions and troubleshooting
- **[GeoZarr Specification](geozarr-specification-contribution.md)** - Our contributions to the GeoZarr spec

## What is EOPF GeoZarr?

The EOPF GeoZarr library bridges the gap between EOPF datasets and the emerging GeoZarr specification, enabling:

✅ **Scientific Accuracy** - Preserves native CRS and data integrity  
✅ **Cloud-Native** - Optimized for S3 and distributed processing  
✅ **Performance** - Intelligent chunking and multiscale pyramids  
✅ **Standards Compliant** - Full GeoZarr 0.4 and CF conventions support  
✅ **Production Ready** - Robust error handling and validation  

## Key Features

### 🌍 Native CRS Preservation

Maintains original coordinate reference systems (UTM, polar stereographic, etc.) without unnecessary reprojection to Web Mercator, preserving scientific accuracy for Earth observation data.

### 📊 Multiscale Pyramids

Automatically generates overview levels with /2 downsampling, creating efficient multiscale pyramids for visualization and analysis at different resolutions.

### 🔧 Intelligent Chunking

Implements aligned chunking strategy that prevents partial chunks, optimizes storage efficiency, and improves I/O performance for both local and cloud storage.

### ☁️ Cloud-Native Design

Full support for AWS S3 and S3-compatible storage with automatic credential detection, retry logic, and optimized multipart uploads.

### 📋 Standards Compliance

- **GeoZarr 0.4 specification** compliance
- **CF conventions** for scientific metadata
- **`_ARRAY_DIMENSIONS`** attributes on all arrays
- **Grid mapping** variables with proper CRS information
- **Multiscales** metadata structure

### 🚀 Performance Optimized

- **Dask integration** for distributed processing
- **Lazy loading** for memory efficiency
- **Band-by-band processing** with validation
- **Retry logic** for robust network operations

## Supported Data

### Satellite Missions

- **Sentinel-2** L1C and L2A products (fully supported)
- **Sentinel-1** (planned support)
- Extensible architecture for additional missions

### Data Formats

- **Input**: EOPF DataTree (Zarr format)
- **Output**: GeoZarr-compliant Zarr with multiscale structure

### Storage Backends

- **Local filesystems**
- **AWS S3**
- **S3-compatible storage** (MinIO, DigitalOcean Spaces, etc.)

## Quick Example

```python
# test: skip
import xarray as xr
from eopf_geozarr import create_geozarr_dataset

# Load EOPF dataset
dt = xr.open_datatree("sentinel2_l2a.zarr", engine="zarr")

# Convert to GeoZarr
dt_geozarr = create_geozarr_dataset(
    dt_input=dt,
    groups=["/measurements/r10m", "/measurements/r20m", "/measurements/r60m"],
    output_path="s3://my-bucket/geozarr.zarr",
    spatial_chunk=4096
)

print("Conversion complete!")
```

Or using the command line:

```bash
eopf-geozarr convert input.zarr s3://bucket/output.zarr
```

## Architecture Overview

The library is organized into focused modules:

- **`conversion/`** - Core conversion engine and algorithms
  - `geozarr.py` - Main conversion functions
  - `fs_utils.py` - Storage backend abstraction
  - `utils.py` - Processing utilities and chunking
- **`cli.py`** - Command-line interface
- **`data_api/`** - Future data access API (pydantic-zarr integration)

## Implementation Highlights

Based on our experience and contributions to the GeoZarr specification (see [ADR-101](https://github.com/DevelopmentSeed/sentinel-zarr-explorer-coordination/blob/main/docs/adr/ADR-101-geozarr-specification-implementation-approach.md)), this library implements:

### Native CRS Tile Matrix Sets

Creates custom tile matrix sets for arbitrary coordinate reference systems, not just Web Mercator, enabling scientific applications that require native projections.

### Aligned Chunking Strategy

Implements intelligent chunk size calculation that prevents partial chunks and optimizes for both storage efficiency and processing performance.

### Hierarchical Data Organization

Uses a sibling-based structure (`/0`, `/1`, `/2`) for resolution levels that complies with xarray DataTree requirements while maintaining GeoZarr specification compliance.

### Robust Cloud Integration

Production-ready S3 integration with credential validation, error handling, and performance optimization for large-scale Earth observation workflows.

## Getting Started

1. **[Install](installation.md)** the library
2. **[Quick Start](quickstart.md)** with your first conversion
3. **[Explore Examples](examples.md)** for your specific use case
4. **[Read the User Guide](converter.md)** for advanced usage

## Community and Support

- **Documentation**: Comprehensive guides and API reference
- **GitHub**: [eopf-explorer/data-model](https://github.com/eopf-explorer/data-model)
- **Issues**: Report bugs and request features
- **Contributions**: Help improve the library and specification

The EOPF GeoZarr library is actively developed and maintained by [Development Seed](https://developmentseed.org/), with ongoing contributions to the GeoZarr specification to better serve the Earth observation community.
