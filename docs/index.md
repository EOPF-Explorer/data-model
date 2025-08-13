# EOPF GeoZarr Documentation

Welcome to the EOPF GeoZarr library documentation. This library provides tools to convert EOPF (Earth Observation Processing Framework) datasets to GeoZarr-spec 0.4 compliant format.

## Table of Contents

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [API Reference](api.md)
- [GeoZarr Specification Compliance](geozarr-compliance.md)
- [Examples](examples.md)

## Overview

The EOPF GeoZarr library enables conversion of EOPF datasets to the GeoZarr specification while:

- Maintaining native coordinate reference systems (no reprojection to TMS)
- Supporting multiscale data with COG-style /2 downsampling
- Ensuring full CF conventions compliance
- Providing robust processing with validation and retry logic

## Key Features

### GeoZarr Specification Compliance

- Full compliance with GeoZarr spec 0.4
- `_ARRAY_DIMENSIONS` attributes on all arrays
- CF standard names for all variables
- `grid_mapping` attributes referencing CF grid_mapping variables
- `GeoTransform` attributes in grid_mapping variables
- Proper multiscales metadata structure

### Native CRS Preservation

- Maintains native CRS (e.g., UTM zones) throughout all overview levels
- Avoids reprojection to Web Mercator, preserving scientific accuracy
- Custom tile matrix sets using native CRS

### Band Organization

- Spectral bands stored as separate DataArray variables
- Enables band-specific metadata and selective access
- Supports different processing chains per spectral band

### Chunking Strategy

- Aligned chunking to optimize storage efficiency and I/O performance
- Prevents partial chunks that waste storage space
- Reduces memory fragmentation

### Hierarchical Structure

- All resolution levels stored as siblings (`/0`, `/1`, `/2`, etc.)
- Multiscales metadata in parent group attributes
- Complies with xarray DataTree alignment requirements

### Robust Processing

- Band-by-band writing with validation
- Retry logic for network operations

## Architecture

The library is organized into the following modules:

- **`conversion`**: Core conversion tools for EOPF to GeoZarr transformation
- **`data_api`**: Data access API (future development with pydantic-zarr)
- **`cli`**: Command-line interface for easy usage

## Getting Started

See the [Quick Start](quickstart.md) guide to begin using the library, or check out the [Examples](examples.md) for common use cases.

## Support

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/eopf-explorer/data-model).
