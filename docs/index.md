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

- No reprojection to TMS required
- Maintains original coordinate reference systems
- Native CRS tile matrix sets

### Multiscale Support

- COG-style /2 downsampling logic
- Overview levels as children groups
- Configurable minimum dimensions and tile widths

### Robust Processing

- Band-by-band writing with validation
- Retry logic for network operations
- Comprehensive error handling

## Architecture

The library is organized into the following modules:

- **`conversion`**: Core conversion tools for EOPF to GeoZarr transformation
- **`data_api`**: Data access API (future development with pydantic-zarr)
- **`cli`**: Command-line interface for easy usage

## Getting Started

See the [Quick Start](quickstart.md) guide to begin using the library, or check out the [Examples](examples.md) for common use cases.

## Support

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/eopf-explorer/data-model).
