# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-11-04

### Added
- `eopf_geozarr.s2_optimization` module with streaming multiscale generation, CLI commands, and validation for Sentinel-2 L2A.
- End-to-end sharding support spanning CLI flags, conversion helpers, Dask execution, and encoding metadata.
- Geo Projection attribute extension documentation plus schema to lock GeoZarr metadata expectations.

### Changed
- Tightened spatial chunk and shard defaults to cut write overhead on large scenes.
- Relocated the entire test suite under `src/eopf_geozarr/tests` and broadened type coverage for tooling.
- Smoothed multiscale metadata handling during streaming writes to keep Sentinel datasets consistent.

### Fixed
- Preserved coordinate dtypes in overview levels and stopped auxiliary coordinate write failures.
- Prevented streaming metadata consolidation from overwriting existing groups between runs.

## [0.2.0] - 2025-09-22

### Added
- Sentinel-1 GRD integration tests and CLI wiring to enforce GeoZarr compliance end to end.
- Reprojection utilities with GCP selection and grid-mapping output for Sentinel-1 converts.

### Changed
- Extended `create_geozarr_dataset` to understand VV/VH polarization groups and build GCP-backed overviews.
- Tuned chunk-size calculation and encoding helpers so shard dimensions and auxiliaries align.

### Fixed
- Stopped auxiliary coordinate writes from failing in overviews when chunked.
- Silenced noisy CLI warnings and aligned launch configs with the packaged tests.

## [0.1.0] - 2025-01-25

### Added
- Initial release of EOPF GeoZarr library
- Core conversion functionality from EOPF datasets to GeoZarr-spec 0.4 compliant format
- Command-line interface with `convert`, `info`, and `validate` commands
- GeoZarr specification compliance features:
  - `_ARRAY_DIMENSIONS` attributes on all arrays
  - CF standard names for all variables
  - `grid_mapping` attributes referencing CF grid_mapping variables
  - `GeoTransform` attributes in grid_mapping variables
  - Proper multiscales metadata structure
- Native CRS preservation (no reprojection to TMS required)
- Multiscale support with COG-style /2 downsampling logic
- Utility functions for data processing:
  - `downsample_2d_array` for block averaging and subsampling
  - `calculate_aligned_chunk_size` for optimal chunking
  - `calculate_overview_levels` for multiscale generation
  - `validate_existing_band_data` for data validation
- Comprehensive test suite with 11 test cases
- Documentation structure with API reference
- Apache 2.0 license
- PyPI package configuration with proper dependencies

### Features
- **Conversion Module**: Core tools for EOPF to GeoZarr transformation
  - `create_geozarr_dataset`: Main conversion function
  - `setup_datatree_metadata_geozarr_spec_compliant`: Metadata setup for GeoZarr compliance
  - `recursive_copy`: Efficient data copying with retry logic
  - `consolidate_metadata`: Zarr metadata consolidation
- **Data API Module**: Foundation for future pydantic-zarr integration
- **CLI Module**: User-friendly command-line interface
- **Utility Functions**: Helper functions for data processing and validation

### Technical Details
- Built on xarray, zarr, and rioxarray
- Supports Python 3.11+
- Follows CF conventions for geospatial metadata
- Implements GeoZarr specification 0.4
- Includes comprehensive error handling and retry logic
- Band-by-band processing for memory efficiency

### Dependencies
- xarray >= 2025.7.1
- zarr >= 3.0.10
- rioxarray >= 0.13.0
- numpy >= 2.3.1
- dask[array,distributed] >= 2025.5.1
- pydantic-zarr (from git)
- cf-xarray >= 0.8.0
- aiohttp >= 3.8.1

### Development
- Pre-commit hooks for code quality
- Black, isort, flake8, and mypy for code formatting and linting
- Pytest for testing with coverage reporting
- Comprehensive CI/CD setup ready
