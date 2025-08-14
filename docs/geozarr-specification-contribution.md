# GeoZarr Specification Contribution

This document outlines our contribution to the GeoZarr specification based on our implementation experience with the EOPF GeoZarr data model.

## Overview

Our implementation of GeoZarr-compliant data conversion for Earth Observation data has revealed several areas where the current specification could be improved to better support scientific use cases. We have contributed feedback to the GeoZarr specification development process through detailed GitHub issues.

## Key Issues Identified and Reported

### 1. Arbitrary Coordinate Systems Support

**Issue:** [zarr-developers/geozarr-spec#81](https://github.com/zarr-developers/geozarr-spec/issues/81)

**Problem:** The current specification has an implicit bias toward web mapping tile schemes (WebMercatorQuad), which may discourage scientific applications that work with native coordinate reference systems.

**Our Solution:** Our implementation successfully demonstrates:

- Creation of "Native CRS Tile Matrix Sets" for arbitrary projections
- Multiscale pyramids working with UTM and other scientific projections
- Proper scale denominator calculations for non-web CRS
- Chunking strategies optimized for native coordinate systems

**Impact:** This is critical for Earth observation data that often comes in UTM zones, polar stereographic, or other scientific projections where preserving the native CRS maintains scientific accuracy.

### 2. Chunking Performance Optimization

**Issue:** [zarr-developers/geozarr-spec#82](https://github.com/zarr-developers/geozarr-spec/issues/82)

**Problem:** The specification requires strict 1:1 mapping between Zarr chunks and tile matrix tiles, which prevents optimal chunking strategies for different data types and storage backends.

**Our Solution:** We implemented sophisticated chunk alignment logic:

```python
def calculate_aligned_chunk_size(dimension_size: int, target_chunk_size: int) -> int:
    """Calculate a chunk size that divides evenly into the dimension size."""
    if target_chunk_size >= dimension_size:
        return dimension_size
    
    # Find the largest divisor that is <= target_chunk_size
    for chunk_size in range(target_chunk_size, 0, -1):
        if dimension_size % chunk_size == 0:
            return chunk_size
    return 1
```

**Impact:** This approach prevents chunk overlap issues with Dask while optimizing for actual data dimensions rather than arbitrary tile sizes, significantly improving performance.

### 3. Multiscale Hierarchy Structure Clarification

**Issue:** [zarr-developers/geozarr-spec#83](https://github.com/zarr-developers/geozarr-spec/issues/83)

**Problem:** The specification describes multiscale encoding but doesn't clearly define the exact hierarchical structure and relationship between parent groups and zoom level children.

**Our Solution:** We implemented a clear hierarchy structure:

```
/measurements/r10m/          # Parent group with multiscales metadata
├── 0/                       # Native resolution (zoom level 0)
│   ├── band1
│   ├── band2
│   └── spatial_ref
├── 1/                       # First overview level
│   ├── band1
│   ├── band2
│   └── spatial_ref
└── 2/                       # Second overview level
    ├── band1
    ├── band2
    └── spatial_ref
```

**Impact:** This provides a concrete, tested pattern for implementing multiscale hierarchies that other implementations can follow.

## Implementation Evidence

Our implementation provides concrete evidence for these improvements:

### Native CRS Preservation

- **Function:** `create_native_crs_tile_matrix_set()`
- **Purpose:** Creates custom tile matrix sets for arbitrary coordinate reference systems
- **Benefit:** Maintains scientific accuracy without unnecessary reprojection

### Robust Processing

- **Function:** `write_dataset_band_by_band_with_validation()`
- **Purpose:** Handles large datasets with retry logic and validation
- **Benefit:** Production-ready robustness for real-world data processing

### Comprehensive Metadata Handling

- **Function:** `_add_coordinate_metadata()`
- **Purpose:** Handles diverse coordinate types (time, angle, band, detector)
- **Benefit:** Supports the full range of Earth observation data structures

### Cloud Storage Optimization

- **Features:** S3 support with credential validation, storage options handling
- **Benefit:** Enables cloud-native workflows with proper error handling

## Specification Sections Addressed

Our contributions target specific sections of the GeoZarr specification:

- **Section 9.7.3** (Tile Matrix Set Representation) - Native CRS support
- **Section 9.7.4** (Chunk Layout Alignment) - Flexible chunking
- **Section 9.7.1** (Hierarchical Layout) - Clear structure definition
- **Section 9.7.2** (Metadata Encoding) - Metadata placement guidance

## Benefits for the Earth Observation Community

These contributions specifically benefit Earth observation and scientific data applications:

1. **Scientific Accuracy:** Preserving native CRS prevents distortion from unnecessary reprojections
2. **Performance:** Optimized chunking improves processing speed and reduces memory usage
3. **Clarity:** Clear hierarchy definitions enable consistent implementations
4. **Robustness:** Production patterns support real-world deployment scenarios

## Future Work

We continue to monitor the specification development and will contribute additional feedback as our implementation evolves. Areas for potential future contribution include:

- Cloud storage optimization patterns
- Coordinate variable handling for diverse data types
- Integration with STAC metadata standards
- Guidance for time dimension handling

## Related Documentation

- [Converter Documentation](converter.md) - Technical details of our implementation
- [Architecture](architecture.md) - Technical architecture and design principles
- [API Reference](api-reference.md) - Complete Python API documentation

## Links

- [GeoZarr Specification Repository](https://github.com/zarr-developers/geozarr-spec)
- [Our GitHub Issues](https://github.com/zarr-developers/geozarr-spec/issues?q=is%3Aissue+author%3Aemmanuelmathot)
- [Project Issue #74](https://github.com/developmentseed/sentinel-zarr-explorer-coordination/issues/74)
