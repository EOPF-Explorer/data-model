# GeoZarr Specification Contribution

This document outlines our contribution to the GeoZarr specification based on our implementation experience with the EOPF GeoZarr data model.

## Overview

Our implementation of GeoZarr-compliant data conversion for Earth Observation data has revealed several areas where the current specification could be improved to better support scientific use cases. We have contributed feedback to the GeoZarr specification development process through detailed GitHub issues.

Our implementation follows the [GeoZarr Mini Spec](geozarr-minispec.md), which documents the specific subset of the GeoZarr specification that we implement, including implementation-specific details for chunking requirements, CF compliance, and multiscale dataset organization.

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

### 3. GeoZarr Hierarchy Boundaries & Root Identification

**Issue:** [zarr-developers/geozarr-spec#132](https://github.com/zarr-developers/geozarr-spec/issues/132)

**Problem:** The current draft does not define how a GeoZarr store is bounded as a set of paths. The base Zarr spec treats the `.zarr` suffix as advisory only, so two implementations can disagree on whether nested stores (e.g. `a.zarr/b/c.zarr/`) are allowed, how a client given a deep URL recovers the root, and where a traversal should stop.

**Our Solution:** We adopted an explicit set of rules in our [Store Root section](geozarr-minispec.md#hierarchy--identification): single root, root prefix ends with `.zarr`, the suffix occurs at most once in the hierarchy, and an enumerated list of terminal-path conditions.

**Impact:** Clients reading a sub-path like `https://example.org/foo.zarr/measurements/reflectance/r10m` can now reliably recover the store root and read its summary `spatial:bbox` + `proj:code`. This also resolves a recurring URL-parsing question raised in [EOPF-Explorer/data-model#124](https://github.com/EOPF-Explorer/data-model/issues/124) without needing fragment-based URL workarounds.

### 4. Store-root Summary Footprint (`spatial:bbox` + `proj:code`)

**Issue:** [EOPF-Explorer/data-model#156](https://github.com/EOPF-Explorer/data-model/issues/156) — to be surfaced upstream once [#132](https://github.com/zarr-developers/geozarr-spec/issues/132) lands.

**Problem:** Without a top-level summary footprint, clients have to walk into child groups to discover where a store sits geographically. That is expensive over network, and prevents STAC-style discovery patterns.

**Our Solution:** A mandatory `spatial:bbox` plus an explicit CRS (one of `proj:code`, `proj:wkt2`, or `proj:projjson`) at the store root, defined in the [Store Root section](geozarr-minispec.md#store-root). The CRS is always declared explicitly — there is no implicit default.

**Impact:** A single read of the root `zarr.json` is enough for catalogues and viewers to place a store on a map, without disturbing per-group `spatial:` attributes which remain authoritative for individual variables.

### 5. STAC-style `spatial:extent` (multi-bbox at root)

**Issue:** [zarr-developers/geozarr-spec#133](https://github.com/zarr-developers/geozarr-spec/issues/133)

**Problem:** A single union `spatial:bbox` at the store root loses per-asset footprint information. Stores that mix several Datasets / Multiscale Datasets cannot expose individual extents without forcing clients to walk every child group.

**Our Solution (proposed):** Mirror STAC's `extent.spatial.bbox` by introducing `spatial:extent` as a list of bboxes — first entry is the global union, subsequent entries are per-child extents. Suggested by @vincentsarago during review of #4 above; deferred until the upstream `spatial:` convention adopts it.

**Impact:** Lets clients fetch all per-child footprints in one request, aligning Zarr discovery with STAC conventions.

### 6. Multiscale Hierarchy Structure Clarification

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
