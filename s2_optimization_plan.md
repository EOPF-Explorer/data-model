# Sentinel-2 Zarr Conversion Optimization Plan

## Overview
This plan outlines the development of an optimized Sentinel-2 converter (`convert_s2`) that dramatically simplifies the dataset structure while maintaining scientific integrity and improving storage efficiency.

## Current State Analysis

### Problems with Current Structure
- **File proliferation**: Multiple resolution groups create numerous zarr chunks
- **Data redundancy**: Repeated metadata across similar resolution groups  
- **Complex navigation**: Deep nested structure makes data discovery difficult
- **Storage inefficiency**: Many small groups instead of consolidated datasets
- **Inconsistent multiscale**: Each resolution treated separately rather than as pyramid

### Current Structure Issues
```
root/
├── conditions/          # Mixed resolution data scattered
├── quality/             # Atmosphere data duplicated across resolutions
└── measurements/        # Resolution-based grouping creates complexity
    └── reflectance/
        ├── r10m/ (+ 6 pyramid levels)
        ├── r20m/ (+ 5 pyramid levels) 
        └── r60m/ (+ 3 pyramid levels)
```

## Proposed Optimized Structure

### New Simplified Structure
```
root/
├── measurements/        # Single multiscale group
│   ├── 0/              # 10m native (b02,b03,b04,b08 + all derived data)
│   ├── 1/              # 20m native (all b01-b12,b8a + derived data)  
│   ├── 2/              # 60m native (minimal additional bands)
│   ├── 3/              # 120m (downsampled)
│   ├── 4/              # 240m (downsampled)
│   ├── 5/              # 480m (downsampled)
│   └── 6/              # 960m (downsampled)
├── geometry/           # Consolidated geometric data
└── meteorology/        # Consolidated weather data
```

## Technical Design Specifications

### 1. Band Distribution Strategy

#### Level 0 (10m) - Native Resolution
**Variables:**
- `b02`, `b03`, `b04`, `b08` (native 10m bands)
- `detector_footprint_b02`, `detector_footprint_b03`, `detector_footprint_b04`, `detector_footprint_b08`
- `quality_b02`, `quality_b03`, `quality_b04`, `quality_b08`

#### Level 1 (20m) - Native Resolution  
**Variables:**
- `b02`, `b03`, `b04`, `b05`, `b06`, `b07`, `b08`, `b11`, `b12`, `b8a` (native 20m bands + all 10m bands downsampled)
- detector footprints of all 10m and 20m bands
- `aot`, `wvp` (native 20m atmosphere quality)
- `scl` (native 20m classification)
- `cld`, `snw` (cloud and snow probability)
- All quality masks for each band (downsampled from 10m where applicable)

#### Level 2 (60m) - Native Resolution
**Variables:**
- All Bands: `b01`, `b02`, `b03`, `b04`, `b05`, `b06`, `b07`, `b08`, `b09`, `b11`, `b12`, `b8a`
- Detector footprints for all bands
- `scl` (native 60m classification)
- `aot`, `wvp` (downsampled from 20m)
- `cld`, `snw` (downsampled from 20m)
- Quality masks for all bands (downsampled from 20m where applicable)

#### Levels 3-6 (120m, 240m, 480m, 960m)
**Variables:**
- All bands downsampled using appropriate resampling methods
- All quality and classification data downsampled accordingly

### 2. Data Consolidation Rules

#### Measurements Group
- **Resampling Strategy**: 
  - Upsampling: Bilinear interpolation for reflectance bands
  - Downsampling: Block averaging for reflectance, mode for classifications
  - Quality data: Logical operations (any/all) for binary masks
  
- **Variable Naming Convention**:
  - Spectral bands: `b01`, `b02`, ..., `b12`, `b8a`
  - Detector footprints: `detector_footprint_{band}`
  - Atmosphere quality: `aot`, `wvp` 
  - Classification: `scl`
  - Probability: `cld`, `snw`
  - Quality: `quality_{band}`

#### Geometry Group
- `sun_angles` (consolidated)
- `viewing_incidence_angles` (consolidated) 
- `mean_sun_angles`, `mean_viewing_incidence_angles`
- `spatial_ref`

#### Meteorology Group  
- CAMS data: `aod*`, `*aod550`, etc.
- ECMWF data: `msl`, `tco3`, `tcwv`, `u10`, `v10`, `r`

### 3. Multiscale Implementation

#### Pyramid Strategy
- **Native data preservation**: Only store at native resolution
- **No upsampling**: Higher resolution levels only contain natively available bands
- **Consistent downsampling**: /2 decimation between consecutive levels where appropriate
- **Smart resampling**: 
  - Reflectance: Bilinear → Block average
  - Classifications: Mode 
  - Quality masks: Logical operations
  - Probabilities: Average

#### Storage Optimization
- **Chunking**: Align chunks across all levels (e.g., 1024×1024)
- **Compression**: Consistent codec across all variables
- **Sharding**: Enable for the full shape dimension
- **Metadata consolidation**: Single zarr.json with complete multiscales info

## Implementation Plan

### Phase 1: Core Infrastructure (New Files)

#### File: `s2_converter.py`
```python
class S2OptimizedConverter:
    """Optimized Sentinel-2 to GeoZarr converter"""
    
    def __init__(self, enable_sharding=True, spatial_chunk=1024):
        self.enable_sharding = enable_sharding
        self.spatial_chunk = spatial_chunk
        
    def convert_s2(self, dt_input, output_path, **kwargs):
        """Main conversion entry point"""
        
    def _create_optimized_structure(self, dt_input):
        """Reorganize data into 3 main groups"""
        
    def _create_measurements_multiscale(self, measurements_data, output_path):
        """Create consolidated multiscale measurements"""
        
    def _consolidate_geometry_data(self, dt_input):
        """Consolidate all geometry-related data"""
        
    def _consolidate_meteorology_data(self, dt_input):
        """Consolidate CAMS and ECMWF data"""
```

#### File: `s2_band_mapping.py`
```python
# Band availability by native resolution
NATIVE_BANDS = {
    10: ['b02', 'b03', 'b04', 'b08'],
    20: ['b05', 'b06', 'b07', 'b11', 'b12', 'b8a'],
    60: ['b01', 'b09']  # Only these are truly native to 60m
}

# Quality data mapping
QUALITY_DATA_MAPPING = {
    'atmosphere': ['aot', 'wvp'],
    'classification': ['scl'],
    'probability': ['cld', 'snw'],
    'detector_footprint': NATIVE_BANDS,
    'quality_masks': NATIVE_BANDS
}
```

#### File: `s2_resampling.py`
```python
class S2ResamplingEngine:
    """Handles all resampling operations for S2 data"""
    
    def upsample_reflectance(self, data, target_resolution):
        """Bilinear upsampling for reflectance bands (REMOVED - no upsampling)"""
        raise NotImplementedError("Upsampling disabled - only native resolution data stored")
        
    def downsample_reflectance(self, data, target_resolution):
        """Block averaging for reflectance bands"""
        
    def resample_classification(self, data, target_resolution, method='mode'):
        """Mode-based resampling for classification data"""
        
    def resample_quality_masks(self, data, target_resolution, operation='any'):
        """Logical operations for quality masks"""
```

### Phase 2: Data Processing Pipeline

#### Measurement Processing Pipeline
1. **Data Inventory**: Scan input structure and catalog all variables
2. **Resolution Analysis**: Determine native resolution for each variable
3. **Level Planning**: Calculate which variables belong to each pyramid level
4. **Resampling Execution**: Apply appropriate resampling for each variable/level combination
5. **Multiscale Writing**: Write consolidated datasets with proper metadata

#### Quality Assurance
- **Data Integrity**: Verify no data loss during consolidation
- **Coordinate Consistency**: Ensure all variables share consistent coordinate systems
- **Metadata Compliance**: Maintain GeoZarr-spec compliance
- **Performance Validation**: Measure storage and access improvements

### Phase 3: Advanced Features

#### Smart Chunking Strategy
```python
def calculate_optimal_chunks(self, resolution_level, data_shape):
    """Calculate chunks that optimize both storage and access patterns"""
    base_chunk = 1024
    level_factor = 2 ** resolution_level
    optimal_chunk = min(base_chunk, data_shape[-1] // level_factor)
    return (1, optimal_chunk, optimal_chunk)  # For 3D (band, y, x)
```

#### Compression Optimization
```python
COMPRESSION_CONFIG = {
    'reflectance': {'codec': 'blosc', 'level': 5, 'shuffle': True},
    'classification': {'codec': 'blosc', 'level': 9, 'shuffle': False},
    'quality': {'codec': 'blosc', 'level': 7, 'shuffle': True},
    'probability': {'codec': 'blosc', 'level': 6, 'shuffle': True}
}
```

## Expected Benefits Analysis

### Storage Optimization
- **Estimated reduction**: 40-60% fewer zarr chunks
- **Metadata efficiency**: ~90% reduction in .zmetadata files
- **Redundancy elimination**: Remove duplicate spatial reference data
- **Compression synergy**: Better compression ratios with consolidated data

### Access Pattern Improvements  
- **Faster discovery**: 3 top-level groups vs current 15+
- **Consistent multiscale**: Single pyramid instead of separate resolution trees
- **Simplified APIs**: Users access data by scale level, not resolution group
- **Better caching**: Consolidated chunks improve filesystem performance

### Scientific Workflow Benefits
- **Band co-registration**: All bands at same level guaranteed co-registered
- **Quality correlation**: Quality data co-located with measurements
- **Scale-aware processing**: Natural support for multi-resolution analysis
- **Simplified subsetting**: Single coordinate system across all variables at each level

## Technical Challenges & Solutions

### Challenge 1: Mixed Resolution Data
**Problem**: Different bands have different native resolutions
**Solution**: Store only at native resolution, use resampling for access at other levels

### Challenge 2: Quality Data Alignment
**Problem**: Quality data needs to align with corresponding measurement bands
**Solution**: Resample quality data to match measurement resolution at each level

### Challenge 3: Coordinate System Consistency
**Problem**: Ensuring consistent coordinates across all variables
**Solution**: Use master coordinate grid for each level, snap all data to this grid

### Challenge 4: Backward Compatibility
**Problem**: Existing tools expect current structure
**Solution**: Provide mapping utilities and clear migration documentation

## Implementation Timeline

### Week 1-2: Infrastructure
- Create new module files
- Implement band mapping and resolution logic
- Develop resampling engine
- Create basic converter structure

### Week 3-4: Core Conversion
- Implement measurements multiscale creation
- Develop geometry and meteorology consolidation
- Add multiscale metadata generation
- Create chunking and compression optimization

### Week 5-6: Validation & Testing
- Implement data integrity checks
- Create performance benchmarking
- Add error handling and edge cases
- Develop unit and integration tests

### Week 7-8: Documentation & Examples
- Create user documentation
- Develop example notebooks
- Add CLI integration
- Performance comparison analysis

## Usage Interface

### CLI Integration
```bash
# New optimized conversion
eopf-geozarr convert-s2-optimized input.zarr output.zarr \
    --spatial-chunk 1024 \
    --enable-sharding \
    --compression-level 5

# Validation
eopf-geozarr validate-s2 output.zarr --check-optimization
```

### Python API
```python
from eopf_geozarr.s2_converter import convert_s2_optimized

# Convert with optimization
convert_s2_optimized(
    input_datatree=dt,
    output_path="optimized.zarr",
    spatial_chunk=1024,
    enable_sharding=True,
    compression_preset='balanced'
)
```

## Validation Criteria

### Storage Efficiency
- [ ] <50% of original zarr chunk count
- [ ] <30% of original metadata files
- [ ] ≥20% reduction in total storage size
- [ ] Consistent compression ratios across levels

### Data Integrity
- [ ] Bit-exact preservation of native resolution data  
- [ ] Consistent coordinate systems across all levels
- [ ] Proper handling of nodata/fill values
- [ ] Metadata preservation and enhancement

### Performance
- [ ] ≥2x faster dataset opening time
- [ ] ≥1.5x faster band access time  
- [ ] Reduced memory overhead for multiscale operations
- [ ] Better parallel access patterns

### Compliance
- [ ] Full GeoZarr-spec compliance maintained
- [ ] CF conventions adherence
- [ ] STAC metadata compatibility
- [ ] Cloud-optimized access patterns

## Risk Mitigation

### Data Loss Prevention
- Comprehensive validation pipeline
- Bit-level comparison tools  
- Automated regression testing
- Rollback capabilities

### Performance Regression
- Benchmarking against current implementation
- Memory usage monitoring
- Access pattern optimization
- Chunking strategy validation

### User Adoption
- Clear migration guides
- Backward compatibility tools
- Performance demonstrations
- Community feedback integration

This plan provides a roadmap for creating a significantly more efficient and user-friendly Sentinel-2 zarr format while maintaining scientific integrity and improving performance across multiple use cases.