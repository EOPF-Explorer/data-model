# Geo Multiscales Attribute Extension for Zarr

- **Extension Name**: Geo Multiscales Attribute Extension
- **Version**: 0.1.0
- **Extension Type**: Attribute
- **Status**: Proposed
- **Owners**: @emmanuelmathot

## Description

This specification defines a JSON object that encodes multiscale pyramid information for geospatial data stored in Zarr groups under the `multiscales` key within the `geo` dictionary in the attributes of Zarr groups. Additionally, it specifies the hierarchical layout of datasets representing different resolution levels, discovery mechanisms for clients to enumerate available levels and variables, and requirements for consolidated metadata to ensure complete discoverability of the multiscale structure.

## Motivation

- Provides standardized multiscale pyramid encoding for geospatial overviews
- Supports flexible decimation schemes (factor-of-2, factor-of-3, custom factors)
- Compatible with OGC TileMatrixSet 2.0 specification while enabling generic pyramids
- Supports both consistent and inconsistent pyramid structures (where variables may not be available at all resolution levels)
- Enables optimized data access patterns for visualization and analysis at different scales
- Accommodates scientific coordinate systems beyond web mapping (UTM, polar stereographic, etc.)

## Inheritance Model

The `multiscales` key under the `geo` object is defined at the group level and applies to the hierarchical structure within that group.
There is no inheritance of `multiscales` metadata from parent groups to child groups. Each multiscale group defines its own pyramid structure independently.

## Specification

The `multiscales` key under the `geo` dictionary can be added to Zarr groups to define multiscale pyramid information.

<!-- GENERATED_SCHEMA_DOCS_START -->
**`geo -> multiscales` Properties**

|                            | Type               | Description                                        | Required | Reference                                                                             |
| -------------------------- | ------------------ | -------------------------------------------------- | -------- | ------------------------------------------------------------------------------------- |
| **version**                | `string`           | Multiscales metadata version                       | ✓ Yes   | [geo -> multiscales.version](#geo---multiscalesversion)                               |
| **tile_matrix_set**        | `string \| object` | Tile matrix set definition or reference            | No       | [geo -> multiscales.tile_matrix_set](#geo---multiscalestile_matrix_set)               |
| **resampling_method**      | `string`           | Resampling method used for downsampling            | No       | [geo -> multiscales.resampling_method](#geo---multiscalesresampling_method)           |
| **tile_matrix_set_limits** | `object`           | Optional limits for available tiles per zoom level | No       | [geo -> multiscales.tile_matrix_set_limits](#geo---multiscalestile_matrix_set_limits) |

### Field Details

Additional properties are allowed.

#### geo -> multiscales.version

Multiscales metadata version

* **Type**: `string`
* **Required**: ✓ Yes
* **Allowed values**: `0.1`

#### geo -> multiscales.layout

TileMatrixSet definition or reference

* **Type**: `string | object`
* **Required**: No

This field can contain either:
1. **Reference by identifier**: A string identifier referencing a well-known TileMatrixSet (e.g., "WebMercatorQuad")
2. **URI reference**: A URI pointing to a JSON document describing the tile matrix set
3. **Inline definition**: A complete TileMatrixSet JSON object following OGC TileMatrixSet 2.0 specification

**Reference by identifier:**
```json
{
  "tile_matrix_set": "WebMercatorQuad"
}
```

**URI reference:**
```json
{
  "tile_matrix_set": "https://maps.example.org/tileMatrixSets/WebMercatorQuad.json"
}
```

**Inline definition:**
```json
{
  "tile_matrix_set": {
    "id": "Custom_Grid",
    "title": "Custom Grid for Scientific Data",
    "crs": "EPSG:4326",
    "tileMatrices": [
      {
        "id": "0",
        "scaleDenominator": 0.703125,
        "cellSize": 0.0625,
        "pointOfOrigin": [-180.0, 90.0],
        "tileWidth": 256,
        "tileHeight": 256,
        "matrixWidth": 2,
        "matrixHeight": 1
      }
    ]
  }
}
```

#### geo -> multiscales.resampling_method

Resampling method used for downsampling

* **Type**: `string`
* **Required**: No
* **Allowed values**: `"nearest"`, `"average"`, `"bilinear"`, `"cubic"`, `"cubic_spline"`, `"lanczos"`, `"mode"`, `"max"`, `"min"`, `"med"`, `"sum"`, `"q1"`, `"q3"`, `"rms"`, `"gauss"`
* **Default**: `"nearest"`

The same method SHALL apply across all levels.

#### geo -> multiscales.tile_matrix_set_limits

Optional limits for available tiles per zoom level

* **Type**: `object`
* **Required**: No

Defines the available tile ranges for each zoom level. Keys must match TileMatrix.id values from the TileMatrixSet.

```json
{
  "tile_matrix_set_limits": {
    "0": {
      "minTileRow": 0,
      "maxTileRow": 0,
      "minTileCol": 0,
      "maxTileCol": 0
    },
    "1": {
      "minTileRow": 0,
      "maxTileRow": 1,
      "minTileCol": 0,
      "maxTileCol": 1
    }
  }
}
```


<!-- GENERATED_SCHEMA_DOCS_END -->

### Hierarchical Layout

Multiscale datasets follow a specific hierarchical structure that accommodates both native resolution storage and overview levels:

1. **Dataset Group**: Contains native resolution data and multiscales metadata
2. **Overview Level Groups**: Child groups containing overview data at different resolutions

```
/measurements/               # Dataset Group with multiscales metadata
├── 0/                       # First overview level
|   ├── b02                  # Native resolution variable
|   ├── b03                  # Native resolution variable
|   ├── b04                  # Native resolution variable
|   ├── spatial_ref          # Coordinate reference variable
├── 1/                       # First overview level
│   ├── b01                  # All bands available at overview level
│   ├── b02
│   ├── b03
│   ├── ...
│   └── spatial_ref
└── 2/                       # Second overview level
    ├── b01
    ├── b02
    ├── b03
    ├── ...
    └── spatial_ref
```

**Key principles:**
- Native resolution variables are stored directly in the Dataset Group (not in a separate "0/" group)
- Overview levels are stored in child groups with names matching TileMatrix identifiers
- This approach maintains efficiency by avoiding the need to restructure existing datasets when adding overviews

### Group Discovery Methods

The multiscales metadata enables complete discovery of the multiscale collection structure through multiple mechanisms:

1. **TileMatrixSet-based discovery**: 
   - The TileMatrixSet definition specifies the exact set of zoom levels through its tileMatrices array
   - Each TileMatrix.id value corresponds to a child group in the multiscale hierarchy
   - Variable discovery within each zoom level group follows standard Zarr metadata conventions

2. **Generic datasets-based discovery**:
   - The `datasets` array explicitly lists all resolution levels and their paths
   - Scale factors provide resolution relationships

3. **Explicit limits**:
   - `tile_matrix_set_limits` explicitly declares which zoom levels contain data
   - For storage backends that do not support directory listing, this is the primary mechanism for discovering available zoom levels

### Consolidated Metadata Requirements

**Consolidated metadata is MANDATORY for multiscale groups** to ensure complete discoverability of pyramid structure and metadata without requiring individual access to each child dataset.

#### Requirements

1. **Zarr Consolidated Metadata**: The multiscale group SHALL use Zarr's consolidated metadata feature to expose metadata from all child groups and arrays at the group level.

2. **Projection Information Access**: All projection information (CRS, transforms, grid mappings) from child datasets SHALL be accessible through the consolidated metadata at the multiscale group level.

3. **Variable Discovery**: The consolidated metadata SHALL include complete variable listings for all resolution levels, enabling clients to understand the full pyramid structure without traversing child groups.

4. **Coordinate Information**: Coordinate arrays and their metadata from all resolution levels SHALL be included in the consolidated metadata.

#### Client Implementation Guidelines

1. **Priority Order**: Clients SHOULD first attempt to read consolidated metadata, falling back to individual metadata requests only if consolidated metadata is unavailable.

2. **Projection Discovery**: Use consolidated metadata to discover CRS information from any resolution level, typically from the native resolution or a representative overview level.

3. **Variable Enumeration**: Enumerate available variables across all resolution levels using the consolidated metadata catalog.

4. **Fallback Behavior**: When variables are not available at the optimal resolution level, use the consolidated metadata to identify the finest available resolution level containing that variable.

### Validation Rules

- **Consolidated Metadata**: Multiscale groups SHALL provide consolidated metadata as specified above
- **Level Consistency**: Resolution level group names SHALL match either TileMatrix.id values (when using TileMatrixSet) or dataset path values
- **Structural Consistency**: All resolution level groups SHALL have the same member structure for variables they contain
- **Coordinate System Consistency**: All resolution levels SHALL use the same coordinate reference system
- **Chunking Alignment**: Chunks SHALL be aligned with the tile grid (1:1 mapping between chunks and tiles) when using TileMatrixSet

### Decimation Requirements and Custom Scaling

While TileMatrixSet commonly assumes quadtree decimation (scaling by factor of 2), custom TileMatrixSets MAY use alternative decimation factors:

- **Factor of 2 (quadtree)**: Standard web mapping approach where each zoom level has 4x more tiles
- **Factor of 3 (nonary tree)**: Each zoom level has 9x more tiles, useful for certain scientific gridding schemes  
- **Other integer factors**: Application-specific requirements may dictate alternative decimation

When using non-standard decimation factors, the TileMatrixSet definition SHALL explicitly specify the matrixWidth and matrixHeight values for each TileMatrix to ensure correct spatial alignment and resolution relationships.

## Examples

### Example 1: Simple TileMatrixSet Reference

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "geo": {
      "multiscales": {
        "version": "0.1",
        "tile_matrix_set": "WebMercatorQuad",
        "resampling_method": "average",
        "tile_matrix_set_limits": {
          "7": {"minTileRow": 42, "maxTileRow": 43, "minTileCol": 67, "maxTileCol": 68},
          "8": {"minTileRow": 85, "maxTileRow": 87, "minTileCol": 134, "maxTileCol": 137}
        }
      }
    }
  }
}
```

### Example 2: Custom UTM TileMatrixSet

```json
{
  "zarr_format": 3,
  "node_type": "group", 
  "attributes": {
    "geo": {
      "multiscales": {
        "version": "0.1",
        "tile_matrix_set": {
          "id": "UTM_Zone_33N_Custom",
          "title": "UTM Zone 33N for Sentinel-2 native resolution",
          "crs": "EPSG:32633",
          "orderedAxes": ["E", "N"],
          "tileMatrices": [
            {
              "id": "0",
              "scaleDenominator": 35.28,
              "cellSize": 10.0,
              "pointOfOrigin": [299960.0, 9000000.0],
              "tileWidth": 1024,
              "tileHeight": 1024,
              "matrixWidth": 1094,
              "matrixHeight": 1094
            },
            {
              "id": "1",
              "scaleDenominator": 70.56,
              "cellSize": 20.0,
              "pointOfOrigin": [299960.0, 9000000.0],
              "tileWidth": 512,
              "tileHeight": 512,
              "matrixWidth": 547,
              "matrixHeight": 547
            }
          ]
        },
        "resampling_method": "average"
      }
    }
  }
}
```

### Example 3: Generic Datasets-based Pyramid

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "geo": {
      "multiscales": {
        "version": "0.1",
        "datasets": [
          {"path": "", "scale": [1.0, 1.0, 1.0]},
          {"path": "level_1", "scale": [1.0, 2.0, 2.0]},
          {"path": "level_2", "scale": [1.0, 4.0, 4.0]}
        ],
        "resampling_method": "average"
      }
    }
  }
}
```

### Example 4: Factor-of-3 Decimation

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "geo": {
      "multiscales": {
        "version": "0.1",
        "tile_matrix_set": {
          "id": "Custom_Nonary_Grid",
          "crs": "EPSG:4326",
          "tileMatrices": [
            {
              "id": "0",
              "matrixWidth": 1,
              "matrixHeight": 1,
              "tileWidth": 256,
              "tileHeight": 256
            },
            {
              "id": "1",
              "matrixWidth": 3,
              "matrixHeight": 3,
              "tileWidth": 256,
              "tileHeight": 256
            },
            {
              "id": "2",
              "matrixWidth": 9,
              "matrixHeight": 9,
              "tileWidth": 256,
              "tileHeight": 256
            }
          ]
        },
        "resampling_method": "average"
      }
    }
  }
}
```

## Versioning and Compatibility

This specification uses semantic versioning (SemVer) for version management:

- **Major version** changes indicate backward-incompatible changes to the attribute schema
- **Minor version** changes add new optional fields while maintaining backward compatibility  
- **Patch version** changes fix documentation, clarify behavior, or make other non-breaking updates

### Compatibility Guarantees

- Parsers MUST support all fields defined in their major version
- Parsers SHOULD gracefully handle unknown optional fields from newer minor versions
- Producers SHOULD include the `version` field to indicate specification compliance level

## Implementation Notes

### Inconsistent Pyramid Support

When implementing support for inconsistent pyramids:

1. **Variable Discovery**: Scan all resolution levels to build a comprehensive variable catalog
2. **Fallback Logic**: When a variable is not available at the optimal resolution level, fall back to the finest resolution level containing that variable
3. **Metadata Consistency**: Ensure that coordinate system and chunking information remains consistent across levels

### Performance Considerations

- **Chunking Alignment**: For TileMatrixSet-based pyramids, chunks SHOULD be aligned with the tile grid (1:1 mapping between chunks and tiles)
- **Chunk Sizes**: Chunk sizes SHOULD match the `tileWidth` and `tileHeight` declared in the TileMatrix
- **Compression**: Use compression codecs appropriate for the data type and use case
- **Access Patterns**: Structure data to optimize common access patterns (spatial locality, multi-resolution queries)

### TileMatrixSet Integration

- **CRS Consistency**: The spatial reference system declared in `supportedCRS` SHALL match the one declared in the corresponding `grid_mapping` of the data variables
- **Group Naming**: Group names in the multiscale hierarchy SHALL correspond exactly to the TileMatrix identifier values
- **Conflict Avoidance**: Additional groups or arrays MAY be present alongside zoom level groups, but SHALL NOT use names that conflict with TileMatrix identifiers

## Compatibility Notes

- The specification supports both TileMatrixSet-based and generic datasets-based approaches for maximum flexibility
- Consolidated metadata at the multiscale group level provides complete information about all child datasets
- Integration with existing `geo.proj` attributes provides complete geospatial metadata coverage
- Native resolution storage in the Dataset Group maintains efficiency and compatibility with existing datasets

## References

- [OGC TileMatrixSet 2.0 Specification](https://docs.ogc.org/is/17-083r4/17-083r4.html)
- [GeoZarr Specification](https://github.com/zarr-developers/geozarr-spec)
- [OME-NGFF Multiscale Specification](https://ngff.openmicroscopy.org/latest/#multiscale-md)
- [STAC Projection Extension](https://github.com/stac-extensions/projection)
