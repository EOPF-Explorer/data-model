# Geo Multiscales Attribute Extension for Zarr

- **Extension Name**: Geo Multiscales Attribute Extension
- **Version**: 0.1.0
- **Extension Type**: Attribute
- **Status**: Proposed
- **Owners**: @emmanuelmathot

## Description

This specification defines a JSON object that encodes multiscale pyramid information for geospatial data stored in Zarr groups under the `multiscales` key within the `geo` dictionary in the attributes of Zarr groups. Additionally, it specifies the hierarchical layout of Zarr groups representing different resolution levels, discovery mechanisms for clients to enumerate available levels and variables, and requirements for consolidated metadata to ensure complete discoverability of the multiscale structure.

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

|                       | Type       | Description                                          | Required | Reference                                                                   |
| --------------------- | ---------- | ---------------------------------------------------- | -------- | --------------------------------------------------------------------------- |
| **version**           | `string`   | Multiscales metadata version                         | ✓ Yes    | [geo -> multiscales.version](#geo---multiscalesversion)                     |
| **layout**            | `[object]` | Array of objects representing the pyramid layout | ✓ Yes    | [geo -> multiscales.layout](#geo---multiscaleslayout)                       |
| **resampling_method** | `string`   | Resampling method used for downsampling              | No       | [geo -> multiscales.resampling_method](#geo---multiscalesresampling_method) |

### Field Details

Additional properties are allowed.

#### geo -> multiscales.version

Multiscales metadata version

* **Type**: `string`
* **Required**: ✓ Yes
* **Allowed values**: `0.1.0`

#### geo -> multiscales.layout

Array of objects representing the pyramid layout and decimation relationships

* **Type**: array of `object`
* **Required**: Yes

This field SHALL describe the pyramid hierarchy with an array of objects representing each resolution level, ordered from highest to lowest resolution. Each object contains:

- **`group`** (required): Group name for this resolution level
- **`from_group`** (optional): Source group used to generate this level 
- **`factors`** (optional): Array of decimation factors per axis (e.g., `[2, 2]` for 2x decimation in X and Y)
- **`resampling_method`** (optional): Resampling method for this specific level

The first level typically contains only the `group` field (native resolution), while subsequent levels include derivation information.

#### geo -> multiscales.resampling_method

Resampling method used for downsampling

* **Type**: `string`
* **Required**: No
* **Allowed values**: `"nearest"`, `"average"`, `"bilinear"`, `"cubic"`, `"cubic_spline"`, `"lanczos"`, `"mode"`, `"max"`, `"min"`, `"med"`, `"sum"`, `"q1"`, `"q3"`, `"rms"`, `"gauss"`
* **Default**: `"nearest"`

The same method SHALL apply across all levels.

<!-- GENERATED_SCHEMA_DOCS_END -->

### Hierarchical Layout

Multiscale datasets SHOULD follow a specific hierarchical structure that accommodates both native resolution storage and overview levels:

1. **Multiscale Group**: Contains `multiscales` metadata
2. **Overview Level Groups**: Child groups containing overview data at different resolutions

```
multiscales/                 # Group with `multiscales` metadata
├── 0/                       # First overview level
│   ├── b01                  # Native resolution variable
|   ├── b02                  # Native resolution variable
|   ├── b03                  # Native resolution variable
|   ├── b04                  # Native resolution variable
|   ├── y          # Coordinate variable
|   ├── x          # Coordinate variable
├── 1/                       # Second overview level
│   ├── b01                  # All bands available at overview level
│   ├── b02
│   ├── b03
│   ├── ...
│   ├── y
│   └── x
└── 2/                       # Third overview level
    ├── b01
    ├── b02
    ├── b03
    ├── ...
    ├── y
    └── x
```

All levels SHOULD be stored in child groups with names matching layout keys (e.g., `0`, `1`, `2`, or custom names)

> [!Note] Layout can describe native resolution stored in the multiscale group directly by using the key `.` (dot) to represent the current group. This is not recommended but MAY be used for backward compatibility with existing datasets that are augmented with multiscale metadata. It is important to acknowledge that this layout is less optimal for clients and MAY lead to errors. For instance, xarray's `open_dataset` function does not support data tree where parent and children shape do not align.

### Group Discovery Methods

The multiscales metadata enables complete discovery of the multiscale collection structure through a simple layout mechanisms:

- The `layout` definition specifies the exact set of zoom levels through its array of group names
- Each group name corresponds to a child group in the multiscale hierarchy
- Variable discovery within each zoom level group follows standard Zarr metadata conventions and should use the consolidated metadata feature for efficiency

### Consolidated Metadata Requirements

**Consolidated metadata is HIGHLY RECOMMENDED for multiscale groups** to ensure complete discoverability of pyramid structure and metadata without requiring individual access to each child dataset.

#### Requirements

1. **Zarr Consolidated Metadata**: The multiscale group SHALL use Zarr's consolidated metadata feature to expose metadata from all child groups and arrays at the group level.

2. **Variable Discovery**: The consolidated metadata SHALL include complete variable listings for all resolution levels, enabling clients to understand the full pyramid structure without traversing child groups.

3. **Projection Information Access**: All projection information via the [`geo/proj` attribute](../proj/README.md) from child datasets SHALL be accessible through the consolidated metadata at the multiscale group level. According to the attributes provided, the client shall be able to discover the CRS, bounding box, and resolution information from any resolution level.

#### Client Implementation Guidelines

1. **Priority Order**: Clients SHOULD first attempt to read consolidated metadata, falling back to individual metadata requests only if consolidated metadata is unavailable.

2. **Projection Discovery**: Use consolidated metadata to discover CRS information from any resolution level, typically from the native resolution or a representative overview level.

3. **Variable Enumeration**: Enumerate available variables across all resolution levels using the consolidated metadata catalog.

4. **Fallback Behavior**: When variables are not available at the optimal resolution level, use the consolidated metadata to identify the finest available resolution level containing that variable.

### Validation Rules

- **Consolidated Metadata**: Multiscale groups SHALL provide consolidated metadata as specified above
- **Level Consistency**: Resolution level group names SHALL match children group path values in the `layout` array
- **Coordinate System Consistency**: All resolution levels SHALL use the same coordinate reference system

## Examples

### Example 1: Simple Multiscale Pyramid with UTM Grid

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "geo": {
      "proj": {
        "epsg": 32633,
        "bbox": [500000.0, 0.0, 600000.0, 1000000.0],
      },
      "multiscales": {
        "version": "0.1.0",
        "layout": [
          {"group": "0"}, 
          {"group": "1", "from_group": "0", "factors": [2, 2], "resampling_method": "average"},
          {"group": "2", "from_group": "1", "factors": [2, 2], "resampling_method": "average"},
          {"group": "3", "from_group": "2", "factors": [2, 2], "resampling_method": "average"}
        ]
      }
    }
  },
  "consolidated_metadata": {
    "kind": "inline",
    "must_understand": false,
    "metadata": { 
      "0": {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
          "geo": {
            "proj": {
              "epsg": 32633,
              "bbox": [50000.0, 0.0, 60000.0, 100000.0],
              "transform": [10.0, 0.0, 50000.0, 0.0, -10.0, 100000.0, 0.0, 0.0, 1.0]
            }
          }
        }
      },
      "0/b01": {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10000, 10000],
        "dtype": "<u2",
        "fill_value": 0,
        "codecs": [...],
        "attributes": {}
      },
      ...
      "1": {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
          "geo": {
            "proj": {
              "epsg": 32633,
              "bbox": [50000.0, 0.0, 60000.0, 100000.0],
              "transform": [20.0, 0.0, 50000.0, 0.0, -20.0, 100000.0, 0.0, 0.0, 1.0]
            }
          }
        }
      },
      "1/b01": {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [5000, 5000],
        "dtype": "<u2",
        "fill_value": 0,
        "codecs": [...],
        "attributes": {}
      },
      ...
      "2": {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
          "geo": {
            "proj": {
              "epsg": 32633,
              "bbox": [50000.0, 0.0, 60000.0, 100000.0],
              "transform": [40.0, 0.0, 50000.0, 0.0, -40.0, 100000.0, 0.0, 0.0, 1.0]
            }
          }
        }
      },
      "2/b01": {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [2500, 2500],
        "dtype": "<u2",
        "fill_value": 0,
        "codecs": [...],
        "attributes": {}
      },
      ...
    }
  }
}
```

### Example 2: WebMercatorQuad TileMatrixSet-Compatible Pyramid

This example shows a multiscale pyramid that follows the OGC WebMercatorQuad TileMatrixSet structure, commonly used for web mapping applications.

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "geo": {
      "proj": {
        "epsg": 3857,
        "bbox": [-20037508.34, -20037508.34, 20037508.34, 20037508.34]
      },
      "multiscales": {
        "version": "0.1.0",
        "layout": [
          {"group": "18"}, 
          {"group": "17", "from_group": "18", "factors": [2, 2], "resampling_method": "average"},
          {"group": "16", "from_group": "17", "factors": [2, 2], "resampling_method": "average"},
          {"group": "15", "from_group": "16", "factors": [2, 2], "resampling_method": "average"},
          {"group": "14", "from_group": "15", "factors": [2, 2], "resampling_method": "average"}
        ]
      }
    }
  },
  "consolidated_metadata": {
    "kind": "inline",
    "must_understand": false,
    "metadata": {
      "18": {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
          "geo": {
            "proj": {
              "epsg": 3857,
              "bbox": [-20037508.34, -20037508.34, 20037508.34, 20037508.34],
              "transform": [0.5971642834779395, 0.0, -20037508.34, 0.0, -0.5971642834779395, 20037508.34, 0.0, 0.0, 1.0]
            }
          }
        }
      },
      "18/red": {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [8192, 8192],
        "dtype": "<u1",
        "fill_value": 0,
        "codecs": [...],
        "attributes": {}
      },
      ...
      "17": {
        "zarr_format": 3,
        "node_type": "group", 
        "attributes": {
          "geo": {
            "proj": {
              "epsg": 3857,
              "bbox": [-20037508.34, -20037508.34, 20037508.34, 20037508.34],
              "transform": [1.1943285669558790, 0.0, -20037508.34, 0.0, -1.1943285669558790, 20037508.34, 0.0, 0.0, 1.0]
            }
          }
        }
      },
      "17/red": {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4096, 4096],
        "dtype": "<u1",
        "fill_value": 0,
        "codecs": [...],
        "attributes": {}
      },
      ...
    }
  }
}
```

**Key aspects of WebMercatorQuad compatibility:**

- **EPSG:3857 projection**: Uses Web Mercator coordinate system
- **Zoom level naming**: Uses tile matrix identifiers (18, 17, 16, etc.) as group names
- **Power-of-2 decimation**: Each level has half the resolution of the previous level
- **256x256 chunk alignment**: Chunks align with standard web map tile size
- **Global extent**: Covers the full Web Mercator extent

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

## Compatibility Notes

- The specification uses a simple layout-based approach for maximum flexibility and ease of implementation
- Consolidated metadata at the multiscale group level provides complete information about all child datasets
- Integration with existing `geo.proj` attributes provides complete geospatial metadata coverage
- Hierarchical storage with explicit layout definition enables efficient discovery and access patterns

## References

- [OGC TileMatrixSet 2.0 Specification](https://docs.ogc.org/is/17-083r4/17-083r4.html)
- [GeoZarr Specification](https://github.com/zarr-developers/geozarr-spec)
- [OME-NGFF Multiscale Specification](https://ngff.openmicroscopy.org/latest/#multiscale-md)
- [GDAL Raster Data Model - Overviews](https://gdal.org/en/stable/user/raster_data_model.html#overviews)
