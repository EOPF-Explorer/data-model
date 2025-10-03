# Multiscales Attribute Extension for Zarr

- **Extension Name**: Multiscales Attribute Extension
- **Version**: 0.1.0
- **Extension Type**: Attribute
- **Status**: Proposed
- **Owners**: @emmanuelmathot

## Description

This specification defines a JSON object that encodes multiscale pyramid information for data stored in Zarr groups under the `multiscales` key in the attributes of Zarr groups. This is a domain-agnostic specification that describes the hierarchical layout of Zarr groups representing different resolution levels and the transformations between them.

## Motivation

- Provides standardized multiscale pyramid encoding applicable across domains (geospatial, bioimaging, etc.)
- Supports flexible decimation schemes (factor-of-2, factor-of-3, custom factors)
- Explicitly captures scale and translation transformations induced by downsampling
- Enables optimized data access patterns for visualization and analysis at different scales
- Composable with domain-specific metadata (e.g., geo/proj for geospatial CRS information)

## Background

This specification emerged from discussions between the geospatial and bioimaging communities about multiscale data representation in Zarr. While both domains need multiscale pyramids, they differ in how spatial metadata is handled:

- **Bioimaging** (OME-NGFF): Multiscale metadata includes all spatial transformation information
- **Geospatial**: Coordinate Reference System (CRS) information is typically separate from multiscale metadata

This generic specification captures the **transformation induced by downsampling** (scale and translation), allowing domain-specific extensions to provide additional spatial metadata as needed.

## Inheritance Model

The `multiscales` key is defined at the group level and applies to the hierarchical structure within that group. There is no inheritance of `multiscales` metadata from parent groups to child groups. Each multiscale group defines its own pyramid structure independently.

## Specification

The `multiscales` key can be added to Zarr group attributes to define multiscale pyramid information.

<!-- GENERATED_SCHEMA_DOCS_START -->
**`multiscales` Properties**

|                       | Type       | Description                                          | Required | Reference                                                    |
| --------------------- | ---------- | ---------------------------------------------------- | -------- | ------------------------------------------------------------ |
| **version**           | `string`   | Multiscales metadata version                         | ✓ Yes    | [multiscales.version](#multiscalesversion)                   |
| **layout**            | `[object]` | Array of objects representing the pyramid layout     | ✓ Yes    | [multiscales.layout](#multiscaleslayout)                     |
| **resampling_method** | `string`   | Resampling method used for downsampling              | No       | [multiscales.resampling_method](#multiscalesresampling_method) |

### Field Details

Additional properties are allowed.

#### multiscales.version

Multiscales metadata version

* **Type**: `string`
* **Required**: ✓ Yes
* **Allowed values**: `0.1.0`

#### multiscales.layout

Array of objects representing the pyramid layout and transformation relationships

* **Type**: array of `object`
* **Required**: Yes

This field SHALL describe the pyramid hierarchy with an array of objects representing each resolution level, ordered from highest to lowest resolution. Each object contains:

- **`group`** (required): Group name for this resolution level
- **`from_group`** (optional): Source group used to generate this level
- **`factors`** (optional): Array of decimation factors per axis (e.g., `[2, 2]` for 2x decimation)
- **`scale`** (optional): Array of scale factors per axis describing the resolution change
- **`translation`** (optional): Array of translation offsets per axis in the coordinate space
- **`resampling_method`** (optional): Resampling method for this specific level

The first level typically contains only the `group` field (native resolution), while subsequent levels include transformation information.

**Transformation Semantics**:

The `scale` and `translation` parameters describe how to map from array indices to a coordinate space at each level. For downsampling operations:

- **Scale** represents the multiplicative factor applied to coordinates (e.g., scale of 2.0 means one pixel represents twice the coordinate span)
- **Translation** represents the coordinate offset, useful when downsampling takes a subset of the original sampling grid

These transformations allow clients to determine the spatial extent of each pyramid level without needing to understand the specific downsampling algorithm.

#### multiscales.resampling_method

Resampling method used for downsampling

* **Type**: `string`
* **Required**: No
* **Allowed values**: `"nearest"`, `"average"`, `"bilinear"`, `"cubic"`, `"cubic_spline"`, `"lanczos"`, `"mode"`, `"max"`, `"min"`, `"med"`, `"sum"`, `"q1"`, `"q3"`, `"rms"`, `"gauss"`
* **Default**: `"nearest"`

The same method SHALL apply across all levels unless overridden at the level-specific `resampling_method`.

<!-- GENERATED_SCHEMA_DOCS_END -->

### Hierarchical Layout

Multiscale datasets SHOULD follow a specific hierarchical structure:

1. **Multiscale Group**: Contains `multiscales` metadata
2. **Resolution Level Groups**: Child groups containing data at different resolutions

```
multiscales/                 # Group with `multiscales` metadata
├── 0/                       # First resolution level (highest resolution)
│   ├── data                 # Data variable
│   └── ...
├── 1/                       # Second resolution level
│   ├── data                 # Data at lower resolution
│   └── ...
└── 2/                       # Third resolution level
    ├── data
    └── ...
```

All levels SHOULD be stored in child groups with names matching layout keys (e.g., `0`, `1`, `2`, or custom names).

### Group Discovery Methods

The multiscales metadata enables complete discovery of the multiscale collection structure through the layout mechanism:

- The `layout` definition specifies the exact set of resolution levels through its array of group names
- Each group name corresponds to a child group in the multiscale hierarchy
- Variable discovery within each level follows standard Zarr metadata conventions

### Consolidated Metadata

**Consolidated metadata is HIGHLY RECOMMENDED for multiscale groups** to ensure complete discoverability of pyramid structure and metadata without requiring individual access to each child dataset.

#### Requirements

1. **Zarr Consolidated Metadata**: The multiscale group SHOULD use Zarr's consolidated metadata feature to expose metadata from all child groups and arrays at the group level.

2. **Variable Discovery**: The consolidated metadata SHOULD include complete variable listings for all resolution levels, enabling clients to understand the full pyramid structure without traversing child groups.

### Validation Rules

- **Level Consistency**: Resolution level group names SHALL match children group path values in the `layout` array
- **Transformation Consistency**: If both `factors` and `scale` are provided, they SHOULD be consistent with each other

## Examples

### Example 1: Simple Power-of-2 Pyramid

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "multiscales": {
      "version": "0.1.0",
      "layout": [
        {
          "group": "0"
        },
        {
          "group": "1",
          "from_group": "0",
          "factors": [2, 2],
          "scale": [2.0, 2.0],
          "translation": [0.0, 0.0],
          "resampling_method": "average"
        },
        {
          "group": "2",
          "from_group": "1",
          "factors": [2, 2],
          "scale": [4.0, 4.0],
          "translation": [0.0, 0.0],
          "resampling_method": "average"
        }
      ],
      "resampling_method": "average"
    }
  }
}
```

### Example 2: Custom Pyramid Levels

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "multiscales": {
      "version": "0.1.0",
      "layout": [
        {
          "group": "full"
        },
        {
          "group": "half",
          "from_group": "full",
          "scale": [2.0, 2.0]
        },
        {
          "group": "quarter",
          "from_group": "half",
          "scale": [4.0, 4.0]
        },
        {
          "group": "eighth",
          "from_group": "quarter",
          "scale": [8.0, 8.0]
        }
      ]
    }
  }
}
```

## Composition with Domain-Specific Metadata

This generic multiscales specification is designed to be composed with domain-specific metadata:

### Geospatial Data

For geospatial data, combine with `geo/proj` attributes to specify the Coordinate Reference System:

```json
{
  "zarr_format": 3,
  "node_type": "group",
  "attributes": {
    "multiscales": {
      "version": "0.1.0",
      "layout": [
        {"group": "0"},
        {"group": "1", "from_group": "0", "factors": [2, 2], "scale": [2.0, 2.0]}
      ]
    },
    "geo": {
      "proj": {
        "version": "0.1",
        "code": "EPSG:32633",
        "transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0],
        "bbox": [500000.0, 4900000.0, 600000.0, 5000000.0]
      }
    }
  }
}
```

At each resolution level, the `geo/proj` metadata would specify the appropriate transform for that resolution.

### Bioimaging Data

For bioimaging, spatial metadata can be integrated directly at each level following OME-NGFF conventions while using this specification for the pyramid structure.

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

### Scale and Translation Parameters

The `scale` and `translation` parameters explicitly capture the transformation induced by downsampling. This approach has several advantages:

1. **Explicit vs. Implicit**: Clients don't need to infer transformations from decimation factors
2. **Flexibility**: Supports arbitrary downsampling schemes beyond simple decimation
3. **Composability**: Domain-specific coordinate systems can build upon these transformations

### Relationship to Decimation Factors

The `factors` field is provided for convenience and documentation purposes. The `scale` field is the authoritative source for the resolution relationship. When both are present, they should be consistent.

## References

- [OME-NGFF Multiscale Specification](https://ngff.openmicroscopy.org/latest/#multiscale-md)
- [Zarr Specifications Discussion on Multiscales](https://github.com/zarr-developers/zarr-specs/issues/50)
- [GeoZarr Specification](https://github.com/zarr-developers/geozarr-spec)
