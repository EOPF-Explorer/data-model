# GeoZarr Mini Spec

This document specifies the GeoZarr model used in this repository. It is a "mini" version of the emerging [GeoZarr specification](https://geozarr.org/) that documents the specific subset of conventions this implementation supports, along with implementation-specific details.

GeoZarr is a set of modular, composable [Zarr conventions](https://geozarr.org/conventions) for storing multidimensional georeferenced grids. The specification is developed by the [OGC GeoZarr Standards Working Group](https://geozarr.org/) and is on track for Architecture Board review in summer 2026 (see [roadmap](https://geozarr.org/roadmap)). All three core conventions are currently at **Proposal** maturity, targeting **Candidate** status (3+ independent implementations) for GeoZarr V1.

> **Evolution note**: Earlier versions of this mini-spec documented a "V0 maximalist" approach based on GeoZarr 0.4 (TileMatrixSet multiscales, mandatory CF conventions, `grid_mapping` 0D arrays). That approach has been superseded by the modular Zarr Conventions described below, which were established at the Zarr Summit in Rome (December 2025) and are the basis of this implementation since v1.0.

## Relationship to Other Documentation

This mini spec is referenced by and aligns with:

- **[geozarr.org](https://geozarr.org/)** — Canonical home for the GeoZarr specification effort
- **[geozarr.org/conventions](https://geozarr.org/conventions)** — Core conventions reference (geo-proj, spatial, multiscales)
- **[zarr-conventions-spec](https://github.com/zarr-conventions/zarr-conventions-spec)** — The Zarr Conventions meta-framework
- **[EOPF Explorer Data Model](https://explorer.eopf.copernicus.eu/software-services/datamodel/)** — Project context and community leadership
- **[Architecture](architecture.md)** — Technical implementation details that follow this specification
- **[GeoZarr Specification Contribution](geozarr-specification-contribution.md)** — Our contributions to the GeoZarr spec based on this implementation
- **[Main Documentation](index.md)** — General library documentation and usage guides

## Zarr Conventions Framework

GeoZarr is built on the [Zarr Conventions Framework](https://github.com/zarr-conventions/zarr-conventions-spec). A convention is a set of attributes on a Zarr array or group that confer special meaning about the contained data. Each node that uses a convention **must** declare it in a `zarr_conventions` array within its attributes.

This implementation uses three core GeoZarr conventions:

| Convention | Namespace | UUID | Purpose |
| ---------- | --------- | ---- | ------- |
| [multiscales](https://github.com/zarr-conventions/multiscales) | `multiscales` | `d35379db-88df-4056-af3a-620245f8e347` | Pyramid layout for multi-resolution data |
| [geo-proj](https://github.com/zarr-conventions/geo-proj) | `proj:` | `f17cb550-5864-4468-aeb7-f3180cfb622f` | CRS and datum encoding |
| [spatial](https://github.com/zarr-conventions/spatial) | `spatial:` | `689b58e2-cf7b-45e0-9fff-9cfc0883d6b4` | Array index to spatial coordinate relationship |

Conventions are composable: a single Zarr group can declare and use all three simultaneously, with `spatial:*` and `proj:*` properties extending `multiscales` layout entries. See [geozarr.org/conventions](https://geozarr.org/conventions) for the full conventions reference and maturity framework.

## Spec conventions

### Array and Group attributes

This document only defines rules for a finite subset of the keys in Zarr array
and group attributes. Unless otherwise stated, any external keys in Zarr array and group attributes are consistent with this specification. This means this specification composes with the presence of, e.g., [CF metadata](https://cfconventions.org/), at different levels of the Zarr hierarchy. CF metadata is no longer required by this specification but remains fully compatible and may be included alongside the Zarr conventions described here.

Convention properties are placed at the root `attributes` level of the Zarr node, following the [Zarr Conventions Specification](https://github.com/zarr-conventions/zarr-conventions-spec).

## Organization

GeoZarr defines a Zarr hierarchy — a particular arrangement of Zarr arrays and groups and
their attributes. This document defines that hierarchy from the bottom-up, starting with arrays
and their attributes before moving to higher-level structures like groups and their attributes.

This specification targets **Zarr V3** exclusively. The geo-proj, spatial, and multiscales conventions are all Zarr V3.

## DataArray

A DataArray is a Zarr V3 array with named axes.

This section contains the rules for *individual* DataArrays. Additional
constraints on groups of DataArrays are defined in the section on [Datasets](#dataset).

### Attributes

No particular attributes are required for DataArrays.

### Array metadata

DataArrays must have at least 1 dimension — scalar arrays are not allowed. The
`dimension_names` field must be set, all elements must be strings, and they must all be unique.

| attribute         | constraint                 | notes                                  |
| ----------------- | -------------------------- | -------------------------------------- |
| `shape`           | at least 1 element         | No scalar arrays allowed               |
| `dimension_names` | an array of unique strings | All array axes must be uniquely named  |

### Example

```json
{
    "zarr.json": {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10,11,12],
        "data_type": "uint8",
        "chunk_key_encoding": {"name": "default", "configuration": {"separator" : "/"}},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10,11,12]}},
        "codecs": [{"name": "bytes"}],
        "dimension_names": ["lat", "lon", "time"],
        "storage_transformers": []
        }
}
```

## Dataset

A GeoZarr dataset is a Zarr group that contains Zarr arrays that together describe a measured quantity,
as well as arbitrary sub-groups.

### Attributes

There are no required attributes for Datasets. To qualify as a GeoZarr Dataset, the group must carry geospatial metadata via the `proj:` and `spatial:` conventions declared in its `zarr_conventions` attribute.

#### Geospatial Metadata

Geospatial reference information is encoded through two complementary conventions declared in the `zarr_conventions` array. Both conventions follow the [geo-proj](https://github.com/zarr-conventions/geo-proj) and [spatial](https://github.com/zarr-conventions/spatial) specifications.

**`proj:` convention** — defines the coordinate reference system (CRS) for the dataset. Defined once at the group level; it is inherited by all direct child arrays (which can override it individually).

| key | type | required | notes |
| --- | ---- | -------- | ----- |
| `proj:code` | string | conditional* | Authority:code identifier, e.g. `"EPSG:32633"`. Pattern: `^[A-Z]+:[0-9]+$` |
| `proj:wkt2` | string | conditional* | WKT2 (ISO 19162) CRS representation |
| `proj:projjson` | object | conditional* | PROJJSON CRS representation |

\* At least one of `proj:code`, `proj:wkt2`, or `proj:projjson` MUST be provided.

**`spatial:` convention** — defines the relationship between array indices and spatial coordinates. Can be set at the group level (applying to all direct child arrays) or per-array.

| key | type | required | notes |
| --- | ---- | -------- | ----- |
| `spatial:dimensions` | string[] | yes | Names of spatial dimensions in row-major order, e.g. `["Y", "X"]` |
| `spatial:transform` | number[6] | no | Affine transform coefficients `[a, b, c, d, e, f]` (Rasterio/Affine ordering). Maps array index `(i, j)` to coordinates via `x = a*i + b*j + c`, `y = d*i + e*j + f` |
| `spatial:bbox` | number[] | no | Bounding box `[xmin, ymin, xmax, ymax]` in the CRS coordinate space |
| `spatial:shape` | integer[] | no | Shape of spatial dimensions `[height, width]` |
| `spatial:registration` | string | no | Grid cell registration type: `"pixel"` (default, cell-registered, equivalent to GeoTIFF PixelIsArea) or `"node"` (grid-registered, equivalent to GeoTIFF PixelIsPoint) |

> [!Note]
> The `spatial:transform` uses **Rasterio/Affine coefficient ordering** `[a, b, c, d, e, f]`, which differs from GDAL's `GetGeoTransform` ordering `[c, a, b, f, d, e]`. Converting from GDAL: `spatial_transform = [GT(1), GT(2), GT(0), GT(4), GT(5), GT(3)]`.

> [!Note]
> CF metadata (`standard_name`, `grid_mapping` variables, coordinate variable attributes) is no longer required by this specification but remains compatible and may be included alongside the Zarr conventions. See [CF conventions](https://cfconventions.org) for reference.

### Members

If any member of a GeoZarr Dataset is an array, then it must comply with the [DataArray](#dataarray) definition.

If the Dataset contains a DataArray `D`, then for each dimension name `N` in the list of `D`'s named dimensions, 
the Dataset must contain a one-dimensional DataArray named `N` with a shape that matches the the length 
of `D` along the axis named by `N`. In this case, `D` is called a "data variable", and the each 
of `D` along the axis named by `N`. In this case, `D` is called a "data variable", and each 
DataArray matching a dimension names of `D` is called a "coordinate variable". 

> [!Note]
> These two definitions are not mutually exclusive, as a 1-dimensional DataArray named `D` with
dimension names `["D"]` is both a coordinate variable and a data variable.


#### Examples

This example shows a geospatial Dataset with two spatial data variables. The `zarr_conventions` array at the group level declares the `proj:` and `spatial:` conventions. The group-level `proj:code` applies to all child arrays, and `spatial:dimensions`, `spatial:transform`, and `spatial:bbox` describe the georeferencing.

```json
{
    "zarr.json": {
        "node_type": "group",
        "zarr_format": 3,
        "attributes": {
            "zarr_conventions": [
                {
                    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
                    "schema_url": "https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v1/schema.json",
                    "spec_url": "https://github.com/zarr-experimental/geo-proj/blob/v1/README.md",
                    "name": "proj:",
                    "description": "Coordinate reference system information for geospatial data"
                },
                {
                    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
                    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
                    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
                    "name": "spatial:",
                    "description": "Spatial coordinate information"
                }
            ],
            "proj:code": "EPSG:32633",
            "spatial:dimensions": ["Y", "X"],
            "spatial:transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0],
            "spatial:bbox": [500000.0, 4900000.0, 600000.0, 5000000.0]
        }
    },
    "red/zarr.json": {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10000, 10000],
        "data_type": "uint16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1024, 1024]}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "dimension_names": ["Y", "X"],
        "attributes": {}
    },
    "nir/zarr.json": {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10000, 10000],
        "data_type": "uint16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1024, 1024]}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "dimension_names": ["Y", "X"],
        "attributes": {}
    }
}
```

This example demonstrates a minimal Dataset with a single 1D array. A single array is only permitted if that array is one-dimensional, and the name of the DataArray in the Dataset matches the (single) dimension name defined for that DataArray. In this case `lat` is both a coordinate variable and a data variable.

```json
{
    "zarr.json": {
        "node_type": "group",
        "zarr_format": 3
    },
    "lat/zarr.json": {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "float64",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "dimension_names": ["lat"],
        "attributes": {}
    }
}
```

## Multiscale Dataset

A Multiscale Dataset is a Zarr group that contains multiple resolution levels (a pyramid) of the same data, following the [multiscales Zarr convention](https://github.com/zarr-conventions/multiscales).

Downsampling is a process in which a collection of localized data points is resampled on a subset of its original sampling locations. In the case of arrays, downsampling generally reduces an array's shape along at least one dimension. To downsample the contents of a Dataset `D` and generate a new Dataset `E`, all of the coordinate variable – data variable relationships in `D` must be preserved in `E`.

The downsampling transformation is thus well-defined for Datasets. Downsampling is often applied multiple times in a series to generate multiple levels of detail for a data variable.

### Implementation Approach

The implementation uses a **pyramid-based downsampling approach** with the following characteristics:

- **Variable Downsampling Factors**: Overview levels use optimal downsampling factors based on data characteristics (e.g., 2x, 3x). For Sentinel-2, this results in resolution levels: 10m → 20m (2x) → 60m (3x) → 120m (2x) → 360m (3x) → 720m (2x)
- **Pyramid Generation**: Overview levels are created sequentially from the previous level
- **Minimum Dimension Threshold**: Overview generation stops when the smallest dimension falls below a configurable threshold (default: 256 pixels)
- **Native CRS Preservation**: All overview levels maintain the same CRS, expressed via `proj:code` at the group level
- **Consistent Variable Structure**: Each overview level contains the same set of variables as the native resolution level

### Attributes

The `zarr_conventions` array at the Multiscale Dataset's root group MUST declare all three conventions: `multiscales`, `proj:`, and `spatial:`. The `multiscales` key then holds the pyramid layout.

| key | type | required | notes |
| --- | ---- | -------- | ----- |
| `zarr_conventions` | array | yes | Declares the multiscales, proj: and spatial: conventions |
| `multiscales` | [MultiscalesMetadata](#multiscalesmetadata) | yes | Pyramid layout for this group |
| `proj:code` | string | conditional* | CRS for all resolution levels (group-level inheritance) |
| `spatial:dimensions` | string[] | yes | Names of spatial dimensions, e.g. `["Y", "X"]` |
| `spatial:bbox` | number[] | no | Overall bounding box in the CRS coordinate space |

\* At least one of `proj:code`, `proj:wkt2`, or `proj:projjson` must be provided for geospatial datasets.

#### MultiscalesMetadata

The `multiscales` object has the following structure:

| key | type | required | notes |
| --- | ---- | -------- | ----- |
| `layout` | [[LayoutObject](#layoutobject), ...] | yes | Ordered array of resolution levels; must not be empty |
| `resampling_method` | string | no | Default resampling method used across all levels (e.g. `"average"`, `"nearest"`). Can be overridden per level |

#### LayoutObject

Each object in the `layout` array represents one resolution level:

| key | type | required | notes |
| --- | ---- | -------- | ----- |
| `asset` | string | yes | Path to the Zarr group (or array) for this level, relative to the multiscales group. E.g. `"r10m"`, `"0"` |
| `derived_from` | string | no | Path to the source level from which this level was generated. Required to define the transform |
| `transform` | [TransformObject](#transformobject) | conditional | Required when `derived_from` is present. Describes the relative coordinate transformation from the source level |
| `resampling_method` | string | no | Per-level override of the default resampling method |
| `spatial:shape` | integer[] | no | Shape of the spatial dimensions at this level `[height, width]` |
| `spatial:transform` | number[6] | no | Affine transform coefficients for this level, in Rasterio/Affine ordering `[a, b, c, d, e, f]` |

#### TransformObject

The `transform` object inside a `LayoutObject` captures the **relative** coordinate transformation from the `derived_from` level to this level. Note that this is separate from `spatial:transform`, which encodes the **absolute** georeferencing of a level.

| key | type | required | notes |
| --- | ---- | -------- | ----- |
| `scale` | number[] | no | Scale factors per axis. A scale of `2.0` means this level covers the same extent with half the resolution |
| `translation` | number[] | no | Translation offsets per axis in the coordinate space. For georeferenced pyramids sharing the same origin, this is `[0.0, 0.0]` |

> [!Note]
> `transform.scale` and `transform.translation` describe how coordinates change **between levels** (relative). `spatial:transform` placed directly on a layout entry describes the absolute affine georeferencing of that level. When composing with the spatial convention, always specify `spatial:transform` and `spatial:shape` explicitly at each level.

### Members

All of the members declared in the `multiscales.layout` must comply with the [Dataset](#dataset) definition. All resolution levels must have the exact same set of variable names within their group.

A Multiscale Dataset should not contain any members that are not declared in the `multiscales.layout`. Any additional Zarr arrays and groups are considered external to the GeoZarr model.

#### Chunking Recommendations

- **Consistent Chunking Strategy**: All data variables within a resolution level should use the same chunk shape to maintain spatial coherence
- **Memory Constraints**: Keep individual chunks under 100 MB
- **Cloud-Optimal Access**: Use `"/"` as separator in `chunk_key_encoding` to allow prefix-based range requests
- **Shard Alignment**: When using Zarr V3 sharding, align shard boundaries with the level's spatial extent

### Examples

#### Simple Power-of-2 Pyramid

A minimal 2-level geospatial pyramid — well-suited for web mapping or non-EO data:

```json
{
    "zarr.json": {
        "node_type": "group",
        "zarr_format": 3,
        "attributes": {
            "zarr_conventions": [
                {
                    "uuid": "d35379db-88df-4056-af3a-620245f8e347",
                    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json",
                    "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v1/README.md",
                    "name": "multiscales",
                    "description": "Multiscale layout of zarr datasets"
                },
                {
                    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
                    "schema_url": "https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v1/schema.json",
                    "spec_url": "https://github.com/zarr-experimental/geo-proj/blob/v1/README.md",
                    "name": "proj:",
                    "description": "Coordinate reference system information for geospatial data"
                },
                {
                    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
                    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
                    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
                    "name": "spatial:",
                    "description": "Spatial coordinate information"
                }
            ],
            "multiscales": {
                "layout": [
                    {
                        "asset": "0",
                        "transform": {"scale": [1.0, 1.0], "translation": [0.0, 0.0]},
                        "spatial:shape": [1024, 1024],
                        "spatial:transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0]
                    },
                    {
                        "asset": "1",
                        "derived_from": "0",
                        "transform": {"scale": [2.0, 2.0], "translation": [0.0, 0.0]},
                        "spatial:shape": [512, 512],
                        "spatial:transform": [20.0, 0.0, 500000.0, 0.0, -20.0, 5000000.0]
                    }
                ],
                "resampling_method": "average"
            },
            "proj:code": "EPSG:32633",
            "spatial:dimensions": ["Y", "X"],
            "spatial:bbox": [500000.0, 4890240.0, 510240.0, 5000000.0]
        }
    }
}
```

#### Sentinel-2 Multi-Resolution Pyramid

A complete Sentinel-2 L2A scene with 6 resolution levels (variable downsampling factors 2x and 3x). Resolution levels are named after pixel size: `r10m`, `r20m`, `r60m`, `r120m`, `r360m`, `r720m`. Each level's `derived_from` records the actual parent in the downsampling chain, and `transform.scale` records the factor used.

```json
{
    "zarr.json": {
        "node_type": "group",
        "zarr_format": 3,
        "attributes": {
            "zarr_conventions": [
                {
                    "uuid": "d35379db-88df-4056-af3a-620245f8e347",
                    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json",
                    "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v1/README.md",
                    "name": "multiscales",
                    "description": "Multiscale layout of zarr datasets"
                },
                {
                    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
                    "schema_url": "https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v1/schema.json",
                    "spec_url": "https://github.com/zarr-experimental/geo-proj/blob/v1/README.md",
                    "name": "proj:",
                    "description": "Coordinate reference system information for geospatial data"
                },
                {
                    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
                    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
                    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
                    "name": "spatial:",
                    "description": "Spatial coordinate information"
                }
            ],
            "multiscales": {
                "layout": [
                    {
                        "asset": "r10m",
                        "transform": {"scale": [1.0, 1.0]},
                        "spatial:shape": [10980, 10980],
                        "spatial:transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0]
                    },
                    {
                        "asset": "r20m",
                        "derived_from": "r10m",
                        "transform": {"scale": [2.0, 2.0], "translation": [0.0, 0.0]},
                        "spatial:shape": [5490, 5490],
                        "spatial:transform": [20.0, 0.0, 500000.0, 0.0, -20.0, 5000000.0]
                    },
                    {
                        "asset": "r60m",
                        "derived_from": "r10m",
                        "transform": {"scale": [6.0, 6.0], "translation": [0.0, 0.0]},
                        "spatial:shape": [1830, 1830],
                        "spatial:transform": [60.0, 0.0, 500000.0, 0.0, -60.0, 5000000.0]
                    },
                    {
                        "asset": "r120m",
                        "derived_from": "r60m",
                        "transform": {"scale": [2.0, 2.0], "translation": [0.0, 0.0]},
                        "spatial:shape": [915, 915],
                        "spatial:transform": [120.0, 0.0, 500000.0, 0.0, -120.0, 5000000.0]
                    },
                    {
                        "asset": "r360m",
                        "derived_from": "r120m",
                        "transform": {"scale": [3.0, 3.0], "translation": [0.0, 0.0]},
                        "spatial:shape": [305, 305],
                        "spatial:transform": [360.0, 0.0, 500000.0, 0.0, -360.0, 5000000.0]
                    },
                    {
                        "asset": "r720m",
                        "derived_from": "r360m",
                        "transform": {"scale": [2.0, 2.0], "translation": [0.0, 0.0]},
                        "spatial:shape": [153, 153],
                        "spatial:transform": [720.0, 0.0, 500000.0, 0.0, -720.0, 5000000.0]
                    }
                ],
                "resampling_method": "average"
            },
            "proj:code": "EPSG:32633",
            "spatial:dimensions": ["Y", "X"],
            "spatial:bbox": [500000.0, 4890220.0, 609800.0, 5000000.0]
        }
    }
}
```

This closely mirrors the [sentinel-2-multiresolution.json](https://github.com/zarr-conventions/multiscales/blob/main/examples/sentinel-2-multiresolution.json) example from the multiscales convention repository.

#### File System Layout

The Sentinel-2 example above corresponds to the following directory structure on disk:

```
S2_scene.zarr/
├── zarr.json                    # Root group: zarr_conventions + multiscales + proj: + spatial: metadata
├── r10m/                        # Native resolution (10m)
│   ├── zarr.json                # Group metadata
│   ├── B02/                     # Blue band
│   │   ├── zarr.json            # Array metadata (shape: [10980, 10980], chunks: [1024, 1024])
│   │   └── c/                   # Chunk files
│   ├── B03/                     # Green band
│   ├── B04/                     # Red band
│   └── B08/                     # NIR band
├── r20m/                        # 20m overview (2x from r10m)
│   ├── zarr.json
│   ├── B02/
│   ├── B05/                     # Red-Edge band (native 20m)
│   └── ...
├── r60m/                        # 60m overview (6x from r10m)
├── r120m/                       # 120m overview (2x from r60m)
├── r360m/                       # 360m overview (3x from r120m)
└── r720m/                       # 720m overview (2x from r360m)
```

Key aspects:
- **Root metadata**: `zarr.json` at the root carries all convention declarations, `proj:code`, `spatial:dimensions`, `spatial:bbox`, and the full `multiscales.layout`
- **Level groups**: Directories (`r10m/`, `r20m/`, ...) match the `asset` paths in the layout
- **Consistent variables**: Each level contains the same set of variable names
- **No `spatial_ref` array**: CRS is expressed via `proj:code` at the group level, not as a 0D array

## Appendix

### Definitions

#### MultiscalesMetadata

| key | type | required | notes |
| --- | ---- | -------- | ----- |
| `layout` | [LayoutObject[], ...] | yes | Must not be empty |
| `resampling_method` | string | no | Default resampling method |

#### LayoutObject

| key | type | required | notes |
| --- | ---- | -------- | ----- |
| `asset` | string | yes | Relative path to the level group or array |
| `derived_from` | string | no | Relative path to the source level |
| `transform` | TransformObject | conditional | Required when `derived_from` is present |
| `resampling_method` | string | no | Per-level resampling override |
| `spatial:shape` | integer[] | no | Spatial dimensions shape `[height, width]` at this level |
| `spatial:transform` | number[6] | no | Absolute affine transform for this level (Rasterio/Affine ordering) |

#### TransformObject

| key | type | required | notes |
| --- | ---- | -------- | ----- |
| `scale` | number[] | no | Scale factors per axis (relative to `derived_from` level). Scale > 1.0 means lower resolution |
| `translation` | number[] | no | Translation offsets per axis |

#### proj: Properties

| key | type | required | notes |
| --- | ---- | -------- | ----- |
| `proj:code` | string | conditional* | Authority:code CRS identifier, e.g. `"EPSG:4326"`. Pattern: `^[A-Z]+:[0-9]+$` |
| `proj:wkt2` | string | conditional* | WKT2 (ISO 19162) CRS string |
| `proj:projjson` | object | conditional* | PROJJSON object |

\* At least one of these MUST be provided. Full spec: [github.com/zarr-conventions/geo-proj](https://github.com/zarr-conventions/geo-proj)

#### spatial: Properties

| key | type | required | notes |
| --- | ---- | -------- | ----- |
| `spatial:dimensions` | string[] | yes | Names of spatial dimensions in row-major order |
| `spatial:transform` | number[6] | no | Affine coefficients `[a, b, c, d, e, f]`. `x = a*i + b*j + c`, `y = d*i + e*j + f` |
| `spatial:bbox` | number[] | no | Bounding box `[xmin, ymin, xmax, ymax]` in CRS coordinates |
| `spatial:shape` | integer[] | no | Spatial shape `[height, width]` |
| `spatial:registration` | string | no | `"pixel"` (default) or `"node"` |

Full spec: [github.com/zarr-conventions/spatial](https://github.com/zarr-conventions/spatial)

#### ResamplingMethod

A string describing the algorithm used to produce a downsampled or upsampled level. Common values: `"nearest"`, `"average"`, `"bilinear"`, `"cubic"`, `"lanczos"`, `"mode"`, `"max"`, `"min"`, `"sum"`. The implementation defaults to `"average"` for creating overview levels. Any string is valid; see the [multiscales convention spec](https://github.com/zarr-conventions/multiscales) for the full list of common values.
