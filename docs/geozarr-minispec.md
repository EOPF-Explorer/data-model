# GeoZarr Mini Spec

This document specifies the GeoZarr model used in this repository. It's a "mini" version of the [official GeoZarr spec](https://zarr.dev/geozarr-spec/documents/standard/template/geozarr-spec.html) that documents the specific subset of the specification that this implementation supports, along with implementation-specific details.

## Relationship to Other Documentation

This mini spec is referenced by and aligns with:

- **[Architecture](architecture.md)** - Technical implementation details that follow this specification
- **[GeoZarr Specification Contribution](geozarr-specification-contribution.md)** - Our contributions to the official spec based on this implementation
- **[Main Documentation](index.md)** - General library documentation and usage guides

The implementation described in this mini spec addresses specific requirements for Earth observation data processing while maintaining compliance with the broader GeoZarr specification.

## Spec conventions

### Array and Group attributes
This document only defines rules for a finite subset of the keys in Zarr array 
and group attributes. Unless otherwise stated, any external keys in Zarr array and group attributes are consistent with this specification. This means this specification composes with the presence of, e.g., [CF metadata](https://cfconventions.org/), at different levels of the Zarr hierarchy.

## Organization

GeoZarr defines a Zarr hierarchy, i.e. a particular arrangements of Zarr arrays and groups, and 
their attributes. This document defines that hierarchy from the bottom-up, starting with arrays
and their attributes before moving to higher-level structures, like groups and their attributes.

The GeoZarr specification can be implemented in Zarr V2 and V3. The main difference between the Zarr V2 and Zarr V3 implementations is how the dimension names of an array are specified.

## DataArray

A DataArray is a Zarr array with named axes. The structure of a DataArray depends on the Zarr format.

This section contains the rules for *individual* DataArrays. Additional 
constraints on groups of DataArrays are defined in the section on [Datasets](#dataset)

### Zarr V2

#### Attributes

| key               | type                                                         | required | notes                                        |
| ----------------- | ------------------------------------------------------------ | -------- | -------------------------------------------- |
| _ARRAY_DIMENSIONS | array of strings, length matches number of axes of the array | yes      | xarray convention for naming axes in Zarr V2 |

#### Array metadata

Zarr V2 DataArrays must have at least 1 dimension, i.e. scalar Zarr V2 DataArrays are not allowed.

In tabular form: 

| attribute | constraint         | notes                    |
| --------- | ------------------ | ------------------------ |
| `shape`   | at least 1 element | No scalar arrays allowed |

#### Example 

```json
{
    ".zarray": {
        "zarr_format": 2,
        "dtype": "|u1",
        "shape": [10,11,12],
        "chunks": [10,11,12],
        "filters": null
        "compressor": null
        "order": "C"
        "dimension_separator": "/"
        }
    ".zattrs": {
        "_ARRAY_DIMENSIONS": ["lat", "lon", "time"]
        }

}
```

### Zarr V3

#### Attributes

No particular attributes are required for Zarr V3 DataArrays.

#### Array metadata

Zarr V3 DataArrays must have at least 1 dimension, i.e. scalar Zarr V3 DataArrays are not allowed. The 
`dimension_names` attribute of a Zarr V3 DataArray must be set, the elements of `dimension_names` must 
all be strings, and they must all be unique.

In tabular form:

| attribute         | constraint                 | notes                                  |
| ----------------- | -------------------------- | -------------------------------------- |
| `shape`           | at least 1 element         | No scalar arrays allowed               |
| `dimension_names` | an array of unique strings | all array axes must be uniquely named. |


#### Example

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
        "storage_transformers": [],
        }
}
```

## Dataset

A GeoZarr dataset is a Zarr group that contains Zarr arrays that together describe a measured quantity, 
as well as arbitrary sub-groups.

### Attributes

There are no required attributes for Datasets but to qualify as a GeoZarr Dataset, the group must contain at least one DataArray with spatial reference information.
This DataArray is referenced in the `grid_mapping` attribute of the dataset and is usually named `spatial_ref`.

#### CF Compliance Requirements

The implementation enforces CF (Climate and Forecast) metadata conventions compliance:

- **Grid Mapping**: All data variables MUST include a `grid_mapping` attribute that references a coordinate reference system variable
- **Standard Names**: Data variables MUST include CF-compliant `standard_name` attributes. The implementation validates these against the official CF standard names table
- **Coordinate Variables**: Coordinate variables (x, y, time, etc.) MUST include appropriate CF standard names:
  - For projected coordinates: `projection_x_coordinate` and `projection_y_coordinate` 
  - For geographic coordinates: `longitude` and `latitude`
  - Units must be specified (`m` for projected, `degrees_east`/`degrees_north` for geographic)
- **Array Dimensions**: All arrays MUST include `_ARRAY_DIMENSIONS` attributes for Zarr V2 compatibility

More information on spatial reference information can be found in the [CF conventions](https://cfconventions.org). Another interesting resource is the [rioxarray](https://corteva.github.io/rioxarray/stable/) and more specifically the documentation on [Coordinate Reference System Management](https://corteva.github.io/rioxarray/stable/getting_started/crs_management.html).

### Members

If any member of a GeoZarr Dataset is an array, then it must comply with the [DataArray](#dataarray) definition.

If the Dataset contains a DataArray `D`, then for each dimension name `N` in the list of `D`'s named dimensions, 
the Dataset must contain a one-dimensional DataArray named `N` with a shape that matches the the length 
of `D` along the axis named by `N`. In this case, `D` is called a "data variable", and the each 
DataArrays matching a dimension names of `D` is called a "coordinate variable". 

> [!Note]
> These two definitions are not mutually exclusive, as a 1-dimensional DataArray named `D` with
dimension names `["D"]` is both a coordinate variable and a data variable.


#### Examples 

This example demonstrates the stored representation of a valid Dataset. Notice how 
the dimension names defined on the DataArray named `"data"` (i.e., `"lat"` and `"lon"`) are 
the names of one-dimensional DataArrays in the same Zarr group as `"data"`.

In this case, `"data"` is a data variable, and `"lat"` and `"lon"` are coordinate variables.

```json
{
    "zarr.json" : {
        "node_type": "group",
        "zarr_format": 3,
        },
    "data/zarr.json" : {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10,11],
        "data_type": "uint8",
        "chunk_key_encoding": {"name": "default", "configuration": {"separator" : "/"}},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10,11]}},
        "codecs": [{"name": "bytes"}],
        "dimension_names": ["lat", "lon"],
        "storage_transformers": [],
        },
    "lat/zarr.json": {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "uint8",
        "chunk_key_encoding": {"name": "default", "configuration": {"separator" : "/"}},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10]}},
        "codecs": [{"name": "bytes"}],
        "dimension_names": ["lat"],
        "storage_transformers": [],
        },
    "lon/zarr.json": {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [11],
        "data_type": "uint8",
        "chunk_key_encoding": {"name": "default", "configuration": {"separator" : "/"}},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [11]}},
        "codecs": [{"name": "bytes"}],
        "dimension_names": ["lon"],
        "storage_transformers": [],
        },
}
```

This example demonstrates the layout of a Dataset with just one DataArray. A single array
is only permitted if that array is one dimensional, and the name of that DataArray in the Dataset 
matches the (single) dimension name defined for that DataArray. 

In this case `lat` is both a coordinate variable and a data variable.

```json
{
    "zarr.json" : {
        "node_type": "group",
        "zarr_format": 3,
        },
    "lat/zarr.json" : {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "uint8",
        "chunk_key_encoding": {"name": "default", "configuration": {"separator" : "/"}},
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [10,11]}},
        "codecs": [{"name": "bytes"}],
        "dimension_names": ["lat"],
        "storage_transformers": [],
    },
}
```

## Multiscale Dataset 

This implementation supports **two multiscales metadata conventions**:

1. **[Zarr Multiscales Convention](https://github.com/zarr-conventions/multiscales)**: An experimental convention for describing multi-resolution data with simple scale and translation metadata
2. **GeoZarr 0.4 TileMatrixSet Specification**: The experimental GeoZarr specification using OGC TileMatrixSet definitions for geospatial data

**Both conventions can coexist in the same Zarr store**, providing flexibility for different use cases and ensuring compatibility with various tools and workflows. The implementation follows the specifications defined in:

- **[Zarr Multiscales Convention Specification](https://github.com/zarr-conventions/multiscales)** - For the experimental multiscales convention with examples at [https://github.com/zarr-conventions/multiscales/tree/main/examples](https://github.com/zarr-conventions/multiscales/tree/main/examples)
  - See particularly [sentinel-2-multiresolution.json](https://github.com/zarr-conventions/multiscales/blob/main/examples/sentinel-2-multiresolution.json) for Sentinel-2 multi-resolution structure
- **[GeoZarr Specification](https://zarr.dev/geozarr-spec/)** - For the OGC TileMatrixSet-based convention

When both conventions are enabled, they are written to the same group attributes with different keys, allowing tools to use whichever convention they support.

Downsampling is a process in which a collection of localized data points is resampled on a subset of its original sampling locations. 

In the case of arrays, downsampling generally reduces an array's shape along at least one dimension. To downsample the
contents of a Dataset `D` and generate a new Dataset `E`, all of the coordinate variable - data variable 
relationships in `D` must be preserved in `E`. If `D/data` is a data variable with dimension names (`"a"` , `"b"`), then `D/a` and `D/b` are coordinate variables with shapes aligned to the dimensions of `D/data`. If we downsample `D/data` and assign the result to `E/data`, we must also generate (e.g., by more downsampling) coordinate variables `E/a` and `E/b` so that `E` can be a valid Dataset according to the relevant [Dataset members rule](#members).

The downsampling transformation is thus well-defined for Datasets. Downsampling 
is often applied multiple times in a series, e.g. to generate multiple levels of 
detail for a data variable. 

### Implementation Approach

The implementation uses a **pyramid-based downsampling approach** with the following characteristics:

- **Variable Downsampling Factors**: Overview levels use optimal downsampling factors based on data characteristics (e.g., 2x, 3x) rather than strictly factor-of-2. For Sentinel-2, this results in resolution levels: 10m → 20m → 60m → 120m (2x) → 360m (3x) → 720m (2x)
- **Pyramid Generation**: Overview levels are created sequentially, with each level generated from the previous level rather than from the native resolution
- **Minimum Dimension Threshold**: Overview generation stops when the smallest dimension falls below a configurable threshold (default: 256 pixels)
- **Native CRS Preservation**: All overview levels maintain the same coordinate reference system as the native data
- **Consistent Variable Structure**: Each overview level contains the same set of variables as the native resolution level

GeoZarr defines a layout for downsampled Datasets (and the original dataset). Given some source Dataset `s0`, 
that dataset and all downsampled Datasets `s1`, `s2`, ... are stored in a flat layout inside a Multiscale Dataset
 `D`. The presence of downsampled Datsets in `D` is signalled by a [special key](#attributes-3) in the attributes of `D`.

### Attributes

The attributes of a Multiscale Dataset function as an entry point to a collection of downsampled Datasets. Accordingly, the attributes of a Multiscale Dataset declare the names of the downsampled datasets it contains, as well as spatial metadata for those datasets.

| key             | type                                        | required | notes                                                                        |
| --------------- | ------------------------------------------- | -------- | ---------------------------------------------------------------------------- |
| `"multiscales"` | [`MultiscaleMetadata`](#multiscalemetadata) | yes      | this field defines the layout of the multiscale Datasets inside this Dataset |

#### MultiscaleMetadata

`MultiscaleMetadata` is a JSON object that declares the names of the downsampled Datasets inside a Multiscale Dataset, as well as the downsampling method used. This object has the following structure:

| key                   | type                                            | required | notes                                                                                                                                                                                                                                                                                                                                                                                                    |
| --------------------- | ----------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"resampling_method"` | [ResamplingMethod](#resamplingmethod)           | yes      | This is a string that declares the resampling method used to create the downsampled datasets.                                                                                                                                                                                                                                                                                                            |
| `"tile_matrix_set"`   | [TileMatrixSet](#tilematrixset) or string       | yes      | This object declares the names of the downsampled Datasets. If `"tile_matrix_set"` is a string, it must be the name of a well-known [`TileMatrixSet`](https://docs.ogc.org/is/17-083r4/17-083r4.html#toc48), which must resolve to a JSON object consistent with the `[TileMatrixSet](#tilematrixset)` definition. For scientific coordinate systems, custom inline TileMatrixSet objects are supported. |
| `"tile_matrix_limits"` | {`string`: [TileMatrixLimit](#tilematrixlimit)} | no       | Optional limits for each tile matrix level |

### Members

All of the members declared in the `multiscales` attribute must comply with the [Dataset](#dataset) definition. All of these Datasets must
have the exact same set of member names. The names of the downsampled Datasets are specified by 
the `"id"` field of each `TileMatrix` object in the `"tileMatrices"` field in the `"TileMatrixSet"` object in the `tile_matrix_set` field in the [`MultiscaleMetadata`](#multiscalemetadata) object in the `"multiscales"` field in the attributes of the Multiscale Dataset. Or, more compactly, using a path-like JSON query:

`attributes.multiscales.tile_matrix_set.tileMatrices[$idx].id`

#### Chunking Requirements for Downsampled Datasets

When creating downsampled datasets in a multiscale hierarchy, careful consideration must be given to chunk sizes to ensure optimal performance and storage efficiency. The chunk dimensions should be aligned with the tile dimensions specified in the corresponding `TileMatrix` definition to enable efficient tile-based access patterns.

Key chunking considerations:

- **Chunk-Tile Alignment**: Chunk sizes should match or be divisible by the `tileWidth` and `tileHeight` values defined in the `TileMatrix` for each zoom level
- **Consistent Chunking Strategy**: All data variables within a zoom level should use the same chunking scheme to maintain spatial coherence
- **Memory Constraints**: Chunk sizes should be chosen to balance I/O efficiency with memory usage, typically keeping individual chunks under 100MB
- **Decimation Factor Alignment**: When downsampling by integer factors (e.g., 2x, 3x), chunk boundaries should align across zoom levels to enable efficient pyramid generation

For example, if a `TileMatrix` specifies `tileWidth: 1024` and `tileHeight: 1024`, the corresponding data arrays should use chunk shapes of `[1024, 1024]` or compatible subdivisions like `[512, 512]`.

#### Extra members

A multiscale Dataset should not contain any members that are not explicitly declared in the `"multiscales"` field for that multiscale Dataset. Any additional Zarr arrays and groups should be considered external to the GeoZarr model.  

### Custom Coordinate Reference Systems

GeoZarr explicitly supports custom TileMatrixSet definitions for arbitrary coordinate reference systems, encouraging preservation of native CRS in Earth observation data. This is particularly useful for scientific projections including UTM zones, polar stereographic, sinusoidal, and other non-web coordinate systems.

For a dataset to be GeoZarr compliant, data variables MUST include a `grid_mapping` attribute that references a coordinate reference system variable. This `grid_mapping` variable defines the spatial referencing information and MUST be consistent with the CRS specified in the TileMatrixSet.

#### Custom TileMatrixSet Example

For custom coordinate systems, the `tile_matrix_set` should be defined as an inline JSON object following the OGC TileMatrixSet v2.0 specification:

```json
{
  "multiscales": {
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
```

#### Custom Decimation Factors

While standard web mapping assumes quadtree decimation (scaling by factor of 2), custom TileMatrixSets may use alternative decimation factors:

- **Factor of 2 (quadtree)**: Standard web mapping approach where each zoom level has 4x more tiles
- **Factor of 3 (nonary tree)**: Each zoom level has 9x more tiles, useful for certain scientific gridding schemes  
- **Other integer factors**: Application-specific requirements may dictate alternative decimation

Example with factor-of-3 decimation:

```json
{
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
}
```

#### Custom CRS Multiscale Dataset Layout Example

Here's a complete example of a multiscale dataset using a custom UTM coordinate reference system:

```json
{
  "zarr.json": {
    "node_type": "group",
    "zarr_format": 3,
    "attributes": {
      "multiscales": {
        "tile_matrix_set": {
          "id": "UTM_Zone_33N_Sentinel2",
          "title": "UTM Zone 33N for Sentinel-2 L2A",
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
  },
  "0/zarr.json": {
    "node_type": "group",
    "zarr_format": 3
  },
  "0/red/zarr.json": {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [1094, 1094],
    "data_type": "uint16",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1024, 1024]}},
    "codecs": [{"name": "bytes"}],
    "dimension_names": ["y", "x"],
    "attributes": {
      "grid_mapping": "spatial_ref"
    }
  },
  "0/nir/zarr.json": {
    "zarr_format": 3,
    "node_type": "array", 
    "shape": [1094, 1094],
    "data_type": "uint16",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1024, 1024]}},
    "codecs": [{"name": "bytes"}],
    "dimension_names": ["y", "x"],
    "attributes": {
      "grid_mapping": "spatial_ref"
    }
  },
  "0/spatial_ref/zarr.json": {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [],
    "data_type": "int32",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": []}},
    "codecs": [{"name": "bytes"}],
    "dimension_names": [],
    "attributes": {
      "crs_wkt": "PROJCS[\"WGS 84 / UTM zone 32N\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",9],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"32632\"]]",
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.314245179,
    "inverse_flattening": 298.257223563,
    "reference_ellipsoid_name": "WGS 84",
    "longitude_of_prime_meridian": 0.0,
    "prime_meridian_name": "Greenwich",
    "geographic_crs_name": "WGS 84",
    "horizontal_datum_name": "World Geodetic System 1984",
    "projected_crs_name": "WGS 84 / UTM zone 32N",
    "grid_mapping_name": "transverse_mercator",
    "latitude_of_projection_origin": 0.0,
    "longitude_of_central_meridian": 9.0,
    "false_easting": 500000.0,
    "false_northing": 0.0,
    "scale_factor_at_central_meridian": 0.9996,
    "spatial_ref": "PROJCS[\"WGS 84 / UTM zone 32N\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",9],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"32632\"]]",
    "_ARRAY_DIMENSIONS": [],
    "GeoTransform": "300000.0 10.0 0.0 5000040.0 0.0 -10.0"
    }
  },
  "0/x/zarr.json": {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [1094],
    "data_type": "float64",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1094]}},
    "codecs": [{"name": "bytes"}],
    "dimension_names": ["x"]
  },
  "0/y/zarr.json": {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [1094],
    "data_type": "float64", 
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1094]}},
    "codecs": [{"name": "bytes"}],
    "dimension_names": ["y"]
  },
  "1/zarr.json": {
    "node_type": "group",
    "zarr_format": 3
  },
  "1/red/zarr.json": {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [547, 547],
    "data_type": "uint16",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [512, 512]}},
    "codecs": [{"name": "bytes"}],
    "dimension_names": ["y", "x"],
    "attributes": {
      "grid_mapping": "spatial_ref"
    }
  },
  "1/nir/zarr.json": {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [547, 547], 
    "data_type": "uint16",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [512, 512]}},
    "codecs": [{"name": "bytes"}],
    "dimension_names": ["y", "x"],
    "attributes": {
      "grid_mapping": "spatial_ref"
    }
  },
  "1/spatial_ref/zarr.json": {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [],
    "data_type": "int32",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": []}},
    "codecs": [{"name": "bytes"}],
    "dimension_names": [],
    "attributes": {
      "crs_wkt": "PROJCS[\"WGS 84 / UTM zone 32N\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",9],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"32632\"]]",
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.314245179,
    "inverse_flattening": 298.257223563,
    "reference_ellipsoid_name": "WGS 84",
    "longitude_of_prime_meridian": 0.0,
    "prime_meridian_name": "Greenwich",
    "geographic_crs_name": "WGS 84",
    "horizontal_datum_name": "World Geodetic System 1984",
    "projected_crs_name": "WGS 84 / UTM zone 32N",
    "grid_mapping_name": "transverse_mercator",
    "latitude_of_projection_origin": 0.0,
    "longitude_of_central_meridian": 9.0,
    "false_easting": 500000.0,
    "false_northing": 0.0,
    "scale_factor_at_central_meridian": 0.9996,
    "spatial_ref": "PROJCS[\"WGS 84 / UTM zone 32N\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",9],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"32632\"]]",
    "_ARRAY_DIMENSIONS": [],
    "GeoTransform": "300000.0 10.0 0.0 5000040.0 0.0 -10.0"
    }
  },
  "1/x/zarr.json": {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [547],
    "data_type": "float64",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [547]}},
    "codecs": [{"name": "bytes"}],
    "dimension_names": ["x"]
  },
  "1/y/zarr.json": {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [547],
    "data_type": "float64",
    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [547]}},
    "codecs": [{"name": "bytes"}],
    "dimension_names": ["y"]
  }
}
```

This example demonstrates:
- **Custom CRS**: Uses EPSG:32633 (UTM Zone 33N) instead of web mapping CRS
- **Scientific Resolution**: Native 10m pixel size typical for Sentinel-2 L2A data
- **Custom Tile Sizes**: 1024x1024 for native, 512x512 for overview to match scientific data characteristics
- **Consistent Structure**: Both zoom levels (`0` and `1`) contain the same variables (`red`, `nir`, `x`, `y`)
- **Coordinate Variables**: UTM coordinates in meters stored as `x` and `y` arrays
- **Chunk Alignment**: Chunk sizes match the `tileWidth` and `tileHeight` from the TileMatrix definition

#### File System Hierarchy Example

The same custom CRS multiscale dataset would appear as the following directory structure on disk:

```
sentinel2_utm33n.zarr/
├── zarr.json                    # Root group with multiscales metadata
├── 0/                          # Native resolution (10m) zoom level
│   ├── zarr.json               # Group metadata for zoom level 0
│   ├── red/                    # Red band data variable
│   │   ├── zarr.json           # Array metadata
│   │   └── c/                  # Chunk directory
│   │       ├── 0/0             # Chunk files (1024x1024 chunks)
│   │       ├── 0/1
│   │       └── ...
│   ├── nir/                    # Near-infrared band data variable
│   │   ├── zarr.json           # Array metadata
│   │   └── c/                  # Chunk directory
│   │       ├── 0/0             # Chunk files (1024x1024 chunks)
│   │       ├── 0/1
│   │       └── ...
│   ├── spatial_ref/            # Spatial reference system variable
│   │   ├── zarr.json           # Array metadata with CRS information
│   │   └── c/                  # Chunk directory
│   │       └── 0               # Single chunk (scalar)
│   ├── x/                      # X coordinate variable (UTM Easting)
│   │   ├── zarr.json           # Array metadata
│   │   └── c/                  # Chunk directory
│   │       └── 0               # Single chunk (1094 elements)
│   └── y/                      # Y coordinate variable (UTM Northing)
│       ├── zarr.json           # Array metadata
│       └── c/                  # Chunk directory
│           └── 0               # Single chunk (1094 elements)
└── 1/                          # Overview level (20m) zoom level
    ├── zarr.json               # Group metadata for zoom level 1
    ├── red/                    # Red band data variable
    │   ├── zarr.json           # Array metadata
    │   └── c/                  # Chunk directory
    │       ├── 0/0             # Chunk files (512x512 chunks)
    │       ├── 0/1
    │       └── ...
    ├── nir/                    # Near-infrared band data variable
    │   ├── zarr.json           # Array metadata
    │   └── c/                  # Chunk directory
    │       ├── 0/0             # Chunk files (512x512 chunks)
    │       ├── 0/1
    │       └── ...
    ├── spatial_ref/            # Spatial reference system variable
    │   ├── zarr.json           # Array metadata with CRS information
    │   └── c/                  # Chunk directory
    │       └── 0               # Single chunk (scalar)
    ├── x/                      # X coordinate variable (UTM Easting)
    │   ├── zarr.json           # Array metadata
    │   └── c/                  # Chunk directory
    │       └── 0               # Single chunk (547 elements)
    └── y/                      # Y coordinate variable (UTM Northing)
        ├── zarr.json           # Array metadata
        └── c/                  # Chunk directory
            └── 0               # Single chunk (547 elements)
```

Key aspects of this file system layout:
- **Root metadata**: The `zarr.json` at the root contains the `multiscales` attribute defining the custom UTM TileMatrixSet
- **Zoom level groups**: Directories `0/` and `1/` correspond exactly to the TileMatrix `id` values
- **Consistent variables**: Each zoom level contains the same set of variables (`red`, `nir`, `x`, `y`)
- **Chunk organization**: Data is stored in chunks that align with the tile dimensions specified in the TileMatrixSet
- **Coordinate preservation**: UTM coordinates are maintained at each resolution level

## Appendix

### Definitions

#### TileMatrixLimit

| key            | type   | required | notes |
| -------------- | ------ | -------- | ----- |
| `"tileMatrix"` | string | yes      |       |
| `"minTileCol"` | int    | yes      |       |  |
| `"minTileRow"` | int    | yes      |       |
| `"maxTileCol"` | int    | yes      |       |
| `"maxTileRow"` | int    | yes      |       |

#### TileMatrix

| key                  | type           | required | notes |
| -------------------- | -------------- | -------- | ----- |
| `"id"`               | string         | yes      |       |
| `"scaleDenominator"` | float          | yes      |       |
| `"cellSize"`         | float          | yes      |       |
| `"pointOfOrigin"`    | [float, float] | yes      |       |
| `"tileWidth"`        | int            | yes      |       |
| `"tileHeight"`       | int            | yes      |       |
| `"matrixWidth"`      | int            | yes      |       |
| `"matrixHeight"`     | int            | yes      |       |


#### TileMatrixSet

| key              | type                             | required | notes            |
| ---------------- | -------------------------------- | -------- | ---------------- |
| `"id"`           | string                           | yes      |                  |
| `"title"`        | string                           | no       |                  |
| `"crs"`          | string                           | no       |                  |
| `"supportedCRS"` | string                           | no       |                  |
| `"orderedAxes"`  | [str, str]                       | no       |                  |
| `"tileMatrices"` | [[TileMatrix](#tilematrix), ...] | yes      | May not be empty |

#### ResamplingMethod

This is a string literal defined [here](https://zarr.dev/geozarr-spec/documents/standard/template/geozarr-spec.html#_71eeacb0-5e4e-8a8e-5714-02fc0838075b).

The implementation defaults to `"average"` for creating overview levels in multiscale datasets.
