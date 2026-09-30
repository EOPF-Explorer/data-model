---
title: Standalone converter (eopf-geozarr CLI)
description: Reference for the standalone eopf-geozarr command line and Python API. Convert EOPF Zarr products for Sentinel-1, Sentinel-2 and Sentinel-3 OLCI to GeoZarr, ingest Sentinel-1 GRD RTC, validate and inspect GeoZarr stores.
---

# Standalone converter (`eopf-geozarr` CLI)

The `eopf-geozarr` command converts EOPF Zarr products to GeoZarr without
EOPF CPM. It reads local stores, `s3://` URLs and HTTP URLs, for example the
products of the [EOPF Sentinel Zarr Samples Service](https://zarr.eopf.copernicus.eu/).
It runs the same pipelines as the [GeoZarr driver for EOPF CPM](cpm-driver.md), without needing [EOPF CPM](https://cpm.pages.eopf.copernicus.eu/eopf-cpm).

## Commands

| Command | Purpose |
|---|---|
| [`convert`](#convert) | Convert any EOPF Zarr product. Detects Sentinel-2 and Sentinel-3 OLCI; other products use the generic pipeline. |
| [`convert-s2-optimized`](#sentinel-2-optimized-conversion) | Convert a Sentinel-2 L1C or L2A product, with all Sentinel-2 options. |
| [`convert-s3-olci-optimized`](#sentinel-3-olci-conversion) | Convert a Sentinel-3 OLCI L1 EFR or ERR product, with all OLCI options. |
| [`ingest-s1`, `ingest-s1-conditions`, `consolidate-s1`, `generate-stac-s1`](#sentinel-1-grd-rtc-ingestion) | Build a Sentinel-1 GRD RTC GeoZarr store from S1Tiling (Orfeo ToolBox) COGs. |
| [`validate`](#validate) | Check a store against the GeoZarr mini spec. |
| [`info`](#info) | Show the structure of a product, optionally as HTML. |

Run `eopf-geozarr <command> --help` for the full list of options.

## `convert`

```bash
eopf-geozarr convert input.zarr output.zarr
eopf-geozarr convert input.zarr s3://my-bucket/output.zarr   # S3 output
eopf-geozarr convert input.zarr output.zarr --dask-cluster   # local Dask cluster
```

`convert` selects the pipeline from the product:

- **Sentinel-2 L1C/L2A**: the [Sentinel-2 optimized pipeline](#sentinel-2-optimized-conversion)
  with its default options. `--groups`, `--crs-groups`, `--gcp-group` and
  `--min-dimension` do not apply.
- **Sentinel-3 OLCI**: the [OLCI pipeline](#sentinel-3-olci-conversion). It
  uses `--min-dimension`. Detection is strict: if a product is not detected,
  use `convert-s3-olci-optimized`.
- **Other products, including Sentinel-1 GRD**: the generic pipeline, which
  converts the groups given with `--groups`.

`--no-s2-optimized` and `--no-s3-olci-optimized` force the generic pipeline.

| Option | Default | Description |
|---|---|---|
| `--groups GROUP [GROUP ...]` | — | Groups to convert (generic pipeline). |
| `--spatial-chunk N` | 4096 | Spatial chunk size. |
| `--min-dimension N` | 256 | Minimum dimension of the coarsest overview level. |
| `--crs-groups [GROUP ...]` | — | Groups that get CRS information added, for example `/conditions/geometry`. |
| `--gcp-group GROUP` | — | Group with ground control points (Sentinel-1), for example `conditions/gcp`. |
| `--enable-sharding` | off | Shard the spatial dimensions of each variable. |
| `--max-retries N` | 3 | Retries for network operations. |
| `--dask-cluster` | off | Start a local Dask cluster for parallel processing. |
| `--no-s2-optimized`, `--no-s3-olci-optimized` | off | Force the generic pipeline. |
| `--verbose` | off | Print more details. |

### Sentinel-1 GRD

Sentinel-1 GRD products use the generic pipeline. The ground control points
georeference the measurements:

```bash
eopf-geozarr convert S1A_IW_GRDH_….zarr output.zarr \
    --groups measurements --gcp-group conditions/gcp
```

### Generic output layout

The generic pipeline writes the native resolution at the group root and adds
factor-of-two overviews as sibling groups `r2`, `r4`, `r8`, … Each overview is
a complete dataset with its own coordinates and `spatial:`/`proj:`
attributes. The parent group's `multiscales` metadata lists every level.

## Sentinel-2 optimized conversion

`convert-s2-optimized` (Python: `convert_s2_optimized`) reuses the native
Sentinel-2 resolutions and adds coarser overviews:

```
output.zarr/
└── measurements/
    └── reflectance/
        ├── r10m/     # native 10 m bands
        ├── r20m/     # native 20 m bands (+ b08 from 10 m)
        ├── r60m/     # native 60 m bands (+ finer bands)
        ├── r120m/    # 2x from r60m
        ├── r360m/    # 3x from r120m
        └── r720m/    # 2x from r360m
```

- The native levels (10 m, 20 m, 60 m) are the ESA resolutions, reused as they
  are.
- The overview factors (2, 3, 2) keep whole chunks and shards at every level.
- Coarser levels also get the finer bands (for example b08 at r20m, and all
  bands at r60m for L1C), so every level has a complete band set.
- Each variable type gets its own resampling: mean for reflectance and
  probabilities (nodata excluded), subsampling for classifications, maximum
  for quality masks.

```bash
eopf-geozarr convert-s2-optimized S2B_MSIL2A_….zarr output.zarr --enable-sharding
```

| Option | Default | Description |
|---|---|---|
| `--spatial-chunk N` | 256 | Spatial chunk size. |
| `--enable-sharding` | off | Enable Zarr v3 sharding. |
| `--compression-level 1-9` | 3 | Blosc zstd compression level. |
| `--scale-offset-codec` | off | Store the packing with the Zarr scale-offset codecs instead of the ESA layout (see [Encoding](#encoding)). |
| `--skip-validation` | off | Do not validate the output. |
| `--dask-cluster` | off | Start a local Dask cluster. |
| `--verbose` | off | Print more details. |

```python
# test: skip (needs a Sentinel-2 product)
import xarray as xr

from eopf_geozarr.s2_optimization.s2_converter import convert_s2_optimized

dt_input = xr.open_datatree("S2B_MSIL2A_….zarr", engine="zarr", chunks={})
dt_output = convert_s2_optimized(
    dt_input,
    output_path="output.zarr",
    enable_sharding=True,
    spatial_chunk=256,
    compression_level=3,
    validate_output=True,
    scale_offset_codec=False,  # default: ESA layout; True packs with Zarr codecs
)
```

### Encoding

Every level of the reflectance pyramid keeps the packing of the source product
(for example `uint16`, `scale_factor` 0.0001, `add_offset` -0.1, nodata 0 for
S2 L2A). One option selects how that packing is stored:

| Mode | Option | On disk | Metadata |
|---|---|---|---|
| Default: ESA layout | none | Source integer dtype, no scale codecs | CF `scale_factor`, `add_offset` and `_FillValue` on the arrays; `raster:scale`, `raster:offset` and `nodata` on the STAC `reflectance` asset |
| Zarr codec | `--scale-offset-codec` / `scale_offset_codec=True` | Source integer dtype, packed by the Zarr `scale_offset` + `cast_value` codecs; the array's logical dtype is `float32` and its fill value is NaN | No CF scale attributes and no STAC scale fields: Zarr readers return decoded values |

What a reader needs:

- ESA layout (default): a reader that uses the CF attributes. xarray
  (`mask_and_scale=True`, the default) and titiler apply them, and GDAL reports
  them as band scale and offset; zarrita (used by the OpenLayers `GeoZarr`
  source) ignores them. This is also the encoding of the
  [EOPF Sentinel Zarr Samples Service](https://zarr.eopf.copernicus.eu/)
  products.
- Zarr codec: a Zarr library that supports the `scale_offset` and `cast_value`
  codecs, for example zarr-python with the `cast-value-rs` extra, or zarrita.
  GDAL does not support them yet
  ([OSGeo/gdal#15170](https://github.com/OSGeo/gdal/issues/15170)).

The input can be opened raw (`mask_and_scale=False`, as in the CPM path) or
decoded (as in the CLI). Both give the same output. Arrays without a packing,
such as the classification and quality masks, keep their integer dtype and
fill value in both modes.

In the CPM writer (`eopf convert-geozarr --scale-offset-codec` or
`target_store_kwargs={"scale_offset_codec": True}`), the former
`keep_scale_offset` option is still accepted for one release with a
`DeprecationWarning`: `keep_scale_offset=False` selects the codecs.

## Sentinel-3 OLCI conversion

Sentinel-3 OLCI L1 EFR and ERR products keep their **native swath geometry**
by default: a per-pixel 2-D latitude/longitude grid, with no reprojection.
`--output-grid <CRS>` warps the swath once onto a regular grid (for example
`EPSG:4326`), so the output is a standard, tileable GeoZarr raster. In both
modes the converter adds /2 overviews, averaged block by block with nodata
excluded.

```bash
eopf-geozarr convert S3A_OL_1_EFR____….zarr output.zarr                      # auto-detected
eopf-geozarr convert-s3-olci-optimized S3A_OL_1_EFR____….zarr output.zarr --output-grid EPSG:4326
```

| Option | Default | Description |
|---|---|---|
| `--output-grid` | `native` | `native` keeps the instrument geometry; a CRS string warps onto a regular grid. |
| `--min-dimension N` | 256 | Minimum dimension of the coarsest overview level. |
| `--spatial-chunk`, `--enable-sharding`, `--compression-level` | — | Accepted but not applied yet. |
| `--verbose` | off | Print more details. |

```
output.zarr/
├── measurements/   # multiscales + spatial: metadata (+ proj: with a CRS)
│   ├── r0/         # native-resolution radiance bands
│   ├── r2/         # 1/2 resolution
│   ├── r4/         # 1/4 resolution
│   └── ...
├── conditions/     # copied unchanged
└── quality/        # copied unchanged
```

- **Native mode**: bands keep per-pixel 2-D `latitude`/`longitude`/`altitude`
  and per-row `time_stamp`, with no projected CRS.
- **Regular grid**: bands get 1-D `y`/`x` coordinates and a `spatial_ref`
  variable referenced by `grid_mapping`. The per-row `time_stamp` is dropped;
  it stays in the source product.
- Radiance keeps the source packing (`uint16` with CF `scale_factor` and
  `_FillValue`). The tie-point groups in `conditions` are copied, not converted
  to the GeoZarr conventions.

## Sentinel-1 GRD RTC ingestion

These commands build a Sentinel-1 GRD γ⁰ RTC GeoZarr store from the
S1Tiling (Orfeo ToolBox) γ⁰ RTC Cloud Optimized GeoTIFFs (COGs), named like
`s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif`. The store has
one time series per orbit direction, multiscale overviews and full GeoZarr
metadata.

```bash
# 1. Add one acquisition (VV, VH and border mask GeoTIFFs)
eopf-geozarr ingest-s1 --vv VV.tif --vh VH.tif --mask BorderMask.tif \
    --store s1-rtc.zarr --orbit-dir ascending

# 2. Add the condition arrays of a relative orbit (all optional)
eopf-geozarr ingest-s1-conditions --store s1-rtc.zarr --orbit-dir ascending \
    --relative-orbit 37 --gamma-area gamma_area.tif --lia lia.tif \
    --incidence-angle incidence.tif

# 3. Consolidate the metadata after the last acquisition
eopf-geozarr consolidate-s1 --store s1-rtc.zarr --orbit-dir ascending

# 4. Print a STAC item for the store
eopf-geozarr generate-stac-s1 --store s1-rtc.zarr --collection sentinel-1-grd-rtc
```

## `validate`

```bash
eopf-geozarr validate output.zarr
```

The validator checks the [GeoZarr mini spec](geozarr-minispec.md) rules: the
store root (convention declarations, `spatial:bbox`, CRS), every multiscale
group (complete layout, georeferencing of each level), every node that uses
the `proj:` or `spatial:` conventions, and the dataset structure (no scalar
arrays, unique `dimension_names`, a 1-D coordinate array for every dimension).
It reports each violation with its Zarr path and exits with a non-zero code
when the store is not compliant.

!!! note "Stores from eopf-geozarr 0.10.x and earlier"
    These stores do not have the store-root `zarr_conventions` declaration, so
    `validate` reports them as non-compliant. Convert them again, or add the
    root metadata with
    `eopf_geozarr.conversion.utils.write_store_root_geo_metadata`.

## `info`

```bash
eopf-geozarr info input.zarr
eopf-geozarr info input.zarr --html-output info.html   # HTML view of the tree
```

## Python API

The same pipelines are available in Python. See the
[API reference](api-reference.md) for the signatures:

- `eopf_geozarr.s2_optimization.s2_converter.convert_s2_optimized` (Sentinel-2)
- `eopf_geozarr.s3_olci_optimization.olci_converter.convert_olci_optimized` (Sentinel-3 OLCI)
- `eopf_geozarr.create_geozarr_dataset` (generic pipeline, Sentinel-1 GRD)
- `eopf_geozarr.conversion.s1_ingest` (Sentinel-1 GRD RTC ingestion)
