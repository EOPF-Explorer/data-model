---
title: GeoZarr driver for EOPF CPM
description: Use eopf-geozarr as the GeoZarr driver for ESA's EOPF CPM. Convert Sentinel-1, Sentinel-2 and Sentinel-3 products (SAFE or EOPF Zarr) to cloud-optimized GeoZarr with eopf convert-geozarr or write_datatree(engine="geozarr").
---

# GeoZarr driver for EOPF CPM

eopf-geozarr is a driver for ESA's **[EOPF CPM](https://cpm.pages.eopf.copernicus.eu/eopf-cpm)** (the `eopf` package). It adds a
writer with the engine name `geozarr` to CPM's writer registry and a
`convert-geozarr` command to the `eopf` CLI. With it, CPM converts a product
straight to cloud-optimized GeoZarr: a Zarr v3 store with multiscale pyramids,
native projections and the GeoZarr conventions.

Use the driver when you start from a native product (for example a `.SAFE`) or
from any product that CPM can read. If you already have an EOPF Zarr product,
the [standalone converter](converter.md) does the same conversion without CPM.

## Installation

The driver needs the `cpm` extra, which installs `eopf` 3.x. EOPF CPM supports
Python 3.13 and later only.

=== "pip"

    ```bash
    pip install "eopf-geozarr[cpm]"
    ```

=== "uv"

    ```bash
    uv add "eopf-geozarr[cpm]"
    ```

!!! note
    On Python 3.12 the `cpm` extra installs nothing. `import eopf_geozarr.cpm.writer`
    then raises an `ImportError` that explains how to install the driver.

## Command line: `eopf convert-geozarr`

The driver registers `convert-geozarr` in the `eopf` CLI through the
`eopf.cli` entry point, so it is listed in `eopf --help`.

```bash
# Sentinel-2 L2A SAFE to GeoZarr
eopf convert-geozarr S2B_MSIL2A_20250113T103309_N0511_R108_T32TLQ.SAFE out.zarr

# Sentinel-3 OLCI EFR, warped once onto a regular grid
eopf convert-geozarr S3A_OL_1_EFR____….SEN3 out.zarr --output-grid EPSG:4326

# Write to S3: convert locally, then upload
eopf convert-geozarr S2B_MSIL2A_….SAFE s3://bucket/path/out.zarr --stage-output
```

| Option | Pipelines | Description |
|---|---|---|
| `--spatial-chunk INTEGER` | all | Spatial chunk size. Default: 256 (Sentinel-2), 1024 (Sentinel-3 OLCI), 4096 (generic). |
| `--enable-sharding` | all | Enable Zarr v3 sharding. |
| `--no-scale-offset-codec` | Sentinel-2 | Write packed reflectance as in the ESA product (CF `scale_factor`/`add_offset`/`_FillValue`) instead of the default Zarr scale-offset codecs. See [Encoding](converter.md#encoding). |
| `--output-grid TEXT` | Sentinel-3 OLCI | `native` (default) keeps the instrument swath geometry. A CRS string (for example `EPSG:4326`) warps the swath onto a regular grid. |
| `--min-dimension INTEGER` | Sentinel-3 OLCI, generic | Minimum dimension of the coarsest overview level. Default: 256. |
| `--groups TEXT` | generic | DataTree group to convert. Repeat the option for more groups. Required on the generic pipeline. |
| `--crs-groups TEXT` | generic | Group that gets CRS information added. Repeatable. |
| `--gcp-group TEXT` | generic | Group that holds the ground control points (Sentinel-1). |
| `--no-s2-optimized` | — | Force the generic pipeline for Sentinel-2 inputs. |
| `--no-s3-olci-optimized` | — | Do not use the OLCI pipeline for Sentinel-3 OLCI inputs; fall back to Sentinel-2 or generic detection. |
| `--stage-source` | — | Download the source product to a local temporary folder before converting. |
| `--stage-output` | — | Write locally first, then upload to the target path. Required for S3 targets. |

## Python

Importing `eopf_geozarr.cpm.writer` registers the `geozarr` engine. Select it
by name: CPM's built-in `cpm_zarr` writer already claims the `.zarr`
extension, so the driver is not chosen from the target path.

=== "Convert a product"

    ```python
    # test: skip (needs eopf-cpm and a source product)
    import eopf_geozarr.cpm.writer  # registers the "geozarr" engine
    from eopf.store.convert import convert

    convert(
        "S2B_MSIL2A_20250113T103309_N0511_R108_T32TLQ.SAFE",
        "out.zarr",
        target_store_kwargs={"engine": "geozarr", "enable_sharding": True},
    )
    ```

=== "Write a DataTree"

    ```python
    # test: skip (needs eopf-cpm and a source product)
    import eopf_geozarr.cpm.writer  # registers the "geozarr" engine
    from eopf.store import write_datatree

    write_datatree(dtree, "out.zarr", engine="geozarr", spatial_chunk=1024)
    ```

### Writer options

Pass these options in `target_store_kwargs` (with `convert`) or as keyword
arguments (with `write_datatree`). Unknown options raise `NotImplementedError`.

| Option | Default | Description |
|---|---|---|
| `mode` | `"w"` | `"w"` replaces an existing store; `"w-"` fails if the target exists. |
| `s2_optimized` | `None` | `True` forces the Sentinel-2 pipeline, `False` forces the generic pipeline, `None` detects the product. |
| `s3_olci_optimized` | `None` | `True` forces the Sentinel-3 OLCI pipeline, `False` falls back to Sentinel-2 or generic detection, `None` detects the product. Cannot be `True` together with `s2_optimized=True`. |
| `spatial_chunk` | per pipeline | 256 (Sentinel-2), 1024 (Sentinel-3 OLCI), 4096 (generic). |
| `enable_sharding` | `False` | Enable Zarr v3 sharding. |
| `scale_offset_codec` | `True` | Sentinel-2: pack reflectance with the Zarr `scale_offset` + `cast_value` codecs. `False` writes the ESA layout with CF and STAC scale fields. |
| `compression_level` | `3` | Sentinel-2: Blosc zstd compression level. |
| `validate_output` | `False` | Sentinel-2: validate the output after writing. |
| `output_grid` | `"native"` | Sentinel-3 OLCI: `native` or a CRS string to warp onto. |
| `min_dimension` | `256` | Sentinel-3 OLCI and generic: minimum dimension of the coarsest overview level. |
| `groups` | — | Generic: DataTree groups to convert (required). |
| `crs_groups` | `None` | Generic: groups that get CRS information added. |
| `gcp_group` | `None` | Generic: group with ground control points (Sentinel-1). |
| `max_retries` | `3` | Retries for network operations. |
| `keep_scale_offset` | — | Deprecated, use `scale_offset_codec`. `keep_scale_offset=True` is `scale_offset_codec=False`. |

## Product routing

The driver selects a pipeline from the product type that CPM declares in
`stac_discovery.properties["product:type"]`:

| `product:type` | Pipeline |
|---|---|
| `S02MSIL1C`, `S02MSIL2A` | Sentinel-2 optimized |
| `S03OLCEFR`, `S03OLCERR` | Sentinel-3 OLCI optimized |
| anything else | generic (needs `groups`) |

When a product has no `product:type`, the driver checks its structure: a
`measurements/reflectance` group with `r10m`, `r20m` and `r60m` is Sentinel-2,
and an `oa01_radiance` variable under `measurements` is Sentinel-3 OLCI.
Sentinel-2 is checked first, then Sentinel-3 OLCI.

## Limitations

- The target must be a local path. For S3, use `--stage-output`
  (`stage_target=True` in Python). S3 credentials come from the environment.
- The driver writes Zarr format 3 only, with consolidated metadata, and writes
  synchronously (`compute=False` is not supported).
- Select the driver by name (`engine="geozarr"`); it is not discovered from
  the target extension.

## See also

- [Standalone converter](converter.md): the same pipelines without CPM,
  including Sentinel-1 GRD RTC ingestion.
- [GeoZarr mini spec](geozarr-minispec.md): the conventions the output follows.
- [EOPF CPM documentation](https://cpm.pages.eopf.copernicus.eu/eopf-cpm).
