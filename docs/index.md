---
title: eopf-geozarr — the GeoZarr driver for EOPF CPM
description: eopf-geozarr is the GeoZarr driver for ESA's EOPF CPM. It converts Sentinel-1, Sentinel-2 and Sentinel-3 products to cloud-optimized GeoZarr (Zarr v3) with multiscale pyramids and native projections, from CPM or as a standalone converter.
---

# eopf-geozarr — the GeoZarr driver for EOPF CPM

**eopf-geozarr** is the GeoZarr driver for ESA's **[EOPF CPM](https://cpm.pages.eopf.copernicus.eu/eopf-cpm)**, the Core Python
Modules of the Copernicus
[Earth Observation Processing Framework](https://eopf.copernicus.eu/). It
converts **Sentinel-1, Sentinel-2 and Sentinel-3** products into
cloud-optimized **GeoZarr**: Zarr v3 stores with multiscale pyramids, native
projections and the OGC GeoZarr conventions.

## Two ways to convert

Both paths run the same pipelines and write the same GeoZarr output.

=== "CPM driver"

    Start from a native product (for example a `.SAFE`) or from any product
    that CPM can read. Needs Python 3.13 or later.

    ```bash
    pip install "eopf-geozarr[cpm]"
    eopf convert-geozarr S2B_MSIL2A_….SAFE out.zarr
    ```

    Read the [CPM driver guide](cpm-driver.md).

=== "Standalone converter"

    Start from an EOPF Zarr product, for example from the
    [EOPF Sentinel Zarr Samples Service](https://zarr.eopf.copernicus.eu/)
    hosted by EODC. The converter reads these products directly from their
    URLs. Needs Python 3.12 or later.

    ```bash
    pip install eopf-geozarr
    eopf-geozarr convert S2B_MSIL2A_….zarr out.zarr
    ```

    Read the [standalone converter guide](converter.md).

## Supported products

| Mission | Product | CPM `product:type` | Pipeline |
|---|---|---|---|
| Sentinel-2 MSI | L1C, L2A | `S02MSIL1C`, `S02MSIL2A` | Sentinel-2 optimized: native r10m/r20m/r60m levels plus r120m/r360m/r720m overviews |
| Sentinel-3 OLCI | L1 EFR, L1 ERR | `S03OLCEFR`, `S03OLCERR` | Sentinel-3 OLCI optimized: native swath geometry, or a regular grid |
| Sentinel-1 | GRD | — | Generic pipeline with ground control points |
| Sentinel-1 | GRD RTC (S1Tiling / Orfeo ToolBox COGs) | — | `eopf-geozarr ingest-s1` and related commands (standalone only) |

## What the output looks like

- **GeoZarr conventions**: `zarr_conventions` declarations, `geo-proj`
  (`proj:code`), `spatial` (`spatial:transform`, `spatial:bbox`) and
  `multiscales` on every georeferenced node. See the
  [GeoZarr mini spec](geozarr-minispec.md).
- **Native projections**: UTM and other source CRSs are kept; nothing is
  reprojected to Web Mercator.
- **Multiscale pyramids** for fast visualization at every zoom level.
- **Source packing**: reflectance stays as packed integers with CF
  attributes, as in the ESA products, or packed by the Zarr `scale_offset` +
  `cast_value` codecs on request. See [Encoding](converter.md#encoding).
- **Cloud storage**: write to local paths or S3-compatible object storage.
- **Validation**: `eopf-geozarr validate` checks a store against the mini spec.

## Where to go next

- [Installation](installation.md): pip, uv and the `cpm` extra.
- [Quick start](quickstart.md): your first conversion, with either path.
- [CPM driver](cpm-driver.md): `eopf convert-geozarr`, `engine="geozarr"`,
  options and product routing.
- [Standalone converter](converter.md): every `eopf-geozarr` command.
- [API reference](api-reference.md) and [examples](examples.md).
- [Architecture](architecture.md) and [FAQ](faq.md).
- [GeoZarr mini spec](geozarr-minispec.md) and our
  [contributions to the GeoZarr specification](geozarr-specification-contribution.md).

## About

eopf-geozarr is open source (Apache 2.0) and developed for the ESA
[EOPF Sentinel Explorer](https://github.com/EOPF-Explorer) by
[Development Seed](https://developmentseed.org/) and
[EODC](https://eodc.eu/) (Earth Observation Data Centre), which also hosts the
[EOPF Sentinel Zarr Samples Service](https://zarr.eopf.copernicus.eu/). Source
code and issues:
[EOPF-Explorer/data-model](https://github.com/EOPF-Explorer/data-model).
