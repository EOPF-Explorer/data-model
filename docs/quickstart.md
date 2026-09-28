---
title: Quick start
description: Convert your first Sentinel product to GeoZarr in minutes, with the eopf-geozarr driver for EOPF CPM or the standalone eopf-geozarr converter.
---

# Quick start

This page converts one Sentinel-2 L2A product to GeoZarr, then checks and
opens the result. Install eopf-geozarr first (see [Installation](installation.md)).

## 1. Convert

=== "CPM driver"

    Start from a native product (for example a `.SAFE`):

    ```bash
    eopf convert-geozarr S2B_MSIL2A_20250113T103309_N0511_R108_T32TLQ.SAFE out.zarr
    ```

    Or in Python:

    ```python
    # test: skip (needs eopf-cpm and a source product)
    import eopf_geozarr.cpm.writer  # registers the "geozarr" engine
    from eopf.store.convert import convert

    convert(
        "S2B_MSIL2A_20250113T103309_N0511_R108_T32TLQ.SAFE",
        "out.zarr",
        target_store_kwargs={"engine": "geozarr"},
    )
    ```

=== "Standalone converter"

    Start from an EOPF Zarr product, local or from a URL of the
    [EOPF Sentinel Zarr Samples Service](https://zarr.eopf.copernicus.eu/):

    ```bash
    eopf-geozarr convert S2B_MSIL2A_20250113T103309_N0511_R108_T32TLQ.zarr out.zarr
    ```

    `convert` detects Sentinel-2 and Sentinel-3 OLCI products and selects the
    optimized pipeline. Or in Python:

    ```python
    # test: skip (needs a Sentinel-2 product)
    import xarray as xr

    from eopf_geozarr.s2_optimization.s2_converter import convert_s2_optimized

    dt = xr.open_datatree("S2B_MSIL2A_20250113T103309_N0511_R108_T32TLQ.zarr", engine="zarr", chunks={})
    convert_s2_optimized(
        dt,
        output_path="out.zarr",
        enable_sharding=True,
        spatial_chunk=256,
        compression_level=3,
        validate_output=True,
    )
    ```

The output has the native levels `r10m`, `r20m` and `r60m` plus the overviews
`r120m`, `r360m` and `r720m` under `measurements/reflectance`.

## 2. Validate

```bash
eopf-geozarr validate out.zarr
```

The validator checks the store against the [GeoZarr mini spec](geozarr-minispec.md).
It lists every violation with its Zarr path and exits with a non-zero code when
the store is not compliant.

## 3. Open the result

```python
# test: skip (needs a converted store)
import xarray as xr

dt = xr.open_datatree("out.zarr", engine="zarr")
reflectance = dt["measurements/reflectance"]
print(reflectance.attrs["multiscales"]["layout"][0])  # first pyramid level
print(reflectance["r720m"].ds)  # coarsest overview, decoded to reflectance
```

Reflectance is stored as packed integers. xarray and zarr-python return the
decoded values. See [Encoding](converter.md#encoding) for the two storage
modes.

## Next steps

- [CPM driver](cpm-driver.md): all `eopf convert-geozarr` options and product routing.
- [Standalone converter](converter.md): every `eopf-geozarr` command, S3 output
  and Sentinel-1.
- [Examples](examples.md) and [API reference](api-reference.md).
- [FAQ](faq.md) for common problems.
