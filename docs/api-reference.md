---
title: API reference
description: Python API of eopf-geozarr — the GeoZarr driver for EOPF CPM and the standalone converters for Sentinel-1, Sentinel-2 and Sentinel-3 OLCI, plus validation and utilities.
---

# API reference

This page lists the public entry points. Each function has a full docstring in
the source code.

## GeoZarr driver for EOPF CPM

`eopf_geozarr.cpm.writer.GeoZarrWriter` is the CPM writer registered under the
engine name `geozarr`. You do not call it directly: import
`eopf_geozarr.cpm.writer` and select `engine="geozarr"` in
`eopf.store.write_datatree` or `eopf.store.convert.convert`. See the
[CPM driver guide](cpm-driver.md) for the options and the product routing.

```python
# test: skip (needs eopf-cpm)
import eopf_geozarr.cpm.writer  # registers the "geozarr" engine
from eopf.store import write_datatree

write_datatree(dtree, "out.zarr", engine="geozarr", enable_sharding=True)
```

`eopf_geozarr.cpm.routing` (importable without eopf-cpm) has the routing
helpers: `select_pipeline(dtree)`, `product_type_of(dtree)`,
`looks_like_sentinel2(dtree)` and `looks_like_sentinel3_olci(dtree)`.

## Sentinel-2: `convert_s2_optimized`

```python
# test: skip
from eopf_geozarr.s2_optimization.s2_converter import convert_s2_optimized

convert_s2_optimized(
    dt_input: xr.DataTree,
    *,
    output_path: str,
    enable_sharding: bool,
    spatial_chunk: int,
    compression_level: int,
    validate_output: bool,
    scale_offset_codec: bool = False,
    max_retries: int = 3,
) -> xr.DataTree
```

Converts a Sentinel-2 L1C or L2A DataTree to the optimized pyramid
(`r10m` … `r720m`). All arguments after `dt_input` are keyword-only, and all
without a default are required.

| Argument | Description |
|---|---|
| `output_path` | Local path or `s3://` URL of the output store. |
| `enable_sharding` | Enable Zarr v3 sharding. |
| `spatial_chunk` | Spatial chunk size (the CLI default is 256). |
| `compression_level` | Blosc zstd level, 1–9. |
| `validate_output` | Validate the output after writing. |
| `scale_offset_codec` | `False` (default): ESA layout with CF and STAC scale fields. `True`: pack reflectance with the Zarr `scale_offset` + `cast_value` codecs. See [Encoding](converter.md#encoding). |
| `max_retries` | Retries for network operations. |

`create_multiscale_from_datatree(dt_input, *, output_group, enable_sharding,
spatial_chunk, crs=None, scale_offset_codec=False)` in
`eopf_geozarr.s2_optimization.s2_multiscale` is the lower-level function that
writes the pyramid into an open `zarr.Group`.

## Sentinel-3 OLCI: `convert_olci_optimized`

```python
# test: skip
from eopf_geozarr.s3_olci_optimization.olci_converter import convert_olci_optimized

convert_olci_optimized(
    dt_input: xr.DataTree,
    *,
    output_path: str,
    enable_sharding: bool = False,
    spatial_chunk: int = 1024,
    compression_level: int = 3,
    min_dimension: int = 256,
    output_grid: str = "native",
) -> xr.DataTree
```

Converts a Sentinel-3 OLCI L1 EFR or ERR DataTree. Open the input with
`mask_and_scale=False`. `output_grid="native"` keeps the swath geometry; a CRS
string (for example `"EPSG:4326"`) warps onto a regular grid.
`enable_sharding`, `spatial_chunk` and `compression_level` are accepted but not
applied yet.

## Generic pipeline and Sentinel-1 GRD: `create_geozarr_dataset`

```python
# test: skip
from eopf_geozarr import create_geozarr_dataset

create_geozarr_dataset(
    dt_input: xr.DataTree,
    groups: Iterable[str],
    output_path: str,
    spatial_chunk: int = 4096,
    min_dimension: int = 256,
    max_retries: int = 3,
    crs_groups: Iterable[str] | None = None,
    gcp_group: str | None = None,
    enable_sharding: bool = False,
) -> xr.DataTree
```

Converts the given `groups` with factor-of-two overviews (`r2`, `r4`, …).
For Sentinel-1 GRD, pass `groups=["measurements"]` and
`gcp_group="conditions/gcp"`.

## Sentinel-1 GRD RTC ingestion

In `eopf_geozarr.conversion.s1_ingest`:

| Function | Description |
|---|---|
| `ingest_s1tiling_acquisition(vv_path, vh_path, border_mask_path, store_path, orbit_direction, allow_out_of_order=False) -> int` | Add one S1Tiling acquisition to the store; returns its time index. |
| `ingest_s1tiling_conditions(store_path, orbit_direction, relative_orbit, gamma_area_path=None, lia_path=None, incidence_angle_path=None)` | Write the time-invariant condition arrays. |
| `consolidate_s1_store(store_path, orbit_direction)` | Consolidate the metadata of the store. |

`eopf_geozarr.stac.s1_rtc.build_s1_rtc_stac_item(zarr_store, collection_id)`
builds a `pystac.Item` from a consolidated store.

## Validation

```python
# test: skip (needs a converted store)
from eopf_geozarr.data_api.geozarr.validation import validate_store

report = validate_store("out.zarr")
print(report.summary())
if not report.compliant:
    for issue in report.issues:
        print(issue)
```

`validate_store(store, *, storage_options=None)` accepts a path, an `s3://`
URL or an open `zarr.Group` and checks it against the
[GeoZarr mini spec](geozarr-minispec.md). The `eopf-geozarr validate` command
uses it.

## Opening sources

`eopf_geozarr.conversion.open_source.open_source_datatree(path, *,
storage_options=None, cache_dir=None, engine="zarr", mask_and_scale=True)`
opens a local, `s3://` or HTTP product with Dask chunks that match the native
Zarr chunks. The CLI uses it to open its inputs.

## Utilities

In `eopf_geozarr.conversion.utils`:

- `calculate_aligned_chunk_size(dimension_size, target_chunk_size)`: the
  largest chunk size up to the target that divides the dimension evenly.
- `write_store_root_geo_metadata(output_path, storage_options=None)`: write the
  store-root GeoZarr metadata (`zarr_conventions`, `spatial:bbox`,
  `proj:code`), for example on stores from older releases.
- `downsample_2d_array(source_data, target_height, target_width, nodata_value=None)`:
  block average with nodata handling.

```python
from eopf_geozarr.conversion.utils import calculate_aligned_chunk_size

print(calculate_aligned_chunk_size(10980, 4096))
#> 3660
```

In `eopf_geozarr.conversion.fs_utils`: `get_storage_options(path)`,
`get_s3_storage_options(s3_path)`, `validate_s3_access(s3_path)` and
`is_s3_path(path)` handle S3 and fsspec storage options.
