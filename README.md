<p align="center">
  <img src="https://raw.githubusercontent.com/EOPF-Explorer/data-model/main/docs/assets/logo-eopf-sentinel-explorer-dark-icon.png" alt="EOPF Sentinel Explorer" width="96">
</p>

# eopf-geozarr — the GeoZarr driver for EOPF CPM

[![PyPI](https://img.shields.io/pypi/v/eopf-geozarr)](https://pypi.org/project/eopf-geozarr/)
[![CI](https://github.com/EOPF-Explorer/data-model/actions/workflows/ci.yml/badge.svg)](https://github.com/EOPF-Explorer/data-model/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-eopf--explorer.github.io-blue)](https://eopf-explorer.github.io/data-model/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/EOPF-Explorer/data-model/blob/main/LICENSE)

**eopf-geozarr** is the GeoZarr driver for ESA's **[EOPF CPM](https://cpm.pages.eopf.copernicus.eu/eopf-cpm)**, the Core Python
Modules of the Copernicus
[Earth Observation Processing Framework](https://eopf.copernicus.eu/)
(Python package `eopf`). It converts
**Sentinel-1, Sentinel-2 and Sentinel-3** products into cloud-optimized
**GeoZarr**: Zarr v3 stores with multiscale pyramids, native projections and
the OGC GeoZarr conventions (`multiscales`, `geo-proj`, `spatial`). Use it as a
CPM driver (`eopf convert-geozarr`) or on its own (`eopf-geozarr convert`).

📖 **Documentation: <https://eopf-explorer.github.io/data-model/>**

## Two ways to convert

| | **CPM driver** | **Standalone converter** |
|---|---|---|
| Input | Native products (for example `.SAFE`) or any product CPM can read | EOPF Zarr products (for example from the [EOPF Sentinel Zarr Samples Service](https://zarr.eopf.copernicus.eu/)) |
| Install | `pip install "eopf-geozarr[cpm]"` (Python ≥ 3.13) | `pip install eopf-geozarr` (Python ≥ 3.12) |
| Command | `eopf convert-geozarr S2B_MSIL2A_….SAFE out.zarr` | `eopf-geozarr convert S2B_MSIL2A_….zarr out.zarr` |
| Python | `write_datatree(dtree, "out.zarr", engine="geozarr")` | `convert_s2_optimized(dt, output_path="out.zarr", …)` |

Both paths run the same conversion pipelines and write the same GeoZarr output.
The standalone converter reads the EOPF Zarr products published by the
[EOPF Sentinel Zarr Samples Service](https://zarr.eopf.copernicus.eu/),
hosted by EODC, directly from their URLs.

### CPM driver

Installing the `cpm` extra registers the `geozarr` engine in CPM's writer
registry and adds the `eopf convert-geozarr` command to the `eopf` CLI:

```bash
pip install "eopf-geozarr[cpm]"

eopf convert-geozarr S2B_MSIL2A_20250113T103309_N0511_R108_T32TLQ.SAFE out.zarr
# S3 targets: write locally, then upload
eopf convert-geozarr S2B_MSIL2A_….SAFE s3://bucket/out.zarr --stage-output
```

```python
import eopf_geozarr.cpm.writer  # registers the "geozarr" engine
from eopf.store.convert import convert

convert("S2B_MSIL2A_….SAFE", "out.zarr", target_store_kwargs={"engine": "geozarr"})
```

See the [CPM driver guide](https://eopf-explorer.github.io/data-model/cpm-driver/)
for all options, the product routing and the Python API.

### Standalone converter

```bash
pip install eopf-geozarr

eopf-geozarr convert S2B_MSIL2A_….zarr out.zarr          # auto-detects S2 and S3 OLCI
eopf-geozarr convert S2B_MSIL2A_….zarr s3://bucket/out.zarr
eopf-geozarr validate out.zarr                         # check GeoZarr compliance
```

See the [standalone converter guide](https://eopf-explorer.github.io/data-model/converter/)
for every command, including Sentinel-1 GRD RTC ingestion.

## Supported products

| Mission | Product | CPM `product:type` | Pipeline |
|---|---|---|---|
| Sentinel-2 MSI | L1C, L2A | `S02MSIL1C`, `S02MSIL2A` | S2 optimized: native r10m/r20m/r60m levels plus r120m/r360m/r720m overviews |
| Sentinel-3 OLCI | L1 EFR, L1 ERR | `S03OLCEFR`, `S03OLCERR` | OLCI optimized: native swath geometry, or a regular grid with `--output-grid` |
| Sentinel-1 | GRD | — | Generic pipeline with ground control points (`--groups`, `--gcp-group`) |
| Sentinel-1 | GRD RTC (S1Tiling / Orfeo ToolBox COGs) | — | `eopf-geozarr ingest-s1` and related commands (standalone only) |

## Output

- **GeoZarr conventions** on every georeferenced node: `zarr_conventions`,
  `geo-proj` (`proj:code`), `spatial` (`spatial:transform`, `spatial:bbox`) and
  `multiscales`, as described in the
  [GeoZarr mini spec](https://eopf-explorer.github.io/data-model/geozarr-minispec/).
- **Native projections**: no reprojection to Web Mercator.
- **Multiscale pyramids** for fast visualization at every zoom level.
- **Source packing kept**: reflectance stays as packed integers with CF
  `scale_factor`/`add_offset`/`_FillValue` and STAC
  `raster:scale`/`raster:offset`/`nodata`, as in the ESA products.
  `--scale-offset-codec` stores the packing with the Zarr `scale_offset` +
  `cast_value` codecs instead, so Zarr readers get decoded values.
- **Validation**: `eopf-geozarr validate` checks a store against the mini spec.

GeoZarr is a set of modular [Zarr conventions](https://geozarr.org/conventions)
from the OGC GeoZarr Standards Working Group. GeoZarr V1 is not released yet
(see the [roadmap](https://geozarr.org/roadmap)), so this driver follows the
current conventions.

## Development

```bash
git clone https://github.com/EOPF-Explorer/data-model.git
cd data-model
uv sync                    # add --extra cpm on Python 3.13+ for the CPM driver
uv run pre-commit install
uv run pytest -m "not network"
uv run mkdocs serve        # documentation
```

Contributions are welcome: open an
[issue](https://github.com/EOPF-Explorer/data-model/issues) or a pull request
against `main`.

## License and acknowledgments

Apache License 2.0, see
[LICENSE](https://github.com/EOPF-Explorer/data-model/blob/main/LICENSE).

eopf-geozarr is developed for the ESA
[EOPF Sentinel Explorer](https://github.com/EOPF-Explorer) by
[Development Seed](https://developmentseed.org/) and
[EODC](https://eodc.eu/) (Earth Observation Data Centre), which also hosts the
[EOPF Sentinel Zarr Samples Service](https://zarr.eopf.copernicus.eu/). It is
built on [xarray](https://xarray.dev/), [zarr-python](https://zarr.readthedocs.io/)
and [EOPF CPM](https://cpm.pages.eopf.copernicus.eu/eopf-cpm).
