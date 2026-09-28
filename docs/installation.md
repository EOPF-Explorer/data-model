---
title: Installation
description: Install eopf-geozarr with pip or uv, as the standalone GeoZarr converter (Python 3.12+) or as the GeoZarr driver for EOPF CPM with the cpm extra (Python 3.13+).
---

# Installation

## Requirements

| Use | Python | Package |
|---|---|---|
| Standalone converter (`eopf-geozarr`) | 3.12 or later | `eopf-geozarr` |
| GeoZarr driver for EOPF CPM (`eopf convert-geozarr`) | 3.13 or later | `eopf-geozarr[cpm]` (installs `eopf` 3.x) |

Linux and macOS are tested. Windows works through WSL.

## Install

=== "Standalone converter"

    ```bash
    pip install eopf-geozarr
    # or
    uv add eopf-geozarr
    ```

=== "CPM driver"

    ```bash
    pip install "eopf-geozarr[cpm]"
    # or
    uv add "eopf-geozarr[cpm]"
    ```

    The `cpm` extra installs EOPF CPM (`eopf`), registers the `geozarr` engine
    in CPM's writer registry and adds `convert-geozarr` to the `eopf` CLI. On
    Python 3.12 the extra installs nothing.

## Check the installation

```bash
eopf-geozarr --version
eopf convert-geozarr --help   # CPM driver only
```

```python
import eopf_geozarr

print(eopf_geozarr.__version__)
```

## Development installation

The project uses [uv](https://docs.astral.sh/uv/) and pre-commit (ruff,
pyright).

```bash
git clone https://github.com/EOPF-Explorer/data-model.git
cd data-model
uv sync                      # add --extra cpm on Python 3.13+ for the CPM driver
uv run pre-commit install
uv run pytest -m "not network"
uv run mkdocs serve          # build and serve this documentation
```

## Cloud storage credentials

Both paths read and write S3-compatible object storage with the standard AWS
environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
# S3-compatible providers (OVH, MinIO, ...):
export AWS_ENDPOINT_URL=https://your-endpoint.example
```

The standalone converter writes to `s3://` URLs directly. The CPM driver
writes locally and uploads with `--stage-output` (`stage_target=True` in
Python).

## Troubleshooting

**`ImportError` about `eopf` when you import `eopf_geozarr.cpm.writer`**
: The CPM driver needs Python 3.13 or later and the `cpm` extra:
  `pip install "eopf-geozarr[cpm]"`.

**`eopf: command not found` or no `convert-geozarr` in `eopf --help`**
: Install the `cpm` extra in the same environment as `eopf`.

**Dependency conflicts**
: Install into a fresh virtual environment (`python -m venv .venv` or
  `uv venv`).

## Next steps

Continue with the [quick start](quickstart.md).
