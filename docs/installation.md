# Installation

This guide covers the installation of the EOPF GeoZarr library and its dependencies.

## Requirements

- Python 3.11 or higher
- Operating System: Linux, macOS, or Windows

## Installation Methods

### Using pip (Recommended)

Install the latest stable version from PyPI:

```bash
pip install eopf-geozarr
```

### Using uv (Fast Alternative)

If you have [uv](https://docs.astral.sh/uv/) installed:

```bash
uv add eopf-geozarr
```

### Development Installation

For development or to get the latest features:

```bash
git clone https://github.com/eopf-explorer/data-model.git
cd data-model
pip install -e .
```

## Dependencies

The library automatically installs the following key dependencies:

- **pydantic-zarr** (≥0.8.0) - Zarr data validation
- **zarr** (≥3.1.1) - Zarr format support
- **xarray** (≥2025.7.1) - N-dimensional labeled arrays
- **dask** (≥2025.5.1) - Parallel computing
- **rioxarray** (≥0.13.0) - Geospatial xarray extension
- **s3fs** (≥2024.6.0) - S3 filesystem support
- **pyproj** (≥3.7.0) - Cartographic projections

## Optional Dependencies

### Development Tools

For development work, install additional tools:

```bash
pip install eopf-geozarr[dev]
```

This includes:

- Testing frameworks (pytest, pytest-cov)
- Code formatting (black, isort)
- Linting (flake8, mypy)
- Security scanning (bandit, safety)

### Documentation

To build documentation locally:

```bash
pip install eopf-geozarr[docs]
```

## Verification

Verify your installation by running:

```bash
eopf-geozarr --version
```

Or in Python:

```python
import eopf_geozarr
print(eopf_geozarr.__version__)
```

## Cloud Storage Setup

### AWS S3 Configuration

For S3 support, configure your AWS credentials:

```bash
# Using AWS CLI
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### Alternative S3-Compatible Storage

For other S3-compatible services (MinIO, DigitalOcean Spaces, etc.):

```bash
export AWS_ENDPOINT_URL=https://your-endpoint.com
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'eopf_geozarr'**

- Ensure you're using the correct Python environment
- Verify installation with `pip list | grep eopf-geozarr`

**Permission errors during installation**

- Use `pip install --user eopf-geozarr` for user-level installation
- Or use a virtual environment (recommended)

**Dependency conflicts**

- Create a fresh virtual environment
- Use `pip install --upgrade eopf-geozarr` to update dependencies

### Virtual Environment Setup

Recommended approach using venv:

```bash
python -m venv eopf-env
source eopf-env/bin/activate  # On Windows: eopf-env\Scripts\activate
pip install eopf-geozarr
```

### System-Specific Notes

**macOS with Apple Silicon**

- Some dependencies may require Rosetta 2 or native ARM builds
- Consider using conda for better compatibility

**Windows**

- Ensure Visual C++ Build Tools are installed for some dependencies
- Use Windows Subsystem for Linux (WSL) for best compatibility

## Next Steps

After installation, proceed to the [Quick Start](quickstart.md) guide to begin using the library.
