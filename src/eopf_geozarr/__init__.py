"""EOPF GeoZarr - GeoZarr compliant data model for EOPF datasets."""

try:
    from importlib.metadata import version as _load_version
except Exception:  # pragma: no cover - extremely unlikely
    _load_version = None  # type: ignore

from .conversion import (
    calculate_aligned_chunk_size,
    create_geozarr_dataset,
    downsample_2d_array,
    is_grid_mapping_variable,
    setup_datatree_metadata_geozarr_spec_compliant,
    validate_existing_band_data,
)
from .conversion.metadata import async_consolidate_metadata, consolidate_metadata

try:
    __version__ = _load_version("eopf-geozarr") if _load_version is not None else "0.0.0+local"
except Exception:  # pragma: no cover
    __version__ = "0.0.0+local"

__all__ = [
    "__version__",
    "create_geozarr_dataset",
    "setup_datatree_metadata_geozarr_spec_compliant",
    "consolidate_metadata",
    "async_consolidate_metadata",
    "downsample_2d_array",
    "calculate_aligned_chunk_size",
    "is_grid_mapping_variable",
    "validate_existing_band_data",
]
