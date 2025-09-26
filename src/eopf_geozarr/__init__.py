"""EOPF GeoZarr - GeoZarr compliant data model for EOPF datasets."""

from importlib.metadata import version

from .conversion import (
    async_consolidate_metadata,
    calculate_aligned_chunk_size,
    consolidate_metadata,
    create_geozarr_dataset,
    downsample_2d_array,
    is_grid_mapping_variable,
    iterative_copy,
    setup_datatree_metadata_geozarr_spec_compliant,
    validate_existing_band_data,
)
from .validator import validate_geozarr_store

__version__ = version("eopf-geozarr")

__all__ = [
    "__version__",
    "async_consolidate_metadata",
    "calculate_aligned_chunk_size",
    "consolidate_metadata",
    "create_geozarr_dataset",
    "downsample_2d_array",
    "is_grid_mapping_variable",
    "iterative_copy",
    "setup_datatree_metadata_geozarr_spec_compliant",
    "validate_existing_band_data",
    "validate_geozarr_store",
]
