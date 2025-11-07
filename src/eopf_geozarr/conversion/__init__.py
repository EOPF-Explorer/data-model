"""Conversion tools for EOPF datasets to GeoZarr compliant format."""

from .fs_utils import (
    get_s3_credentials_info,
    is_s3_path,
    open_s3_zarr_group,
    parse_s3_path,
    s3_path_exists,
    validate_s3_access,
    write_s3_json_metadata,
)
from .geozarr import (
    async_consolidate_metadata,
    calculate_overview_levels,
    consolidate_metadata,
    create_geozarr_dataset,
    iterative_copy,
    setup_datatree_metadata_geozarr_spec_compliant,
)
from .utils import (
    calculate_aligned_chunk_size,
    downsample_2d_array,
    is_grid_mapping_variable,
    validate_existing_band_data,
)

__all__ = [
    "create_geozarr_dataset",
    "setup_datatree_metadata_geozarr_spec_compliant",
    "iterative_copy",
    "consolidate_metadata",
    "async_consolidate_metadata",
    "calculate_overview_levels",
    "downsample_2d_array",
    "calculate_aligned_chunk_size",
    "is_grid_mapping_variable",
    "validate_existing_band_data",
    "get_s3_credentials_info",
    "is_s3_path",
    "open_s3_zarr_group",
    "parse_s3_path",
    "s3_path_exists",
    "validate_s3_access",
    "write_s3_json_metadata",
]
