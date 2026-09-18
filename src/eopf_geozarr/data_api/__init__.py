"""Data API for accessing GeoZarr compliant EOPF datasets.

The Sentinel-2 models in this package are zarr-integrated, extending pydantic-zarr's
GroupSpec and ArraySpec classes for direct reading/writing of zarr stores.
"""

from eopf_geozarr.data_api.s1_rtc import S1RtcRoot
from eopf_geozarr.data_api.s2 import (
    ALL_BAND_NAMES,
    NATIVE_BANDS,
    RESOLUTION_TO_METERS,
    BandName,
    QualityDataName,
    ResolutionLevel,
    Sentinel2BandInfo,
    Sentinel2ConditionsGroup,
    Sentinel2CoordinateArray,
    Sentinel2DataArray,
    Sentinel2MeasurementsGroup,
    Sentinel2QualityGroup,
    Sentinel2ReflectanceGroup,
    Sentinel2ResolutionDataset,
    Sentinel2Root,
    Sentinel2RootAttrs,
    VariableType,
)

__all__ = [
    # Constants
    "ALL_BAND_NAMES",
    "NATIVE_BANDS",
    "RESOLUTION_TO_METERS",
    # Type literals
    "BandName",
    "QualityDataName",
    "ResolutionLevel",
    # Models
    "S1RtcRoot",
    "Sentinel2BandInfo",
    "Sentinel2ConditionsGroup",
    "Sentinel2CoordinateArray",
    "Sentinel2DataArray",
    "Sentinel2MeasurementsGroup",
    "Sentinel2QualityGroup",
    "Sentinel2ReflectanceGroup",
    "Sentinel2ResolutionDataset",
    "Sentinel2Root",
    "Sentinel2RootAttrs",
    "VariableType",
]
