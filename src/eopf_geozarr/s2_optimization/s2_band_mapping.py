"""
Band mapping and resolution definitions for Sentinel-2 optimization.
"""

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class BandInfo:
    """Information about a spectral band."""

    name: str
    native_resolution: int  # meters
    data_type: str
    wavelength_center: float  # nanometers
    wavelength_width: float  # nanometers


# Native resolution definitions
NATIVE_BANDS: Dict[int, List[str]] = {
    10: ["b02", "b03", "b04", "b08"],  # Blue, Green, Red, NIR
    20: ["b05", "b06", "b07", "b11", "b12", "b8a"],  # Red Edge, SWIR
    60: ["b01", "b09", "b10"],  # Coastal, Water Vapor, Cirrus
}

# Complete band information
BAND_INFO: Dict[str, BandInfo] = {
    "b01": BandInfo("b01", 60, "uint16", 443, 21),  # Coastal aerosol
    "b02": BandInfo("b02", 10, "uint16", 490, 66),  # Blue
    "b03": BandInfo("b03", 10, "uint16", 560, 36),  # Green
    "b04": BandInfo("b04", 10, "uint16", 665, 31),  # Red
    "b05": BandInfo("b05", 20, "uint16", 705, 15),  # Red Edge 1
    "b06": BandInfo("b06", 20, "uint16", 740, 15),  # Red Edge 2
    "b07": BandInfo("b07", 20, "uint16", 783, 20),  # Red Edge 3
    "b08": BandInfo("b08", 10, "uint16", 842, 106),  # NIR
    "b8a": BandInfo("b8a", 20, "uint16", 865, 21),  # NIR Narrow
    "b09": BandInfo("b09", 60, "uint16", 945, 20),  # Water Vapor
    "b10": BandInfo("b10", 60, "uint16", 1375, 30),  # Cirrus
    "b11": BandInfo("b11", 20, "uint16", 1614, 91),  # SWIR 1
    "b12": BandInfo("b12", 20, "uint16", 2202, 175),  # SWIR 2
}

# Quality data mapping - defines which auxiliary data exists at which resolutions
QUALITY_DATA_NATIVE: Dict[str, int] = {
    "scl": 20,  # Scene Classification Layer - native 20m
    "aot": 20,  # Aerosol Optical Thickness - native 20m
    "wvp": 20,  # Water Vapor - native 20m
    "cld": 20,  # Cloud probability - native 20m
    "snw": 20,  # Snow probability - native 20m
}

# Detector footprint availability - matches spectral bands
DETECTOR_FOOTPRINT_NATIVE: Dict[int, List[str]] = {
    10: ["b02", "b03", "b04", "b08"],
    20: ["b05", "b06", "b07", "b11", "b12", "b8a"],
    60: ["b01", "b09", "b10"],
}


def get_bands_for_level(level: int) -> Set[str]:
    """
    Get all bands available at a given pyramid level.

    Args:
        level: Pyramid level (0=10m, 1=20m, 2=60m, 3+=downsampled)

    Returns:
        Set of band names available at this level
    """
    if level == 0:  # 10m - only native 10m bands
        return set(NATIVE_BANDS[10])
    elif level == 1:  # 20m - all bands (native + downsampled from 10m)
        return set(NATIVE_BANDS[10] + NATIVE_BANDS[20] + NATIVE_BANDS[60])
    elif level == 2:  # 60m - all bands downsampled
        return set(NATIVE_BANDS[10] + NATIVE_BANDS[20] + NATIVE_BANDS[60])
    else:  # Further downsampling - all bands
        return set(NATIVE_BANDS[10] + NATIVE_BANDS[20] + NATIVE_BANDS[60])


def get_quality_data_for_level(level: int) -> Set[str]:
    """Get quality data available at a given level (no upsampling)."""
    if level == 0:  # 10m - no quality data (would require upsampling)
        return set()
    elif level >= 1:  # 20m and below - all quality data available
        return set(QUALITY_DATA_NATIVE.keys())
