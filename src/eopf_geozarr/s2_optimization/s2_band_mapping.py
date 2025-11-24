"""
Band mapping and resolution definitions for Sentinel-2 optimization.
"""

from dataclasses import dataclass
from typing import TypedDict


@dataclass
class BandInfo:
    """Information about a spectral band."""

    name: str
    native_resolution: int  # meters
    data_type: str
    wavelength_center: float  # nanometers
    wavelength_width: float  # nanometers


# Native resolution definitions
NATIVE_BANDS: dict[int, list[str]] = {
    10: ["b02", "b03", "b04", "b08"],  # Blue, Green, Red, NIR
    20: ["b05", "b06", "b07", "b11", "b12", "b8a"],  # Red Edge, SWIR
    60: ["b01", "b09", "b10"],  # Coastal, Water Vapor, Cirrus
}


class AllBands(TypedDict):
    b01: BandInfo  # Coastal aerosol
    b02: BandInfo  # Blue
    b03: BandInfo  # Green
    b04: BandInfo  # Red
    b05: BandInfo  # Red Edge 1
    b06: BandInfo  # Red Edge 2
    b07: BandInfo  # Red Edge 3
    b08: BandInfo  # NIR
    b8a: BandInfo  # NIR Narrow
    b09: BandInfo  # Water Vapor
    b10: BandInfo  # Cirrus
    b11: BandInfo  # SWIR 1
    b12: BandInfo  # SWIR 2


# Complete band information
BAND_INFO: AllBands = {
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


class QualityDatNative(TypedDict):
    scl: int  # Scene Classification Layer
    aot: int  # Aerosol Optical Thickness
    wvp: int  # Water Vapor
    cld: int  # Cloud probability
    snw: int  # Snow probability


# Quality data mapping - defines which auxiliary data exists at which resolutions
QUALITY_DATA_NATIVE: dict[str, int] = {
    "scl": 20,  # Scene Classification Layer - native 20m
    "aot": 20,  # Aerosol Optical Thickness - native 20m
    "wvp": 20,  # Water Vapor - native 20m
    "cld": 20,  # Cloud probability - native 20m
    "snw": 20,  # Snow probability - native 20m
}
