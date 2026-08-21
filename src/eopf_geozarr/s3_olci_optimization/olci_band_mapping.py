"""Band definitions for Sentinel-3 OLCI L1 EFR.

OLCI has 21 radiance bands (Oa01..Oa21), all delivered at the same full
resolution (~300 m) on a single swath grid.
"""

from dataclasses import dataclass

RADIANCE_DTYPE = "uint16"

# Band index -> central wavelength in nm (OLCI Oa01..Oa21).
_WAVELENGTHS_NM: tuple[float, ...] = (
    400.0,
    412.5,
    442.5,
    490.0,
    510.0,
    560.0,
    620.0,
    665.0,
    673.75,
    681.25,
    708.75,
    753.75,
    761.25,
    764.375,
    767.5,
    778.75,
    865.0,
    885.0,
    900.0,
    940.0,
    1020.0,
)

OLCI_BANDS: tuple[str, ...] = tuple(f"oa{i:02d}_radiance" for i in range(1, 22))


@dataclass(frozen=True)
class OlciBandInfo:
    """Spectral characterization of a single OLCI radiance band."""

    name: str
    data_type: str
    wavelength_center: float  # nanometers


OLCI_BAND_INFO: dict[str, OlciBandInfo] = {
    name: OlciBandInfo(name=name, data_type=RADIANCE_DTYPE, wavelength_center=wl)
    for name, wl in zip(OLCI_BANDS, _WAVELENGTHS_NM, strict=True)
}
