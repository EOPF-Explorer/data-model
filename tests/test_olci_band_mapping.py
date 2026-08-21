from eopf_geozarr.s3_olci_optimization.olci_band_mapping import (
    OLCI_BAND_INFO,
    OLCI_BANDS,
    RADIANCE_DTYPE,
    OlciBandInfo,
)


def test_there_are_21_olci_bands() -> None:
    assert len(OLCI_BANDS) == 21
    assert OLCI_BANDS[0] == "oa01_radiance"
    assert OLCI_BANDS[-1] == "oa21_radiance"


def test_every_band_has_info() -> None:
    assert set(OLCI_BAND_INFO) == set(OLCI_BANDS)
    for name, info in OLCI_BAND_INFO.items():
        assert isinstance(info, OlciBandInfo)
        assert info.name == name
        assert info.data_type == RADIANCE_DTYPE
        assert info.wavelength_center > 0


def test_first_band_wavelength() -> None:
    assert OLCI_BAND_INFO["oa01_radiance"].wavelength_center == 400.0
