from eopf_geozarr.s2_optimization.s2_band_mapping import (
    BAND_INFO,
    NATIVE_BANDS,
    QUALITY_DATA_NATIVE,
    BandInfo,
)


def test_bandinfo_initialization():
    band = BandInfo("b01", 60, "uint16", 443, 21)
    assert band.name == "b01"
    assert band.native_resolution == 60
    assert band.data_type == "uint16"
    assert band.wavelength_center == 443
    assert band.wavelength_width == 21


def test_native_bands():
    assert NATIVE_BANDS[10] == ["b02", "b03", "b04", "b08"]
    assert NATIVE_BANDS[20] == ["b05", "b06", "b07", "b11", "b12", "b8a"]
    assert NATIVE_BANDS[60] == ["b01", "b09", "b10"]


def test_band_info():
    assert BAND_INFO["b01"].name == "b01"
    assert BAND_INFO["b01"].native_resolution == 60
    assert BAND_INFO["b01"].data_type == "uint16"
    assert BAND_INFO["b01"].wavelength_center == 443
    assert BAND_INFO["b01"].wavelength_width == 21


def test_quality_data_native():
    assert QUALITY_DATA_NATIVE["scl"] == 20
    assert QUALITY_DATA_NATIVE["aot"] == 20
    assert QUALITY_DATA_NATIVE["wvp"] == 20
    assert QUALITY_DATA_NATIVE["cld"] == 20
    assert QUALITY_DATA_NATIVE["snw"] == 20
