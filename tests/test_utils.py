from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import xarray as xr
import zarr

from eopf_geozarr.conversion.utils import create_uniform_encoding, rechunk_dataset_for_encoding

if TYPE_CHECKING:
    from eopf_geozarr.data_api.geozarr.types import XarrayDataArrayEncoding


_SCALE_FACTOR = 0.01
_ADD_OFFSET = 0.0


def _packed_band(dtype: str, nodata: int | None, *, form: str) -> xr.DataArray:
    """A 4x4 CF-packed band with one nodata pixel and one valid 0 pixel.

    `form` is how the nodata is declared on input:
    * "eopf_attr": raw integers, nodata only in the EOPF `fill_value` attribute
      (CPM SAFE reader / Zarr v3 products opened with `mask_and_scale=False`).
    * "cf_attr": raw integers, nodata in the CF `_FillValue` attribute.
    * "decoded": decoded floats with NaN for nodata, CF values in `.encoding`
      (products opened with `mask_and_scale=True`, as the CLI does).
    """
    info = np.iinfo(dtype)
    raw = np.full((4, 4), 7, dtype=dtype)
    raw[0, 1] = 0  # a valid 0: must not be read as nodata
    if nodata is not None:
        raw[0, 0] = nodata
    elif info.min < 0:
        raw[0, 0] = info.min  # no nodata declared: an ordinary value

    if form == "decoded":
        values = raw * _SCALE_FACTOR + _ADD_OFFSET
        if nodata is not None:
            values = np.where(raw == nodata, np.nan, values)
        da = xr.DataArray(values, dims=("y", "x"))
        da.encoding = {
            "scale_factor": _SCALE_FACTOR,
            "add_offset": _ADD_OFFSET,
            "dtype": np.dtype(dtype),
        }
        if nodata is not None:
            da.encoding["_FillValue"] = nodata
        return da

    attrs: dict[str, Any] = {"scale_factor": _SCALE_FACTOR, "add_offset": _ADD_OFFSET}
    if nodata is not None:
        attrs["fill_value" if form == "eopf_attr" else "_FillValue"] = nodata
    return xr.DataArray(raw, dims=("y", "x"), attrs=attrs)


@pytest.mark.parametrize("form", ["eopf_attr", "cf_attr", "decoded"])
@pytest.mark.parametrize(
    ("dtype", "nodata"),
    [
        ("uint16", 0),  # S2 reflectance: the zarr default happens to be right
        ("uint16", 65535),  # S3 OLCI radiance
        ("uint8", 255),  # S3 OLCI radiance uncertainty
        ("int32", -2147483648),  # S3 OLCI angles (oaa, saa)
        ("uint16", 1000),  # arbitrary value inside the valid range
    ],
    ids=["u16-0", "u16-65535", "u8-255", "i32-min", "u16-1000"],
)
def test_packed_nodata_sets_zarr_fill_value(dtype: str, nodata: int, form: str) -> None:
    """ESA-packed variables store their nodata as both the CF `_FillValue` and the zarr `fill_value`.

    The zarr `fill_value` must not fall back to the dtype default (0): for a
    nodata other than 0, readers that use it (zarr-python, GDAL) would treat
    valid 0 pixels as nodata and the real nodata pixels as valid.
    """
    ds = xr.Dataset({"band": _packed_band(dtype, nodata, form=form)}).chunk({"y": 2, "x": 2})

    encoding: dict[str, XarrayDataArrayEncoding] = create_uniform_encoding(
        ds, enable_sharding=False, scale_offset_codec=False
    )

    # 1. Encoding passed to `to_zarr`: both keys, same value.
    assert "_FillValue" in encoding["band"]
    assert "fill_value" in encoding["band"]
    assert encoding["band"]["_FillValue"] == nodata
    assert encoding["band"]["fill_value"] == nodata

    # 2. On disk: same nodata at the zarr level and in the CF attribute; the
    #    data stay packed integers, and the EOPF attribute does not survive.
    stored = rechunk_dataset_for_encoding(ds, encoding).to_zarr(
        {}, encoding=encoding, zarr_format=3
    )
    arr = stored.zarr_group["band"]
    assert isinstance(arr, zarr.Array)
    assert arr.dtype == np.dtype(dtype)
    assert arr.metadata.fill_value == nodata
    assert arr.attrs["_FillValue"] == nodata
    assert "fill_value" not in arr.attrs

    # 3. Values: the nodata pixel is stored as nodata, the valid 0 as 0.
    raw = np.asarray(arr[:])
    assert raw[0, 0] == nodata
    assert raw[0, 1] == 0


@pytest.mark.parametrize("dtype", ["uint16", "int32"])
def test_packed_without_nodata_leaves_zarr_fill_value_unset(dtype: str) -> None:
    """With no declared nodata, neither key is set: no nodata value is invented."""
    ds = xr.Dataset({"band": _packed_band(dtype, None, form="eopf_attr")}).chunk({"y": 2, "x": 2})

    encoding = create_uniform_encoding(ds, enable_sharding=False, scale_offset_codec=False)

    assert "_FillValue" not in encoding["band"]
    assert "fill_value" not in encoding["band"]
