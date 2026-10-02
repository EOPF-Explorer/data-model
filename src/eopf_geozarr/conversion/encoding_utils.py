from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from zarr.codecs import CastValue

from eopf_geozarr.codecs.scale_offset import scale_offset_from_cf
from eopf_geozarr.conversion import utils

if TYPE_CHECKING:
    import xarray as xr


class Packing(NamedTuple):
    """CF packing of a variable: `decoded = packed * scale_factor + add_offset`."""

    scale_factor: float
    add_offset: float
    fill_value: int | None
    dtype: np.dtype[Any]


def find_fill_value(var: xr.DataArray) -> Any:
    """Return the nodata value of `var` in stored units, or None when it has none.

    This is the CF `_FillValue` (in `.encoding` for decoded input, in `.attrs`
    for raw input) or, on Zarr v3 CPM products, only the EOPF `fill_value`
    attribute.
    """
    candidates = (
        var.encoding.get("_FillValue"),
        var.attrs.get("_FillValue"),
        var.attrs.get("fill_value"),
    )
    return next((value for value in candidates if value is not None), None)


def packing_of(var: xr.DataArray) -> Packing | None:
    """Return the CF packing of `var`, or None when it is not packed.

    The CF values are in `.encoding` for decoded input (`mask_and_scale=True`)
    and in `.attrs` for raw input (`mask_and_scale=False`). A trivial packing
    (`scale_factor` 1, `add_offset` 0) counts as not packed. See
    `find_fill_value` for the nodata value.
    """
    scale = var.encoding.get("scale_factor", var.attrs.get("scale_factor"))
    offset = var.encoding.get("add_offset", var.attrs.get("add_offset"))
    if scale is None and offset is None:
        return None
    scale = 1.0 if scale is None else float(scale)
    offset = 0.0 if offset is None else float(offset)
    dtype = np.dtype(var.encoding.get("dtype", var.dtype))
    if (scale == 1.0 and offset == 0.0) or not np.issubdtype(dtype, np.integer):
        return None
    fill = find_fill_value(var)
    return Packing(scale, offset, None if fill is None else int(fill), dtype)


def _encode_packed(var: xr.DataArray, packing: Packing) -> xr.DataArray:
    """Return `var` as the packed integers of `packing`, with the CF values in `.attrs`.

    The inverse of `_decode_packed`, for the ESA layout. Raw integer input is
    kept as it is (no cast); decoded input is packed again, so both input forms
    end up identical. The nodata value goes to `.encoding["_FillValue"]`.
    """
    if np.issubdtype(var.dtype, np.integer):
        values = var
    else:
        # Decoded Zarr v3 input is not masked, but its nodata packs back to the
        # same stored value.
        values = ((var - packing.add_offset) / packing.scale_factor).round()
        if packing.fill_value is not None:
            values = values.fillna(packing.fill_value)
        values = values.astype(packing.dtype)
    encoded = values.copy(deep=False)
    # Same attributes as the decoded form, so both modes describe the data alike.
    encoded.attrs = {
        **utils.sanitize_array_attrs(var.attrs, is_decoded_float=True),
        "scale_factor": packing.scale_factor,
        "add_offset": packing.add_offset,
    }
    encoded.encoding = {
        key: value for key, value in var.encoding.items() if key in ("chunks", "preferred_chunks")
    }
    if packing.fill_value is not None:
        encoded.encoding["_FillValue"] = packing.fill_value
    return encoded


def normalize_packed(
    var: xr.DataArray, packing: Packing, *, scale_offset_codec: bool
) -> xr.DataArray:
    """Return `var` in the form that the selected encoding mode writes.

    The Zarr codecs take decoded float32 values; the ESA layout keeps the
    packed integers, so it does not cast the data to float.
    """
    if scale_offset_codec:
        return _decode_packed(var, packing)
    return _encode_packed(var, packing)


def _decode_packed(var: xr.DataArray, packing: Packing) -> xr.DataArray:
    """Return `var` as decoded float32 with NaN for nodata, and `packing` in `.encoding`.

    Both input forms end up identical, so each encoding mode packs the same values.
    """
    if np.issubdtype(var.dtype, np.integer):
        values = var.astype("float64")
        if packing.fill_value is not None:
            values = values.where(var != packing.fill_value)
        values = values * packing.scale_factor + packing.add_offset
    else:
        values = var
        if packing.fill_value is not None:
            # Decoded Zarr v3 input is not masked: its nodata is only in attributes.
            decoded_fill = packing.fill_value * packing.scale_factor + packing.add_offset
            values = values.where(abs(values - decoded_fill) > packing.scale_factor / 2)
    decoded = values.astype("float32")
    decoded.attrs = {
        key: value
        for key, value in utils.sanitize_array_attrs(var.attrs, is_decoded_float=True).items()
        if key not in ("scale_factor", "add_offset")
    }
    decoded.encoding = {
        key: value for key, value in var.encoding.items() if key in ("chunks", "preferred_chunks")
    }
    decoded.encoding.update(
        {
            "scale_factor": packing.scale_factor,
            "add_offset": packing.add_offset,
            "dtype": packing.dtype,
        }
    )
    if packing.fill_value is not None:
        decoded.encoding["_FillValue"] = packing.fill_value
    return decoded


def _drop_trivial_scaling(var: xr.DataArray) -> xr.DataArray:
    """Remove a `scale_factor` 1 / `add_offset` 0 pair and keep the integer dtype.

    Zarr v3 CPM products declare it on classification masks such as SCL; with
    it, readers would decode those integers to floats.
    """
    keys = ("scale_factor", "add_offset")
    if not any(key in var.attrs or key in var.encoding for key in keys):
        return var
    dtype = np.dtype(var.encoding.get("dtype", var.dtype))
    if np.issubdtype(var.dtype, np.floating) and np.issubdtype(dtype, np.integer):
        fill = find_fill_value(var)
        fill = 0 if fill is None else fill
        encoding = var.encoding
        var = var.fillna(fill).astype(dtype)
        var.encoding = encoding
    var.attrs = {key: value for key, value in var.attrs.items() if key not in keys}
    var.encoding = {key: value for key, value in var.encoding.items() if key not in keys}
    return var


def _scale_offset_filters(packing: Packing) -> tuple[Any, CastValue]:
    """Zarr codecs that store decoded floats as the packed integers of `packing`."""
    # CastValue refuses to cast NaN to an integer without a mapping: map it to the
    # source nodata value, or to the lowest integer when the source has none.
    nan_sentinel = (
        packing.fill_value if packing.fill_value is not None else int(np.iinfo(packing.dtype).min)
    )
    return (
        scale_offset_from_cf(scale_factor=packing.scale_factor, add_offset=packing.add_offset),
        CastValue(
            data_type=packing.dtype.name,
            rounding="nearest-even",
            scalar_map={"encode": [("NaN", nan_sentinel)], "decode": [(nan_sentinel, "NaN")]},
        ),
    )
