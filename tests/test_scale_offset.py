"""Tests for the CF-to-codec helper.

The `ScaleOffset` and `CastValue` codecs themselves ship with
zarr-python >= 3.2.0; they have their own upstream test suite. This module
only covers the small `scale_offset_from_cf` helper that maps CF-convention
`scale_factor`/`add_offset` attributes to `ScaleOffset` constructor args.
"""

import numpy as np
import zarr
import zarr.storage
from zarr.codecs import BytesCodec, CastValue

from eopf_geozarr.codecs.scale_offset import scale_offset_from_cf


def test_scale_offset_from_cf_round_trip() -> None:
    """`scale_offset_from_cf` + `CastValue` should round-trip CF-encoded data
    through a zarr codec pipeline back to its decoded float values."""
    scale_factor = 0.01
    add_offset = 273.15
    packed_dtype = "int16"

    packed_values = np.arange(-1000, 1001, dtype=packed_dtype)
    unpacked_values = packed_values * scale_factor + add_offset

    so_codec = scale_offset_from_cf(scale_factor=scale_factor, add_offset=add_offset)
    cv_codec = CastValue(data_type=packed_dtype, rounding="nearest-even")

    store = zarr.storage.MemoryStore()
    arr = zarr.open_array(
        store,
        mode="w",
        shape=unpacked_values.shape,
        dtype=unpacked_values.dtype,
        codecs=[so_codec, cv_codec, BytesCodec()],
    )

    arr[:] = unpacked_values
    np.testing.assert_array_almost_equal(arr[:], unpacked_values)
