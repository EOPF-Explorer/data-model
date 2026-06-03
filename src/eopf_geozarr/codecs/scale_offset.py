"""
CF-to-zarr-codec helper for the `scale_offset` codec.

The `scale_offset` codec itself ships with zarr-python >= 3.2.0
(`zarr.codecs.ScaleOffset`); this module only provides the small mapping from
CF-convention `scale_factor` / `add_offset` attributes to `ScaleOffset`
constructor arguments.
"""

from __future__ import annotations

from zarr.codecs import ScaleOffset


def scale_offset_from_cf(*, scale_factor: float, add_offset: float) -> ScaleOffset:
    """
    Convert CF-convention scale_factor and add_offset to a ScaleOffset codec.

    CF convention: unpacked = packed * scale_factor + add_offset

    ScaleOffset convention:
        encode: out = (in - offset) * scale
        decode: out = (in / scale) + offset

    To match CF: offset = add_offset, scale = 1 / scale_factor.
    """
    return ScaleOffset(offset=add_offset, scale=1.0 / scale_factor)
