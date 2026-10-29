"""
The validation scope for GeoZarr metadata documents.

pydantic-zarr validates each Zarr V3 array spec as a whole metadata document,
judging every extension-point name (data type, chunk grid, chunk key encoding,
codec, storage transformer) by the definition that claims it in a scope, and
leaving unclaimed names unjudged. GeoZarr stores written by zarr-python hold
fixed-length unicode arrays (band and angle coordinates) under zarr-python's
provisional ``fixed_length_utf32`` spelling, which no ``zarr-metadata`` scope
claims yet; this module's scope extends the default with a definition for it,
so GeoZarr documents can be judged strictly — every extension-point name
claimed — instead of leniently.
"""

from __future__ import annotations

from typing import Literal, NotRequired

from typing_extensions import TypedDict
from zarr_metadata.v3.definition import CORE_AND_EXTENSIONS, DataTypeDefinition


class FixedLengthUtf32Configuration(TypedDict, closed=True):
    """The configuration of zarr-python's ``fixed_length_utf32`` data type."""

    length_bytes: int


class FixedLengthUtf32Object(TypedDict, closed=True):
    """``fixed_length_utf32`` data type metadata in object form, mirroring the
    shape of ``zarr-metadata``'s per-extension ``*Object`` types."""

    name: Literal["fixed_length_utf32"]
    configuration: FixedLengthUtf32Configuration
    must_understand: NotRequired[bool]


class NullTerminatedBytesConfiguration(TypedDict, closed=True):
    """The configuration of zarr-python's ``null_terminated_bytes`` data type."""

    length_bytes: int


class NullTerminatedBytesObject(TypedDict, closed=True):
    """``null_terminated_bytes`` data type metadata in object form."""

    name: Literal["null_terminated_bytes"]
    configuration: NullTerminatedBytesConfiguration
    must_understand: NotRequired[bool]


FIXED_LENGTH_UTF32 = DataTypeDefinition(
    name="fixed_length_utf32",
    configuration=FixedLengthUtf32Configuration,
    fill_value=str,
)
"""zarr-python's provisional fixed-length unicode data type (numpy's ``U``)."""

NULL_TERMINATED_BYTES = DataTypeDefinition(
    name="null_terminated_bytes",
    configuration=NullTerminatedBytesConfiguration,
    fill_value=str,
)
"""zarr-python's provisional fixed-length bytes data type (numpy's ``S``);
the fill value is the base64 spelling of the bytes."""

GEOZARR_SCOPE = CORE_AND_EXTENSIONS.update(
    # update() replaces a slot whole, so splice the new definitions in after
    # the default scope's own data types
    data_types=[
        *CORE_AND_EXTENSIONS.data_types.values(),
        FIXED_LENGTH_UTF32,
        NULL_TERMINATED_BYTES,
    ],
)
"""The default scope plus the data types zarr-python writes that it lacks."""
