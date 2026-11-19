"""
The validation scope for GeoZarr metadata documents.

pydantic-zarr validates each Zarr V3 array spec as a whole metadata document,
judging every extension-point name (data type, chunk grid, chunk key encoding,
codec, storage transformer) by the definition that claims it in a scope, and
leaving unclaimed names unjudged. GeoZarr stores written by zarr-python hold
fixed-length unicode and bytes arrays (band and angle coordinates) under
zarr-python's ``fixed_length_utf32`` and ``null_terminated_bytes`` data types.
``zarr-metadata``'s ``ZARR_PYTHON_CONTEXT`` claims everything zarr-python
writes, both of these included, so GeoZarr documents can be judged strictly —
every extension-point name claimed — instead of leniently.
"""

from __future__ import annotations

from zarr_metadata.v3.definition import ZARR_PYTHON_CONTEXT

GEOZARR_SCOPE = ZARR_PYTHON_CONTEXT
"""The scope of what zarr-python writes: the core spec, the registered
extensions, and zarr-python's own data types."""
