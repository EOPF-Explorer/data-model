"""
Round-trip tests for Sentinel-2 pydantic-zarr integrated models.

These tests verify that Sentinel-2 data can be:
1. Loaded from example JSON data using direct instantiation
2. Validated through Pydantic models
3. Round-tripped without data loss

Note: Documentation code examples are tested separately via pytest-examples
from the markdown files in docs/models/sentinel2.md
"""

from eopf_geozarr.data_api.s2 import Sentinel2Root

# basic validation for zarr v2 version -> generally can all be removed as zarrv3 will be supoprted

# `Sentinel2Root` validates only the Zarr V2 shape of an EOPF Sentinel-2
# product; there is no V3 counterpart yet. Round-trip fidelity (does
# validate -> dump -> validate -> dump lose or corrupt anything) is a
# property of the model's field definitions, not of any particular real
# product, so a small hand-built structure that satisfies the schema is
# enough to exercise it: one group is left empty (measurements/quality
# structure) and one carries a single band array (reflectance/r10m/b02) so
# both the group- and array-level fields get exercised.
MINIMAL_SENTINEL2_GROUP: dict[str, object] = {
    "zarr_format": 2,
    "attributes": {"other_metadata": {}, "stac_discovery": {}},
    "members": {
        "measurements": {
            "zarr_format": 2,
            "attributes": {},
            "members": {
                "reflectance": {
                    "zarr_format": 2,
                    "attributes": {},
                    "members": {
                        "r10m": {
                            "zarr_format": 2,
                            "attributes": {},
                            "members": {
                                "b02": {
                                    "zarr_format": 2,
                                    "attributes": {
                                        "_ARRAY_DIMENSIONS": ["y", "x"],
                                        "scale_factor": 0.0001,
                                        "add_offset": 0.0,
                                    },
                                    "shape": [2, 2],
                                    "chunks": [2, 2],
                                    "dtype": "<u2",
                                    "fill_value": 0,
                                    "order": "C",
                                    "filters": None,
                                    "dimension_separator": "/",
                                    "compressor": {
                                        "id": "blosc",
                                        "cname": "zstd",
                                        "clevel": 3,
                                        "shuffle": 2,
                                        "blocksize": 0,
                                    },
                                }
                            },
                        }
                    },
                }
            },
        },
        "quality": {"zarr_format": 2, "attributes": {}, "members": {}},
        "conditions": {
            "zarr_format": 2,
            "attributes": {},
            "members": {
                "geometry": {"zarr_format": 2, "attributes": {}, "members": {}},
                "mask": {"zarr_format": 2, "attributes": {}, "members": {}},
                "meteorology": {"zarr_format": 2, "attributes": {}, "members": {}},
            },
        },
    },
}


def test_sentinel2_roundtrip() -> None:
    """A minimal, schema-valid Sentinel-2 group round-trips without loss."""
    model1 = Sentinel2Root.model_validate(MINIMAL_SENTINEL2_GROUP)
    dumped = model1.model_dump()
    model2 = Sentinel2Root.model_validate(dumped)
    assert model1.model_dump() == model2.model_dump()
