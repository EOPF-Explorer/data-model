"""Pydantic-zarr model for the Sentinel-3 OLCI L1 EFR EOPF Zarr structure.

Mirrors data_api/s2.py: GroupSpec + closed TypedDict members. Used for
structural product detection (an EOPF product is OLCI iff it validates here).
"""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003

from pydantic import BaseModel
from typing_extensions import TypedDict

from eopf_geozarr.data_api.geozarr.common import DatasetAttrs
from eopf_geozarr.pyz.v2 import ArraySpec, GroupSpec


class Sentinel3OlciRootAttrs(BaseModel):
    """Root-level attributes for an OLCI DataTree (not validated in detail)."""

    other_metadata: dict[str, object]
    stac_discovery: dict[str, object]


class Sentinel3OlciMeasurementsMembers(TypedDict, closed=True, total=False):
    """Members of the OLCI measurements group.

    The 21 radiance bands and the per-pixel geolocation coordinate arrays are
    required in practice but typed optional so partial/variant products still
    validate; the converter checks for the bands it needs.
    """

    latitude: ArraySpec[object]
    longitude: ArraySpec[object]
    altitude: ArraySpec[object]
    time_stamp: ArraySpec[object]
    orphans: GroupSpec[Mapping[str, object], Mapping[str, ArraySpec[object]]]
    oa01_radiance: ArraySpec[object]
    oa02_radiance: ArraySpec[object]
    oa03_radiance: ArraySpec[object]
    oa04_radiance: ArraySpec[object]
    oa05_radiance: ArraySpec[object]
    oa06_radiance: ArraySpec[object]
    oa07_radiance: ArraySpec[object]
    oa08_radiance: ArraySpec[object]
    oa09_radiance: ArraySpec[object]
    oa10_radiance: ArraySpec[object]
    oa11_radiance: ArraySpec[object]
    oa12_radiance: ArraySpec[object]
    oa13_radiance: ArraySpec[object]
    oa14_radiance: ArraySpec[object]
    oa15_radiance: ArraySpec[object]
    oa16_radiance: ArraySpec[object]
    oa17_radiance: ArraySpec[object]
    oa18_radiance: ArraySpec[object]
    oa19_radiance: ArraySpec[object]
    oa20_radiance: ArraySpec[object]
    oa21_radiance: ArraySpec[object]


class Sentinel3OlciMeasurementsGroup(GroupSpec[DatasetAttrs, Sentinel3OlciMeasurementsMembers]):
    """OLCI measurements group: 21 radiance bands + 2-D geolocation."""


class Sentinel3OlciRootMembers(TypedDict, closed=True, total=False):
    """Members of the OLCI root group."""

    measurements: Sentinel3OlciMeasurementsGroup
    quality: GroupSpec[
        Mapping[str, object],
        Mapping[
            str,
            GroupSpec[Mapping[str, object], Mapping[str, ArraySpec[object]]] | ArraySpec[object],
        ],
    ]
    conditions: GroupSpec[
        Mapping[str, object],
        Mapping[
            str,
            GroupSpec[Mapping[str, object], Mapping[str, ArraySpec[object]]] | ArraySpec[object],
        ],
    ]


class Sentinel3OlciRoot(GroupSpec[Sentinel3OlciRootAttrs, Sentinel3OlciRootMembers]):
    """Complete Sentinel-3 OLCI L1 EFR EOPF Zarr hierarchy."""

    @property
    def measurements(self) -> Sentinel3OlciMeasurementsGroup:
        """Get the measurements group."""
        group = self.members.get("measurements")
        if group is None:
            raise KeyError("measurements")
        return group
