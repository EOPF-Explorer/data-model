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

    latitude: ArraySpec[Mapping[str, object]]
    longitude: ArraySpec[Mapping[str, object]]
    altitude: ArraySpec[Mapping[str, object]]
    time_stamp: ArraySpec[Mapping[str, object]]
    orphans: GroupSpec[Mapping[str, object], Mapping[str, ArraySpec[Mapping[str, object]]]]
    oa01_radiance: ArraySpec[Mapping[str, object]]
    oa02_radiance: ArraySpec[Mapping[str, object]]
    oa03_radiance: ArraySpec[Mapping[str, object]]
    oa04_radiance: ArraySpec[Mapping[str, object]]
    oa05_radiance: ArraySpec[Mapping[str, object]]
    oa06_radiance: ArraySpec[Mapping[str, object]]
    oa07_radiance: ArraySpec[Mapping[str, object]]
    oa08_radiance: ArraySpec[Mapping[str, object]]
    oa09_radiance: ArraySpec[Mapping[str, object]]
    oa10_radiance: ArraySpec[Mapping[str, object]]
    oa11_radiance: ArraySpec[Mapping[str, object]]
    oa12_radiance: ArraySpec[Mapping[str, object]]
    oa13_radiance: ArraySpec[Mapping[str, object]]
    oa14_radiance: ArraySpec[Mapping[str, object]]
    oa15_radiance: ArraySpec[Mapping[str, object]]
    oa16_radiance: ArraySpec[Mapping[str, object]]
    oa17_radiance: ArraySpec[Mapping[str, object]]
    oa18_radiance: ArraySpec[Mapping[str, object]]
    oa19_radiance: ArraySpec[Mapping[str, object]]
    oa20_radiance: ArraySpec[Mapping[str, object]]
    oa21_radiance: ArraySpec[Mapping[str, object]]


class Sentinel3OlciMeasurementsGroup(GroupSpec[DatasetAttrs, Sentinel3OlciMeasurementsMembers]):
    """OLCI measurements group: 21 radiance bands + 2-D geolocation."""


class Sentinel3OlciRootMembers(TypedDict, closed=True, total=False):
    """Members of the OLCI root group."""

    measurements: Sentinel3OlciMeasurementsGroup
    quality: GroupSpec[
        Mapping[str, object],
        Mapping[
            str,
            GroupSpec[Mapping[str, object], Mapping[str, ArraySpec[Mapping[str, object]]]]
            | ArraySpec[Mapping[str, object]],
        ],
    ]
    conditions: GroupSpec[
        Mapping[str, object],
        Mapping[
            str,
            GroupSpec[Mapping[str, object], Mapping[str, ArraySpec[Mapping[str, object]]]]
            | ArraySpec[Mapping[str, object]],
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
