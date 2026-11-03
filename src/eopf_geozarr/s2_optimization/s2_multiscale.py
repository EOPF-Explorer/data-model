"""
The Sentinel-2 multiscale policy tables.

This module holds the declarative side of the S2 multiscale layout: the aux
pyramid groups and levels (`_AUX_MULTISCALES`, mirrored by the plan's
`AUX_MULTISCALE_LEVELS`) and the SCL flag vocabulary. The pyramids themselves
are planned by `eopf_geozarr.data_api.s2_plan` (whose
`REFLECTANCE_SCALE_LEVELS` is the reflectance policy) and written by
`eopf_geozarr.data_api.s2_write`; the former DataTree-based write path was
removed once the plan-driven writer reproduced its stores byte for byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from eopf_geozarr.cpm.routing import product_type_of

if TYPE_CHECKING:
    import xarray as xr

SCL_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
SCL_LABELS = [
    "NO_DATA",
    "SATURATED_OR_DEFECTIVE",
    "CAST_SHADOWS",
    "CLOUD_SHADOWS",
    "VEGETATION",
    "NOT_VEGETATED",
    "WATER",
    "UNCLASSIFIED",
    "CLOUD_MEDIUM_PROBABILITY",
    "CLOUD_HIGH_PROBABILITY",
    "THIN_CIRRUS",
    "SNOW_ICE",
]

SCL_ATTRS = {
    "flag_values": SCL_VALUES,
    "flag_meanings": SCL_LABELS,
}
SCL_NODATA = SCL_VALUES[SCL_LABELS.index("NO_DATA")]


class S2Type(StrEnum):
    L1C = "L1C"
    L2A = "L2A"

    @classmethod
    def from_filename(cls, filename: str | None) -> S2Type | None:
        if not filename:
            return None
        for member in cls:
            if member.value in filename:
                return member
        return None

    @classmethod
    def from_datatree(cls, dt: xr.DataTree) -> S2Type | None:
        """Product level from `stac_discovery` `product:type`, else from the tree name.

        A tree opened from a Zarr store has no name, so the name alone is not enough.
        """
        return cls.from_filename(product_type_of(dt)) or cls.from_filename(dt.name)


@dataclass(frozen=True)
class MultiscaleSpec:
    """A group that gets a multiscale pyramid."""

    base_path: str
    """Group holding the pyramid levels, e.g. `/measurements/reflectance`."""
    scale_levels: tuple[int, ...]
    """Level resolutions in metres, finest first.

    The finest level must exist in the input. Coarser levels present in the input
    are reused; the others are coarsened from the level before them.
    """
    additional_attributes: dict[str, list[Any]] | None = None
    """Attributes written on every variable of every level (e.g. SCL flag meanings)."""
    product_levels: frozenset[S2Type] | None = None
    """Products the group exists in; `None` means every product."""

    @property
    def finest_key(self) -> str:
        return f"r{self.scale_levels[0]}m"

    def applies_to(self, s2_type: S2Type | None) -> bool:
        return self.product_levels is None or s2_type in self.product_levels


# Auxiliary (non-reflectance) groups. Groups missing from the input are skipped.
_AUX_MULTISCALES: tuple[MultiscaleSpec, ...] = (
    MultiscaleSpec(
        base_path="/conditions/mask/l2a_classification",
        scale_levels=(20, 60, 120, 360, 720),
        additional_attributes=SCL_ATTRS,
        product_levels=frozenset({S2Type.L2A}),
    ),
    MultiscaleSpec(
        base_path="/quality/probability",
        scale_levels=(20, 60, 120, 360, 720),
        product_levels=frozenset({S2Type.L2A}),
    ),
    MultiscaleSpec(
        base_path="/conditions/mask/l1c_classification",
        scale_levels=(60, 120, 360, 720),
    ),
)
