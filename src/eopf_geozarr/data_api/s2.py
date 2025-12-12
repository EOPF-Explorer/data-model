"""
Pydantic-zarr integrated models for Sentinel-2 EOPF Zarr data structure.

Uses the new pyz.GroupSpec with TypedDict members to enforce strict structure validation.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from pyproj import CRS
from typing_extensions import TypedDict

from eopf_geozarr.data_api.geozarr.common import (
    BaseDataArrayAttrs,
    CFStandardName,
    DatasetAttrs,
)
from eopf_geozarr.pyz.v2 import ArraySpec, GroupSpec
from eopf_geozarr.s2_optimization.s2_band_mapping import BAND_INFO

# ============================================================================
# Sentinel-2 Array Attributes Models
# ============================================================================


class OtherMetadataDict(TypedDict, total=False):
    """Sentinel-2 other_metadata attributes container.

    This is intentionally flexible to accommodate variations across different
    Sentinel-2 product types (L1C, L2A) and processing versions.

    Only horizontal_CRS_code is required as it's the load-bearing element
    needed by the API to extract CRS information.
    """

    # Required field - needed for CRS extraction
    horizontal_CRS_code: str

    # All other fields are optional and unvalidated
    # Different S2 products have different metadata structures


class Sentinel2ArrayAttributes(BaseModel):
    """Array attributes for Sentinel-2 measurement data.

    Contains structured metadata from Sentinel-2 products including:
    - other_metadata: Detailed processing and band information
    - stac_discovery: STAC-compliant discovery metadata
    """

    other_metadata: dict[str, object]  # unvalidated
    stac_discovery: dict[str, object]  # unvalidated


# Resolution levels
ResolutionLevel = Literal["r10m", "r20m", "r60m", "r120m", "r360m", "r720m"]


# Resolution-level members for probability data arrays
class ProbabilityArrayMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for probability arrays at a specific resolution (r10m, r20m, r60m).

    Closed TypedDict - contains probability arrays (cld, snw) and per-band/coordinate arrays.
    All fields are optional since not all data products are always present.
    """

    cld: ArraySpec[Any]
    snw: ArraySpec[Any]
    band: Any  # Per-band probability data (may be group or array)
    x: ArraySpec[Any]  # Coordinate arrays
    y: ArraySpec[Any]


# Probability resolution groups (r10m, r20m, r60m)
class ProbabilityResolutionMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for probability data containing resolution-level groups (r10m, r20m, r60m).

    Closed TypedDict - contains resolution groups as subgroups.
    All fields are optional since not all resolutions are always present.
    """

    r10m: GroupSpec[Any, ProbabilityArrayMembers]  # type: ignore[type-var]
    r20m: GroupSpec[Any, ProbabilityArrayMembers]  # type: ignore[type-var]
    r60m: GroupSpec[Any, ProbabilityArrayMembers]  # type: ignore[type-var]


# Resolution-level members for quicklook data arrays
class QuicklookArrayMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for quicklook arrays at a specific resolution.

    Closed TypedDict - typically contains TCI (True Color Image) and optional band/coordinate arrays.
    All fields are optional for flexibility.
    """

    tci: ArraySpec[Any]
    band: Any  # May contain per-band data
    x: ArraySpec[Any]  # Coordinate arrays
    y: ArraySpec[Any]


# Quicklook resolution groups (r10m, r20m, r60m)
class QuicklookResolutionMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for quicklook data containing resolution-level groups (r10m, r20m, r60m).

    Closed TypedDict - contains resolution groups as subgroups.
    All fields are optional since not all resolutions are always present.
    """

    r10m: GroupSpec[Any, QuicklookArrayMembers]  # type: ignore[type-var]
    r20m: GroupSpec[Any, QuicklookArrayMembers]  # type: ignore[type-var]
    r60m: GroupSpec[Any, QuicklookArrayMembers]  # type: ignore[type-var]


# Mask members - contains resolution-level groups or various classification/detector groups
class ConditionsMaskMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for mask subgroup in conditions.

    Closed TypedDict - can contain either:
    1. Resolution-level groups (r10m, r20m, r60m) with mask arrays, OR
    2. Classification and detector footprint groups/arrays

    All fields are optional since not all mask types are always present.
    Using Any to handle flexible deep hierarchies.
    """

    r10m: Any  # Resolution-level mask group
    r20m: Any  # Resolution-level mask group
    r60m: Any  # Resolution-level mask group
    detector_footprint: Any  # May have r10m/r20m/r60m array subgroups
    l1c_classification: Any  # May have r20m/r60m array subgroups
    l2a_classification: Any  # May have r20m/r60m array subgroups


# Geometry members - contains angle and orientation groups/arrays
class GeometryMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for geometry group containing sun and viewing angles.

    Closed TypedDict - contains angle and geometry groups/arrays with flexible internal structure.
    Common members include sun angles, viewing angles, band angles, detector angles.
    All fields are optional since not all angles are always present.

    Note: Members can be either GroupSpec (for hierarchical data) or ArraySpec (for direct arrays),
    allowing for flexible deep hierarchies. Using Any to handle Union[GroupSpec[Any, Any], ArraySpec[Any]].
    """

    angle: Any  # Contains resolution-level angle data (group or array)
    band: Any  # Contains per-band angle data (group or array)
    detector: Any  # Contains per-detector angle data (group or array)
    sun_angles: Any  # Sun geometry data (group or array)
    viewing_incidence_angles: Any  # Viewing geometry data (group or array)
    mean_sun_angles: Any  # Mean sun angles (group or array)
    mean_viewing_incidence_angles: Any  # Mean viewing angles (group or array)
    x: Any  # Coordinate arrays (group or array)
    y: Any
    time: Any


# Meteorology members - contains CAMS and ECMWF atmospheric data
class MeteorologyMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for meteorology group containing CAMS and ECMWF atmospheric data.

    Closed TypedDict - contains subgroups for different meteorological data sources.
    All fields are optional since different meteorological data sources have different variables.

    Note: Both CAMS and ECMWF subgroups can have flexible internal structure with
    atmospheric parameter arrays/groups.
    """

    cams: GroupSpec[Any, Any]  # Contains CAMS atmospheric data
    ecmwf: GroupSpec[Any, Any]  # Contains ECMWF atmospheric data


# Sentinel-2 spectral band identifiers
BandName = Literal[
    # 10m bands
    "b02",  # Blue
    "b03",  # Green
    "b04",  # Red
    "b08",  # NIR
    # 20m bands
    "b05",  # Red Edge 1
    "b06",  # Red Edge 2
    "b07",  # Red Edge 3
    "b8a",  # NIR Narrow
    "b11",  # SWIR 1
    "b12",  # SWIR 2
    # 60m bands
    "b01",  # Coastal aerosol
    "b09",  # Water Vapor
    "b10",  # Cirrus
]

# All valid band names
ALL_BAND_NAMES = {
    "b01",
    "b02",
    "b03",
    "b04",
    "b05",
    "b06",
    "b07",
    "b08",
    "b8a",
    "b09",
    "b10",
    "b11",
    "b12",
}

# Native band mappings for validation
NATIVE_BANDS = {
    10: {"b02", "b03", "b04", "b08"},
    20: {"b05", "b06", "b07", "b8a", "b11", "b12"},
    60: {"b01", "b09", "b10"},
}

# Resolution name to meters mapping
RESOLUTION_TO_METERS = {
    "r10m": 10,
    "r20m": 20,
    "r60m": 60,
    "r120m": 120,
    "r360m": 360,
    "r720m": 720,
}

# Quality data names
QualityDataName = Literal[
    "scl",  # Scene Classification Layer
    "aot",  # Aerosol Optical Thickness
    "wvp",  # Water Vapor
    "cld",  # Cloud probability
    "snw",  # Snow probability
]

# Variable type for determining resampling strategy
VariableType = Literal[
    "reflectance",  # Block averaging
    "classification",  # Nearest neighbor
    "quality_mask",  # Logical OR
    "probability",  # Averaged clamping
]


class Sentinel2BandInfo(BaseModel):
    """Complete information about a Sentinel-2 spectral band."""

    name: BandName
    native_resolution: Annotated[int, Field(ge=10, le=60)]
    data_type: str = "uint16"
    wavelength_center: Annotated[float, Field(gt=0, description="nm")]
    wavelength_width: Annotated[float, Field(gt=0, description="nm")]
    long_name: str
    standard_name: str = "toa_bidirectional_reflectance"
    units: str = "1"

    @classmethod
    def from_band_name(cls, band_name: BandName) -> Sentinel2BandInfo:
        """Create BandInfo from band name."""
        if band_name not in BAND_INFO:
            raise ValueError(f"Unknown band name: {band_name}")
        info = BAND_INFO[band_name]
        return cls(
            name=band_name,
            native_resolution=info.native_resolution,
            wavelength_center=info.wavelength_center,
            wavelength_width=info.wavelength_width,
            long_name=f"Band {band_name.upper()}",
        )


class Sentinel2DataArrayAttrs(BaseDataArrayAttrs):
    """Extended attributes for Sentinel-2 data arrays."""

    long_name: str
    standard_name: CFStandardName | str | None = None
    units: str = "1"


class Sentinel2RootAttrs(BaseModel):
    """Root-level attributes for Sentinel-2 DataTree."""

    other_metadata: OtherMetadataDict
    stac_discovery: dict[str, object]  # no validation


class Sentinel2DataArray(ArraySpec[Sentinel2DataArrayAttrs]):
    """Sentinel-2 data array integrated with pydantic-zarr."""


class Sentinel2CoordinateArray(ArraySpec[Sentinel2DataArrayAttrs]):
    """Coordinate array for Sentinel-2 data."""


# TypedDict definitions for members structure
class Sentinel2ResolutionMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members dict for a resolution dataset (r10m, r20m, r60m).

    Closed TypedDict - no extra keys are allowed beyond those explicitly defined.
    All fields are optional since not all bands are present in every resolution.
    """

    x: ArraySpec[Any]
    y: ArraySpec[Any]
    time: ArraySpec[Any]
    b01: ArraySpec[Any]
    b02: ArraySpec[Any]
    b03: ArraySpec[Any]
    b04: ArraySpec[Any]
    b05: ArraySpec[Any]
    b06: ArraySpec[Any]
    b07: ArraySpec[Any]
    b08: ArraySpec[Any]
    b8a: ArraySpec[Any]
    b09: ArraySpec[Any]
    b10: ArraySpec[Any]
    b11: ArraySpec[Any]
    b12: ArraySpec[Any]


class Sentinel2ResolutionDataset(GroupSpec[DatasetAttrs, Sentinel2ResolutionMembers]):  # type: ignore[type-var]
    """A single resolution dataset within reflectance (r10m, r20m, r60m)."""


class Sentinel2ReflectanceMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for reflectance group.

    Closed TypedDict - only r10m, r20m, r60m keys are allowed.
    """

    r10m: Sentinel2ResolutionDataset
    r20m: Sentinel2ResolutionDataset
    r60m: Sentinel2ResolutionDataset


class Sentinel2ReflectanceGroup(GroupSpec[DatasetAttrs, Sentinel2ReflectanceMembers]):  # type: ignore[type-var]
    """Reflectance data organized by resolution."""


class Sentinel2MeasurementsMembers(TypedDict, closed=True):  # type: ignore[call-arg]
    """Members for measurements group.

    Closed TypedDict - only 'reflectance' key is allowed.
    """

    reflectance: Sentinel2ReflectanceGroup


class Sentinel2MeasurementsGroup(GroupSpec[DatasetAttrs, Sentinel2MeasurementsMembers]):  # type: ignore[type-var]
    """Measurements group containing reflectance data."""

    @property
    def reflectance(self) -> Sentinel2ReflectanceGroup:
        """Get reflectance subgroup."""
        return self.members["reflectance"]


# Quality data groups - need resolution-level typed groups


class Sentinel2AtmosphereResolutionMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for atmosphere data at a specific resolution.

    Closed TypedDict - may contain aot and/or wvp arrays depending on available data.
    All fields are optional since not all data products are always present.
    """

    aot: ArraySpec[Any]
    wvp: ArraySpec[Any]
    x: ArraySpec[Any]  # Coordinate arrays
    y: ArraySpec[Any]


class Sentinel2AtmosphereResolutionDataset(
    GroupSpec[DatasetAttrs, Sentinel2AtmosphereResolutionMembers]  # type: ignore[type-var]
):
    """Atmosphere data at a single resolution."""


class Sentinel2AtmosphereMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for atmosphere group containing resolution datasets."""

    r10m: Sentinel2AtmosphereResolutionDataset
    r20m: Sentinel2AtmosphereResolutionDataset
    r60m: Sentinel2AtmosphereResolutionDataset


class Sentinel2AtmosphereDataset(GroupSpec[DatasetAttrs, Sentinel2AtmosphereMembers]):  # type: ignore[type-var]
    """Atmosphere quality data (AOT, WVP) at multiple resolutions."""


class Sentinel2ProbabilityDataset(
    GroupSpec[DatasetAttrs, ProbabilityResolutionMembers]  # type: ignore[type-var]
):
    """Probability data (cloud, snow) at multiple resolutions."""


class Sentinel2QuicklookDataset(GroupSpec[DatasetAttrs, QuicklookResolutionMembers]):  # type: ignore[type-var]
    """True Color Image (TCI) quicklook data at multiple resolutions."""


class Sentinel2MaskDataset(GroupSpec[DatasetAttrs, ConditionsMaskMembers]):  # type: ignore[type-var]
    """Mask data containing classification and detector footprints."""


class Sentinel2QualityMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for quality group.

    Closed TypedDict with optional fields to accommodate different product levels:
    - Sentinel-2A/C (L2A): atmosphere, probability, l2a_quicklook, mask
    - Sentinel-2B (L1C): l1c_quicklook, mask
    """

    atmosphere: Sentinel2AtmosphereDataset
    probability: Sentinel2ProbabilityDataset
    l2a_quicklook: Sentinel2QuicklookDataset
    l1c_quicklook: Sentinel2QuicklookDataset
    mask: Sentinel2MaskDataset


class Sentinel2QualityGroup(GroupSpec[DatasetAttrs, Sentinel2QualityMembers]):  # type: ignore[type-var]
    """Quality group containing atmosphere, probability, classification, and quicklook data.

    Supports both L2A products (Sentinel-2A, 2C) and L1C products (Sentinel-2B).
    """

    def atmosphere(self) -> Sentinel2AtmosphereDataset | None:
        """Get atmosphere subgroup (L2A only)."""
        return self.members.get("atmosphere")

    def probability(self) -> Sentinel2ProbabilityDataset | None:
        """Get probability subgroup (L2A only)."""
        return self.members.get("probability")

    def quicklook(self) -> Sentinel2QuicklookDataset | None:
        """Get quicklook subgroup (L2A: l2a_quicklook or L1C: l1c_quicklook)."""
        return self.members.get("l2a_quicklook") or self.members.get("l1c_quicklook")

    def mask(self) -> Sentinel2MaskDataset | None:
        """Get mask subgroup."""
        return self.members.get("mask")


# Conditions groups
class Sentinel2GeometryGroup(GroupSpec[DatasetAttrs, GeometryMembers]):  # type: ignore[type-var]
    """Geometry group containing sun and viewing angles."""


class Sentinel2MeteorologyGroup(GroupSpec[DatasetAttrs, MeteorologyMembers]):  # type: ignore[type-var]
    """Meteorology group containing CAMS and ECMWF atmospheric data."""


class Sentinel2ConditionsMaskGroup(GroupSpec[DatasetAttrs, ConditionsMaskMembers]):  # type: ignore[type-var]
    """Mask subgroup in conditions."""


class Sentinel2ConditionsMembers(TypedDict, closed=True):  # type: ignore[call-arg]
    """Members for conditions group.

    Closed TypedDict - only geometry, mask, meteorology keys are allowed.
    """

    geometry: Sentinel2GeometryGroup
    mask: Sentinel2ConditionsMaskGroup
    meteorology: Sentinel2MeteorologyGroup


class Sentinel2ConditionsGroup(GroupSpec[DatasetAttrs, Sentinel2ConditionsMembers]):  # type: ignore[type-var]
    """Conditions group containing geometry and meteorology data."""

    def geometry(self) -> Sentinel2GeometryGroup | None:
        """Get geometry subgroup."""
        return self.members.get("geometry")

    def mask(self) -> Sentinel2ConditionsMaskGroup | None:
        """Get mask subgroup."""
        return self.members.get("mask")

    def meteorology(self) -> Sentinel2MeteorologyGroup | None:
        """Get meteorology subgroup."""
        return self.members.get("meteorology")


# Root model
class Sentinel2RootMembers(TypedDict, closed=True):  # type: ignore[call-arg]
    """Members for Sentinel-2 root group.

    Closed TypedDict - only measurements, quality, conditions keys are allowed.
    """

    measurements: Sentinel2MeasurementsGroup
    quality: Sentinel2QualityGroup
    conditions: Sentinel2ConditionsGroup


class Sentinel2Root(GroupSpec[Sentinel2RootAttrs, Sentinel2RootMembers]):  # type: ignore[type-var]
    """Complete Sentinel-2 EOPF Zarr hierarchy.

    The hierarchy follows EOPF organization:
    Root
    ├── measurements/
    │   └── reflectance/
    │       ├── r10m/
    │       ├── r20m/
    │       └── r60m/
    ├── quality/
    │   ├── atmosphere/
    │   ├── l2a_quicklook/
    │   ├── mask/
    │   └── probability/
    └── conditions/
        ├── geometry/
        ├── mask/
        └── meteorology/
    """

    @property
    def measurements(self) -> Sentinel2MeasurementsGroup:
        """Get measurements group."""
        return self.members["measurements"]

    @property
    def quality(self) -> Sentinel2QualityGroup:
        """Get quality group."""
        return self.members["quality"]

    @property
    def conditions(self) -> Sentinel2ConditionsGroup:
        """Get conditions group."""
        return self.members["conditions"]

    @property
    def crs(self) -> CRS:
        """Get the coordinate reference system (CRS) for this product"""
        crs_code = self.attributes.other_metadata["horizontal_CRS_code"]
        # Handle both "EPSG:32635" and "32635" formats
        crs_code = crs_code.removeprefix("EPSG:")  # Remove "EPSG:" prefix
        return CRS.from_epsg(int(crs_code))
