"""
Pydantic-zarr integrated models for Sentinel-2 EOPF Zarr data structure.

Uses the new pyz.GroupSpec with TypedDict members to enforce strict structure validation.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
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


class BandMetadata(BaseModel):
    """Metadata for a single Sentinel-2 spectral band.

    Contains physical and spectral characterization of a band.
    """

    bandwidth: Annotated[float | str, Field(description="Bandwidth in nm")]
    central_wavelength: Annotated[float, Field(description="Central wavelength in nm")]
    onboard_compression_rate: Annotated[
        str | float, Field(description="Compression rate")
    ]
    onboard_integration_time: Annotated[
        str | float, Field(description="Integration time")
    ]
    physical_gain: Annotated[str | float, Field(description="Physical gain factor")]
    spectral_response_step: Annotated[str, Field(description="Spectral response step")]
    spectral_response_values: Annotated[
        str, Field(description="Spectral response curve values")
    ]
    units: Annotated[str, Field(description="Unit of measurement")]
    wavelength_max: Annotated[float, Field(description="Maximum wavelength in nm")]
    wavelength_min: Annotated[float, Field(description="Minimum wavelength in nm")]


class BandDescription(BaseModel):
    """Collection of band metadata for all Sentinel-2 bands.

    Maps Sentinel-2 band identifiers to their spectral and technical metadata.
    All standard bands should be present in Sentinel-2 L1C and L2A products.

    Preserves both numeric (01, 02, ..., 8A) and named (b01, b02, ..., b8a) band keys
    from the original JSON structure for round-trip compatibility.
    """

    # Numeric band keys (JSON format)
    band_01: Annotated[BandMetadata, Field(alias="01")]
    band_02: Annotated[BandMetadata, Field(alias="02")]
    band_03: Annotated[BandMetadata, Field(alias="03")]
    band_04: Annotated[BandMetadata, Field(alias="04")]
    band_05: Annotated[BandMetadata, Field(alias="05")]
    band_06: Annotated[BandMetadata, Field(alias="06")]
    band_07: Annotated[BandMetadata, Field(alias="07")]
    band_08: Annotated[BandMetadata, Field(alias="08")]
    band_09: Annotated[BandMetadata, Field(alias="09")]
    band_10: Annotated[BandMetadata, Field(alias="10")]
    band_11: Annotated[BandMetadata, Field(alias="11")]
    band_12: Annotated[BandMetadata, Field(alias="12")]
    band_8a: Annotated[BandMetadata, Field(alias="8A")]

    # Named band keys (human-readable format)
    band_b01: Annotated[BandMetadata, Field(alias="b01")]
    band_b02: Annotated[BandMetadata, Field(alias="b02")]
    band_b03: Annotated[BandMetadata, Field(alias="b03")]
    band_b04: Annotated[BandMetadata, Field(alias="b04")]
    band_b05: Annotated[BandMetadata, Field(alias="b05")]
    band_b06: Annotated[BandMetadata, Field(alias="b06")]
    band_b07: Annotated[BandMetadata, Field(alias="b07")]
    band_b08: Annotated[BandMetadata, Field(alias="b08")]
    band_b09: Annotated[BandMetadata, Field(alias="b09")]
    band_b10: Annotated[BandMetadata, Field(alias="b10")]
    band_b11: Annotated[BandMetadata, Field(alias="b11")]
    band_b12: Annotated[BandMetadata, Field(alias="b12")]
    band_b8a: Annotated[BandMetadata, Field(alias="b8a")]

    model_config = {"populate_by_name": True}


class OtherMetadata(BaseModel):
    """Sentinel-2 product metadata container.

    Stores various metadata about the product including:
    - Quality information (L0/L2A, processing details)
    - Band descriptions (spectral characteristics)
    - Atmospheric corrections (AOT, water vapor)
    - Geolocation and timing information
    """

    # Core metadata fields
    AOT_retrieval_model: Annotated[
        str, Field(description="Aerosol Optical Thickness retrieval model")
    ]
    L0_ancillary_data_quality: Annotated[
        str, Field(description="L0 ancillary data quality indicator")
    ]
    L0_ephemeris_data_quality: Annotated[
        str, Field(description="L0 ephemeris data quality")
    ]
    NUC_table_ID: Annotated[
        int | str, Field(description="Non-Uniformity Correction table ID")
    ]
    SWIR_rearrangement_flag: Annotated[
        str | None, Field(description="SWIR band rearrangement flag")
    ]
    UTM_zone_identification: Annotated[str, Field(description="UTM zone identifier")]
    absolute_location_assessment_from_AOCS: Annotated[
        str, Field(description="Location assessment")
    ]

    # Band information
    band_description: Annotated[
        BandDescription, Field(description="Spectral band metadata for all bands")
    ]

    # Accuracy declarations
    declared_accuracy_of_AOT_model: Annotated[
        float | None, Field(description="AOT model accuracy")
    ]
    declared_accuracy_of_radiative_transfer_model: Annotated[
        float | None, Field(description="Radiative transfer accuracy")
    ]
    declared_accuracy_of_water_vapour_model: Annotated[
        float | None, Field(description="Water vapor model accuracy")
    ]

    # Correction flags
    electronic_crosstalk_correction_flag: Annotated[
        str | bool, Field(description="Electronic crosstalk correction")
    ]
    optical_crosstalk_correction_flag: Annotated[
        str | bool, Field(description="Optical crosstalk correction")
    ]
    onboard_compression_flag: Annotated[
        str | bool, Field(description="Onboard compression applied")
    ]
    onboard_equalization_flag: Annotated[
        str | bool, Field(description="Onboard equalization applied")
    ]

    # Product and geometry information
    eopf_category: Annotated[str, Field(description="EOPF product category")]
    geometric_refinement: Annotated[
        dict[str, Any] | str | None,
        Field(description="Geometric refinement information"),
    ]
    history: Annotated[
        list[dict[str, Any]] | str | None, Field(description="Processing history")
    ]
    horizontal_CRS_code: Annotated[
        str, Field(description="Coordinate Reference System code")
    ]
    horizontal_CRS_name: Annotated[str, Field(description="CRS name")]
    mean_sensing_time: Annotated[str | None, Field(description="Mean acquisition time")]

    # Sun/sensor geometry
    mean_sun_azimuth_angle_in_deg_for_all_bands_all_detectors: Annotated[
        float, Field(description="Mean sun azimuth in degrees")
    ]
    mean_sun_zenith_angle_in_deg_for_all_bands_all_detectors: Annotated[
        float, Field(description="Mean sun zenith in degrees")
    ]

    # Atmospheric parameters
    mean_value_of_aerosol_optical_thickness: Annotated[
        float | None, Field(description="Mean AOT value")
    ]
    mean_value_of_total_water_vapour_content: Annotated[
        float | None, Field(description="Mean water vapor content")
    ]

    # Meteo information (flexible structure)
    meteo: Annotated[
        dict[str, Any] | None, Field(description="Meteorological parameters")
    ]

    # Quality assessments
    multispectral_registration_assessment: Annotated[
        str | None, Field(description="Registration quality")
    ]
    product_quality_status: Annotated[str, Field(description="Product quality status")]
    planimetric_stability_assessment_from_AOCS: Annotated[
        str | None, Field(description="Planimetric stability")
    ]

    # Data degradation
    percentage_of_degraded_MSI_data: Annotated[
        float | None, Field(description="Percentage of degraded data")
    ]

    # Ozone information
    ozone_source: Annotated[str | None, Field(description="Ozone data source")]
    ozone_value: Annotated[float | str | None, Field(description="Ozone value")]

    # Reference band
    spectral_band_of_reference: Annotated[
        str, Field(description="Reference spectral band")
    ]

    # Radiometric correction
    reflectance_correction_factor_from_the_Sun_Earth_distance_variation_computed_using_the_acquisition_date: Annotated[
        float | None,
        Field(
            default=None,
            alias="reflectance_correction_factor_from_the_Sun-Earth_distance_variation_computed_using_the_acquisition_date",
            description="Reflectance correction factor for Sun-Earth distance variation",
        ),
    ]

    model_config = {"populate_by_name": True}


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

    other_metadata: dict[str, object]  # no validation
    stac_discovery: dict[str, object]  # no validation


class Sentinel2DataArray(ArraySpec[Sentinel2DataArrayAttrs]):
    """Sentinel-2 data array integrated with pydantic-zarr."""

    ...


class Sentinel2CoordinateArray(ArraySpec[Sentinel2DataArrayAttrs]):
    """Coordinate array for Sentinel-2 data."""

    ...


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

    pass


class Sentinel2AtmosphereMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for atmosphere group containing resolution datasets."""

    r10m: Sentinel2AtmosphereResolutionDataset
    r20m: Sentinel2AtmosphereResolutionDataset
    r60m: Sentinel2AtmosphereResolutionDataset


class Sentinel2AtmosphereDataset(GroupSpec[DatasetAttrs, Sentinel2AtmosphereMembers]):  # type: ignore[type-var]
    """Atmosphere quality data (AOT, WVP) at multiple resolutions."""

    pass


class Sentinel2ProbabilityDataset(
    GroupSpec[DatasetAttrs, ProbabilityResolutionMembers]  # type: ignore[type-var]
):
    """Probability data (cloud, snow) at multiple resolutions."""

    pass


class Sentinel2QuicklookDataset(GroupSpec[DatasetAttrs, QuicklookResolutionMembers]):  # type: ignore[type-var]
    """True Color Image (TCI) quicklook data at multiple resolutions."""

    pass


class Sentinel2MaskDataset(GroupSpec[DatasetAttrs, ConditionsMaskMembers]):  # type: ignore[type-var]
    """Mask data containing classification and detector footprints."""

    pass


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

    pass


class Sentinel2MeteorologyGroup(GroupSpec[DatasetAttrs, MeteorologyMembers]):  # type: ignore[type-var]
    """Meteorology group containing CAMS and ECMWF atmospheric data."""

    pass


class Sentinel2ConditionsMaskGroup(GroupSpec[DatasetAttrs, ConditionsMaskMembers]):  # type: ignore[type-var]
    """Mask subgroup in conditions."""

    pass


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
