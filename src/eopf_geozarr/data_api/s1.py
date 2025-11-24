"""
Pydantic-zarr integrated models for Sentinel-1A EOPF Zarr data structure.

Uses the new pyz.GroupSpec with TypedDict members to enforce strict structure validation.
"""

from __future__ import annotations

from typing import Any, Mapping, Union

from pydantic import BaseModel
from typing_extensions import TypedDict

from eopf_geozarr.data_api.geozarr.common import (
    BaseDataArrayAttrs,
    CFStandardName,
    DatasetAttrs,
)
from eopf_geozarr.pyz.v2 import ArraySpec, GroupSpec

# Member type for groups with any nested structures (groups or arrays)
# Used for groups with dynamic or variable nested structures
AnyMembers = Mapping[str, Union[GroupSpec[Any, Any], ArraySpec[Any]]]


class Sentinel1DataArrayAttrs(BaseDataArrayAttrs):
    """Extended attributes for Sentinel-1 data arrays."""

    long_name: str
    standard_name: CFStandardName | str | None = None
    units: str = "1"


class Sentinel1RootAttrs(BaseModel):
    """Root-level attributes for Sentinel-1 DataTree."""

    other_metadata: dict[str, object]
    stac_discovery: dict[str, object]


class Sentinel1DataArray(ArraySpec[Sentinel1DataArrayAttrs]):
    """Sentinel-1 data array integrated with pydantic-zarr."""

    pass


# Conditions groups
class Sentinel1AntennaPatternMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for antenna_pattern group.

    All fields are optional to support different product variants.
    """

    azimuth_time: ArraySpec[Any]
    count: ArraySpec[Any]
    elevation_angle: ArraySpec[Any]
    incidence_angle: ArraySpec[Any]
    roll: ArraySpec[Any]
    slant_range_time: ArraySpec[Any]  # S1C variant
    slant_range_time_ap: ArraySpec[Any]
    swath: ArraySpec[Any]
    terrain_height: ArraySpec[Any]


class Sentinel1AntennaPatternGroup(
    GroupSpec[DatasetAttrs, Sentinel1AntennaPatternMembers]  # type: ignore[type-var]
):
    """Antenna pattern group containing antenna characteristics."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        return self.members["azimuth_time"]

    @property
    def count(self) -> ArraySpec[Any]:
        """Get count array."""
        return self.members["count"]

    @property
    def elevation_angle(self) -> ArraySpec[Any]:
        """Get elevation_angle array."""
        return self.members["elevation_angle"]

    @property
    def incidence_angle(self) -> ArraySpec[Any]:
        """Get incidence_angle array."""
        return self.members["incidence_angle"]

    @property
    def roll(self) -> ArraySpec[Any]:
        """Get roll array."""
        return self.members["roll"]

    @property
    def slant_range_time_ap(self) -> ArraySpec[Any]:
        """Get slant_range_time_ap array."""
        return self.members["slant_range_time_ap"]

    @property
    def swath(self) -> ArraySpec[Any]:
        """Get swath array."""
        return self.members["swath"]

    @property
    def terrain_height(self) -> ArraySpec[Any]:
        """Get terrain_height array."""
        return self.members["terrain_height"]


class Sentinel1AttitudeMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for attitude group."""

    azimuth_time: ArraySpec[Any]
    pitch: ArraySpec[Any]
    q0: ArraySpec[Any]
    q1: ArraySpec[Any]
    q2: ArraySpec[Any]
    q3: ArraySpec[Any]
    roll: ArraySpec[Any]
    wx: ArraySpec[Any]
    wy: ArraySpec[Any]
    wz: ArraySpec[Any]
    yaw: ArraySpec[Any]


class Sentinel1AttitudeGroup(GroupSpec[DatasetAttrs, Sentinel1AttitudeMembers]):  # type: ignore[type-var]
    """Attitude group containing spacecraft attitude data."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        return self.members["azimuth_time"]

    @property
    def pitch(self) -> ArraySpec[Any]:
        """Get pitch array."""
        return self.members["pitch"]

    @property
    def q0(self) -> ArraySpec[Any]:
        """Get q0 array."""
        return self.members["q0"]

    @property
    def q1(self) -> ArraySpec[Any]:
        """Get q1 array."""
        return self.members["q1"]

    @property
    def q2(self) -> ArraySpec[Any]:
        """Get q2 array."""
        return self.members["q2"]

    @property
    def q3(self) -> ArraySpec[Any]:
        """Get q3 array."""
        return self.members["q3"]

    @property
    def roll(self) -> ArraySpec[Any]:
        """Get roll array."""
        return self.members["roll"]

    @property
    def wx(self) -> ArraySpec[Any]:
        """Get wx array."""
        return self.members["wx"]

    @property
    def wy(self) -> ArraySpec[Any]:
        """Get wy array."""
        return self.members["wy"]

    @property
    def wz(self) -> ArraySpec[Any]:
        """Get wz array."""
        return self.members["wz"]

    @property
    def yaw(self) -> ArraySpec[Any]:
        """Get yaw array."""
        return self.members["yaw"]


class Sentinel1AzimuthFmRateMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for azimuth_fm_rate group."""

    azimuth_fm_rate_polynomial: ArraySpec[Any]
    azimuth_time: ArraySpec[Any]
    t0: ArraySpec[Any]


class Sentinel1AzimuthFmRateGroup(
    GroupSpec[DatasetAttrs, Sentinel1AzimuthFmRateMembers]  # type: ignore[type-var]
):
    """Azimuth FM rate group."""

    @property
    def azimuth_fm_rate_polynomial(self) -> ArraySpec[Any]:
        """Get azimuth_fm_rate_polynomial array."""
        return self.members["azimuth_fm_rate_polynomial"]

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        return self.members["azimuth_time"]

    @property
    def t0(self) -> ArraySpec[Any]:
        """Get t0 array."""
        return self.members["t0"]


class Sentinel1CoordinateConversionMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for coordinate_conversion group."""

    azimuth_time: ArraySpec[Any]
    gr0: ArraySpec[Any]
    grsr_coefficients: ArraySpec[Any]
    slant_range_time: ArraySpec[Any]
    sr0: ArraySpec[Any]
    srgr_coefficients: ArraySpec[Any]


class Sentinel1CoordinateConversionGroup(
    GroupSpec[DatasetAttrs, Sentinel1CoordinateConversionMembers]  # type: ignore[type-var]
):
    """Coordinate conversion group."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        return self.members["azimuth_time"]

    @property
    def gr0(self) -> ArraySpec[Any]:
        """Get gr0 array."""
        return self.members["gr0"]

    @property
    def grsr_coefficients(self) -> ArraySpec[Any]:
        """Get grsr_coefficients array."""
        return self.members["grsr_coefficients"]

    @property
    def slant_range_time(self) -> ArraySpec[Any]:
        """Get slant_range_time array."""
        return self.members["slant_range_time"]

    @property
    def sr0(self) -> ArraySpec[Any]:
        """Get sr0 array."""
        return self.members["sr0"]

    @property
    def srgr_coefficients(self) -> ArraySpec[Any]:
        """Get srgr_coefficients array."""
        return self.members["srgr_coefficients"]


class Sentinel1DopplerCentroidMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for doppler_centroid group."""

    azimuth_time: ArraySpec[Any]
    data_dc_polynomial: ArraySpec[Any]
    data_dc_rms_error: ArraySpec[Any]
    data_dc_rms_error_above_threshold: ArraySpec[Any]
    degree: ArraySpec[Any]
    fine_dce_azimuth_start_time: ArraySpec[Any]
    fine_dce_azimuth_stop_time: ArraySpec[Any]
    geometry_dc_polynomial: ArraySpec[Any]
    t0: ArraySpec[Any]


class Sentinel1DopplerCentroidGroup(
    GroupSpec[DatasetAttrs, Sentinel1DopplerCentroidMembers]  # type: ignore[type-var]
):
    """Doppler centroid group."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        return self.members["azimuth_time"]

    @property
    def data_dc_polynomial(self) -> ArraySpec[Any]:
        """Get data_dc_polynomial array."""
        return self.members["data_dc_polynomial"]

    @property
    def data_dc_rms_error(self) -> ArraySpec[Any]:
        """Get data_dc_rms_error array."""
        return self.members["data_dc_rms_error"]

    @property
    def data_dc_rms_error_above_threshold(self) -> ArraySpec[Any]:
        """Get data_dc_rms_error_above_threshold array."""
        return self.members["data_dc_rms_error_above_threshold"]

    @property
    def degree(self) -> ArraySpec[Any]:
        """Get degree array."""
        return self.members["degree"]

    @property
    def fine_dce_azimuth_start_time(self) -> ArraySpec[Any]:
        """Get fine_dce_azimuth_start_time array."""
        return self.members["fine_dce_azimuth_start_time"]

    @property
    def fine_dce_azimuth_stop_time(self) -> ArraySpec[Any]:
        """Get fine_dce_azimuth_stop_time array."""
        return self.members["fine_dce_azimuth_stop_time"]

    @property
    def geometry_dc_polynomial(self) -> ArraySpec[Any]:
        """Get geometry_dc_polynomial array."""
        return self.members["geometry_dc_polynomial"]

    @property
    def t0(self) -> ArraySpec[Any]:
        """Get t0 array."""
        return self.members["t0"]


class Sentinel1GcpMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for GCP (Ground Control Points) group.

    All fields are optional to support different product variants (S1A, S1C).
    """

    azimuth_time: ArraySpec[Any]
    azimuth_time_gcp: ArraySpec[Any]
    elevation_angle: ArraySpec[Any]
    ground_range: ArraySpec[Any]
    height: ArraySpec[Any]
    incidence_angle: ArraySpec[Any]
    latitude: ArraySpec[Any]
    line: ArraySpec[Any]
    longitude: ArraySpec[Any]
    pixel: ArraySpec[Any]
    slant_range_time: ArraySpec[Any]  # S1C variant
    slant_range_time_gcp: ArraySpec[Any]


class Sentinel1GcpGroup(GroupSpec[DatasetAttrs, Sentinel1GcpMembers]):  # type: ignore[type-var]
    """Ground Control Points (GCP) group."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        return self.members["azimuth_time"]

    @property
    def azimuth_time_gcp(self) -> ArraySpec[Any]:
        """Get azimuth_time_gcp array."""
        return self.members["azimuth_time_gcp"]

    @property
    def elevation_angle(self) -> ArraySpec[Any]:
        """Get elevation_angle array."""
        return self.members["elevation_angle"]

    @property
    def ground_range(self) -> ArraySpec[Any]:
        """Get ground_range array."""
        return self.members["ground_range"]

    @property
    def height(self) -> ArraySpec[Any]:
        """Get height array."""
        return self.members["height"]

    @property
    def incidence_angle(self) -> ArraySpec[Any]:
        """Get incidence_angle array."""
        return self.members["incidence_angle"]

    @property
    def latitude(self) -> ArraySpec[Any]:
        """Get latitude array."""
        return self.members["latitude"]

    @property
    def line(self) -> ArraySpec[Any]:
        """Get line array."""
        return self.members["line"]

    @property
    def longitude(self) -> ArraySpec[Any]:
        """Get longitude array."""
        return self.members["longitude"]

    @property
    def pixel(self) -> ArraySpec[Any]:
        """Get pixel array."""
        return self.members["pixel"]

    @property
    def slant_range_time_gcp(self) -> ArraySpec[Any]:
        """Get slant_range_time_gcp array."""
        return self.members["slant_range_time_gcp"]


class Sentinel1OrbitMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for orbit group."""

    axis: ArraySpec[Any]
    azimuth_time: ArraySpec[Any]
    position: ArraySpec[Any]
    velocity: ArraySpec[Any]


class Sentinel1OrbitGroup(GroupSpec[DatasetAttrs, Sentinel1OrbitMembers]):  # type: ignore[type-var]
    """Orbit group containing spacecraft position and velocity."""

    @property
    def axis(self) -> ArraySpec[Any]:
        """Get axis array."""
        return self.members["axis"]

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        return self.members["azimuth_time"]

    @property
    def position(self) -> ArraySpec[Any]:
        """Get position array."""
        return self.members["position"]

    @property
    def velocity(self) -> ArraySpec[Any]:
        """Get velocity array."""
        return self.members["velocity"]


class Sentinel1ReferenceReplicaMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for reference_replica group.

    Closed TypedDict - only reference replica coefficient array keys are allowed.
    All fields are optional since not all reference replica data may be present.
    """

    azimuth_time: ArraySpec[Any]
    reference_replica_amplitude_coefficients: ArraySpec[Any]
    reference_replica_phase_coefficients: ArraySpec[Any]


class Sentinel1ReferenceReplicaGroup(
    GroupSpec[DatasetAttrs, Sentinel1ReferenceReplicaMembers]  # type: ignore[type-var]
):
    """Reference replica group."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        return self.members["azimuth_time"]

    @property
    def reference_replica_amplitude_coefficients(self) -> ArraySpec[Any]:
        """Get reference_replica_amplitude_coefficients array."""
        return self.members["reference_replica_amplitude_coefficients"]

    @property
    def reference_replica_phase_coefficients(self) -> ArraySpec[Any]:
        """Get reference_replica_phase_coefficients array."""
        return self.members["reference_replica_phase_coefficients"]


class Sentinel1ReplicaMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for replica group.

    Closed TypedDict - only pulse replica data array keys are allowed.
    All fields are optional since not all replica data may be present.
    """

    absolute_pg_product_valid_flag: ArraySpec[Any]
    azimuth_time: ArraySpec[Any]
    cross_correlation_peak_location: ArraySpec[Any]
    cross_correlation_pslr: ArraySpec[Any]
    internal_time_delay: ArraySpec[Any]
    model_pg_product_amplitude: ArraySpec[Any]
    model_pg_product_phase: ArraySpec[Any]
    pg_product_amplitude: ArraySpec[Any]
    pg_product_phase: ArraySpec[Any]
    reconstructed_replica_valid_flag: ArraySpec[Any]
    relative_pg_product_valid_flag: ArraySpec[Any]


class Sentinel1ReplicaGroup(GroupSpec[DatasetAttrs, Sentinel1ReplicaMembers]):  # type: ignore[type-var]
    """Replica group containing pulse replica data."""

    @property
    def absolute_pg_product_valid_flag(self) -> ArraySpec[Any]:
        """Get absolute_pg_product_valid_flag array."""
        return self.members["absolute_pg_product_valid_flag"]

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        return self.members["azimuth_time"]

    @property
    def cross_correlation_peak_location(self) -> ArraySpec[Any]:
        """Get cross_correlation_peak_location array."""
        return self.members["cross_correlation_peak_location"]

    @property
    def cross_correlation_pslr(self) -> ArraySpec[Any]:
        """Get cross_correlation_pslr array."""
        return self.members["cross_correlation_pslr"]

    @property
    def internal_time_delay(self) -> ArraySpec[Any]:
        """Get internal_time_delay array."""
        return self.members["internal_time_delay"]

    @property
    def model_pg_product_amplitude(self) -> ArraySpec[Any]:
        """Get model_pg_product_amplitude array."""
        return self.members["model_pg_product_amplitude"]

    @property
    def model_pg_product_phase(self) -> ArraySpec[Any]:
        """Get model_pg_product_phase array."""
        return self.members["model_pg_product_phase"]

    @property
    def pg_product_amplitude(self) -> ArraySpec[Any]:
        """Get pg_product_amplitude array."""
        return self.members["pg_product_amplitude"]

    @property
    def pg_product_phase(self) -> ArraySpec[Any]:
        """Get pg_product_phase array."""
        return self.members["pg_product_phase"]

    @property
    def reconstructed_replica_valid_flag(self) -> ArraySpec[Any]:
        """Get reconstructed_replica_valid_flag array."""
        return self.members["reconstructed_replica_valid_flag"]

    @property
    def relative_pg_product_valid_flag(self) -> ArraySpec[Any]:
        """Get relative_pg_product_valid_flag array."""
        return self.members["relative_pg_product_valid_flag"]


class Sentinel1TerrainHeightMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for terrain_height group."""

    azimuth_time: ArraySpec[Any]
    terrain_height: ArraySpec[Any]


class Sentinel1TerrainHeightGroup(
    GroupSpec[DatasetAttrs, Sentinel1TerrainHeightMembers]  # type: ignore[type-var]
):
    """Terrain height group."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        return self.members["azimuth_time"]

    @property
    def terrain_height(self) -> ArraySpec[Any]:
        """Get terrain_height array."""
        return self.members["terrain_height"]


class Sentinel1ConditionsMembers(TypedDict, closed=True):  # type: ignore[call-arg]
    """Members for conditions group.

    Closed TypedDict - only antenna_pattern, attitude, azimuth_fm_rate, etc. keys are allowed.
    """

    antenna_pattern: Sentinel1AntennaPatternGroup
    attitude: Sentinel1AttitudeGroup
    azimuth_fm_rate: Sentinel1AzimuthFmRateGroup
    coordinate_conversion: Sentinel1CoordinateConversionGroup
    doppler_centroid: Sentinel1DopplerCentroidGroup
    gcp: Sentinel1GcpGroup
    orbit: Sentinel1OrbitGroup
    reference_replica: Sentinel1ReferenceReplicaGroup
    replica: Sentinel1ReplicaGroup
    terrain_height: Sentinel1TerrainHeightGroup


class Sentinel1ConditionsGroup(GroupSpec[DatasetAttrs, Sentinel1ConditionsMembers]):  # type: ignore[type-var]
    """Conditions group containing acquisition and processing metadata."""

    def get_antenna_pattern(self) -> Sentinel1AntennaPatternGroup | None:
        """Get antenna pattern subgroup."""
        return self.members["antenna_pattern"]

    def get_attitude(self) -> Sentinel1AttitudeGroup | None:
        """Get spacecraft attitude subgroup."""
        return self.members["attitude"]

    def get_azimuth_fm_rate(self) -> Sentinel1AzimuthFmRateGroup | None:
        """Get azimuth FM rate subgroup."""
        return self.members["azimuth_fm_rate"]

    def get_coordinate_conversion(self) -> Sentinel1CoordinateConversionGroup | None:
        """Get coordinate conversion subgroup."""
        return self.members["coordinate_conversion"]

    def get_doppler_centroid(self) -> Sentinel1DopplerCentroidGroup | None:
        """Get Doppler centroid subgroup."""
        return self.members["doppler_centroid"]

    def get_gcp(self) -> Sentinel1GcpGroup | None:
        """Get Ground Control Points subgroup."""
        return self.members["gcp"]

    def get_orbit(self) -> Sentinel1OrbitGroup | None:
        """Get orbit subgroup."""
        return self.members["orbit"]

    def get_reference_replica(self) -> Sentinel1ReferenceReplicaGroup | None:
        """Get reference replica subgroup."""
        return self.members["reference_replica"]

    def get_replica(self) -> Sentinel1ReplicaGroup | None:
        """Get replica subgroup."""
        return self.members["replica"]

    def get_terrain_height(self) -> Sentinel1TerrainHeightGroup | None:
        """Get terrain height subgroup."""
        return self.members["terrain_height"]


# Quality groups
class Sentinel1CalibrationMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for calibration group."""

    azimuth_time: ArraySpec[Any]
    beta_nought: ArraySpec[Any]
    dn: ArraySpec[Any]
    gamma: ArraySpec[Any]
    ground_range: ArraySpec[Any]
    line: ArraySpec[Any]
    pixel: ArraySpec[Any]
    sigma_nought: ArraySpec[Any]


class Sentinel1CalibrationGroup(GroupSpec[DatasetAttrs, Sentinel1CalibrationMembers]):  # type: ignore[type-var]
    """Calibration group containing radiometric calibration data."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        return self.members["azimuth_time"]

    @property
    def beta_nought(self) -> ArraySpec[Any]:
        """Get beta_nought array."""
        return self.members["beta_nought"]

    @property
    def dn(self) -> ArraySpec[Any]:
        """Get dn array."""
        return self.members["dn"]

    @property
    def gamma(self) -> ArraySpec[Any]:
        """Get gamma array."""
        return self.members["gamma"]

    @property
    def ground_range(self) -> ArraySpec[Any]:
        """Get ground_range array."""
        return self.members["ground_range"]

    @property
    def line(self) -> ArraySpec[Any]:
        """Get line array."""
        return self.members["line"]

    @property
    def pixel(self) -> ArraySpec[Any]:
        """Get pixel array."""
        return self.members["pixel"]

    @property
    def sigma_nought(self) -> ArraySpec[Any]:
        """Get sigma_nought array."""
        return self.members["sigma_nought"]


class Sentinel1NoiseMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for noise group."""

    azimuth_time: ArraySpec[Any]
    noise_power_correction_factor: ArraySpec[Any]
    number_of_noise_lines: ArraySpec[Any]


class Sentinel1NoiseGroup(GroupSpec[DatasetAttrs, Sentinel1NoiseMembers]):  # type: ignore[type-var]
    """Noise group containing noise estimation data."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        return self.members["azimuth_time"]

    @property
    def noise_power_correction_factor(self) -> ArraySpec[Any]:
        """Get noise_power_correction_factor array."""
        return self.members["noise_power_correction_factor"]

    @property
    def number_of_noise_lines(self) -> ArraySpec[Any]:
        """Get number_of_noise_lines array."""
        return self.members["number_of_noise_lines"]


class Sentinel1NoiseAzimuthMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for noise_azimuth group."""

    first_azimuth_time: ArraySpec[Any]
    first_range_sample: ArraySpec[Any]
    last_azimuth_time: ArraySpec[Any]
    last_range_sample: ArraySpec[Any]
    line: ArraySpec[Any]
    noise_azimuth_lut: ArraySpec[Any]
    swath: ArraySpec[Any]


class Sentinel1NoiseAzimuthGroup(GroupSpec[DatasetAttrs, Sentinel1NoiseAzimuthMembers]):  # type: ignore[type-var]
    """Noise azimuth group containing azimuth noise vectors."""

    @property
    def first_azimuth_time(self) -> ArraySpec[Any]:
        """Get first_azimuth_time array."""
        return self.members["first_azimuth_time"]

    @property
    def first_range_sample(self) -> ArraySpec[Any]:
        """Get first_range_sample array."""
        return self.members["first_range_sample"]

    @property
    def last_azimuth_time(self) -> ArraySpec[Any]:
        """Get last_azimuth_time array."""
        return self.members["last_azimuth_time"]

    @property
    def last_range_sample(self) -> ArraySpec[Any]:
        """Get last_range_sample array."""
        return self.members["last_range_sample"]

    @property
    def line(self) -> ArraySpec[Any]:
        """Get line array."""
        return self.members["line"]

    @property
    def noise_azimuth_lut(self) -> ArraySpec[Any]:
        """Get noise_azimuth_lut array."""
        return self.members["noise_azimuth_lut"]

    @property
    def swath(self) -> ArraySpec[Any]:
        """Get swath array."""
        return self.members["swath"]


class Sentinel1NoiseRangeMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for noise_range group."""

    azimuth_time: ArraySpec[Any]
    ground_range: ArraySpec[Any]
    line: ArraySpec[Any]
    noise_range_lut: ArraySpec[Any]
    pixel: ArraySpec[Any]


class Sentinel1NoiseRangeGroup(GroupSpec[DatasetAttrs, Sentinel1NoiseRangeMembers]):  # type: ignore[type-var]
    """Noise range group containing range noise vectors."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        return self.members["azimuth_time"]

    @property
    def ground_range(self) -> ArraySpec[Any]:
        """Get ground_range array."""
        return self.members["ground_range"]

    @property
    def line(self) -> ArraySpec[Any]:
        """Get line array."""
        return self.members["line"]

    @property
    def noise_range_lut(self) -> ArraySpec[Any]:
        """Get noise_range_lut array."""
        return self.members["noise_range_lut"]

    @property
    def pixel(self) -> ArraySpec[Any]:
        """Get pixel array."""
        return self.members["pixel"]


class Sentinel1QualityMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for quality group.

    Closed TypedDict with optional fields to support different product variants:
    - S1A: calibration, noise, noise_azimuth, noise_range
    - S1C: calibration, noise (no noise_azimuth or noise_range)
    """

    calibration: Sentinel1CalibrationGroup
    noise: Sentinel1NoiseGroup
    noise_azimuth: Sentinel1NoiseAzimuthGroup
    noise_range: Sentinel1NoiseRangeGroup


class Sentinel1QualityGroup(GroupSpec[DatasetAttrs, Sentinel1QualityMembers]):  # type: ignore[type-var]
    """Quality group containing quality assurance and calibration data.

    Supports both S1A (with noise_azimuth, noise_range) and S1C (without them) products.
    """

    def get_calibration(self) -> Sentinel1CalibrationGroup | None:
        """Get calibration subgroup."""
        return self.members.get("calibration")

    def get_noise(self) -> Sentinel1NoiseGroup | None:
        """Get noise subgroup."""
        return self.members.get("noise")

    def get_noise_azimuth(self) -> Sentinel1NoiseAzimuthGroup | None:
        """Get noise azimuth subgroup (S1A only)."""
        return self.members.get("noise_azimuth")

    def get_noise_range(self) -> Sentinel1NoiseRangeGroup | None:
        """Get noise range subgroup (S1A only)."""
        return self.members.get("noise_range")


# Measurements
class Sentinel1MeasurementsMembers(TypedDict, closed=True, total=False):  # type: ignore[call-arg]
    """Members for measurements group."""

    azimuth_time: ArraySpec[Any]
    grd: ArraySpec[Any]
    ground_range: ArraySpec[Any]
    line: ArraySpec[Any]
    pixel: ArraySpec[Any]


class Sentinel1MeasurementsGroup(GroupSpec[DatasetAttrs, Sentinel1MeasurementsMembers]):  # type: ignore[type-var]
    """Measurements group containing SAR imagery data."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        return self.members["azimuth_time"]

    @property
    def grd(self) -> ArraySpec[Any]:
        """Get grd array."""
        return self.members["grd"]

    @property
    def ground_range(self) -> ArraySpec[Any]:
        """Get ground_range array."""
        return self.members["ground_range"]

    @property
    def line(self) -> ArraySpec[Any]:
        """Get line array."""
        return self.members["line"]

    @property
    def pixel(self) -> ArraySpec[Any]:
        """Get pixel array."""
        return self.members["pixel"]


# Polarization group
class Sentinel1PolarizationMembers(TypedDict, closed=True):  # type: ignore[call-arg]
    """Members for polarization group.

    Closed TypedDict - only conditions, measurements, quality keys are allowed.
    """

    conditions: Sentinel1ConditionsGroup
    measurements: Sentinel1MeasurementsGroup
    quality: Sentinel1QualityGroup


class Sentinel1PolarizationGroup(GroupSpec[DatasetAttrs, Sentinel1PolarizationMembers]):  # type: ignore[type-var]
    """Polarization-specific group containing all data for one polarization."""

    @property
    def conditions(self) -> Sentinel1ConditionsGroup | None:
        """Get the conditions group."""
        return self.members["conditions"]

    @property
    def measurements(self) -> Sentinel1MeasurementsGroup | None:
        """Get the measurements group."""
        return self.members["measurements"]

    @property
    def quality(self) -> Sentinel1QualityGroup | None:
        """Get the quality group."""
        return self.members["quality"]


# Root model - uses any members since polarizations can have variable names (VH_xxx, VV_xxx)
class Sentinel1Root(
    GroupSpec[Sentinel1RootAttrs, dict[str, Sentinel1PolarizationGroup]]
):
    """Complete Sentinel-1 EOPF Zarr hierarchy.

    The hierarchy follows EOPF organization with separate groups for each
    polarization (VH and VV):

    Root
    ├── S01SIWGRD_[timestamp]_..._VH/ (VH Polarization)
    │   ├── conditions/
    │   ├── measurements/ (GRD imagery)
    │   └── quality/
    └── S01SIWGRD_[timestamp]_..._VV/ (VV Polarization)
        ├── conditions/
        ├── measurements/
        └── quality/
    """

    def get_polarization_groups(self) -> dict[str, Sentinel1PolarizationGroup]:
        """Get all polarization groups (VH, VV, etc.)."""
        pol_groups = {}
        for name, member in self.members.items():
            if isinstance(member, Sentinel1PolarizationGroup):
                pol_groups[name] = member
        return pol_groups

    def get_vh_group(self) -> Sentinel1PolarizationGroup | None:
        """Get the VH polarization group."""
        for name, member in self.members.items():
            if "VH" in name:
                if isinstance(member, Sentinel1PolarizationGroup):
                    return member
        return None

    def get_vv_group(self) -> Sentinel1PolarizationGroup | None:
        """Get the VV polarization group."""
        for name, member in self.members.items():
            if "VV" in name:
                if isinstance(member, Sentinel1PolarizationGroup):
                    return member
        return None
