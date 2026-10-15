"""
Pydantic-zarr integrated models for Sentinel-1A EOPF Zarr data structure.

Uses the new pyz.GroupSpec with TypedDict members to enforce strict structure validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from pydantic.experimental.missing_sentinel import MISSING as MISSING

from eopf_geozarr.data_api.geozarr.common import BaseDataArrayAttrs, DatasetAttrs
from eopf_geozarr.pyz.v2 import ArraySpec, GroupSpec

# Member type for groups with any nested structures (groups or arrays)
# Used for groups with dynamic or variable nested structures
AnyMembers = Mapping[str, GroupSpec[Any, Any] | ArraySpec[Any]]


class Sentinel1DataArrayAttrs(BaseDataArrayAttrs):
    """Extended attributes for Sentinel-1 data arrays."""

    long_name: str
    standard_name: str | None = None
    units: str = "1"


class Sentinel1RootAttrs(BaseModel):
    """Root-level attributes for Sentinel-1 DataTree."""

    other_metadata: dict[str, object]
    stac_discovery: dict[str, object]


class Sentinel1DataArray(ArraySpec[Sentinel1DataArrayAttrs]):
    """Sentinel-1 data array integrated with pydantic-zarr."""


# Conditions groups
class Sentinel1AntennaPatternMembers(TypedDict, closed=True, total=False):
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


class Sentinel1AntennaPatternGroup(GroupSpec[DatasetAttrs, Sentinel1AntennaPatternMembers]):
    """Antenna pattern group containing antenna characteristics."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        value = self.members.get("azimuth_time")
        if value is None:
            raise KeyError("azimuth_time")
        return value

    @property
    def count(self) -> ArraySpec[Any]:
        """Get count array."""
        value = self.members.get("count")
        if value is None:
            raise KeyError("count")
        return value

    @property
    def elevation_angle(self) -> ArraySpec[Any]:
        """Get elevation_angle array."""
        value = self.members.get("elevation_angle")
        if value is None:
            raise KeyError("elevation_angle")
        return value

    @property
    def incidence_angle(self) -> ArraySpec[Any]:
        """Get incidence_angle array."""
        value = self.members.get("incidence_angle")
        if value is None:
            raise KeyError("incidence_angle")
        return value

    @property
    def roll(self) -> ArraySpec[Any]:
        """Get roll array."""
        value = self.members.get("roll")
        if value is None:
            raise KeyError("roll")
        return value

    @property
    def slant_range_time_ap(self) -> ArraySpec[Any]:
        """Get slant_range_time_ap array."""
        value = self.members.get("slant_range_time_ap")
        if value is None:
            raise KeyError("slant_range_time_ap")
        return value

    @property
    def swath(self) -> ArraySpec[Any]:
        """Get swath array."""
        value = self.members.get("swath")
        if value is None:
            raise KeyError("swath")
        return value

    @property
    def terrain_height(self) -> ArraySpec[Any]:
        """Get terrain_height array."""
        value = self.members.get("terrain_height")
        if value is None:
            raise KeyError("terrain_height")
        return value


class Sentinel1AttitudeMembers(TypedDict, closed=True, total=False):
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


class Sentinel1AttitudeGroup(GroupSpec[DatasetAttrs, Sentinel1AttitudeMembers]):
    """Attitude group containing spacecraft attitude data."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        value = self.members.get("azimuth_time")
        if value is None:
            raise KeyError("azimuth_time")
        return value

    @property
    def pitch(self) -> ArraySpec[Any]:
        """Get pitch array."""
        value = self.members.get("pitch")
        if value is None:
            raise KeyError("pitch")
        return value

    @property
    def q0(self) -> ArraySpec[Any]:
        """Get q0 array."""
        value = self.members.get("q0")
        if value is None:
            raise KeyError("q0")
        return value

    @property
    def q1(self) -> ArraySpec[Any]:
        """Get q1 array."""
        value = self.members.get("q1")
        if value is None:
            raise KeyError("q1")
        return value

    @property
    def q2(self) -> ArraySpec[Any]:
        """Get q2 array."""
        value = self.members.get("q2")
        if value is None:
            raise KeyError("q2")
        return value

    @property
    def q3(self) -> ArraySpec[Any]:
        """Get q3 array."""
        value = self.members.get("q3")
        if value is None:
            raise KeyError("q3")
        return value

    @property
    def roll(self) -> ArraySpec[Any]:
        """Get roll array."""
        value = self.members.get("roll")
        if value is None:
            raise KeyError("roll")
        return value

    @property
    def wx(self) -> ArraySpec[Any]:
        """Get wx array."""
        value = self.members.get("wx")
        if value is None:
            raise KeyError("wx")
        return value

    @property
    def wy(self) -> ArraySpec[Any]:
        """Get wy array."""
        value = self.members.get("wy")
        if value is None:
            raise KeyError("wy")
        return value

    @property
    def wz(self) -> ArraySpec[Any]:
        """Get wz array."""
        value = self.members.get("wz")
        if value is None:
            raise KeyError("wz")
        return value

    @property
    def yaw(self) -> ArraySpec[Any]:
        """Get yaw array."""
        value = self.members.get("yaw")
        if value is None:
            raise KeyError("yaw")
        return value


class Sentinel1AzimuthFmRateMembers(TypedDict, closed=True, total=False):
    """Members for azimuth_fm_rate group."""

    azimuth_fm_rate_polynomial: ArraySpec[Any]
    azimuth_time: ArraySpec[Any]
    t0: ArraySpec[Any]


class Sentinel1AzimuthFmRateGroup(GroupSpec[DatasetAttrs, Sentinel1AzimuthFmRateMembers]):
    """Azimuth FM rate group."""

    @property
    def azimuth_fm_rate_polynomial(self) -> ArraySpec[Any]:
        """Get azimuth_fm_rate_polynomial array."""
        value = self.members.get("azimuth_fm_rate_polynomial")
        if value is None:
            raise KeyError("azimuth_fm_rate_polynomial")
        return value

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        value = self.members.get("azimuth_time")
        if value is None:
            raise KeyError("azimuth_time")
        return value

    @property
    def t0(self) -> ArraySpec[Any]:
        """Get t0 array."""
        value = self.members.get("t0")
        if value is None:
            raise KeyError("t0")
        return value


class Sentinel1CoordinateConversionMembers(TypedDict, closed=True, total=False):
    """Members for coordinate_conversion group."""

    azimuth_time: ArraySpec[Any]
    gr0: ArraySpec[Any]
    grsr_coefficients: ArraySpec[Any]
    slant_range_time: ArraySpec[Any]
    sr0: ArraySpec[Any]
    srgr_coefficients: ArraySpec[Any]


class Sentinel1CoordinateConversionGroup(
    GroupSpec[DatasetAttrs, Sentinel1CoordinateConversionMembers]
):
    """Coordinate conversion group."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        value = self.members.get("azimuth_time")
        if value is None:
            raise KeyError("azimuth_time")
        return value

    @property
    def gr0(self) -> ArraySpec[Any]:
        """Get gr0 array."""
        value = self.members.get("gr0")
        if value is None:
            raise KeyError("gr0")
        return value

    @property
    def grsr_coefficients(self) -> ArraySpec[Any]:
        """Get grsr_coefficients array."""
        value = self.members.get("grsr_coefficients")
        if value is None:
            raise KeyError("grsr_coefficients")
        return value

    @property
    def slant_range_time(self) -> ArraySpec[Any]:
        """Get slant_range_time array."""
        value = self.members.get("slant_range_time")
        if value is None:
            raise KeyError("slant_range_time")
        return value

    @property
    def sr0(self) -> ArraySpec[Any]:
        """Get sr0 array."""
        value = self.members.get("sr0")
        if value is None:
            raise KeyError("sr0")
        return value

    @property
    def srgr_coefficients(self) -> ArraySpec[Any]:
        """Get srgr_coefficients array."""
        value = self.members.get("srgr_coefficients")
        if value is None:
            raise KeyError("srgr_coefficients")
        return value


class Sentinel1DopplerCentroidMembers(TypedDict, closed=True, total=False):
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


class Sentinel1DopplerCentroidGroup(GroupSpec[DatasetAttrs, Sentinel1DopplerCentroidMembers]):
    """Doppler centroid group."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        value = self.members.get("azimuth_time")
        if value is None:
            raise KeyError("azimuth_time")
        return value

    @property
    def data_dc_polynomial(self) -> ArraySpec[Any]:
        """Get data_dc_polynomial array."""
        value = self.members.get("data_dc_polynomial")
        if value is None:
            raise KeyError("data_dc_polynomial")
        return value

    @property
    def data_dc_rms_error(self) -> ArraySpec[Any]:
        """Get data_dc_rms_error array."""
        value = self.members.get("data_dc_rms_error")
        if value is None:
            raise KeyError("data_dc_rms_error")
        return value

    @property
    def data_dc_rms_error_above_threshold(self) -> ArraySpec[Any]:
        """Get data_dc_rms_error_above_threshold array."""
        value = self.members.get("data_dc_rms_error_above_threshold")
        if value is None:
            raise KeyError("data_dc_rms_error_above_threshold")
        return value

    @property
    def degree(self) -> ArraySpec[Any]:
        """Get degree array."""
        value = self.members.get("degree")
        if value is None:
            raise KeyError("degree")
        return value

    @property
    def fine_dce_azimuth_start_time(self) -> ArraySpec[Any]:
        """Get fine_dce_azimuth_start_time array."""
        value = self.members.get("fine_dce_azimuth_start_time")
        if value is None:
            raise KeyError("fine_dce_azimuth_start_time")
        return value

    @property
    def fine_dce_azimuth_stop_time(self) -> ArraySpec[Any]:
        """Get fine_dce_azimuth_stop_time array."""
        value = self.members.get("fine_dce_azimuth_stop_time")
        if value is None:
            raise KeyError("fine_dce_azimuth_stop_time")
        return value

    @property
    def geometry_dc_polynomial(self) -> ArraySpec[Any]:
        """Get geometry_dc_polynomial array."""
        value = self.members.get("geometry_dc_polynomial")
        if value is None:
            raise KeyError("geometry_dc_polynomial")
        return value

    @property
    def t0(self) -> ArraySpec[Any]:
        """Get t0 array."""
        value = self.members.get("t0")
        if value is None:
            raise KeyError("t0")
        return value


class Sentinel1GcpMembers(TypedDict, closed=True, total=False):
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


class Sentinel1GcpGroup(GroupSpec[DatasetAttrs, Sentinel1GcpMembers]):
    """Ground Control Points (GCP) group."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        value = self.members.get("azimuth_time")
        if value is None:
            raise KeyError("azimuth_time")
        return value

    @property
    def azimuth_time_gcp(self) -> ArraySpec[Any]:
        """Get azimuth_time_gcp array."""
        value = self.members.get("azimuth_time_gcp")
        if value is None:
            raise KeyError("azimuth_time_gcp")
        return value

    @property
    def elevation_angle(self) -> ArraySpec[Any]:
        """Get elevation_angle array."""
        value = self.members.get("elevation_angle")
        if value is None:
            raise KeyError("elevation_angle")
        return value

    @property
    def ground_range(self) -> ArraySpec[Any]:
        """Get ground_range array."""
        value = self.members.get("ground_range")
        if value is None:
            raise KeyError("ground_range")
        return value

    @property
    def height(self) -> ArraySpec[Any]:
        """Get height array."""
        value = self.members.get("height")
        if value is None:
            raise KeyError("height")
        return value

    @property
    def incidence_angle(self) -> ArraySpec[Any]:
        """Get incidence_angle array."""
        value = self.members.get("incidence_angle")
        if value is None:
            raise KeyError("incidence_angle")
        return value

    @property
    def latitude(self) -> ArraySpec[Any]:
        """Get latitude array."""
        value = self.members.get("latitude")
        if value is None:
            raise KeyError("latitude")
        return value

    @property
    def line(self) -> ArraySpec[Any]:
        """Get line array."""
        value = self.members.get("line")
        if value is None:
            raise KeyError("line")
        return value

    @property
    def longitude(self) -> ArraySpec[Any]:
        """Get longitude array."""
        value = self.members.get("longitude")
        if value is None:
            raise KeyError("longitude")
        return value

    @property
    def pixel(self) -> ArraySpec[Any]:
        """Get pixel array."""
        value = self.members.get("pixel")
        if value is None:
            raise KeyError("pixel")
        return value

    @property
    def slant_range_time_gcp(self) -> ArraySpec[Any]:
        """Get slant_range_time_gcp array."""
        value = self.members.get("slant_range_time_gcp")
        if value is None:
            raise KeyError("slant_range_time_gcp")
        return value


class Sentinel1OrbitMembers(TypedDict, closed=True, total=False):
    """Members for orbit group."""

    axis: ArraySpec[Any]
    azimuth_time: ArraySpec[Any]
    position: ArraySpec[Any]
    velocity: ArraySpec[Any]


class Sentinel1OrbitGroup(GroupSpec[DatasetAttrs, Sentinel1OrbitMembers]):
    """Orbit group containing spacecraft position and velocity."""

    @property
    def axis(self) -> ArraySpec[Any]:
        """Get axis array."""
        value = self.members.get("axis")
        if value is None:
            raise KeyError("axis")
        return value

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        value = self.members.get("azimuth_time")
        if value is None:
            raise KeyError("azimuth_time")
        return value

    @property
    def position(self) -> ArraySpec[Any]:
        """Get position array."""
        value = self.members.get("position")
        if value is None:
            raise KeyError("position")
        return value

    @property
    def velocity(self) -> ArraySpec[Any]:
        """Get velocity array."""
        value = self.members.get("velocity")
        if value is None:
            raise KeyError("velocity")
        return value


class Sentinel1ReferenceReplicaMembers(TypedDict, closed=True, total=False):
    """Members for reference_replica group.

    Closed TypedDict - only reference replica coefficient array keys are allowed.
    All fields are optional since not all reference replica data may be present.
    """

    azimuth_time: ArraySpec[Any]
    reference_replica_amplitude_coefficients: ArraySpec[Any]
    reference_replica_phase_coefficients: ArraySpec[Any]


class Sentinel1ReferenceReplicaGroup(GroupSpec[DatasetAttrs, Sentinel1ReferenceReplicaMembers]):
    """Reference replica group."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        value = self.members.get("azimuth_time")
        if value is None:
            raise KeyError("azimuth_time")
        return value

    @property
    def reference_replica_amplitude_coefficients(self) -> ArraySpec[Any]:
        """Get reference_replica_amplitude_coefficients array."""
        value = self.members.get("reference_replica_amplitude_coefficients")
        if value is None:
            raise KeyError("reference_replica_amplitude_coefficients")
        return value

    @property
    def reference_replica_phase_coefficients(self) -> ArraySpec[Any]:
        """Get reference_replica_phase_coefficients array."""
        value = self.members.get("reference_replica_phase_coefficients")
        if value is None:
            raise KeyError("reference_replica_phase_coefficients")
        return value


class Sentinel1ReplicaMembers(TypedDict, closed=True, total=False):
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


class Sentinel1ReplicaGroup(GroupSpec[DatasetAttrs, Sentinel1ReplicaMembers]):
    """Replica group containing pulse replica data."""

    @property
    def absolute_pg_product_valid_flag(self) -> ArraySpec[Any]:
        """Get absolute_pg_product_valid_flag array."""
        value = self.members.get("absolute_pg_product_valid_flag")
        if value is None:
            raise KeyError("absolute_pg_product_valid_flag")
        return value

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        value = self.members.get("azimuth_time")
        if value is None:
            raise KeyError("azimuth_time")
        return value

    @property
    def cross_correlation_peak_location(self) -> ArraySpec[Any]:
        """Get cross_correlation_peak_location array."""
        value = self.members.get("cross_correlation_peak_location")
        if value is None:
            raise KeyError("cross_correlation_peak_location")
        return value

    @property
    def cross_correlation_pslr(self) -> ArraySpec[Any]:
        """Get cross_correlation_pslr array."""
        value = self.members.get("cross_correlation_pslr")
        if value is None:
            raise KeyError("cross_correlation_pslr")
        return value

    @property
    def internal_time_delay(self) -> ArraySpec[Any]:
        """Get internal_time_delay array."""
        value = self.members.get("internal_time_delay")
        if value is None:
            raise KeyError("internal_time_delay")
        return value

    @property
    def model_pg_product_amplitude(self) -> ArraySpec[Any]:
        """Get model_pg_product_amplitude array."""
        value = self.members.get("model_pg_product_amplitude")
        if value is None:
            raise KeyError("model_pg_product_amplitude")
        return value

    @property
    def model_pg_product_phase(self) -> ArraySpec[Any]:
        """Get model_pg_product_phase array."""
        value = self.members.get("model_pg_product_phase")
        if value is None:
            raise KeyError("model_pg_product_phase")
        return value

    @property
    def pg_product_amplitude(self) -> ArraySpec[Any]:
        """Get pg_product_amplitude array."""
        value = self.members.get("pg_product_amplitude")
        if value is None:
            raise KeyError("pg_product_amplitude")
        return value

    @property
    def pg_product_phase(self) -> ArraySpec[Any]:
        """Get pg_product_phase array."""
        value = self.members.get("pg_product_phase")
        if value is None:
            raise KeyError("pg_product_phase")
        return value

    @property
    def reconstructed_replica_valid_flag(self) -> ArraySpec[Any]:
        """Get reconstructed_replica_valid_flag array."""
        value = self.members.get("reconstructed_replica_valid_flag")
        if value is None:
            raise KeyError("reconstructed_replica_valid_flag")
        return value

    @property
    def relative_pg_product_valid_flag(self) -> ArraySpec[Any]:
        """Get relative_pg_product_valid_flag array."""
        value = self.members.get("relative_pg_product_valid_flag")
        if value is None:
            raise KeyError("relative_pg_product_valid_flag")
        return value


class Sentinel1TerrainHeightMembers(TypedDict, closed=True, total=False):
    """Members for terrain_height group."""

    azimuth_time: ArraySpec[Any]
    terrain_height: ArraySpec[Any]


class Sentinel1TerrainHeightGroup(GroupSpec[DatasetAttrs, Sentinel1TerrainHeightMembers]):
    """Terrain height group."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        value = self.members.get("azimuth_time")
        if value is None:
            raise KeyError("azimuth_time")
        return value

    @property
    def terrain_height(self) -> ArraySpec[Any]:
        """Get terrain_height array."""
        value = self.members.get("terrain_height")
        if value is None:
            raise KeyError("terrain_height")
        return value


class Sentinel1ConditionsMembers(TypedDict, closed=True):
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


class Sentinel1ConditionsGroup(GroupSpec[DatasetAttrs, Sentinel1ConditionsMembers]):
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
class Sentinel1CalibrationMembers(TypedDict, closed=True, total=False):
    """Members for calibration group."""

    azimuth_time: ArraySpec[Any]
    beta_nought: ArraySpec[Any]
    dn: ArraySpec[Any]
    gamma: ArraySpec[Any]
    ground_range: ArraySpec[Any]
    line: ArraySpec[Any]
    pixel: ArraySpec[Any]
    sigma_nought: ArraySpec[Any]


class Sentinel1CalibrationGroup(GroupSpec[DatasetAttrs, Sentinel1CalibrationMembers]):
    """Calibration group containing radiometric calibration data."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        value = self.members.get("azimuth_time")
        if value is None:
            raise KeyError("azimuth_time")
        return value

    @property
    def beta_nought(self) -> ArraySpec[Any]:
        """Get beta_nought array."""
        value = self.members.get("beta_nought")
        if value is None:
            raise KeyError("beta_nought")
        return value

    @property
    def dn(self) -> ArraySpec[Any]:
        """Get dn array."""
        value = self.members.get("dn")
        if value is None:
            raise KeyError("dn")
        return value

    @property
    def gamma(self) -> ArraySpec[Any]:
        """Get gamma array."""
        value = self.members.get("gamma")
        if value is None:
            raise KeyError("gamma")
        return value

    @property
    def ground_range(self) -> ArraySpec[Any]:
        """Get ground_range array."""
        value = self.members.get("ground_range")
        if value is None:
            raise KeyError("ground_range")
        return value

    @property
    def line(self) -> ArraySpec[Any]:
        """Get line array."""
        value = self.members.get("line")
        if value is None:
            raise KeyError("line")
        return value

    @property
    def pixel(self) -> ArraySpec[Any]:
        """Get pixel array."""
        value = self.members.get("pixel")
        if value is None:
            raise KeyError("pixel")
        return value

    @property
    def sigma_nought(self) -> ArraySpec[Any]:
        """Get sigma_nought array."""
        value = self.members.get("sigma_nought")
        if value is None:
            raise KeyError("sigma_nought")
        return value


class Sentinel1NoiseMembers(TypedDict, closed=True, total=False):
    """Members for noise group."""

    azimuth_time: ArraySpec[Any]
    noise_power_correction_factor: ArraySpec[Any]
    number_of_noise_lines: ArraySpec[Any]


class Sentinel1NoiseGroup(GroupSpec[DatasetAttrs, Sentinel1NoiseMembers]):
    """Noise group containing noise estimation data."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        value = self.members.get("azimuth_time")
        if value is None:
            raise KeyError("azimuth_time")
        return value

    @property
    def noise_power_correction_factor(self) -> ArraySpec[Any]:
        """Get noise_power_correction_factor array."""
        value = self.members.get("noise_power_correction_factor")
        if value is None:
            raise KeyError("noise_power_correction_factor")
        return value

    @property
    def number_of_noise_lines(self) -> ArraySpec[Any]:
        """Get number_of_noise_lines array."""
        value = self.members.get("number_of_noise_lines")
        if value is None:
            raise KeyError("number_of_noise_lines")
        return value


class Sentinel1NoiseAzimuthMembers(TypedDict, closed=True, total=False):
    """Members for noise_azimuth group."""

    first_azimuth_time: ArraySpec[Any]
    first_range_sample: ArraySpec[Any]
    last_azimuth_time: ArraySpec[Any]
    last_range_sample: ArraySpec[Any]
    line: ArraySpec[Any]
    noise_azimuth_lut: ArraySpec[Any]
    swath: ArraySpec[Any]


class Sentinel1NoiseAzimuthGroup(GroupSpec[DatasetAttrs, Sentinel1NoiseAzimuthMembers]):
    """Noise azimuth group containing azimuth noise vectors."""

    @property
    def first_azimuth_time(self) -> ArraySpec[Any]:
        """Get first_azimuth_time array."""
        value = self.members.get("first_azimuth_time")
        if value is None:
            raise KeyError("first_azimuth_time")
        return value

    @property
    def first_range_sample(self) -> ArraySpec[Any]:
        """Get first_range_sample array."""
        value = self.members.get("first_range_sample")
        if value is None:
            raise KeyError("first_range_sample")
        return value

    @property
    def last_azimuth_time(self) -> ArraySpec[Any]:
        """Get last_azimuth_time array."""
        value = self.members.get("last_azimuth_time")
        if value is None:
            raise KeyError("last_azimuth_time")
        return value

    @property
    def last_range_sample(self) -> ArraySpec[Any]:
        """Get last_range_sample array."""
        value = self.members.get("last_range_sample")
        if value is None:
            raise KeyError("last_range_sample")
        return value

    @property
    def line(self) -> ArraySpec[Any]:
        """Get line array."""
        value = self.members.get("line")
        if value is None:
            raise KeyError("line")
        return value

    @property
    def noise_azimuth_lut(self) -> ArraySpec[Any]:
        """Get noise_azimuth_lut array."""
        value = self.members.get("noise_azimuth_lut")
        if value is None:
            raise KeyError("noise_azimuth_lut")
        return value

    @property
    def swath(self) -> ArraySpec[Any]:
        """Get swath array."""
        value = self.members.get("swath")
        if value is None:
            raise KeyError("swath")
        return value


class Sentinel1NoiseRangeMembers(TypedDict, closed=True, total=False):
    """Members for noise_range group."""

    azimuth_time: ArraySpec[Any]
    ground_range: ArraySpec[Any]
    line: ArraySpec[Any]
    noise_range_lut: ArraySpec[Any]
    pixel: ArraySpec[Any]


class Sentinel1NoiseRangeGroup(GroupSpec[DatasetAttrs, Sentinel1NoiseRangeMembers]):
    """Noise range group containing range noise vectors."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        value = self.members.get("azimuth_time")
        if value is None:
            raise KeyError("azimuth_time")
        return value

    @property
    def ground_range(self) -> ArraySpec[Any]:
        """Get ground_range array."""
        value = self.members.get("ground_range")
        if value is None:
            raise KeyError("ground_range")
        return value

    @property
    def line(self) -> ArraySpec[Any]:
        """Get line array."""
        value = self.members.get("line")
        if value is None:
            raise KeyError("line")
        return value

    @property
    def noise_range_lut(self) -> ArraySpec[Any]:
        """Get noise_range_lut array."""
        value = self.members.get("noise_range_lut")
        if value is None:
            raise KeyError("noise_range_lut")
        return value

    @property
    def pixel(self) -> ArraySpec[Any]:
        """Get pixel array."""
        value = self.members.get("pixel")
        if value is None:
            raise KeyError("pixel")
        return value


class Sentinel1QualityMembers(TypedDict, closed=True, total=False):
    """Members for quality group.

    Closed TypedDict with optional fields to support different product variants:
    - S1A: calibration, noise, noise_azimuth, noise_range
    - S1C: calibration, noise (no noise_azimuth or noise_range)
    """

    calibration: Sentinel1CalibrationGroup
    noise: Sentinel1NoiseGroup
    noise_azimuth: Sentinel1NoiseAzimuthGroup
    noise_range: Sentinel1NoiseRangeGroup


class Sentinel1QualityGroup(GroupSpec[DatasetAttrs, Sentinel1QualityMembers]):
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
class Sentinel1MeasurementsMembers(TypedDict, closed=True, total=False):
    """Members for measurements group."""

    azimuth_time: ArraySpec[Any]
    grd: ArraySpec[Any]
    ground_range: ArraySpec[Any]
    line: ArraySpec[Any]
    pixel: ArraySpec[Any]


class Sentinel1MeasurementsGroup(GroupSpec[DatasetAttrs, Sentinel1MeasurementsMembers]):
    """Measurements group containing SAR imagery data."""

    @property
    def azimuth_time(self) -> ArraySpec[Any]:
        """Get azimuth_time array."""
        value = self.members.get("azimuth_time")
        if value is None:
            raise KeyError("azimuth_time")
        return value

    @property
    def grd(self) -> ArraySpec[Any]:
        """Get grd array."""
        value = self.members.get("grd")
        if value is None:
            raise KeyError("grd")
        return value

    @property
    def ground_range(self) -> ArraySpec[Any]:
        """Get ground_range array."""
        value = self.members.get("ground_range")
        if value is None:
            raise KeyError("ground_range")
        return value

    @property
    def line(self) -> ArraySpec[Any]:
        """Get line array."""
        value = self.members.get("line")
        if value is None:
            raise KeyError("line")
        return value

    @property
    def pixel(self) -> ArraySpec[Any]:
        """Get pixel array."""
        value = self.members.get("pixel")
        if value is None:
            raise KeyError("pixel")
        return value


# Polarization group
class Sentinel1PolarizationMembers(TypedDict, closed=True):
    """Members for polarization group.

    Closed TypedDict - only conditions, measurements, quality keys are allowed.
    """

    conditions: Sentinel1ConditionsGroup
    measurements: Sentinel1MeasurementsGroup
    quality: Sentinel1QualityGroup


class Sentinel1PolarizationGroup(GroupSpec[DatasetAttrs, Sentinel1PolarizationMembers]):
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
class Sentinel1Root(GroupSpec[Sentinel1RootAttrs, dict[str, Sentinel1PolarizationGroup]]):
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
        return {
            name: member
            for name, member in self.members.items()
            if isinstance(member, Sentinel1PolarizationGroup)
        }

    def get_vh_group(self) -> Sentinel1PolarizationGroup | None:
        """Get the VH polarization group."""
        for name, member in self.members.items():
            if "VH" in name and isinstance(member, Sentinel1PolarizationGroup):
                return member
        return None

    def get_vv_group(self) -> Sentinel1PolarizationGroup | None:
        """Get the VV polarization group."""
        for name, member in self.members.items():
            if "VV" in name and isinstance(member, Sentinel1PolarizationGroup):
                return member
        return None
