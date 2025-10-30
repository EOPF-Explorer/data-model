"""
Pydantic models for PROJ JSON schema v0.7

Based on the schema at: https://proj.org/en/latest/schemas/v0.7/projjson.schema.json
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


class Id(BaseModel):
    authority: str
    code: str | int
    version: str | float | None = None
    authority_citation: str | None = None
    uri: str | None = None


class Unit(BaseModel):
    type: Literal[
        "Unit", "AngularUnit", "LinearUnit", "ScaleUnit", "ParametricUnit", "TimeUnit"
    ] = "Unit"
    name: str
    conversion_factor: float
    id: Id | None = None
    ids: list[Id] | None = None

    @model_validator(mode="after")
    def validate_id_mutually_exclusive(self) -> Unit:
        """Ensure that id and ids are mutually exclusive."""
        if self.id is not None and self.ids is not None:
            raise ValueError("Cannot specify both 'id' and 'ids' fields")
        return self


class Meridian(BaseModel):
    type: Literal["Meridian"] = "Meridian"
    longitude: float | ValueAndUnit
    id: Id | None = None
    ids: list[Id] | None = None


class ValueAndUnit(BaseModel):
    value: float
    unit: Unit


class BBox(BaseModel):
    east_longitude: float
    west_longitude: float
    south_latitude: float
    north_latitude: float


class VerticalExtent(BaseModel):
    minimum: float
    maximum: float
    unit: Unit


class TemporalExtent(BaseModel):
    start: str | float
    end: str | float


class Usage(BaseModel):
    scope: str
    area: str
    bbox: BBox
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None


class Axis(BaseModel):
    type: Literal["Axis"] = "Axis"
    name: str
    abbreviation: str
    direction: Literal[
        "north",
        "northNorthEast",
        "northEast",
        "eastNorthEast",
        "east",
        "eastSouthEast",
        "southEast",
        "southSouthEast",
        "south",
        "southSouthWest",
        "southWest",
        "westSouthWest",
        "west",
        "westNorthWest",
        "northWest",
        "northNorthWest",
        "up",
        "down",
        "geocentricX",
        "geocentricY",
        "geocentricZ",
        "columnPositive",
        "columnNegative",
        "rowPositive",
        "rowNegative",
        "displayRight",
        "displayLeft",
        "displayUp",
        "displayDown",
        "forward",
        "aft",
        "port",
        "starboard",
        "clockwise",
        "counterClockwise",
        "towards",
        "awayFrom",
        "future",
        "past",
        "unspecified",
    ]
    meridian: Meridian | None = None
    unit: Unit | str | None = None
    minimum_value: float | None = None
    maximum_value: float | None = None
    range_meaning: Literal["exact", "wraparound"] | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class CoordinateSystem(BaseModel):
    type: Literal["CoordinateSystem"] = "CoordinateSystem"
    name: str | None = None
    subtype: Literal[
        "Cartesian",
        "spherical",
        "ellipsoidal",
        "vertical",
        "ordinal",
        "parametric",
        "affine",
        "TemporalDateTime",
        "TemporalCount",
        "TemporalMeasure",
    ]
    axis: list[Axis]
    id: Id | None = None
    ids: list[Id] | None = None


class Ellipsoid(BaseModel):
    type: Literal["Ellipsoid"] = "Ellipsoid"
    name: str
    semi_major_axis: float | ValueAndUnit | None = None
    semi_minor_axis: float | ValueAndUnit | None = None
    inverse_flattening: float | None = None
    radius: float | ValueAndUnit | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class PrimeMeridian(BaseModel):
    type: Literal["PrimeMeridian"] = "PrimeMeridian"
    name: str
    longitude: float | ValueAndUnit
    id: Id | None = None
    ids: list[Id] | None = None


class GeodeticReferenceFrame(BaseModel):
    type: Literal["GeodeticReferenceFrame"] = "GeodeticReferenceFrame"
    name: str
    anchor: str | None = None
    anchor_epoch: float | None = None
    ellipsoid: Ellipsoid
    prime_meridian: PrimeMeridian | None = None
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class DynamicGeodeticReferenceFrame(BaseModel):
    type: Literal["DynamicGeodeticReferenceFrame"] = "DynamicGeodeticReferenceFrame"
    name: str
    anchor: str | None = None
    anchor_epoch: float | None = None
    ellipsoid: Ellipsoid
    prime_meridian: PrimeMeridian | None = None
    frame_reference_epoch: float
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class VerticalReferenceFrame(BaseModel):
    type: Literal["VerticalReferenceFrame"] = "VerticalReferenceFrame"
    name: str
    anchor: str | None = None
    anchor_epoch: float | None = None
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class DynamicVerticalReferenceFrame(BaseModel):
    type: Literal["DynamicVerticalReferenceFrame"] = "DynamicVerticalReferenceFrame"
    name: str
    anchor: str | None = None
    anchor_epoch: float | None = None
    frame_reference_epoch: float
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class TemporalDatum(BaseModel):
    type: Literal["TemporalDatum"] = "TemporalDatum"
    name: str
    anchor: str | None = None
    calendar: str | None = None
    origin: str
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class ParametricDatum(BaseModel):
    type: Literal["ParametricDatum"] = "ParametricDatum"
    name: str
    anchor: str | None = None
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class EngineeringDatum(BaseModel):
    type: Literal["EngineeringDatum"] = "EngineeringDatum"
    name: str
    anchor: str | None = None
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class DatumEnsembleMember(BaseModel):
    name: str
    id: Id | None = None
    ids: list[Id] | None = None


class DatumEnsemble(BaseModel):
    type: Literal["DatumEnsemble"] = "DatumEnsemble"
    name: str
    members: list[DatumEnsembleMember]
    ellipsoid: Ellipsoid | None = None
    accuracy: str
    id: Id | None = None
    ids: list[Id] | None = None


class DeformationModel(BaseModel):
    name: str
    id: Id | None = None


class Method(BaseModel):
    type: Literal["OperationMethod"] = "OperationMethod"
    name: str
    id: Id | None = None
    ids: list[Id] | None = None


class ParameterValue(BaseModel):
    type: Literal["ParameterValue"] = "ParameterValue"
    name: str
    value: str | float
    unit: Unit | str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class Conversion(BaseModel):
    type: Literal["Conversion"] = "Conversion"
    name: str
    method: Method
    parameters: list[ParameterValue] | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class GeoidModel(BaseModel):
    name: str
    interpolation_crs: CRS | None = None
    id: Id | None = None


Datum = (
    GeodeticReferenceFrame
    | DynamicGeodeticReferenceFrame
    | VerticalReferenceFrame
    | DynamicVerticalReferenceFrame
    | TemporalDatum
    | ParametricDatum
    | EngineeringDatum
)


class GeodeticCRS(BaseModel):
    type: Literal["GeodeticCRS", "GeographicCRS"]
    name: str
    datum: GeodeticReferenceFrame | DynamicGeodeticReferenceFrame | None = None
    datum_ensemble: DatumEnsemble | None = None
    coordinate_system: CoordinateSystem | None = None
    deformation_models: list[DeformationModel] | None = None
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class ProjectedCRS(BaseModel):
    type: Literal["ProjectedCRS"] = "ProjectedCRS"
    name: str
    base_crs: GeodeticCRS
    conversion: Conversion
    coordinate_system: CoordinateSystem | None = None
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class VerticalCRS(BaseModel):
    type: Literal["VerticalCRS"] = "VerticalCRS"
    name: str
    datum: VerticalReferenceFrame | DynamicVerticalReferenceFrame | None = None
    datum_ensemble: DatumEnsemble | None = None
    coordinate_system: CoordinateSystem | None = None
    geoid_model: GeoidModel | None = None
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class TemporalCRS(BaseModel):
    type: Literal["TemporalCRS"] = "TemporalCRS"
    name: str
    datum: TemporalDatum
    coordinate_system: CoordinateSystem | None = None
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class ParametricCRS(BaseModel):
    type: Literal["ParametricCRS"] = "ParametricCRS"
    name: str
    datum: ParametricDatum
    coordinate_system: CoordinateSystem | None = None
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class EngineeringCRS(BaseModel):
    type: Literal["EngineeringCRS"] = "EngineeringCRS"
    name: str
    datum: EngineeringDatum
    coordinate_system: CoordinateSystem | None = None
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class DerivedGeodeticCRS(BaseModel):
    type: Literal["DerivedGeodeticCRS", "DerivedGeographicCRS"]
    name: str
    base_crs: GeodeticCRS
    conversion: Conversion
    coordinate_system: CoordinateSystem
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class DerivedProjectedCRS(BaseModel):
    type: Literal["DerivedProjectedCRS"] = "DerivedProjectedCRS"
    name: str
    base_crs: ProjectedCRS
    conversion: Conversion
    coordinate_system: CoordinateSystem
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class DerivedVerticalCRS(BaseModel):
    type: Literal["DerivedVerticalCRS"] = "DerivedVerticalCRS"
    name: str
    base_crs: VerticalCRS
    conversion: Conversion
    coordinate_system: CoordinateSystem
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class DerivedTemporalCRS(BaseModel):
    type: Literal["DerivedTemporalCRS"] = "DerivedTemporalCRS"
    name: str
    base_crs: TemporalCRS
    conversion: Conversion
    coordinate_system: CoordinateSystem
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class DerivedParametricCRS(BaseModel):
    type: Literal["DerivedParametricCRS"] = "DerivedParametricCRS"
    name: str
    base_crs: ParametricCRS
    conversion: Conversion
    coordinate_system: CoordinateSystem
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class DerivedEngineeringCRS(BaseModel):
    type: Literal["DerivedEngineeringCRS"] = "DerivedEngineeringCRS"
    name: str
    base_crs: EngineeringCRS
    conversion: Conversion
    coordinate_system: CoordinateSystem
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class CompoundCRS(BaseModel):
    type: Literal["CompoundCRS"] = "CompoundCRS"
    name: str
    components: list[CRS]
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class AbridgedTransformation(BaseModel):
    type: Literal["AbridgedTransformation"] = "AbridgedTransformation"
    name: str
    source_crs: CRS | None = None
    method: Method
    parameters: list[ParameterValue]
    id: Id | None = None
    ids: list[Id] | None = None


class BoundCRS(BaseModel):
    type: Literal["BoundCRS"] = "BoundCRS"
    source_crs: CRS
    target_crs: CRS
    transformation: AbridgedTransformation
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


CRS = (
    BoundCRS
    | CompoundCRS
    | DerivedEngineeringCRS
    | DerivedGeodeticCRS
    | DerivedParametricCRS
    | DerivedProjectedCRS
    | DerivedTemporalCRS
    | DerivedVerticalCRS
    | EngineeringCRS
    | GeodeticCRS
    | ParametricCRS
    | ProjectedCRS
    | TemporalCRS
    | VerticalCRS
)


class SingleOperation(BaseModel):
    type: Literal["Transformation", "Conversion"]
    name: str
    source_crs: CRS | None = None
    target_crs: CRS | None = None
    method: Method
    parameters: list[ParameterValue] | None = None
    accuracy: str | None = None
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class ConcatenatedOperation(BaseModel):
    type: Literal["ConcatenatedOperation"] = "ConcatenatedOperation"
    name: str
    source_crs: CRS
    target_crs: CRS
    steps: list[SingleOperation]
    accuracy: str | None = None
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


class CoordinateMetadata(BaseModel):
    type: Literal["CoordinateMetadata"] = "CoordinateMetadata"
    crs: CRS
    coordinateEpoch: float | None = None


class PointMotionOperation(BaseModel):
    type: Literal["PointMotionOperation"] = "PointMotionOperation"
    name: str
    source_crs: CRS
    method: Method
    parameters: list[ParameterValue]
    accuracy: str | None = None
    scope: str | None = None
    area: str | None = None
    bbox: BBox | None = None
    vertical_extent: VerticalExtent | None = None
    temporal_extent: TemporalExtent | None = None
    usages: list[Usage] | None = None
    remarks: str | None = None
    id: Id | None = None
    ids: list[Id] | None = None


ProjJSON = (
    CRS
    | Datum
    | DatumEnsemble
    | Ellipsoid
    | PrimeMeridian
    | SingleOperation
    | ConcatenatedOperation
    | CoordinateMetadata
)


# Update forward references
GeoidModel.model_rebuild()
CompoundCRS.model_rebuild()
AbridgedTransformation.model_rebuild()
BoundCRS.model_rebuild()
