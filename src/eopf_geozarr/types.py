"""Types and constants for the GeoZarr data API."""

from typing import Any, Final, Literal, NewType, NotRequired, TypedDict

CRSCode = NewType("CRSCode", str)
"""A CRS identifier accepted by pyproj, e.g. ``"EPSG:32631"`` (the GeoZarr ``proj:code`` value)."""

BoundingBox2D = NewType("BoundingBox2D", tuple[float, float, float, float])
"""A ``(xmin, ymin, xmax, ymax)`` bounding box, tagged so it cannot be confused with
other float sequences (e.g. an affine transform)."""


def make_crs_code(value: object) -> CRSCode:
    """Validate an untyped value (e.g. a zarr attribute) as a CRS code string.

    Validation is intentionally light (non-empty string): pyproj's
    ``CRS.from_user_input`` raises ``CRSError`` downstream on values it cannot parse.
    """
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"CRS code must be a non-empty string, got {value!r}")
    return CRSCode(value)


EPSGCode = NewType("EPSGCode", int)
"""A bare EPSG integer code, e.g. ``32631`` (the legacy ``proj:epsg`` attribute value)."""


def make_epsg_code(value: object) -> EPSGCode:
    """Validate an untyped value (e.g. a zarr attribute) as an EPSG integer code.

    Accepts an int (``32631``) or the string forms found in stored attrs and CPM
    metadata (``"32631"``, ``"EPSG:32631"``). Validation is intentionally light:
    pyproj's ``CRS.from_epsg`` raises ``CRSError`` downstream on codes it does
    not recognize.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return EPSGCode(value)
    if isinstance(value, str):
        tail = value.strip().split(":")[-1]
        if tail.isdigit():
            return EPSGCode(int(tail))
    raise TypeError(f"EPSG code must be an int or 'EPSG:<int>' string, got {value!r}")


def make_bounding_box(value: object) -> BoundingBox2D:
    """Validate an untyped value (e.g. a zarr attribute) as a 4-number bounding box.

    Accepts a list or tuple (zarr attrs serialize tuples to JSON arrays and read back
    lists) and returns a float 4-tuple.
    """
    if not isinstance(value, list | tuple):
        raise TypeError(f"Bounding box must be a sequence, got {type(value).__name__}")
    if len(value) != 4:
        raise ValueError(f"Bounding box must have exactly 4 values, got {len(value)}")
    coords: list[float] = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, int | float):
            raise TypeError(f"Bounding box values must be numbers, got {v!r}")
        coords.append(float(v))
    return BoundingBox2D((coords[0], coords[1], coords[2], coords[3]))


class XarrayEncodingJSON(TypedDict):
    fill_value: NotRequired[object]
    chunks: NotRequired[tuple[int, ...]]
    compressors: Any
    shards: NotRequired[Any]


class StandardXCoordAttrsJSON(TypedDict):
    units: Literal["m"]
    long_name: Literal["x coordinate of projection"]
    standard_name: Literal["projection_x_coordinate"]
    _ARRAY_DIMENSIONS: list[Literal["x"]]


class StandardYCoordAttrsJSON(TypedDict):
    units: Literal["m"]
    long_name: Literal["y coordinate of projection"]
    standard_name: Literal["projection_y_coordinate"]
    _ARRAY_DIMENSIONS: list[Literal["y"]]


class StandardLonCoordAttrsJSON(TypedDict):
    units: Literal["degrees_east"]
    long_name: Literal["longitude"]
    standard_name: Literal["longitude"]
    _ARRAY_DIMENSIONS: list[Literal["x"]]


class StandardLatCoordAttrsJSON(TypedDict):
    units: Literal["degrees_north"]
    long_name: Literal["latitude"]
    standard_name: Literal["latitude"]
    _ARRAY_DIMENSIONS: list[Literal["y"]]


class OverviewLevelJSON(TypedDict):
    level: int | str
    width: int
    height: int
    translation_relative: float
    scale_relative: int | float
    zoom: NotRequired[int]
    scale_absolute: NotRequired[int | float]
    spatial_transform: NotRequired[tuple[float, ...] | None]
    spatial_shape: NotRequired[tuple[int, ...]]
    chunks: NotRequired[tuple[tuple[int, ...], ...] | None]


ResamplingMethod = Literal[
    "nearest",
    "average",
    "bilinear",
    "cubic",
    "cubic_spline",
    "lanczos",
    "mode",
    "max",
    "min",
    "med",
    "sum",
    "q1",
    "q3",
    "rms",
    "gauss",
]
"""A string literal indicating a resampling method"""
XARRAY_DIMS_KEY: Final = "_ARRAY_DIMENSIONS"


# Why is endpoint URL specified twice?
class S3ClientOptions(TypedDict):
    """
    S3 client options
    """

    region_name: NotRequired[str]
    endpoint_url: NotRequired[str]


class S3FsOptions(TypedDict):
    """
    S3FS options
    """

    anon: NotRequired[bool]
    use_ssl: NotRequired[bool]
    client_kwargs: NotRequired[S3ClientOptions]
    endpoint_url: NotRequired[str]
    asynchronous: NotRequired[bool]


class S3Credentials(TypedDict):
    """
    S3 credentials
    """

    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    aws_session_token: str | None
    aws_default_region: str
    aws_profile: str | None
    AWS_ENDPOINT_URL: str | None
