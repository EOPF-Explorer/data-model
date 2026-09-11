"""Types and constants for the GeoZarr data API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping


class TileMatrixLimitJSON(TypedDict):
    tileMatrix: str
    minTileCol: int
    minTileRow: int
    maxTileCol: int
    maxTileRow: int


CF_SCALE_OFFSET_KEYS: Final[set[str]] = {"scale_factor", "add_offset", "dtype"}

XARRAY_ENCODING_KEYS: Final[set[str]] = {
    "chunks",
    "preferred_chunks",
    "compressors",
    "filters",
    "shards",
    "_FillValue",
    "fill_value",
} | CF_SCALE_OFFSET_KEYS


class XarrayDataArrayEncoding(TypedDict):
    """
    The dict form of the encoding for xarray.DataArray
    """

    chunks: NotRequired[tuple[int, ...]]
    preferred_chunks: NotRequired[tuple[int, ...]]
    compressors: NotRequired[tuple[object, ...] | None]
    filters: NotRequired[tuple[object, ...]]
    shards: NotRequired[tuple[int, ...] | None]
    _FillValue: NotRequired[object]
    fill_value: NotRequired[object]
    scale_factor: NotRequired[float]
    add_offset: NotRequired[float]
    dtype: NotRequired[object]


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


class S1BackscatterAttrsJSON(TypedDict):
    """CF attributes for the Sentinel-1 RTC backscatter bands (vv/vh).

    ``_FillValue`` is the ``FillValueCoder``-encoded base64 NaN that lets xarray mask
    nodata under ``use_zarr_fill_value_as_mask`` despite xarray #11345 (data-model #172).
    """

    standard_name: Literal["surface_backwards_scattering_coefficient_of_radar_wave"]
    units: Literal["1"]
    _FillValue: str


class S1TimeCoordAttrsJSON(TypedDict):
    """CF datetime-encoding attrs for the S1 RTC ``time`` coordinate.

    Lets xarray / TiTiler's ``open_datatree(decode_times=True)`` decode the int64-ns
    axis to datetime64, enabling per-acquisition selection by exact datetime
    (data-model #192).
    """

    units: Literal["nanoseconds since 1970-01-01"]
    calendar: Literal["proleptic_gregorian"]
    standard_name: Literal["time"]


class TileMatrixJSON(TypedDict):
    id: str
    scaleDenominator: float
    cellSize: float
    pointOfOrigin: tuple[float, float] | list[float]
    tileWidth: int
    tileHeight: int
    matrixWidth: int
    matrixHeight: int


class TileMatrixSetJSON(TypedDict):
    id: str
    title: str | None
    crs: str | None
    supportedCRS: str | None
    orderedAxes: tuple[str, str] | None
    tileMatrices: tuple[TileMatrixJSON, ...]


class TMSMultiscalesJSON(TypedDict):
    """
    Typeddict model of the `multiscales` attribute of Zarr groups that implement the
    OGC TileMatrixSet multiscales structure
    """

    tile_matrix_set: TileMatrixSetJSON
    resampling_method: ResamplingMethod
    tile_matrix_limits: Mapping[str, TileMatrixLimitJSON]


class TMSMultiscalesAttrsJSON(TypedDict):
    multiscales: TMSMultiscalesJSON


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
