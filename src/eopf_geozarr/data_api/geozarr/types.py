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
} | CF_SCALE_OFFSET_KEYS


class XarrayDataArrayEncoding(TypedDict):
    """
    The dict form of the encoding for xarray.DataArray
    """

    chunks: NotRequired[tuple[int, ...]]
    preferred_chunks: NotRequired[dict[str, int]]
    compressors: NotRequired[tuple[object, ...] | None]
    filters: NotRequired[tuple[object, ...]]
    shards: NotRequired[tuple[int, ...] | None]
    _FillValue: NotRequired[object]
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
