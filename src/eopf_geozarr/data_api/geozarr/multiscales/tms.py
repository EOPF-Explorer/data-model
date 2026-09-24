from __future__ import annotations

from pydantic import BaseModel

from eopf_geozarr.data_api.geozarr.types import ResamplingMethod


class TileMatrix(BaseModel):
    id: str
    scaleDenominator: float
    cellSize: float
    pointOfOrigin: tuple[float, float]
    tileWidth: int
    tileHeight: int
    matrixWidth: int
    matrixHeight: int


class TileMatrixSet(BaseModel):
    id: str
    title: str | None = None
    crs: str | None = None
    supportedCRS: str | None = None
    orderedAxes: tuple[str, str] | None = None
    tileMatrices: tuple[TileMatrix, ...]


class TileMatrixLimit(BaseModel):
    """"""

    tileMatrix: str
    minTileCol: int
    minTileRow: int
    maxTileCol: int
    maxTileRow: int


class Multiscales(BaseModel, extra="allow"):
    """
    Multiscale metadata for a GeoZarr dataset based on the OGC TileMatrixSet standard

    Attributes
    ----------
    tile_matrix_set : str
        The tile matrix set identifier for the multiscale dataset.
    resampling_method : ResamplingMethod
        The name of the resampling method for the multiscale dataset.
    tile_matrix_set_limits : dict[str, TileMatrixSetLimits] | None, optional
        The tile matrix set limits for the multiscale dataset.
    """

    tile_matrix_set: TileMatrixSet
    resampling_method: ResamplingMethod
    # TODO: ensure that the keys match tile_matrix_set.tileMatrices[$index].id
    # TODO: ensure that the keys match the tileMatrix attribute
    tile_matrix_limits: dict[str, TileMatrixLimit] | None = None
