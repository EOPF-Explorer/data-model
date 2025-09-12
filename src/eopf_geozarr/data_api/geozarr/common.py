"""Common utilities for GeoZarr data API."""

import io
import urllib
import urllib.request
from typing import Annotated, Final, Literal

from cf_xarray.utils import parse_cf_standard_name_table
from pydantic import AfterValidator, BaseModel
from pydantic.experimental.missing_sentinel import MISSING

XARRAY_DIMS_KEY: Final = "_ARRAY_DIMENSIONS"


def get_cf_standard_names(url: str) -> tuple[str, ...]:
    """Retrieve the set of CF standard names and return them as a tuple."""

    headers = {"User-Agent": "eopf_geozarr"}

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response:
            content = response.read()  # Read the entire response body into memory
            content_fobj = io.BytesIO(content)
    except urllib.error.URLError as e:
        raise e

    _info, table, _aliases = parse_cf_standard_name_table(source=content_fobj)
    return tuple(table.keys())


# This is a URL to the CF standard names table.
CF_STANDARD_NAME_URL = (
    "https://raw.githubusercontent.com/cf-convention/cf-convention.github.io/"
    "master/Data/cf-standard-names/current/src/cf-standard-name-table.xml"
)

# this does IO against github. consider locally storing this data instead if fetching every time
# is problematic.
CF_STANDARD_NAMES = get_cf_standard_names(url=CF_STANDARD_NAME_URL)


def check_standard_name(name: str) -> str:
    """
    Check if the standard name is valid according to the CF conventions.

    Parameters
    ----------
    name : str
        The standard name to check.

    Returns
    -------
    str
        The validated standard name.

    Raises
    ------
    ValueError
        If the standard name is not valid.
    """

    if name in CF_STANDARD_NAMES:
        return name
    raise ValueError(
        f"Invalid standard name: {name}. This name was not found in the list of CF standard names."
    )


CFStandardName = Annotated[str, AfterValidator(check_standard_name)]

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


class GridMappingAttrs(BaseModel, extra="allow"):
    """
    Grid mapping attributes for a GeoZarr grid mapping variable.

    Attributes
    ----------
    grid_mapping_name : str
        The name of the grid mapping.

    Extra fields are permitted.

    Additional attributes might be present depending on the type of grid mapping.

    References
    ----------
    https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#grid-mappings-and-projections
    """

    grid_mapping_name: str


class TileMatrixLimit(BaseModel):
    """"""

    tileMatrix: str
    minTileCol: int
    minTileRow: int
    maxTileCol: int
    maxTileRow: int


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


class Multiscales(BaseModel, extra="allow"):
    """
    Multiscale metadata for a GeoZarr dataset.

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


class DatasetAttrs(BaseModel, extra="allow"):
    """
    Attributes for a GeoZarr dataset.

    A dataset is a collection of DataArrays.
    """

    ...


class MultiscaleAttrs(BaseModel, extra="allow"):
    """
    Attributes for Multiscale GeoZarr dataset.

    A Multiscale dataset is a collection of Dataet

    Attributes
    ----------
    multiscales: MultiscaleAttrs
    """

    multiscales: Multiscales


class BaseDataArrayAttrs(BaseModel, extra="allow"):
    """
    Base attributes for a  GeoZarr DataArray.

    Attributes
    ----------
    """

    grid_mapping: str | MISSING = MISSING
