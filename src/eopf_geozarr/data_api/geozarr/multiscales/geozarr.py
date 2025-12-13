from __future__ import annotations

from typing import NotRequired, Self

from pydantic import BaseModel, model_validator
from pydantic.experimental.missing_sentinel import MISSING
from typing_extensions import TypedDict

from eopf_geozarr.data_api.geozarr.common import ZarrConventionMetadata  # noqa: TC001
from eopf_geozarr.data_api.geozarr.types import ResamplingMethod  # noqa: TC001

from . import tms, zcm


class MultiscaleMeta(BaseModel):
    """
    Attributes for Multiscale GeoZarr dataset. Can be a mix of TMS multiscale
    or ZCM multiscale metadata
    """

    layout: tuple[zcm.ScaleLevel, ...] | MISSING = MISSING
    resampling_method: ResamplingMethod | MISSING = MISSING
    tile_matrix_set: tms.TileMatrixSet | MISSING = MISSING
    tile_matrix_limits: dict[str, tms.TileMatrixLimit] | MISSING = MISSING

    @model_validator(mode="after")
    def valid_zcm(self) -> Self:
        """Validate zcm multiscales when present."""
        if self.layout is not MISSING:
            zcm.Multiscales(**self.model_dump())

        return self

    @model_validator(mode="after")
    def valid_tms(self) -> Self:
        """Validate tms multiscales when present."""
        tms_set = self.tile_matrix_set
        rm = self.resampling_method
        if tms_set is not MISSING and rm is not MISSING:
            tile_matrix_limits = (
                None if self.tile_matrix_limits is MISSING else self.tile_matrix_limits
            )
            tms.Multiscales(
                tile_matrix_set=tms_set,
                resampling_method=rm,
                tile_matrix_limits=tile_matrix_limits,
            )
        return self


class MultiscaleGroupAttrs(BaseModel):
    """
    Attributes for Multiscale GeoZarr dataset.

    A Multiscale dataset is a Zarr group containing multiscale metadata
    That metadata can be either in the Zarr Convention Metadata (ZCM) format, or
    the Tile Matrix Set (TMS) format, or both.

    Attributes
    ----------
    multiscales: MultiscaleAttrs
    """

    zarr_conventions: tuple[ZarrConventionMetadata, ...] | MISSING = MISSING
    multiscales: MultiscaleMeta

    _zcm_multiscales: zcm.Multiscales | None = None
    _tms_multiscales: tms.Multiscales | None = None

    @model_validator(mode="after")
    def valid_zcm_and_tms(self) -> Self:
        """Validate at least one of zcm/tms multiscales."""
        ms = self.multiscales

        if self.zarr_conventions is not MISSING and ms.layout is not MISSING:
            self._zcm_multiscales = zcm.Multiscales(
                layout=ms.layout,
                resampling_method=ms.resampling_method,
            )

        tms_set = ms.tile_matrix_set
        rm = ms.resampling_method
        if tms_set is not MISSING and rm is not MISSING:
            tile_matrix_limits = None if ms.tile_matrix_limits is MISSING else ms.tile_matrix_limits
            self._tms_multiscales = tms.Multiscales(
                tile_matrix_set=tms_set,
                resampling_method=rm,
                tile_matrix_limits=tile_matrix_limits,
            )

        if self._tms_multiscales is None and self._zcm_multiscales is None:
            raise ValueError(
                "Missing multiscales metadata: expected either zcm (zarr_conventions + multiscales.layout) "
                "or tms (multiscales.tile_matrix_set + multiscales.resampling_method)."
            )
        return self

    @property
    def multiscale_meta(self) -> MultiscaleMetaDict:
        out: MultiscaleMetaDict = {}
        if self._tms_multiscales is not None:
            out["tms"] = self._tms_multiscales
        if self._zcm_multiscales is not None:
            out["zcm"] = self._zcm_multiscales
        return out


class MultiscaleMetaDict(TypedDict):
    tms: NotRequired[tms.Multiscales]
    zcm: NotRequired[zcm.Multiscales]
