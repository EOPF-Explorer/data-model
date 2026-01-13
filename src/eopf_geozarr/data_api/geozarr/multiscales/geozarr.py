from __future__ import annotations

from typing import NotRequired, Self

from pydantic import BaseModel, model_validator
from pydantic.experimental.missing_sentinel import MISSING
from typing_extensions import TypedDict

from eopf_geozarr.data_api.geozarr.common import ZarrConventionMetadata  # noqa: TC001

from . import tms, zcm


class MultiscaleMeta(BaseModel):
    """
    Attributes for Multiscale GeoZarr dataset. Can be a mix of TMS multiscale
    or ZCM multiscale metadata
    """

    layout: tuple[zcm.ScaleLevel, ...] | MISSING = MISSING
    resampling_method: str | MISSING = MISSING
    tile_matrix_set: tms.TileMatrixSet | MISSING = MISSING
    tile_matrix_limits: dict[str, tms.TileMatrixLimit] | MISSING = MISSING

    @model_validator(mode="after")
    def valid_zcm(self) -> Self:
        """
        Ensure that the ZCM metadata, if present, is valid
        """
        if self.layout is not MISSING:
            zcm.Multiscales(layout=self.layout, resampling_method=self.resampling_method)

        return self

    @model_validator(mode="after")
    def valid_tms(self) -> Self:
        """
        Ensure that the TMS metadata, if present, is valid
        """
        if self.tile_matrix_set is not MISSING:
            tms.Multiscales(
                tile_matrix_set=self.tile_matrix_set,
                tile_matrix_limits=self.tile_matrix_limits,
                resampling_method=self.resampling_method,  # type: ignore[arg-type]
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
        """
        Ensure that the ZCM metadata, if present, is valid, and that TMS metadata, if present,
        is valid, and that at least one of the two is present.
        """
        if self.zarr_conventions is not MISSING:
            self._zcm_multiscales = zcm.Multiscales(
                layout=self.multiscales.layout,
                resampling_method=self.multiscales.resampling_method,
            )
        if self.multiscales.tile_matrix_limits is not MISSING:
            self._tms_multiscales = tms.Multiscales(
                tile_matrix_limits=self.multiscales.tile_matrix_limits,
                resampling_method=self.multiscales.resampling_method,  # type: ignore[arg-type]
                tile_matrix_set=self.multiscales.tile_matrix_set,
            )
        if self._tms_multiscales is None and self._zcm_multiscales is None:
            raise ValueError("Either ZCM multiscales or TMS multiscales must be present")
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
