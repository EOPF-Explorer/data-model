from typing import Any, Literal

import pytest
from pydantic.experimental.missing_sentinel import MISSING

from eopf_geozarr.data_api.geozarr.multiscales import tms, zcm
from eopf_geozarr.data_api.geozarr.multiscales.geozarr import (
    MultiscaleGroupAttrs,
    MultiscaleMeta,
)


@pytest.mark.parametrize("multiscale_flavor", [{"zcm"}, {"tms"}, {"zcm", "tms"}], ids=str)
def test_multiscale_group_attrs(multiscale_flavor: set[Literal["zcm", "tms"]]) -> None:
    """
    Test that we can create a MultiscaleGroupAttrs with both ZCM and TMS metadata
    """
    zcm_meta: dict[str, object] = {}
    tms_meta: dict[str, object] = {}
    zarr_conventions_meta: MISSING | tuple[Any, ...] = MISSING
    if "zcm" in multiscale_flavor:
        layout = (
            zcm.ScaleLevel(
                asset="level_0",
                transform={"scale": (1.0, 1.0), "translation": (0.0, 0.0)},
            ),
        )
        zcm_meta = zcm.Multiscales(layout=layout, resampling_method="nearest").model_dump()
        zarr_conventions_meta = (zcm.MULTISCALE_CONVENTION_METADATA,)
    if "tms" in multiscale_flavor:
        tile_matrix_set = tms.TileMatrixSet(
            id="example_tms",
            tileMatrices=(
                tms.TileMatrix(
                    id="0",
                    scaleDenominator=559082264.0287178,
                    tileWidth=256,
                    tileHeight=256,
                    matrixWidth=1,
                    matrixHeight=1,
                    cellSize=156543.03392804097,
                    pointOfOrigin=(20037508.342789244, -20037508.342789244),
                ),
            ),
        )
        tms_meta = tms.Multiscales(
            resampling_method="nearest",
            tile_matrix_set=tile_matrix_set,
            tile_matrix_limits={
                "0": tms.TileMatrixLimit(
                    tileMatrix="0",
                    minTileRow=0,
                    maxTileRow=0,
                    minTileCol=0,
                    maxTileCol=0,
                )
            },
        ).model_dump()
    multiscale_meta = MultiscaleMeta(**{**zcm_meta, **tms_meta})
    multiscale_group_attrs = MultiscaleGroupAttrs(
        zarr_conventions=zarr_conventions_meta, multiscales=multiscale_meta
    )
    if "zcm" in multiscale_flavor:
        assert "zcm" in multiscale_group_attrs.multiscale_meta
        assert multiscale_group_attrs.multiscale_meta["zcm"] == zcm.Multiscales(**zcm_meta)
    if "tms" in multiscale_flavor:
        assert "tms" in multiscale_group_attrs.multiscale_meta
        assert multiscale_group_attrs.multiscale_meta["tms"] == tms.Multiscales(**tms_meta)
