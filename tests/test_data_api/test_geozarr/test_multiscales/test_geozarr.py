import json
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic.experimental.missing_sentinel import MISSING
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import GroupSpec

from eopf_geozarr.data_api.geozarr.multiscales import tms, zcm
from eopf_geozarr.data_api.geozarr.multiscales.geozarr import (
    MultiscaleGroupAttrs,
    MultiscaleMeta,
)

OPTIMIZED_GEOZARR_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "optimized_geozarr_examples"
OPTIMIZED_GEOZARR_EXAMPLES = sorted(OPTIMIZED_GEOZARR_EXAMPLES_DIR.glob("*.json"))


@pytest.mark.parametrize("multiscale_flavor", [{"zcm"}, {"tms"}, {"zcm", "tms"}], ids=str)
def test_multiscale_group_attrs(multiscale_flavor: set[Literal["zcm", "tms"]]) -> None:
    """Creates `MultiscaleGroupAttrs` from ZCM/TMS/both metadata."""
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


@pytest.mark.parametrize(
    "example_json",
    OPTIMIZED_GEOZARR_EXAMPLES,
    ids=lambda p: p.stem,
)
def test_optimized_geozarr_reflectance_multiscales_tms_contract(example_json: Path) -> None:
    """Checks that reflectance pyramid group ids match TMS metadata."""
    group_json = tuplify_json(json.loads(example_json.read_text(encoding="utf-8")))
    root = GroupSpec(**group_json)
    flat = root.to_flat()

    reflectance = flat["/measurements/reflectance"]
    attrs = reflectance.attributes
    multiscale_attrs = MultiscaleGroupAttrs(**attrs)

    tms_multiscales = multiscale_attrs.multiscale_meta["tms"]
    tile_matrix_ids = [tm.id for tm in tms_multiscales.tile_matrix_set.tileMatrices]
    limit_keys = set((tms_multiscales.tile_matrix_limits or {}).keys())

    assert set(tile_matrix_ids) == limit_keys

    for tile_matrix_id in tile_matrix_ids:
        assert f"/measurements/reflectance/{tile_matrix_id}" in flat
        assert tms_multiscales.tile_matrix_limits is not None
        assert tms_multiscales.tile_matrix_limits[tile_matrix_id].tileMatrix == tile_matrix_id


def test_missing_required_tms_fields_is_not_processed_when_zcm_present() -> None:
    """Partial TMS must not be treated as valid when ZCM is valid."""
    attrs = {
        "zarr_conventions": (zcm.MultiscaleConventionMetadata(),),
        "multiscales": {
            "layout": (
                {
                    "asset": "r10m",
                    "derived_from": "r10m",
                    "transform": {"scale": (1.0, 1.0), "translation": (0.0, 0.0)},
                },
            ),
            "resampling_method": "nearest",
            "tile_matrix_limits": {
                "r10m": {
                    "tileMatrix": "r10m",
                    "minTileCol": 0,
                    "minTileRow": 0,
                    "maxTileCol": 0,
                    "maxTileRow": 0,
                }
            },
        },
    }

    multiscale_group_attrs = MultiscaleGroupAttrs(**attrs)
    assert "zcm" in multiscale_group_attrs.multiscale_meta
    assert "tms" not in multiscale_group_attrs.multiscale_meta
