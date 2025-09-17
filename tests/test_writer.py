from pathlib import Path

import numpy as np
import xarray as xr

from eopf_geozarr.conversion.geozarr import GeoZarrWriter
from eopf_geozarr.conversion.helpers import build_var_encodings


def test_writer_build_var_encodings_excludes_grid_mapping(tmp_path: Path):
    ds = xr.Dataset(
        {
            "B01": (
                ("y", "x"),
                np.arange(16, dtype="uint16").reshape(4, 4),
                {"grid_mapping": "spatial_ref"},
            ),
            "spatial_ref": ([], 0, {"crs_wkt": "EPSG:4326"}),
        },
        coords={
            "y": ("y", np.arange(4)),
            "x": ("x", np.arange(4)),
        },
    )
    comp = None  # rely on helper returning {}
    enc = build_var_encodings(ds, comp)
    assert enc == {}

    # Now simulate with a dummy compressor object minimal interface
    class DummyCompressor:
        codec_id = "dummy"
        configuration = {}

    dcomp = DummyCompressor()
    enc2 = build_var_encodings(ds, dcomp)
    assert "B01" in enc2
    assert "spatial_ref" not in enc2


def test_write_group_basic(tmp_path: Path):
    ds = xr.Dataset(
        {
            "B01": (("y", "x"), np.ones((4, 4), dtype="uint16")),
            "spatial_ref": ([], 0, {"crs_wkt": "EPSG:4326"}),
        },
        coords={"y": ("y", np.arange(4)), "x": ("x", np.arange(4))},
    )
    # Add grid mapping attr
    ds["B01"].attrs["grid_mapping"] = "spatial_ref"

    writer = GeoZarrWriter(
        output_path=str(tmp_path / "out.zarr"),
        compressor=None,
        spatial_chunk=4096,
        min_dimension=2,
        tile_width=256,
        max_retries=1,
        overwrite="replace",
        max_overview_levels=1,
        skip_overviews=True,
        base_write_mode="single",
    )

    dt = xr.DataTree()
    dt2 = writer.write_group(dt, "/group", ds)
    assert "group" in dt2.children
    # Ensure spatial_ref not duplicated in data vars of on-disk store
    # (implicit check: no exception and dataset can be reopened)
    reopened = xr.open_dataset(str(tmp_path / "out.zarr/group/0"), engine="zarr", chunks="auto")
    assert "B01" in reopened.data_vars
    assert "spatial_ref" in reopened.variables
