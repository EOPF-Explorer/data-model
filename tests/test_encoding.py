import numpy as np
import xarray as xr

from eopf_geozarr.conversion.encoding import create_geozarr_encoding


def _make_ds(shape_2d=(1024, 2048), extra_dims=("band",), bands=3, dtype="uint16"):
    h, w = shape_2d
    coords = {"y": np.arange(h), "x": np.arange(w)}
    data_vars = {}
    if extra_dims:
        coords[extra_dims[0]] = np.arange(bands)
        data = np.random.randint(0, 1000, size=(bands, h, w), dtype=dtype)
        data_vars["data"] = (extra_dims + ("y", "x"), data)
    else:
        data = np.random.randint(0, 1000, size=(h, w), dtype=dtype)
        data_vars["data"] = (("y", "x"), data)
    # Add a grid mapping var and reference it via a data variable's attrs
    data_vars["spatial_ref"] = ((), np.array(0, dtype="int8"))
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds["data"].attrs["grid_mapping"] = "spatial_ref"
    return ds


def test_grid_mapping_excluded_from_compression():
    ds = _make_ds()
    enc = create_geozarr_encoding(ds, compressor="COMP", spatial_chunk=4096)
    assert (
        enc["spatial_ref"].get("compressors") is None
    ), "grid mapping var should not be compressed"
    assert enc["data"]["compressors"], "data variable should have compressors list"


def test_chunk_shape_3d_leading_dims_collapsed():
    ds = _make_ds()
    enc = create_geozarr_encoding(ds, compressor=None, spatial_chunk=512)
    chunks = enc["data"]["chunks"]
    # Expect (1, y_chunk, x_chunk)
    assert len(chunks) == 3 and chunks[0] == 1, f"Leading dim should be 1-sized in chunks: {chunks}"
    assert chunks[1] == chunks[2], "Spatial chunks should be square"
    assert chunks[1] <= 512, "Spatial chunk should not exceed requested size"


def test_chunk_shape_2d():
    ds = _make_ds(extra_dims=None)
    enc = create_geozarr_encoding(ds, compressor=None, spatial_chunk=256)
    chunks = enc["data"]["chunks"]
    assert len(chunks) == 2, "2D var should have 2 chunk dims"
    assert chunks[0] == chunks[1] <= 256


def test_max_chunk_bytes_enforced(monkeypatch):
    ds = _make_ds()
    # Force very low limit so logic must reduce
    monkeypatch.setenv("EOPF_MAX_CHUNK_BYTES", "16384")  # 16 KiB
    enc = create_geozarr_encoding(ds, compressor=None, spatial_chunk=2048)
    chunks = enc["data"]["chunks"]
    # Compute approximate bytes
    dtype_size = ds["data"].dtype.itemsize
    try:
        from math import prod
    except Exception:

        def prod(vals):
            out = 1
            for v in vals:
                out *= v
            return out

    est_bytes = prod(chunks) * dtype_size
    assert est_bytes <= 16384, f"Chunk bytes {est_bytes} exceed limit with chunks {chunks}"


def test_coord_variables_no_compression():
    ds = _make_ds()
    enc = create_geozarr_encoding(ds, compressor=None, spatial_chunk=512)
    for coord in ds.coords:
        assert (
            enc[coord].get("compressors") is None
        ), f"Coordinate {coord} should not have compressors"
