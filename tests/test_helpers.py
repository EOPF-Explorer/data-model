import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.conversion.helpers import (
    build_var_encodings,
    inject_crs_if_requested,
    maybe_step,
    normalize_group_path,
    safely,
)


class DummyRecorder:
    def __init__(self):
        self.steps = []

    class _Ctx:
        def __init__(self, outer, name):
            self.outer = outer
            self.name = name

        def __enter__(self):
            self.outer.steps.append((self.name, "start"))
            return self

        def __exit__(self, exc_type, exc, tb):
            self.outer.steps.append((self.name, "end"))
            return False

    def time_step(self, name):  # mimic recorder interface
        return self._Ctx(self, name)


def test_normalize_group_path():
    assert normalize_group_path("") == "/"
    assert normalize_group_path(".") == "/"
    assert normalize_group_path("/a") == "/a"
    assert normalize_group_path("a") == "/a"


def test_build_var_encodings_excludes_grid_mapping():
    ds = xr.Dataset(
        {
            "B01": ("y", np.arange(5).astype("uint16")),
            "spatial_ref": ([], 0),
        },
        coords={"y": ("y", np.arange(5))},
    )
    # Mark spatial_ref as grid mapping variable
    ds["B01"].attrs["grid_mapping"] = "spatial_ref"
    ds["spatial_ref"].attrs["crs_wkt"] = "WKT"

    class DummyCompressor:
        codec_id = "dummy"
        configuration = {}

    comp = DummyCompressor()

    enc = build_var_encodings(ds, comp)
    assert "B01" in enc
    assert "spatial_ref" not in enc  # grid mapping excluded
    assert enc["B01"]["compressors"][0] is comp
    assert enc["B01"]["dtype"] == str(ds["B01"].dtype)


def test_build_var_encodings_none_compressor():
    ds = xr.Dataset({"B01": ("y", np.arange(3))})
    assert build_var_encodings(ds, None) == {}


def test_inject_crs_if_requested_applies_once():
    ds = xr.Dataset(
        {"B01": (("y", "x"), np.random.rand(4, 4))},
        coords={"y": ("y", np.arange(4)), "x": ("x", np.arange(4))},
    )

    def ref_crs():
        return "EPSG:4326"

    def preparer(ds_in, reference_crs):  # mimic prepare_dataset_with_crs_info minimal
        # attach spatial_ref variable and grid_mapping attrs
        if "spatial_ref" not in ds_in:
            ds_in["spatial_ref"] = xr.DataArray(
                0, attrs={"crs_wkt": reference_crs, "_ARRAY_DIMENSIONS": []}
            )
        ds_in["B01"].attrs["grid_mapping"] = "spatial_ref"
        return ds_in

    requested = {"/group"}
    # Not in requested
    out = inject_crs_if_requested("/other", ds.copy(), requested, ref_crs, preparer)
    assert "spatial_ref" not in out

    # In requested
    out2 = inject_crs_if_requested("group", ds.copy(), requested, ref_crs, preparer)
    assert "spatial_ref" in out2
    assert out2["spatial_ref"].attrs["crs_wkt"] == "EPSG:4326"
    assert out2["B01"].attrs["grid_mapping"] == "spatial_ref"


def test_safely_swallow_and_raise():
    calls = {"n": 0}

    def good():
        calls["n"] += 1
        return 42

    def bad():
        raise RuntimeError("boom")

    assert safely("good", good) == 42
    assert calls["n"] == 1
    # swallow True returns None
    assert safely("bad", bad, swallow=True) is None
    with pytest.raises(RuntimeError):
        safely("bad", bad, swallow=False)


def test_maybe_step_recorder():
    rec = DummyRecorder()
    with maybe_step(rec, "encode"):
        pass
    assert rec.steps == [("encode", "start"), ("encode", "end")]


def test_maybe_step_no_recorder():
    # Should act as no-op
    with maybe_step(None, "noop"):
        pass  # nothing to assert; just ensure no exception
