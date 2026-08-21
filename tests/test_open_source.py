"""Tests for open_source_datatree: native-chunk-aligned reads and the
per-source cache (EOPF-Explorer/data-pipeline#339)."""

import os
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numcodecs
import numpy as np
import pytest
import zarr

from eopf_geozarr.conversion.open_source import open_source_datatree


def _build_source_store(store_path: str, seed: int = 42) -> str:
    """Write a zarr v2 store mimicking a cpm_v270 source group.

    - ``aot``: ONE whole-array chunk (the pathological v270 layout)
    - ``b02``: regular 32x32 tiles

    ``seed`` varies the values so two stores can share a key layout but differ
    in content.
    """
    group = zarr.open_group(store_path, mode="w", zarr_format=2)
    rng = np.random.default_rng(seed)

    aot = group.create_array(
        "aot",
        shape=(128, 128),
        chunks=(128, 128),
        dtype="f8",
        compressors=numcodecs.Blosc(cname="lz4", clevel=5, shuffle=1),
    )
    aot[:] = rng.random((128, 128))
    aot.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]

    b02 = group.create_array(
        "b02",
        shape=(128, 128),
        chunks=(32, 32),
        dtype="f8",
        compressors=numcodecs.Blosc(cname="lz4", clevel=5, shuffle=1),
    )
    b02[:] = rng.random((128, 128))
    b02.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]

    zarr.consolidate_metadata(group.store)
    return store_path


@pytest.fixture
def source_store(tmp_path) -> str:
    return _build_source_store(str(tmp_path / "source.zarr"))


def test_single_chunk_array_is_one_dask_task(source_store) -> None:
    dt = open_source_datatree(source_store)
    assert dt["aot"].data.chunks == ((128,), (128,))


def test_tiled_array_chunks_match_native_grid(source_store) -> None:
    dt = open_source_datatree(source_store)
    assert dt["aot"].data.chunks == ((128,), (128,))
    assert dt["b02"].data.chunks == ((32,) * 4, (32,) * 4)


def test_values_roundtrip(source_store) -> None:
    dt = open_source_datatree(source_store)
    aot_expected = np.asarray(zarr.open_array(source_store, mode="r", path="aot")[:])
    np.testing.assert_array_equal(dt["aot"].values, aot_expected)


def test_explicit_storage_options_are_forwarded_to_xarray(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_open_datatree(path: object, **kwargs: object) -> None:
        captured.update(kwargs)
        raise InterruptedError("stop before any I/O")

    monkeypatch.setattr("xarray.open_datatree", fake_open_datatree)
    sentinel = {"endpoint_url": "https://example.invalid"}
    with pytest.raises(InterruptedError):
        open_source_datatree("/some/store.zarr", storage_options=sentinel)
    assert captured["storage_options"] == sentinel
    assert captured["chunks"] == {}


def test_mask_and_scale_is_forwarded_to_xarray(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_open_datatree(path: object, **kwargs: object) -> None:
        captured.update(kwargs)
        raise InterruptedError("stop before any I/O")

    monkeypatch.setattr("xarray.open_datatree", fake_open_datatree)
    with pytest.raises(InterruptedError):
        open_source_datatree("/some/store.zarr", storage_options={})
    assert captured["mask_and_scale"] is True

    with pytest.raises(InterruptedError):
        open_source_datatree("/some/store.zarr", storage_options={}, mask_and_scale=False)
    assert captured["mask_and_scale"] is False


@pytest.fixture
def http_source(source_store) -> Iterator[str]:
    """Serve the source store over local HTTP (mimics the EODC https source)."""
    root = os.path.dirname(source_store)
    name = os.path.basename(source_store)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=root, **kwargs)  # type: ignore[arg-type]

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/{name}"
    server.shutdown()


def test_cache_dir_reads_correct_data_over_http(http_source, source_store, tmp_path) -> None:
    """cache_dir reads return correct data and populate the cache, with no
    leftover temp files. Atomicity of the cache writes themselves is zarr
    LocalStore's contract (temp file + rename)."""
    cache_dir = str(tmp_path / "source-cache")
    dt = open_source_datatree(http_source, cache_dir=cache_dir)

    aot_expected = np.asarray(zarr.open_array(source_store, mode="r", path="aot")[:])
    np.testing.assert_array_equal(dt["aot"].values, aot_expected)
    assert dt["aot"].data.chunks == ((128,), (128,))  # native alignment preserved

    cached_files = [f for _, _, fs in os.walk(cache_dir) for f in fs]
    assert cached_files, "cache directory should be populated after reads"
    assert not [f for f in cached_files if f.endswith(".partial")]


def test_two_sources_can_share_one_cache_dir(tmp_path) -> None:
    """A shared cache_dir must never let one source serve another's bytes.

    CacheStore keys entries by zarr key ("aot/0.0" is identical for every S2
    product), so open_source_datatree namespaces the cache per source URL.
    """
    first = _build_source_store(str(tmp_path / "first.zarr"), seed=1)
    second = _build_source_store(str(tmp_path / "second.zarr"), seed=2)
    shared_cache = str(tmp_path / "shared-cache")

    dt_first = open_source_datatree(first, cache_dir=shared_cache)
    np.testing.assert_array_equal(
        dt_first["aot"].values,
        np.asarray(zarr.open_array(first, mode="r", path="aot")[:]),
    )

    dt_second = open_source_datatree(second, cache_dir=shared_cache)
    np.testing.assert_array_equal(
        dt_second["aot"].values,
        np.asarray(zarr.open_array(second, mode="r", path="aot")[:]),
    )
