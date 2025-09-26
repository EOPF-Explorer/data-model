"""Tests for metrics helpers in CLI."""

import importlib
import json
from typing import Any

cli_module = importlib.import_module("eopf_geozarr.cli")
_write_metrics = cli_module._write_metrics  # type: ignore[attr-defined]


def test_write_metrics_local_path(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    payload = {"hello": "world", "value": 42}

    _write_metrics(str(metrics_path), payload)

    assert metrics_path.exists()  # noqa: S101 - pytest assertion
    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert loaded == payload  # noqa: S101 - pytest assertion


def test_write_metrics_s3_path(monkeypatch):
    captured: dict[str, Any] = {"path": None, "content": None}

    class DummyStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def write(self, data):
            captured["content"] = data

    def fake_open(path, mode, encoding=None):
        captured["path"] = path
        return DummyStream()

    import fsspec

    monkeypatch.setattr(fsspec, "open", fake_open)

    payload = {"foo": "bar"}
    _write_metrics("s3://bucket/key.json", payload)

    assert captured["path"] == "s3://bucket/key.json"  # noqa: S101 - pytest assertion
    assert json.loads(captured["content"]) == payload  # noqa: S101 - pytest assertion
