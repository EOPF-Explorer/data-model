from __future__ import annotations

import json
import os
import platform
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterator, List, Optional

import xarray as xr


@dataclass
class MetricsRecorder:
    schema_version: str = "convert_metrics/v1"
    run_id: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
    attempt: int = 1
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    steps: List[Dict[str, Any]] = field(default_factory=list)
    input_info: Dict[str, Any] = field(default_factory=dict)
    output_info: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    ended_at: Optional[str] = None

    def set_input(
        self,
        *,
        source_uri: str,
        profile: Optional[str],
        groups: List[str],
        dask: Dict[str, Any],
    ) -> None:
        self.input_info = {
            "source_uri": source_uri,
            "profile": profile,
            "groups": groups,
            "dask": dask,
        }

    def set_environment(self) -> None:
        def _get_version(pkg: str) -> Optional[str]:
            try:
                import importlib.metadata as importlib_metadata

                return importlib_metadata.version(pkg)
            except Exception:
                return None

        self.environment = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                "eopf_geozarr": _get_version("eopf-geozarr")
                or _get_version("eopf_geozarr"),
                "xarray": _get_version("xarray"),
                "zarr": _get_version("zarr"),
                "dask": _get_version("dask"),
                "fsspec": _get_version("fsspec"),
            },
            "num_cpus": os.cpu_count(),
        }

    @contextmanager
    def time_step(self, name: str) -> Iterator[None]:
        t0 = perf_counter()
        try:
            yield
        finally:
            dt = perf_counter() - t0
            self.steps.append({"name": name, "duration_s": dt})

    def build_output_summary(self, output_path: str) -> None:
        info: Dict[str, Any] = {
            "store_uri": output_path,
            "store_total_bytes": None,
            "per_variable": [],
            "multiscales": [],
            "metadata": {"consolidated": None},
        }

        # Size (local only)
        try:
            op = Path(output_path)
            if op.exists():
                total = 0
                for p in op.rglob("*"):
                    if p.is_file():
                        try:
                            total += p.stat().st_size
                        except Exception:
                            pass
                info["store_total_bytes"] = total
        except Exception:
            pass

        # Structure via xarray datatree (works for local and remote)
        try:
            dt = xr.open_datatree(str(output_path), engine="zarr")
            # consolidated metadata flag (best effort)
            try:
                info["metadata"]["consolidated"] = getattr(dt, "_consolidated", None)
            except Exception:
                info["metadata"]["consolidated"] = None

            # walk groups and variables
            def _walk(tree: xr.DataTree, prefix: str = "") -> None:
                if hasattr(tree, "data_vars") and tree.data_vars:
                    for var_name, da in tree.data_vars.items():
                        chunks = None
                        try:
                            enc = getattr(da, "encoding", {}) or {}
                            ch = enc.get("chunks")
                            if isinstance(ch, (list, tuple)):
                                chunks = list(ch)
                        except Exception:
                            pass
                        info["per_variable"].append(
                            {
                                "path": f"{prefix}/{var_name}" if prefix else var_name,
                                "shape": list(da.shape),
                                "chunks": chunks,
                                "dtype": str(da.dtype),
                                "nchunks": None,  # unknown without scanning zarr metadata
                                "compressor": None,
                            }
                        )
                for name, child in (tree.children or {}).items():
                    new_prefix = f"{prefix}/{name}" if prefix else name
                    _walk(child, new_prefix)

            _walk(dt)
        except Exception:
            pass

        # Enrich compressor/chunk info via zarr (best-effort; works with local and remote stores)
        try:
            import zarr

            def _walk_z(g: Any, prefix: str = "") -> None:
                try:
                    # arrays()
                    for name, arr in g.arrays():
                        path = f"{prefix}/{name}" if prefix else name
                        comp_name = None
                        comp_cfg = None
                        try:
                            comp = getattr(arr, "compressor", None)
                            if comp is not None:
                                # numcodecs or blosc-like compressors
                                comp_name = getattr(comp, "codec_id", None) or getattr(
                                    comp, "id", None
                                )
                                comp_cfg = getattr(comp, "get_config", lambda: None)()
                        except Exception:
                            pass
                        # merge
                        merged = False
                        for rec in info["per_variable"]:
                            if rec.get("path") == path:
                                rec["compressor"] = {
                                    "name": comp_name,
                                    "config": comp_cfg,
                                }
                                try:
                                    rec["chunks"] = (
                                        list(arr.chunks)
                                        if getattr(arr, "chunks", None)
                                        else rec.get("chunks")
                                    )
                                except Exception:
                                    pass
                                try:
                                    rec["dtype"] = str(arr.dtype)
                                except Exception:
                                    pass
                                merged = True
                                break
                        if not merged:
                            try:
                                info["per_variable"].append(
                                    {
                                        "path": path,
                                        "shape": list(arr.shape),
                                        "chunks": list(arr.chunks)
                                        if getattr(arr, "chunks", None)
                                        else None,
                                        "dtype": str(arr.dtype),
                                        "nchunks": None,
                                        "compressor": {
                                            "name": comp_name,
                                            "config": comp_cfg,
                                        },
                                    }
                                )
                            except Exception:
                                pass
                    # groups()
                    for name, grp in g.groups():
                        new_prefix = f"{prefix}/{name}" if prefix else name
                        _walk_z(grp, new_prefix)
                except Exception:
                    pass

            z = zarr.open_group(str(output_path), mode="r")
            _walk_z(z)
        except Exception:
            pass

        self.output_info = info

    def finalize(self, status: str, exception: Optional[str] = None) -> Dict[str, Any]:
        self.ended_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        wall = 0.0
        try:
            wall = sum(float(s.get("duration_s", 0.0)) for s in self.steps)
        except Exception:
            wall = 0.0
        return {
            "schema_version": self.schema_version,
            "run": {
                "run_id": self.run_id,
                "attempt": self.attempt,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "status": status,
                "exception": exception,
            },
            "input": self.input_info,
            "output": self.output_info,
            "performance": {"wall_clock_s": wall, "steps": self.steps},
            "environment": self.environment,
        }

    @staticmethod
    def write_json(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
