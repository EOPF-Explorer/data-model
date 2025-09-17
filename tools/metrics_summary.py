#!/usr/bin/env python3
"""
Aggregate conversion run summaries into a CSV for quick benchmarking.

Usage:
  uv run python tools/metrics_summary.py --root ./out --output ./out/metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def find_run_summaries(root: Path) -> List[Path]:
    return list(root.rglob("run_summary.json"))


def flatten_record(path: Path, doc: Dict[str, Any]) -> Dict[str, Any]:
    run = doc.get("run", {})
    inp = doc.get("input", {})
    out = doc.get("output", {})
    perf = doc.get("performance", {})
    env = doc.get("environment", {})

    return {
        "file": str(path),
        "run_id": run.get("run_id"),
        "status": run.get("status"),
        "source_uri": inp.get("source_uri"),
        "groups": ",".join(inp.get("groups", []) or []),
        "dask_mode": (inp.get("dask") or {}).get("mode"),
        "dask_workers": (inp.get("dask") or {}).get("workers"),
        "dask_threads_per_worker": (inp.get("dask") or {}).get("threads_per_worker"),
        "store_uri": out.get("store_uri"),
        "store_total_bytes": out.get("store_total_bytes"),
        "wall_clock_s": perf.get("wall_clock_s"),
        "python": env.get("python"),
        "xarray": (env.get("packages") or {}).get("xarray"),
        "zarr": (env.get("packages") or {}).get("zarr"),
        "dask": (env.get("packages") or {}).get("dask"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root directory to scan")
    ap.add_argument("--output", type=str, required=True, help="Output CSV path")
    args = ap.parse_args()

    root = Path(args.root)
    outp = Path(args.output)
    rows: List[Dict[str, Any]] = []

    for p in find_run_summaries(root):
        try:
            doc = json.loads(p.read_text())
            if not isinstance(doc, dict):
                continue
            rows.append(flatten_record(p, doc))
        except Exception:
            continue

    outp.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        outp.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with outp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
