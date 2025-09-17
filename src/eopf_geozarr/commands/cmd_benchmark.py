import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, cast

import xarray as xr

from ..conversion.fs_utils import get_storage_options
from .dask_utils import setup_dask_cluster


def _open_datatree(path: str, chunks: Optional[str] = None) -> xr.DataTree:
    storage_options = get_storage_options(path)
    return xr.open_datatree(path, engine="zarr", chunks=chunks, storage_options=storage_options)


def _detect_spatial_dims(var: xr.DataArray) -> Optional[Tuple[str, str]]:
    spatial_y = {"y", "lat", "latitude", "rows"}
    spatial_x = {"x", "lon", "longitude", "cols", "col"}
    dims: List[str] = [str(d) for d in var.dims]
    y_dim = next((d for d in dims if d in spatial_y), None)
    x_dim = next((d for d in dims if d in spatial_x), None)
    if y_dim is not None and x_dim is not None:
        return y_dim, x_dim
    if len(dims) >= 2:
        return dims[-2], dims[-1]
    return None


def _first_non_spatial_isel(var: xr.DataArray, yx: Tuple[str, str]) -> Dict[str, int]:
    sel: Dict[str, int] = {}
    for d_any in var.dims:
        d = str(d_any)
        if d not in yx:
            sel[d] = 0
    return sel


def benchmark_command(args: argparse.Namespace) -> None:
    dask_client = setup_dask_cluster(
        enable_dask=getattr(args, "dask_cluster", False), verbose=args.verbose
    )
    try:
        dt_a = _open_datatree(args.input_path, chunks=None)
        dt_b = (
            _open_datatree(args.compare_with, chunks=None)
            if getattr(args, "compare_with", None)
            else None
        )
        groups: List[str] = (
            cast(List[str], args.groups) if args.groups else list(dt_a.children.keys())
        )
        variables: Optional[List[str]] = args.variables
        rng = __import__("random")
        rng.seed(args.seed)

        def run_one(dt: xr.DataTree, tag: str) -> Dict[str, Any]:
            timings: List[float] = []
            bytes_read: List[int] = []
            samples = 0

            def iter_leaf_nodes(path: str, node: xr.DataTree) -> List[Tuple[str, xr.DataTree]]:
                out: List[Tuple[str, xr.DataTree]] = []
                try:
                    ds_local = node.to_dataset()
                except Exception:
                    return out
                if len(ds_local.data_vars) > 0:
                    out.append((path, node))
                for child_name, child in node.children.items():
                    try:
                        child_path = f"{path.rstrip('/')}/{child_name}" if path else str(child_name)
                        out.extend(iter_leaf_nodes(child_path, cast(xr.DataTree, child)))
                    except Exception:
                        continue
                return out

            for g in groups:
                try:
                    node = cast(xr.DataTree, dt[g])
                except Exception:
                    if args.verbose:
                        print(f"Skipping missing group {g} in {tag}")
                    continue
                leaves = iter_leaf_nodes(g, node)
                if not leaves and args.verbose:
                    print(f"No data variables found under group {g} in {tag}")
                for leaf_path, leaf_node in leaves:
                    ds = leaf_node.to_dataset()
                    var_names = variables or [str(n) for n in ds.data_vars]
                    for name in var_names:
                        if name not in ds.data_vars:
                            if args.verbose:
                                print(f"Variable {name} not in {leaf_path} for {tag}")
                            continue
                        var = ds[name]
                        yx = _detect_spatial_dims(var)
                        if not yx:
                            if args.verbose:
                                print(
                                    f"Could not detect spatial dims for {leaf_path}/{name} in {tag}"
                                )
                            continue
                        y_dim, x_dim = yx
                        ny, nx = int(var.sizes[y_dim]), int(var.sizes[x_dim])
                        wy, wx = min(args.window_size, ny), min(args.window_size, nx)
                        if wy <= 0 or wx <= 0:
                            continue
                        for _ in range(args.windows):
                            sy = 0 if ny == wy else rng.randint(0, ny - wy)
                            sx = 0 if nx == wx else rng.randint(0, nx - wx)
                            base_isel = _first_non_spatial_isel(var, (y_dim, x_dim))
                            sel = {
                                **base_isel,
                                y_dim: slice(sy, sy + wy),
                                x_dim: slice(sx, sx + wx),
                            }
                            t0 = time.perf_counter()
                            # Selection mapping keys/values constrained to str -> (int|slice)
                            arr = var.isel(**{k: v for k, v in sel.items()}).values
                            dt_sec = time.perf_counter() - t0
                            timings.append(dt_sec)
                            try:
                                nbytes = int(getattr(arr, "nbytes", arr.size * arr.itemsize))
                            except Exception:
                                nbytes = wy * wx
                            bytes_read.append(nbytes)
                            samples += 1
            if samples == 0:
                return {"tag": tag, "samples": 0}
            timings_sorted = sorted(timings)
            med = timings_sorted[len(timings_sorted) // 2]
            avg = sum(timings) / len(timings)
            avg_mb = (sum(bytes_read) / len(bytes_read)) / (1024 * 1024)
            return {
                "tag": tag,
                "samples": samples,
                "median_s": med,
                "mean_s": avg,
                "mean_window_mb": avg_mb,
            }

        res_a = run_one(dt_a, "A")
        res_b = run_one(dt_b, "B") if dt_b is not None else None
        print("Benchmark summary")
        print("=================")
        print(json.dumps({k: v for k, v in res_a.items() if k != "tag"}, indent=2))
        if res_b:
            print("\nComparison")
            print("-----------")
            print(json.dumps({k: v for k, v in res_b.items() if k != "tag"}, indent=2))
            if res_a.get("samples") and res_b.get("samples"):
                try:
                    speedup = res_a["median_s"] / res_b["median_s"]
                    print(f"\nSpeedup (A/B median): {speedup:.2f}x")
                except Exception:
                    pass
    except Exception as e:
        print(f"Error during benchmark: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
    finally:
        if dask_client is not None:
            try:
                if hasattr(dask_client, "close"):
                    dask_client.close()
                if args.verbose:
                    print("Dask cluster closed")
            except Exception:
                pass
