import argparse
import sys
from pathlib import Path
from typing import Any
from typing import Any as _Any
from typing import Dict, List, cast

import xarray as xr

from .. import create_geozarr_dataset
from ..conversion.fs_utils import (
    get_s3_credentials_info,
    get_storage_options,
    is_s3_path,
    validate_s3_access,
)
from ..conversion.helpers import normalize_group_path
from ._shared import prepare_output_path, resolve_input_path
from .dask_utils import setup_dask_cluster

MetricsRecorder: _Any
try:  # Optional metrics
    from ..metrics import MetricsRecorder as MetricsRecorderCls

    MetricsRecorder = MetricsRecorderCls
except Exception:  # pragma: no cover
    MetricsRecorder = None


def convert_command(args: argparse.Namespace) -> None:
    dask_client = setup_dask_cluster(
        enable_dask=getattr(args, "dask_cluster", False), verbose=args.verbose
    )

    rec = None
    metrics_path = None
    if getattr(args, "metrics_out", None) and MetricsRecorder is not None:
        try:
            rec = MetricsRecorder()
            rec.set_environment()
        except Exception:
            rec = None

    try:
        input_path = resolve_input_path(args.input_path)
        output_path = prepare_output_path(
            args.output_path,
            is_s3_path=is_s3_path,
            validate_s3_access=validate_s3_access,
            get_s3_credentials_info=get_s3_credentials_info,
            verbose=args.verbose,
        )

        # Normalize empty --crs-groups flag (present but with no values) -> []
        if getattr(args, "crs_groups", None) is None and "--crs-groups" in sys.argv:
            # Normalize explicit empty list
            args.crs_groups = []

        if args.verbose:
            print(f"Loading EOPF dataset from: {input_path}")
            print(f"Requested groups: {args.groups}")
            print(f"CRS groups: {args.crs_groups}")
            print(f"Output path: {output_path}")
            print(f"Spatial chunk size: {args.spatial_chunk}")
            print(f"Min dimension: {args.min_dimension}")
            print(f"Tile width: {args.tile_width}")

        print("Loading EOPF dataset...")
        storage_options = get_storage_options(input_path)
        if rec:
            with rec.time_step("open_input"):
                dt = xr.open_datatree(
                    str(input_path),
                    engine="zarr",
                    chunks="auto",
                    storage_options=storage_options,
                )
        else:
            dt = xr.open_datatree(
                str(input_path),
                engine="zarr",
                chunks="auto",
                storage_options=storage_options,
            )

        if args.verbose:
            print(f"Loaded DataTree with {len(dt.children)} groups")
            print("Available top-level groups:")
            for group_name in dt.children:
                print(f"  - {group_name}")

        def iter_leaf_groups(path: str, node: xr.DataTree) -> List[str]:
            out: List[str] = []
            try:
                ds_local = node.to_dataset()
                if len(ds_local.data_vars) > 0:
                    return [normalize_group_path(path)]
            except Exception:
                return out
            for child_name, child in getattr(node, "children", {}).items():
                child_path = f"{path.rstrip('/')}/{child_name}" if path else child_name
                out.extend(iter_leaf_groups(child_path, cast(xr.DataTree, child)))
            return out

        groups_requested: List[str] = args.groups
        groups_expanded: List[str] = []
        for g in groups_requested:
            g_norm = normalize_group_path(g)
            if g_norm in {"/", "//"}:
                try:
                    if len(dt.to_dataset().data_vars) > 0:
                        groups_expanded.append("/")
                except Exception:
                    pass
                for child_name, child in dt.children.items():
                    groups_expanded.extend(iter_leaf_groups(child_name, child))
                continue
            try:
                node = dt[g_norm]
            except Exception:
                try:
                    node = dt[g_norm.lstrip("/")]
                except Exception:
                    if args.verbose:
                        print(f"Warning: Requested group not found: {g_norm}; skipping")
                    continue
            from typing import cast as _cast

            leaves = iter_leaf_groups(g_norm, _cast(xr.DataTree, node))
            if leaves:
                groups_expanded.extend(leaves)
            elif args.verbose:
                print(f"Warning: No data variables found under {g_norm}; skipping")

        if not groups_expanded:
            print("No valid groups with data variables were found to convert. Nothing to do.")
            sys.exit(1)

        if args.verbose:
            print("Groups to convert (expanded to leaves):")
            for g in groups_expanded:
                print(f"  - {g}")

        if rec:
            try:
                dask_meta: Dict[str, Any] = {}
                if dask_client is not None:
                    try:
                        sched = dask_client.scheduler_info()
                        workers = sched.get("workers", {})
                        any_worker: Dict[str, Any] = (
                            cast(Dict[str, Any], next(iter(workers.values()))) if workers else {}
                        )
                        dask_meta = {
                            "mode": "local",
                            "workers": len(workers),
                            "threads_per_worker": any_worker.get("nthreads"),
                        }
                    except Exception:
                        dask_meta = {"mode": "local"}
                rec.set_input(
                    source_uri=input_path,
                    profile=None,
                    groups=groups_expanded,
                    dask=dask_meta,
                )
            except Exception:
                pass

        print("Converting to GeoZarr compliant format...")
        if rec:
            with rec.time_step("convert"):
                dt_geozarr = create_geozarr_dataset(
                    dt_input=dt,
                    groups=groups_expanded,
                    output_path=output_path,
                    spatial_chunk=args.spatial_chunk,
                    min_dimension=args.min_dimension,
                    tile_width=args.tile_width,
                    max_retries=args.max_retries,
                    crs_groups=args.crs_groups,
                    overwrite=getattr(args, "overwrite", "fail"),
                )
        else:
            dt_geozarr = create_geozarr_dataset(
                dt_input=dt,
                groups=groups_expanded,
                output_path=output_path,
                spatial_chunk=args.spatial_chunk,
                min_dimension=args.min_dimension,
                tile_width=args.tile_width,
                max_retries=args.max_retries,
                crs_groups=args.crs_groups,
                overwrite=getattr(args, "overwrite", "fail"),
            )

        print("Conversion completed")
        print(f"Output saved to: {output_path}")

        if rec:
            try:
                with rec.time_step("output_summary"):
                    rec.build_output_summary(output_path)
                payload = rec.finalize("success")
                metrics_path = Path(args.metrics_out)
                if str(metrics_path).endswith("/") or metrics_path.is_dir():
                    metrics_path = metrics_path / "run_summary.json"
                if not metrics_path.suffix:
                    metrics_path = metrics_path.with_name(metrics_path.name + ".json")
                from ..metrics import MetricsRecorder as _MR

                _MR.write_json(metrics_path, payload)
                if args.verbose:
                    print(f"Metrics written to: {metrics_path}")
            except Exception as me:  # pragma: no cover
                if args.verbose:
                    print(f"Warning: failed to write metrics: {me}")

        if args.verbose:
            if hasattr(dt_geozarr, "children"):
                print(f"Converted DataTree has {len(dt_geozarr.children)} groups")
                print("Converted groups:")
                for group_name in dt_geozarr.children:
                    print(f"  - {group_name}")
            else:
                print("Converted dataset (single group)")

    except Exception as e:
        err_msg = str(e)
        print(f"Error during conversion: {err_msg}")
        if (
            "already exists, but encoding was provided" in err_msg
            and getattr(args, "overwrite", "fail") == "fail"
        ):
            print(
                "Hint: Existing arrays detected in output. Re-run with --overwrite replace to force a clean write, "
                "or use --overwrite skip/merge for incremental behavior (if supported)."
            )
        if rec:
            try:
                if not rec.output_info:
                    rec.build_output_summary(output_path)
                payload = rec.finalize("error", exception=str(e)[:500])
                metrics_path = metrics_path or (
                    Path(getattr(args, "metrics_out"))
                    if getattr(args, "metrics_out", None)
                    else None
                )
                if metrics_path is not None:
                    if str(metrics_path).endswith("/"):
                        metrics_path = metrics_path / "run_summary.json"
                    if not metrics_path.suffix:
                        metrics_path = metrics_path.with_name(metrics_path.name + ".json")
                    from ..metrics import MetricsRecorder as _MR

                    _MR.write_json(metrics_path, payload)
                    if args.verbose:
                        print(f"Error metrics written to: {metrics_path}")
            except Exception:
                pass
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
                if args.verbose:
                    print("Warning: Error closing dask cluster")
