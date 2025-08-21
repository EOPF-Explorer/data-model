#!/usr/bin/env python3
"""Command-line interface for eopf-geozarr.

This module provides CLI commands for converting EOPF datasets to GeoZarr compliant format.
"""

import argparse
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import xarray as xr

from . import create_geozarr_dataset
from .conversion.fs_utils import (
    get_s3_credentials_info,
    get_storage_options,
    is_s3_path,
    validate_s3_access,
)


def setup_dask_cluster(
    enable_dask: bool,
    verbose: bool = False,
    mode: str = "threads",  # threads | processes | single-threaded
    n_workers: int = 4,
    threads_per_worker: int = 1,
) -> Optional[Any]:
    """Set up a dask cluster for parallel processing.

    Parameters
    ----------
    enable_dask : bool
        Whether to enable dask cluster
    verbose : bool, default False
        Enable verbose output

    Returns:
    -------
    dask.distributed.Client or None
        Dask client if enabled, None otherwise
    """
    if not enable_dask:
        return None

    try:
        from dask.distributed import Client, LocalCluster

        if mode not in {"threads", "processes", "single-threaded"}:
            raise ValueError(f"Unsupported --dask-mode: {mode}")

        processes = mode == "processes"
        # For single-threaded, use one worker, one thread, processes=False
        if mode == "single-threaded":
            n_workers = 1
            threads_per_worker = 1
            processes = False

        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=processes,
        )
        client = Client(cluster)

        if verbose:
            print(f"🚀 Dask cluster started: {client}")
            print(f"   Dashboard: {client.dashboard_link}")
            print(f"   Workers: {len(client.scheduler_info()['workers'])}")
        else:
            print(
                f"🚀 Dask cluster started "
                f"({mode}; workers={n_workers}, threads/worker={threads_per_worker})"
            )
        return client

    except ImportError:
        print(
            "❌ Error: dask.distributed not available. Install with: pip install 'dask[distributed]'"
        )
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting dask cluster: {e}")
        sys.exit(1)


def _duration_from_startstops(ev: dict, action: str) -> float:
    """Best-effort duration extractor from task_stream 'startstops'."""
    total = 0.0
    for ss in ev.get("startstops", []) or []:
        if ss.get("action") == action:
            s, t = ss.get("start"), ss.get("stop")
            if s is not None and t is not None:
                total += max(0.0, float(t) - float(s))
    return total


def _parse_task_events(task_events: Optional[list]) -> dict:
    """Normalize task_stream events across Dask versions into aggregate numbers."""
    if not task_events:
        return {
            "tasks_observed": 0,
            "compute_time_s_sum": 0.0,
            "transfer_time_s_sum": 0.0,
        }
    comp_sum = 0.0
    xfer_sum = 0.0
    for e in task_events:
        # compute time
        cd = e.get("compute_duration")
        if cd is None:
            cs = e.get("compute_start") or e.get("start")
            ce = e.get("compute_stop") or e.get("stop")
            if cs is not None and ce is not None:
                cd = max(0.0, float(ce) - float(cs))
            else:
                cd = _duration_from_startstops(e, "compute")
        comp_sum += float(cd or 0.0)
        # transfer time
        td = e.get("transfer_duration")
        if td is None:
            ts = e.get("transfer_start")
            te = e.get("transfer_stop")
            if ts is not None and te is not None:
                td = max(0.0, float(te) - float(ts))
            else:
                td = _duration_from_startstops(e, "transfer")
        xfer_sum += float(td or 0.0)
    return {
        "tasks_observed": len(task_events),
        "compute_time_s_sum": comp_sum,
        "transfer_time_s_sum": xfer_sum,
    }


def _summarize_dask_metrics(
    client: Any, task_events: Optional[list], wall_clock_s: float
) -> dict:
    info = client.scheduler_info()
    workers = info.get("workers", {})
    total_threads = sum(int(w.get("nthreads", 0) or 0) for w in workers.values())
    memory_limit = sum(int(w.get("memory_limit", 0) or 0) for w in workers.values())
    memory_used = sum(
        int(w.get("metrics", {}).get("memory", 0) or 0) for w in workers.values()
    )
    spilled = sum(
        int(w.get("metrics", {}).get("spilled_nbytes", 0) or 0)
        for w in workers.values()
    )

    parsed = _parse_task_events(task_events)
    tasks = parsed["tasks_observed"]
    base = {
        "wall_clock_s": wall_clock_s,
        "workers": len(workers),
        "threads_total": total_threads,
        "tasks_observed": tasks,
        "tasks_per_sec": (tasks / wall_clock_s) if tasks and wall_clock_s else 0.0,
        "compute_time_s_sum": parsed["compute_time_s_sum"],
        "transfer_time_s_sum": parsed["transfer_time_s_sum"],
        "memory_used_bytes": memory_used,
        "memory_limit_bytes": memory_limit,
    }
    if spilled:
        base["spilled_nbytes"] = spilled
    if getattr(client, "dashboard_link", None):
        base["dashboard_link"] = client.dashboard_link
    return base


def convert_command(args: argparse.Namespace) -> None:
    """Convert EOPF dataset to GeoZarr compliant format.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    # Set up dask cluster if requested
    dask_client = setup_dask_cluster(
        enable_dask=getattr(args, "dask_cluster", False),
        verbose=args.verbose,
        mode=getattr(args, "dask_mode", "threads"),
        n_workers=getattr(args, "dask_workers", 4),
        threads_per_worker=getattr(args, "dask_threads_per_worker", 1),
    )

    # Prepare for metrics writing even on failures (local outputs only)
    debug_dir: Optional[Path] = None
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    status = "unknown"
    error_msg = None

    try:
        # Validate input path (handle both local paths and URLs)
        input_path_str = args.input_path
        if input_path_str.startswith(("http://", "https://", "s3://", "gs://")):
            # URL - no local validation needed
            input_path = input_path_str
        else:
            # Local path - validate existence
            input_path = Path(input_path_str)
            if not input_path.exists():
                print(f"Error: Input path {input_path} does not exist")
                sys.exit(1)
            input_path = str(input_path)

        # Handle output path validation
        output_path_str = args.output_path
        if is_s3_path(output_path_str):
            # S3 path - validate S3 access
            print("🔍 Validating S3 access...")
            success, error_msg = validate_s3_access(output_path_str)
            if not success:
                print(f"❌ Error: Cannot access S3 path {output_path_str}")
                print(f"   Reason: {error_msg}")
                print("\n💡 S3 Configuration Help:")
                print("   Make sure you have S3 credentials configured:")
                print(
                    "   - Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
                )
                print("   - Set AWS_DEFAULT_REGION (default: us-east-1)")
                print(
                    "   - For custom S3 providers (e.g., OVH Cloud), set AWS_ENDPOINT_URL"
                )
                print("   - Or configure AWS CLI with 'aws configure'")
                print("   - Or use IAM roles if running on EC2")

                if args.verbose:
                    creds_info = get_s3_credentials_info()
                    print("\n🔧 Current AWS configuration:")
                    for key, value in creds_info.items():
                        print(f"   {key}: {value or 'Not set'}")

                sys.exit(1)

            print("✅ S3 access validated successfully")
            output_path = output_path_str
        else:
            # Local path - create directory if it doesn't exist
            output_path = Path(output_path_str)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path = str(output_path)
            # Prepare debug dir for metrics
            debug_dir = Path(output_path) / "debug"
            try:
                debug_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                debug_dir = None

        if args.verbose:
            print(f"Loading EOPF dataset from: {input_path}")
            print(f"Groups to convert: {args.groups}")
            print(f"CRS groups: {args.crs_groups}")
            print(f"Output path: {output_path}")
            print(f"Spatial chunk size: {args.spatial_chunk}")
            print(f"Min dimension: {args.min_dimension}")
            print(f"Tile width: {args.tile_width}")

        # Load the EOPF DataTree with appropriate storage options
        print("Loading EOPF dataset...")
        storage_options = get_storage_options(input_path)
        dt = xr.open_datatree(
            str(input_path),
            engine="zarr",
            chunks="auto",
            storage_options=storage_options,
        )

        if args.verbose:
            print(f"Loaded DataTree with {len(dt.children)} groups")
            print("Available groups:")
            for group_name in dt.children:
                print(f"  - {group_name}")

        # Convert to GeoZarr compliant format
        print("Converting to GeoZarr compliant format...")
        t0 = perf_counter()
        task_events = None
        perf_ctx = nullcontext()
        if dask_client is not None and getattr(args, "dask_perf_html", None):
            from dask.distributed import performance_report

            perf_path = args.dask_perf_html
            if not (perf_path.startswith(("s3://", "gs://", "http://", "https://"))):
                Path(perf_path).parent.mkdir(parents=True, exist_ok=True)
            perf_ctx = performance_report(filename=perf_path)

        capture_tasks = False
        if dask_client is not None:
            try:
                from dask.distributed import get_task_stream

                capture_tasks = True
            except Exception:
                pass

        if capture_tasks:
            with perf_ctx:
                from dask.distributed import get_task_stream

                with get_task_stream(client=dask_client, plot=False) as ts:
                    dt_geozarr = create_geozarr_dataset(
                        dt_input=dt,
                        groups=args.groups,
                        output_path=output_path,
                        spatial_chunk=args.spatial_chunk,
                        min_dimension=args.min_dimension,
                        tile_width=args.tile_width,
                        max_retries=args.max_retries,
                        crs_groups=args.crs_groups,
                    )
                task_events = ts.data
        else:
            with perf_ctx:
                dt_geozarr = create_geozarr_dataset(
                    dt_input=dt,
                    groups=args.groups,
                    output_path=output_path,
                    spatial_chunk=args.spatial_chunk,
                    min_dimension=args.min_dimension,
                    tile_width=args.tile_width,
                    max_retries=args.max_retries,
                    crs_groups=args.crs_groups,
                )

        wall_clock = perf_counter() - t0
        if (
            dask_client is not None
            and getattr(args, "verbose", False)
            and getattr(args, "dask_perf_html", None)
        ):
            print(f"📊 Dask performance report: {args.dask_perf_html}")

        print("✅ Successfully converted EOPF dataset to GeoZarr format")
        print(f"Output saved to: {output_path}")
        status = "ok"

        # Drop a JSON run summary (local paths only)
        try:
            if debug_dir is not None:
                summary_path = Path(output_path) / "debug" / "dask_run_summary.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                from json import dumps as _dumps

                summary = {
                    "dask_enabled": bool(dask_client is not None),
                    "mode": getattr(args, "dask_mode", None),
                    "workers": getattr(args, "dask_workers", None),
                    "threads_per_worker": getattr(
                        args, "dask_threads_per_worker", None
                    ),
                    "perf_report": getattr(args, "dask_perf_html", None),
                    "wall_clock_s": wall_clock if dask_client is not None else None,
                    "groups": args.groups,
                    "spatial_chunk": args.spatial_chunk,
                    "min_dimension": args.min_dimension,
                    "tile_width": args.tile_width,
                }
                summary_path.write_text(_dumps(summary, indent=2))
                if args.verbose:
                    print(f"🧾 Wrote run summary: {summary_path}")
        except Exception as _exc:
            if args.verbose:
                print(f"(debug) could not write run summary: {_exc}")

        if args.verbose:
            # Check if dt_geozarr is a DataTree or Dataset
            if hasattr(dt_geozarr, "children"):
                print(f"Converted DataTree has {len(dt_geozarr.children)} groups")
                print("Converted groups:")
                for group_name in dt_geozarr.children:
                    print(f"  - {group_name}")
            else:
                print("Converted dataset (single group)")

    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        error_msg = str(e)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up dask client if it was created
        if dask_client is not None:
            try:
                if hasattr(dask_client, "close"):
                    dask_client.close()
                if args.verbose:
                    print("🔄 Dask cluster closed")
            except Exception as e:
                if args.verbose:
                    print(f"Warning: Error closing dask cluster: {e}")

        # Best-effort metrics write (works for success or failure, local only)
        try:
            if debug_dir is not None:
                attempt = 1
                try:
                    attempt = 1 + len(list(debug_dir.glob("dask_metrics_*.json")))
                except Exception:
                    pass
                wall = None
                if "t0" in locals():
                    wall = perf_counter() - t0
                metrics_doc = {
                    "status": status,
                    "run_id": run_id,
                    "attempt": attempt,
                    "dask_enabled": bool(dask_client is not None),
                    "mode": getattr(args, "dask_mode", None),
                }
                if dask_client is not None and wall is not None:
                    try:
                        metrics_doc.update(
                            _summarize_dask_metrics(
                                dask_client, locals().get("task_events"), wall
                            )
                        )
                    except Exception:
                        metrics_doc["wall_clock_s"] = wall
                elif wall is not None:
                    metrics_doc["wall_clock_s"] = wall
                if error_msg:
                    metrics_doc["error"] = error_msg

                path_ts = debug_dir / f"dask_metrics_{run_id}_attempt{attempt}.json"
                path_latest = debug_dir / "dask_metrics.json"
                from json import dumps as _dumps

                txt = _dumps(metrics_doc, indent=2)
                path_ts.write_text(txt)
                path_latest.write_text(txt)
                if getattr(args, "verbose", False):
                    print(f"📈 Wrote metrics: {path_ts}")
        except Exception as _mexc:
            if getattr(args, "verbose", False):
                print(f"(debug) could not write metrics snapshot: {_mexc}")


def info_command(args: argparse.Namespace) -> None:
    """Display information about an EOPF dataset.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    # Handle both local paths and URLs
    input_path_str = args.input_path
    if input_path_str.startswith(("http://", "https://", "s3://", "gs://")):
        # URL - no local validation needed
        input_path = input_path_str
    else:
        # Local path - validate existence
        input_path = Path(input_path_str)
        if not input_path.exists():
            print(f"Error: Input path {input_path} does not exist")
            sys.exit(1)
        input_path = str(input_path)

    try:
        print(f"Loading dataset from: {input_path}")
        # Use unified storage options for S3 support
        storage_options = get_storage_options(input_path)
        dt = xr.open_datatree(
            input_path, engine="zarr", chunks="auto", storage_options=storage_options
        )

        if hasattr(args, "html_output") and args.html_output:
            # Generate HTML output
            _generate_html_output(dt, args.html_output, input_path, args.verbose)
        else:
            # Standard console output
            print("\nDataset Information:")
            print("==================")
            print(f"Total groups: {len(dt.children)}")

            print("\nGroup structure:")
            print(dt)

    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def _generate_optimized_tree_html(dt: xr.DataTree) -> str:
    """Generate an optimized, condensed tree HTML representation.

    This function creates a clean tree view that:
    - Hides empty nodes by default
    - Shows only nodes with data variables or meaningful content
    - Provides a more condensed, focused view

    Parameters
    ----------
    dt : xr.DataTree
        DataTree to visualize

    Returns:
    -------
    str
        HTML representation of the optimized tree
    """

    def has_meaningful_content(node: Any) -> bool:
        """Check if a node has meaningful content (data variables, attributes, or meaningful children)."""
        if hasattr(node, "ds") and node.ds is not None:
            # Has data variables
            if hasattr(node.ds, "data_vars") and len(node.ds.data_vars) > 0:
                return True
            # Has meaningful attributes (more than just empty metadata)
            if hasattr(node.ds, "attrs") and node.ds.attrs:
                return True

        # Check if any children have meaningful content
        if hasattr(node, "children") and node.children:
            return any(
                has_meaningful_content(child) for child in node.children.values()
            )

        return False

    def format_dimensions(dims: Any) -> str:
        """Format dimensions in a compact way."""
        if not dims:
            return ""
        return f"({', '.join(f'{k}: {v}' for k, v in dims.items())})"

    def format_data_vars(data_vars: Any) -> str:
        """Format data variables using xarray's rich HTML representation."""
        if not data_vars:
            return ""

        # Create a temporary dataset with just these variables to get xarray's HTML
        import xarray as xr

        temp_ds = xr.Dataset(data_vars)

        # Get xarray's HTML representation and extract just the variables section
        try:
            html_repr = temp_ds._repr_html_()
            # Extract the variables section from xarray's HTML
            # This gives us the rich, interactive variable display
            return f'<div class="xarray-variables">{html_repr}</div>'
        except Exception:
            # Fallback to simple format if xarray HTML fails
            vars_html = []
            for name, var in data_vars.items():
                dims_str = format_dimensions(dict(zip(var.dims, var.shape)))
                dtype_str = str(var.dtype)
                vars_html.append(
                    f"""
                    <div class="tree-variable">
                        <span class="var-name">{name}</span>
                        <span class="var-dims">{dims_str}</span>
                        <span class="var-dtype">{dtype_str}</span>
                    </div>
                """
                )
            return "".join(vars_html)

    def format_attributes(attrs: Any) -> str:
        """Format attributes in a compact way."""
        if not attrs:
            return ""

        # Show only first few attributes to keep it condensed
        items = list(attrs.items())[:5]  # Show max 5 attributes
        attrs_html = []
        for key, value in items:
            # Truncate long values
            value_str = str(value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
            attrs_html.append(
                f"""
                <div class="tree-attribute">
                    <span class="attr-key">{key}:</span>
                    <span class="attr-value">{value_str}</span>
                </div>
            """
            )

        if len(attrs) > 5:
            attrs_html.append(
                f'<div class="tree-attribute-more">... and {len(attrs) - 5} more</div>'
            )

        return "".join(attrs_html)

    def render_node(node: Any, path: str = "", level: int = 0) -> str:
        """Render a single node and its children."""
        if not has_meaningful_content(node):
            return ""  # Skip empty nodes

        node_name = path.split("/")[-1] if path else "root"
        if not node_name:
            node_name = "root"

        # Determine node type and content
        has_data = hasattr(node, "ds") and node.ds is not None
        data_vars_count = (
            len(node.ds.data_vars) if has_data and hasattr(node.ds, "data_vars") else 0
        )
        attrs_count = (
            len(node.ds.attrs) if has_data and hasattr(node.ds, "attrs") else 0
        )
        children_count = (
            len(
                [
                    child
                    for child in node.children.values()
                    if has_meaningful_content(child)
                ]
            )
            if hasattr(node, "children")
            else 0
        )

        # Create node summary
        summary_parts = []
        if data_vars_count > 0:
            summary_parts.append(f"{data_vars_count} variables")
        if attrs_count > 0:
            summary_parts.append(f"{attrs_count} attributes")
        if children_count > 0:
            summary_parts.append(f"{children_count} subgroups")

        summary = " • ".join(summary_parts) if summary_parts else "empty group"

        # Generate HTML for this node
        node_html = f"""
        <div class="tree-node" style="margin-left: {level * 20}px;">
            <details class="tree-details" {'open' if level < 2 else ''}>
                <summary class="tree-summary">
                    <span class="tree-icon">{'📁' if children_count > 0 else '📄'}</span>
                    <span class="tree-name">{node_name}</span>
                    <span class="tree-info">({summary})</span>
                </summary>
                <div class="tree-content">
        """

        # Add data variables if present
        if has_data and hasattr(node.ds, "data_vars") and node.ds.data_vars:
            node_html += f"""
                <div class="tree-section">
                    <h4 class="section-title">Variables</h4>
                    <div class="tree-variables">
                        {format_data_vars(node.ds.data_vars)}
                    </div>
                </div>
            """

        # Add attributes if present
        if has_data and hasattr(node.ds, "attrs") and node.ds.attrs:
            node_html += f"""
                <div class="tree-section">
                    <h4 class="section-title">Attributes</h4>
                    <div class="tree-attributes">
                        {format_attributes(node.ds.attrs)}
                    </div>
                </div>
            """

        # Add children
        if hasattr(node, "children") and node.children:
            children_html = []
            for child_name, child_node in node.children.items():
                child_path = f"{path}/{child_name}" if path else child_name
                child_html = render_node(child_node, child_path, level + 1)
                if child_html:  # Only add if not empty
                    children_html.append(child_html)

            if children_html:
                node_html += f"""
                    <div class="tree-section">
                        <h4 class="section-title">Subgroups</h4>
                        <div class="tree-children">
                            {"".join(children_html)}
                        </div>
                    </div>
                """

        node_html += """
                </div>
            </details>
        </div>
        """

        return node_html

    # Generate the complete tree
    tree_content = render_node(dt)

    # Wrap in container with custom styles
    return f"""
    <div class="optimized-tree">
        <style>
            .optimized-tree {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.5;
            }}

            .tree-node {{
                margin-bottom: 8px;
            }}

            .tree-details {{
                border: 1px solid #e1e5e9;
                border-radius: 6px;
                overflow: hidden;
            }}

            .tree-summary {{
                background: linear-gradient(90deg, #f6f8fa 0%, #ffffff 100%);
                padding: 12px 16px;
                cursor: pointer;
                border: none;
                display: flex;
                align-items: center;
                gap: 8px;
                font-weight: 500;
                color: #24292f;
                transition: background-color 0.2s ease;
            }}

            .tree-summary:hover {{
                background: linear-gradient(90deg, #f1f3f4 0%, #f6f8fa 100%);
            }}

            .tree-icon {{
                font-size: 16px;
            }}

            .tree-name {{
                font-weight: 600;
                color: #0969da;
            }}

            .tree-info {{
                color: #656d76;
                font-size: 0.9em;
                margin-left: auto;
            }}

            .tree-content {{
                padding: 16px;
                background-color: #fafbfc;
                border-top: 1px solid #e1e5e9;
            }}

            .tree-section {{
                margin-bottom: 16px;
            }}

            .tree-section:last-child {{
                margin-bottom: 0;
            }}

            .section-title {{
                margin: 0 0 8px 0;
                font-size: 0.9em;
                font-weight: 600;
                color: #656d76;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}

            .tree-variable {{
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 6px 0;
                border-bottom: 1px solid #f1f3f4;
            }}

            .tree-variable:last-child {{
                border-bottom: none;
            }}

            .var-name {{
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
                font-weight: 600;
                color: #0969da;
                min-width: 120px;
            }}

            .var-dims {{
                color: #656d76;
                font-size: 0.85em;
                font-style: italic;
            }}

            .var-dtype {{
                color: #1a7f37;
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
                font-size: 0.85em;
                font-weight: 500;
                background-color: #f6f8fa;
                padding: 2px 6px;
                border-radius: 3px;
            }}

            .tree-attribute {{
                display: flex;
                gap: 8px;
                padding: 4px 0;
                font-size: 0.9em;
            }}

            .attr-key {{
                font-weight: 600;
                color: #24292f;
                min-width: 100px;
            }}

            .attr-value {{
                color: #656d76;
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
                font-size: 0.85em;
            }}

            .tree-attribute-more {{
                color: #656d76;
                font-style: italic;
                font-size: 0.85em;
                padding: 4px 0;
            }}

            .tree-children {{
                margin-top: 8px;
            }}
        </style>
        {tree_content}
    </div>
    """


def _generate_html_output(
    dt: xr.DataTree, output_path: str, input_path: str, verbose: bool = False
) -> None:
    """Generate HTML output for DataTree visualization.

    Parameters
    ----------
    dt : xr.DataTree
        DataTree to visualize
    output_path : str
        Path for HTML output file
    input_path : str
        Original input path for reference
    verbose : bool, default False
        Enable verbose output
    """
    try:
        # Generate optimized tree structure
        tree_html = _generate_optimized_tree_html(dt)

        # Create a complete HTML document with EOPF-style formatting
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataTree Visualization - {Path(input_path).name}</title>
    <style>
        /* EOPF-inspired styling */
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #fafafa;
            color: #333;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            margin: 0 0 15px 0;
            font-size: 2.2em;
            font-weight: 300;
            letter-spacing: -0.5px;
        }}

        .header-info {{
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            margin-top: 20px;
            font-size: 0.95em;
            opacity: 0.95;
        }}

        .header-info-item {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        .header-info-label {{
            font-weight: 600;
            margin-bottom: 4px;
            text-transform: uppercase;
            font-size: 0.8em;
            letter-spacing: 0.5px;
        }}

        .header-info-value {{
            font-size: 1.1em;
        }}

        .content {{
            padding: 0;
        }}

        .datatree-container {{
            overflow-x: auto;
            padding: 30px;
        }}

        /* Enhanced xarray styling to match EOPF look */
        .xr-wrap {{
            font-family: inherit !important;
        }}

        .xr-header {{
            background-color: #f8f9fa !important;
            border: 1px solid #e9ecef !important;
            border-radius: 6px !important;
            padding: 15px !important;
            margin-bottom: 20px !important;
        }}

        .xr-obj-type {{
            color: #6f42c1 !important;
            font-weight: 600 !important;
            font-size: 1.1em !important;
        }}

        .xr-section-item {{
            margin-bottom: 15px !important;
            border: 1px solid #e9ecef !important;
            border-radius: 6px !important;
            overflow: hidden !important;
        }}

        .xr-section-summary {{
            background: linear-gradient(90deg, #f8f9fa 0%, #ffffff 100%) !important;
            padding: 12px 15px !important;
            border: none !important;
            cursor: pointer !important;
            font-weight: 500 !important;
            color: #495057 !important;
            transition: all 0.2s ease !important;
        }}

        .xr-section-summary:hover {{
            background: linear-gradient(90deg, #e9ecef 0%, #f8f9fa 100%) !important;
            transform: translateX(2px) !important;
        }}

        .xr-section-summary-in {{
            display: flex !important;
            align-items: center !important;
            gap: 10px !important;
        }}

        .xr-section-details {{
            padding: 20px !important;
            background-color: #fdfdfd !important;
            border-top: 1px solid #e9ecef !important;
        }}

        .xr-var-list {{
            margin: 0 !important;
            padding: 0 !important;
        }}

        .xr-var-item {{
            padding: 8px 0 !important;
            border-bottom: 1px solid #f1f3f4 !important;
        }}

        .xr-var-item:last-child {{
            border-bottom: none !important;
        }}

        .xr-var-name {{
            font-weight: 600 !important;
            color: #1a73e8 !important;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
        }}

        .xr-var-dims {{
            color: #5f6368 !important;
            font-style: italic !important;
            font-size: 0.9em !important;
        }}

        .xr-var-dtype {{
            color: #137333 !important;
            font-weight: 500 !important;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
            font-size: 0.9em !important;
        }}

        .xr-attrs {{
            background-color: #f8f9fa !important;
            border-radius: 4px !important;
            padding: 10px !important;
            margin-top: 10px !important;
        }}

        .xr-attrs dt {{
            font-weight: 600 !important;
            color: #495057 !important;
        }}

        .xr-attrs dd {{
            color: #6c757d !important;
            margin-left: 20px !important;
        }}

        /* Collapsible sections styling */
        details {{
            margin-bottom: 10px !important;
        }}

        summary {{
            cursor: pointer !important;
            padding: 10px !important;
            background-color: #f1f3f4 !important;
            border-radius: 4px !important;
            font-weight: 500 !important;
            transition: background-color 0.2s ease !important;
        }}

        summary:hover {{
            background-color: #e8eaed !important;
        }}

        /* Footer styling */
        .footer {{
            background-color: #f8f9fa;
            padding: 20px 30px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
            border-top: 1px solid #e9ecef;
        }}

        /* Responsive design */
        @media (max-width: 768px) {{
            .header-info {{
                flex-direction: column;
                gap: 15px;
            }}

            .datatree-container {{
                padding: 20px;
            }}

            .header {{
                padding: 20px;
            }}

            .header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{Path(input_path).name}</h1>
            <div class="header-info">
                <div class="header-info-item">
                    <div class="header-info-label">Dataset Path</div>
                    <div class="header-info-value">{input_path}</div>
                </div>
                <div class="header-info-item">
                    <div class="header-info-label">Total Groups</div>
                    <div class="header-info-value">{len(dt.children)}</div>
                </div>
                <div class="header-info-item">
                    <div class="header-info-label">Generated</div>
                    <div class="header-info-value">{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                </div>
            </div>
        </div>

        <div class="content">
            <div class="datatree-container">
                {tree_html}
            </div>
        </div>

        <div class="footer">
            Generated by eopf-geozarr CLI • Interactive DataTree Visualization
        </div>
    </div>
</body>
</html>
"""

        # Write HTML file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ HTML visualization generated: {output_file}")

        if verbose:
            print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
            print(f"   Groups included: {len(dt.children)}")

        # Try to open in browser if possible
        try:
            import webbrowser

            webbrowser.open(f"file://{output_file.absolute()}")
            print("🌐 Opening in default browser...")
        except Exception as e:
            if verbose:
                print(f"   Note: Could not auto-open browser: {e}")
            print(f"   You can open the file manually: {output_file.absolute()}")

    except Exception as e:
        print(f"❌ Error generating HTML output: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def validate_command(args: argparse.Namespace) -> None:
    """Validate GeoZarr compliance of a dataset.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    # Handle both local paths and URLs
    input_path_str = args.input_path
    if input_path_str.startswith(("http://", "https://", "s3://", "gs://")):
        # URL - no local validation needed
        input_path = input_path_str
    else:
        # Local path - validate existence
        input_path = Path(input_path_str)
        if not input_path.exists():
            print(f"Error: Input path {input_path} does not exist")
            sys.exit(1)
        input_path = str(input_path)

    try:
        print(f"Validating GeoZarr compliance for: {input_path}")
        # Use unified storage options for S3 support
        storage_options = get_storage_options(input_path)
        dt = xr.open_datatree(
            input_path, engine="zarr", chunks="auto", storage_options=storage_options
        )

        compliance_issues = []
        total_variables = 0
        compliant_variables = 0

        print("\nValidation Results:")
        print("==================")

        for group_name, group in dt.children.items():
            print(f"\nGroup: {group_name}")

            if not hasattr(group, "data_vars") or not group.data_vars:
                print("  ⚠️  No data variables found")
                continue

            for var_name, var in group.data_vars.items():
                total_variables += 1
                issues = []

                # Check for _ARRAY_DIMENSIONS
                if "_ARRAY_DIMENSIONS" not in var.attrs:
                    issues.append("Missing _ARRAY_DIMENSIONS attribute")

                # Check for standard_name
                if "standard_name" not in var.attrs:
                    issues.append("Missing standard_name attribute")

                # Check for grid_mapping (for data variables, not grid_mapping variables)
                if (
                    "grid_mapping" not in var.attrs
                    and "grid_mapping_name" not in var.attrs
                ):
                    issues.append("Missing grid_mapping attribute")

                if issues:
                    print(f"  ❌ {var_name}: {', '.join(issues)}")
                    compliance_issues.extend(issues)
                else:
                    print(f"  ✅ {var_name}: Compliant")
                    compliant_variables += 1

        print("\nSummary:")
        print("========")
        print(f"Total variables checked: {total_variables}")
        print(f"Compliant variables: {compliant_variables}")
        print(f"Non-compliant variables: {total_variables - compliant_variables}")

        if compliance_issues:
            print("\n❌ Dataset is NOT GeoZarr compliant")
            print(f"Issues found: {len(compliance_issues)}")
            if args.verbose:
                print("Detailed issues:")
                for issue in set(compliance_issues):
                    print(f"  - {issue}")
        else:
            print("\n✅ Dataset appears to be GeoZarr compliant")

    except Exception as e:
        print(f"❌ Error validating dataset: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Returns:
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="eopf-geozarr",
        description="Convert EOPF datasets to GeoZarr compliant format",
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert EOPF dataset to GeoZarr compliant format"
    )
    convert_parser.add_argument(
        "input_path", type=str, help="Path to input EOPF dataset (Zarr format)"
    )
    convert_parser.add_argument(
        "output_path",
        type=str,
        help="Path for output GeoZarr dataset (local path or S3 URL like s3://bucket/path)",
    )
    convert_parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=["/measurements/r10m", "/measurements/r20m", "/measurements/r60m"],
        help="Groups to convert (default: Sentinel-2 resolution groups)",
    )
    convert_parser.add_argument(
        "--spatial-chunk",
        type=int,
        default=4096,
        help="Spatial chunk size for encoding (default: 4096)",
    )
    convert_parser.add_argument(
        "--min-dimension",
        type=int,
        default=256,
        help="Minimum dimension for overview levels (default: 256)",
    )
    convert_parser.add_argument(
        "--tile-width",
        type=int,
        default=256,
        help="Tile width for TMS compatibility (default: 256)",
    )
    convert_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for network operations (default: 3)",
    )
    convert_parser.add_argument(
        "--crs-groups",
        type=str,
        nargs="*",
        help="Groups that need CRS information added on best-effort basis (e.g., /conditions/geometry)",
    )
    convert_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    convert_parser.add_argument(
        "--dask-cluster",
        action="store_true",
        help="Start a local dask cluster for parallel processing of chunks",
    )
    convert_parser.add_argument(
        "--dask-mode",
        type=str,
        default="threads",
        choices=["threads", "processes", "single-threaded"],
        help="Local Dask execution mode (default: threads).",
    )
    convert_parser.add_argument(
        "--dask-workers",
        type=int,
        default=4,
        help="Number of Dask workers (default: 4).",
    )
    convert_parser.add_argument(
        "--dask-threads-per-worker",
        type=int,
        default=1,
        help="Threads per worker (default: 1).",
    )
    convert_parser.add_argument(
        "--dask-perf-html",
        type=str,
        help="Write a Dask performance report HTML to this path (local only).",
    )
    convert_parser.set_defaults(func=convert_command)

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Display information about an EOPF dataset"
    )
    info_parser.add_argument("input_path", type=str, help="Path to EOPF dataset")
    info_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    info_parser.add_argument(
        "--html-output",
        type=str,
        help="Generate HTML visualization and save to specified file (e.g., dataset_info.html)",
    )
    info_parser.set_defaults(func=info_command)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate GeoZarr compliance of a dataset"
    )
    validate_parser.add_argument(
        "input_path", type=str, help="Path to dataset to validate"
    )
    validate_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )
    validate_parser.set_defaults(func=validate_command)

    return parser


def main() -> None:
    """Execute main entry point for the CLI."""
    parser = create_parser()

    if len(sys.argv) == 1:
        # Show help if no arguments provided
        parser.print_help()
        return

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
