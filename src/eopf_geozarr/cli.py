"""Minimal CLI entry point wiring subcommands from commands package."""

import argparse
import sys
from typing import Any, Dict, List

from .commands import (
    benchmark_command,
    convert_command,
    info_command,
    stac_command,
    validate_command,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eopf-geozarr",
        description="Convert EOPF datasets to GeoZarr compliant format",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subs = parser.add_subparsers(dest="command", help="Commands")

    command_specs: List[Dict[str, Any]] = [
        {
            "name": "convert",
            "func": convert_command,
            "help": "Convert dataset to GeoZarr",
            "positional": [
                ("input_path", {"type": str}),
                ("output_path", {"type": str}),
            ],
            "options": [
                (
                    "--groups",
                    {
                        "type": str,
                        "nargs": "+",
                        "default": [
                            "/measurements/r10m",
                            "/measurements/r20m",
                            "/measurements/r60m",
                        ],
                    },
                ),
                ("--spatial-chunk", {"type": int, "default": 4096}),
                ("--min-dimension", {"type": int, "default": 256}),
                ("--tile-width", {"type": int, "default": 256}),
                ("--max-retries", {"type": int, "default": 3}),
                (
                    "--crs-groups",
                    {
                        "type": str,
                        "nargs": "*",
                        "help": "Groups that need CRS information added",
                    },
                ),
                (
                    "--overwrite",
                    {
                        "choices": ["fail", "skip", "merge", "replace"],
                        "default": "fail",
                        "help": "Overwrite policy for existing output groups",
                    },
                ),
                ("--verbose", {"action": "store_true"}),
                ("--dask-cluster", {"action": "store_true"}),
                (
                    "--metrics-out",
                    {"type": str, "help": "Write metrics JSON (file or dir ending /)"},
                ),
            ],
        },
        {
            "name": "info",
            "func": info_command,
            "help": "Show dataset info",
            "positional": [("input_path", {"type": str})],
            "options": [
                ("--verbose", {"action": "store_true"}),
                ("--html-output", {"type": str, "help": "Write HTML visualization"}),
            ],
        },
        {
            "name": "validate",
            "func": validate_command,
            "help": "Validate GeoZarr compliance",
            "positional": [("input_path", {"type": str})],
            "options": [("--verbose", {"action": "store_true"})],
        },
        {
            "name": "benchmark",
            "func": benchmark_command,
            "help": "Benchmark random window reads",
            "positional": [("input_path", {"type": str})],
            "options": [
                ("--compare-with", {"type": str}),
                ("--groups", {"type": str, "nargs": "*"}),
                ("--variables", {"type": str, "nargs": "*"}),
                ("--window-size", {"type": int, "default": 512}),
                ("--windows", {"type": int, "default": 5}),
                ("--seed", {"type": int, "default": 42}),
                ("--verbose", {"action": "store_true"}),
                ("--dask-cluster", {"action": "store_true"}),
            ],
        },
        {
            "name": "stac",
            "func": stac_command,
            "help": "Generate minimal STAC JSON",
            "positional": [("input_path", {"type": str}), ("output", {"type": str})],
            "options": [
                ("--type", {"choices": ["item", "collection"], "default": "item"}),
                ("--id", {"type": str}),
                ("--title", {"type": str}),
                (
                    "--bbox",
                    {
                        "type": float,
                        "nargs": 4,
                        "metavar": ("MINX", "MINY", "MAXX", "MAXY"),
                    },
                ),
                ("--datetime", {"type": str}),
                ("--collection-id", {"type": str}),
                ("--license", {"type": str}),
                ("--start-datetime", {"type": str}),
                ("--end-datetime", {"type": str}),
            ],
        },
    ]

    for spec in command_specs:
        p = subs.add_parser(spec["name"], help=spec["help"])
        for arg, kwargs in spec.get("positional", []):
            p.add_argument(arg, **kwargs)
        for arg, kwargs in spec.get("options", []):
            p.add_argument(arg, **kwargs)
        p.set_defaults(func=spec["func"])
    return parser


def main() -> None:
    parser = create_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
