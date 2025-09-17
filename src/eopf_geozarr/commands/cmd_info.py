import argparse
import sys

import xarray as xr

from ..conversion.fs_utils import get_storage_options
from ._shared import resolve_input_path
from .html_utils import generate_html_output


def info_command(args: argparse.Namespace) -> None:
    input_path = resolve_input_path(args.input_path)

    try:
        print(f"Loading dataset from: {input_path}")
        storage_options = get_storage_options(input_path)
        dt = xr.open_datatree(
            input_path, engine="zarr", chunks="auto", storage_options=storage_options
        )
        if getattr(args, "html_output", None):
            generate_html_output(dt, args.html_output, input_path, args.verbose)
        else:
            print("\nDataset Information:")
            print("==================")
            print(f"Total groups: {len(dt.children)}")
            print("\nGroup structure:")
            print(dt)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
