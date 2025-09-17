import argparse
import sys
from typing import Iterable, List, Tuple

import xarray as xr

from ..conversion.fs_utils import get_storage_options
from ._shared import resolve_input_path


def validate_command(args: argparse.Namespace) -> None:
    input_path = resolve_input_path(args.input_path)

    try:
        print(f"Validating GeoZarr compliance for: {input_path}")
        storage_options = get_storage_options(input_path)
        dt = xr.open_datatree(
            input_path, engine="zarr", chunks="auto", storage_options=storage_options
        )
        compliance_issues: List[str] = []
        total_variables = 0
        compliant_variables = 0
        print("\nValidation Results:")
        print("==================")

        def walk(node: xr.DataTree, path: str = "") -> Iterable[Tuple[str, xr.DataTree]]:
            # Depth-first traversal yielding (path, node)
            yield path, node
            children = getattr(node, "children", None)
            if children:
                for child_name, child in children.items():
                    child_path = f"{path}/{child_name}" if path else child_name
                    yield from walk(child, child_path)

        any_reported = False

        def check_var(var: xr.DataArray) -> List[str]:
            issues: List[str] = []
            attrs = var.attrs
            if "_ARRAY_DIMENSIONS" not in attrs:
                issues.append("Missing _ARRAY_DIMENSIONS attribute")
            if "standard_name" not in attrs:
                issues.append("Missing standard_name attribute")
            if "grid_mapping" not in attrs and "grid_mapping_name" not in attrs:
                issues.append("Missing grid_mapping attribute")
            return issues

        for path, node in walk(dt):
            try:
                ds = node.to_dataset()
            except Exception:
                continue
            if not len(ds.data_vars):
                continue
            any_reported = True
            print(f"\nGroup: /{path.lstrip('/')}")
            for var_name, var in ds.data_vars.items():
                if var_name in {"spatial_ref", "crs"}:
                    continue
                total_variables += 1
                issues = check_var(var)
                if issues:
                    print(f"  FAIL {var_name}: {', '.join(issues)}")
                    compliance_issues.extend(issues)
                else:
                    print(f"  OK   {var_name}: Compliant")
                    compliant_variables += 1

        print("\nSummary:")
        print("========")
        if not any_reported:
            print("No data variables found in any groups. Dataset may be metadata-only or empty.")
        print(f"Total variables checked: {total_variables}")
        print(f"Compliant variables: {compliant_variables}")
        print(f"Non-compliant variables: {total_variables - compliant_variables}")
        if compliance_issues:
            print("\nDataset is NOT GeoZarr compliant")
            print(f"Issues found: {len(compliance_issues)}")
            if args.verbose:
                print("Detailed issues:")
                for issue in set(compliance_issues):
                    print(f"  - {issue}")
        else:
            print("\nDataset appears to be GeoZarr compliant")
    except Exception as e:
        print(f"Error validating dataset: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
