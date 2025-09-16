from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

REMOTE_PREFIXES = ("http://", "https://", "s3://", "gs://")


def resolve_input_path(path_str: str) -> str:
    if path_str.startswith(REMOTE_PREFIXES):
        return path_str
    p = Path(path_str)
    if not p.exists():
        print(f"Error: Input path {p} does not exist")
        sys.exit(1)
    return str(p)


def prepare_output_path(
    path_str: str,
    is_s3_path: Callable[[str], bool],
    validate_s3_access: Callable[[str], Tuple[bool, Optional[str]]],
    get_s3_credentials_info: Callable[[], Dict[str, Optional[str]]],
    verbose: bool = False,
) -> str:
    if is_s3_path(path_str):
        print("Validating S3 access...")
        success, error_msg = validate_s3_access(path_str)
        if not success:
            print(f"Error: Cannot access S3 path {path_str}")
            print(f"   Reason: {error_msg}")
            print("\nS3 configuration help:")
            print("   Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION")
            print("   Set AWS_ENDPOINT_URL for custom providers or configure AWS CLI")
            if verbose:
                creds_info = get_s3_credentials_info()
                print("\nCurrent AWS configuration:")
                for key, value in creds_info.items():
                    print(f"   {key}: {value or 'Not set'}")
            sys.exit(1)
        print("S3 access validated successfully")
        return path_str
    out = Path(path_str)
    out.parent.mkdir(parents=True, exist_ok=True)
    return str(out)
