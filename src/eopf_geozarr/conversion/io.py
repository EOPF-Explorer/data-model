from __future__ import annotations

from typing import Optional

import xarray as xr

from . import fs_utils


def load_existing_dataset(path: str) -> Optional[xr.Dataset]:
    """Load existing dataset if it exists.

    Mirrors previous _load_existing_dataset behavior but with a public name.
    """
    try:
        if fs_utils.path_exists(path):
            storage_options = fs_utils.get_storage_options(path)
            return xr.open_dataset(
                path,
                zarr_format=3,
                storage_options=storage_options,
                engine="zarr",
                chunks="auto",
                decode_coords="all",
                mask_and_scale=False,
                decode_cf=False,
            )
    except Exception as e:
        print(f"Warning: Could not open existing dataset at {path}: {e}")
    return None
