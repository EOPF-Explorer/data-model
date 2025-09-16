from __future__ import annotations

from typing import Any, Mapping

import xarray as xr


def write_dataset_zarr(
    ds: xr.Dataset,
    store_path: str,
    *,
    group: str | None = None,
    mode: str = "a",
    consolidated: bool = True,
    zarr_format: int = 3,
    encoding: Mapping[str, Mapping[str, Any]] | None = None,
    align_chunks: bool = True,
    safe_chunks: bool = False,
    storage_options: Mapping[str, Any] | None = None,
) -> None:
    """Typed thin wrapper around xarray.Dataset.to_zarr.

    Centralizes parameters and narrows types to avoid repeated mypy ignore noise
    caused by incomplete upstream type stubs for xarray/zarr interactions.
    """
    ds.to_zarr(
        store_path,
        group=group,
        mode=mode,
        consolidated=consolidated,
        zarr_format=zarr_format,
        encoding=encoding,  # already filtered by caller
        align_chunks=align_chunks,
        safe_chunks=safe_chunks,
        storage_options=storage_options,
    )
