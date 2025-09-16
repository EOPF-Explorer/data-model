from __future__ import annotations

import asyncio
from typing import Any, Optional, Union

import zarr
from zarr.storage import StoreLike

from . import fs_utils


def consolidate_metadata(
    store: StoreLike,
    path: Optional[str] = None,
    zarr_format: Optional[Any] = None,
) -> zarr.Group:
    """Consolidate metadata using public Zarr APIs (v2/v3 friendly).

    Attempts to use zarr.convenience.consolidate_metadata or zarr.consolidate_metadata.
    If unavailable or failing, logs a warning and returns an opened group.
    """
    try:
        _consolidate: Any = None
        try:
            from zarr.convenience import consolidate_metadata as _cm

            _consolidate = _cm
        except Exception:
            _consolidate = getattr(zarr, "consolidate_metadata", None)
        if callable(_consolidate):
            try:
                _consolidate(store, path=path)
            except TypeError:
                _consolidate(store)
        else:
            print("⚠️  Consolidation API not available; skipping consolidation")
    except Exception as e:  # pragma: no cover
        print(f"⚠️  Consolidation failed: {e}")

    # Return a group handle for callers expecting it
    try:
        return zarr.open_group(store, mode="r+", zarr_format=3)
    except Exception:
        return zarr.open_group(store, mode="r+")


async def async_consolidate_metadata(
    target: Union[str, StoreLike, zarr.Group],
    path: Optional[str] = None,
    zarr_format: Optional[Any] = None,
    **kwargs: Any,
) -> zarr.Group:
    """Async wrapper for consolidate_metadata (backwards compatible).

    Accepts a path string, a StoreLike, or a zarr.Group. Any extra kwargs are ignored
    for compatibility with previous signatures that accepted storage options.
    """

    def _resolve_store() -> StoreLike:
        if hasattr(target, "store"):
            return getattr(target, "store")
        if isinstance(target, str):
            group_path = (
                fs_utils.normalize_path(target)
                if not path
                else fs_utils.normalize_path(f"{target.rstrip('/')}/{path.lstrip('/')}")
            )
            zgroup = fs_utils.open_zarr_group(group_path, mode="r+")
            return zgroup.store
        return target  # assume StoreLike

    store_like = _resolve_store()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: consolidate_metadata(store_like, path=path, zarr_format=zarr_format),
    )
