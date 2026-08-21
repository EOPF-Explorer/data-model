"""Open source EOPF zarr stores with dask chunks aligned to native chunks.

With ``chunks="auto"``, dask may split one on-disk zarr chunk into several
read tasks that each fetch and decompress the same object key — multiplying
egress and racing on shared cache files (EOPF-Explorer/data-pipeline#339).
``open_source_datatree`` opens with ``chunks={}`` (the store's native chunk
grid), so each zarr chunk is read by exactly one dask task; downstream
conversion code rechunks to its own output encoding anyway. Note that one
task then materializes one whole native chunk (~241 MB raw for cpm_v270
aot/wvp) — under tight memory budgets, lower dask task concurrency rather
than sub-splitting stored chunks.
"""

import hashlib
from pathlib import Path
from typing import Any

import xarray as xr

from .fs_utils import S3FsOptions, get_storage_options


def open_source_datatree(
    path: str,
    *,
    storage_options: S3FsOptions | dict[str, Any] | None = None,
    cache_dir: str | None = None,
    engine: str = "zarr",
    mask_and_scale: bool = True,
) -> xr.DataTree:
    """Open a source datatree with dask chunks matching the native zarr chunks.

    Parameters
    ----------
    path : str
        Source store URL (local path, s3://, or https://).
    storage_options : dict, optional
        fsspec storage options. Defaults to ``get_storage_options(path)``
        (S3 credentials/endpoint for s3:// paths, None otherwise).
    cache_dir : str, optional
        Local directory for an on-disk read cache of source objects, backed
        by zarr's ``CacheStore`` over a ``LocalStore``. LocalStore writes are
        atomic (temp file + rename), so a concurrent read never sees a
        partially downloaded object. Entries are namespaced per source URL
        (CacheStore keys entries by relative zarr key, so different sources
        sharing a directory would otherwise collide) and never expire; use an
        ephemeral directory.
    engine : str
        xarray backend engine, default ``"zarr"``.
    mask_and_scale : bool
        Forwarded to ``xr.open_datatree``. Pass ``False`` to keep variables
        packed (e.g. uint16 with CF scale_factor/add_offset in ``.attrs``)
        instead of CF-decoding to floats, as the OLCI converter requires.

    Returns
    -------
    xr.DataTree
        Datatree whose dask arrays are chunked exactly on the store's native
        chunk grid: each on-disk zarr chunk is read by exactly one dask task.
    """
    if storage_options is None:
        storage_options = get_storage_options(path)
    if cache_dir is None:
        return xr.open_datatree(
            path,
            engine=engine,
            chunks={},
            storage_options=storage_options,
            mask_and_scale=mask_and_scale,
        )
    # Imported lazily: CacheStore is zarr's experimental API, so a relocation
    # in a future zarr release must not break importing this package.
    try:
        from zarr.experimental.cache_store import CacheStore
    except ImportError as exc:
        raise ImportError(
            "cache_dir requires zarr.experimental.cache_store.CacheStore "
            "(present in zarr 3.2.x). Your zarr version no longer provides "
            "it; pin zarr accordingly or open without cache_dir."
        ) from exc
    from zarr.storage import FsspecStore, LocalStore

    source = FsspecStore.from_url(
        path,
        storage_options=dict(storage_options) if storage_options else None,
        read_only=True,
    )
    # CacheStore keys entries by relative zarr key ("aot/0.0" is the same key
    # for every S2 product), so namespace the cache per source URL. Trailing
    # slashes are stripped so equivalent URLs share a namespace.
    cache_key = hashlib.sha256(path.rstrip("/").encode()).hexdigest()[:16]
    source_cache = Path(cache_dir) / cache_key
    cached = CacheStore(source, cache_store=LocalStore(source_cache))
    # xarray's open_datatree accepts a zarr store at runtime, but its stub does
    # not list Store among the accepted input types.
    return xr.open_datatree(
        cached,  # pyright: ignore[reportArgumentType]
        engine=engine,
        chunks={},
        mask_and_scale=mask_and_scale,
    )
