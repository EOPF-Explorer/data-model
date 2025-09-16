import os
from typing import Any, Dict, Tuple

import xarray as xr

from . import utils
from .helpers import iter_str


def create_geozarr_encoding(
    ds: xr.Dataset, compressor: Any, spatial_chunk: int
) -> Dict[str, Dict[str, Any]]:
    """Create encoding for GeoZarr dataset variables (spec-aware)."""
    encoding: Dict[str, Dict[str, Any]] = {}
    # Optional safety cap for chunk bytes to avoid OOM. Default ~8 MiB unless overridden.
    try:
        max_chunk_bytes = int(os.environ.get("EOPF_MAX_CHUNK_BYTES", str(8 * 1024 * 1024)))
    except Exception:
        max_chunk_bytes = 8 * 1024 * 1024
    for var in iter_str(ds.data_vars):
        if utils.is_grid_mapping_variable(ds, var):
            encoding[var] = {"compressors": None}
        else:
            data_shape = ds[var].shape
            dtype_size = getattr(ds[var].dtype, "itemsize", 1) or 1
            if len(data_shape) >= 2:
                height, width = data_shape[-2:]
                spatial_chunk_aligned = min(
                    spatial_chunk,
                    utils.calculate_aligned_chunk_size(width, spatial_chunk),
                    utils.calculate_aligned_chunk_size(height, spatial_chunk),
                )
            else:
                spatial_chunk_aligned = spatial_chunk

            # Build chunk tuple matching variable dimensionality.
            # Use 1 for all leading (non-spatial) dims, and spatial_chunk_aligned for the last two.
            if len(data_shape) == 1:
                chunks: Tuple[int, ...] = (min(spatial_chunk_aligned, data_shape[0]),)
            elif len(data_shape) == 2:
                chunks = (spatial_chunk_aligned, spatial_chunk_aligned)
            else:
                leading: Tuple[int, ...] = tuple(1 for _ in range(len(data_shape) - 2))
                chunks = leading + (spatial_chunk_aligned, spatial_chunk_aligned)

            # Enforce max_chunk_bytes by reducing spatial chunk if needed
            # Estimate total bytes per chunk as product(chunks) * dtype_size
            from math import prod as _prod

            est_bytes = _prod(chunks) * dtype_size
            if max_chunk_bytes and est_bytes > max_chunk_bytes and len(chunks) >= 1:
                # Reduce spatial chunks proportionally (keep leading dims as-is)
                lead = chunks[:-2] if len(chunks) > 2 else tuple()
                yc, xc = (chunks[-2], chunks[-1]) if len(chunks) >= 2 else (chunks[-1], 1)
                factor = (est_bytes / max_chunk_bytes) ** 0.5
                new_y = max(1, int(yc / factor))
                new_x = max(1, int(xc / factor))
                chunks = lead + (new_y, new_x) if len(chunks) >= 2 else (new_y,)

            encoding[var] = {
                "chunks": chunks,
                "compressors": ([compressor] if compressor is not None else None),
            }

    for coord in iter_str(ds.coords):
        encoding[coord] = {"compressors": None}

    return encoding
