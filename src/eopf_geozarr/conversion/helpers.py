from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol, Set

import xarray as xr

from . import utils


def iter_str(keys: Any) -> list[str]:
    """Return list of string keys from an iterable of Hashable items.

    xarray exposes mapping-like objects whose iteration type is Hashable; we only
    operate on string variable names, so cast/filter accordingly for type safety.
    """
    out: list[str] = []
    try:
        for k in keys:  # runtime iteration; filter for str
            if isinstance(k, str):
                out.append(k)
    except Exception:
        pass
    return out


def normalize_group_path(group: str) -> str:
    if group in {"", "."}:
        return "/"
    return group if group.startswith("/") else f"/{group}"


def build_var_encodings(ds: xr.Dataset, compressor: Any | None) -> Dict[str, Dict[str, Any]]:
    if compressor is None:
        return {}

    def enc_for(var: str) -> Dict[str, Any]:
        base: Dict[str, Any] = {"compressors": [compressor]}
        try:
            base["dtype"] = str(ds[var].dtype)
        except Exception:
            pass
        return base

    return {
        var: enc_for(var)
        for var in iter_str(ds.data_vars)
        if not utils.is_grid_mapping_variable(ds, var)
    }


def safely(
    label: str,
    func: Callable[[], Any],
    *,
    verbose: bool = False,
    swallow: bool = True,
) -> Any:
    try:
        return func()
    except Exception as e:  # pragma: no cover
        if verbose:
            print(f"Warning: {label} failed: {e}")
        if not swallow:
            raise
        return None


class _CRSPreparer(Protocol):
    def __call__(self, ds: xr.Dataset, *, reference_crs: Any) -> xr.Dataset: ...


def inject_crs_if_requested(
    group: str,
    ds: xr.Dataset,
    requested: Optional[Set[str]],
    reference_crs_finder: Callable[[], Any],
    preparer: _CRSPreparer,
    *,
    verbose: bool = False,
) -> xr.Dataset:
    if not requested:
        return ds
    norm = normalize_group_path(group)
    if norm not in requested:
        return ds
    try:
        if verbose:
            print(f"Adding CRS information to group: {norm}")
        ref_crs = reference_crs_finder()
        if verbose:
            print("Inferred reference CRS from measurements:", ref_crs)
        return preparer(ds, reference_crs=ref_crs)
    except Exception as e:  # pragma: no cover
        if verbose:
            print(f"Warning: CRS preparation skipped for {norm}: {e}")
        return ds


class MaybeStep:
    def __init__(self, recorder: Any, name: str):  # recorder is external metrics-like object
        self.rec: Any = recorder
        self.name: str = name
        self.ctx: Any | None = None

    def __enter__(self) -> "MaybeStep":  # pragma: no cover - trivial
        if self.rec:
            try:
                self.ctx = self.rec.time_step(self.name)
                self.ctx.__enter__()
            except Exception:
                self.ctx = None
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:  # pragma: no cover - trivial
        if self.ctx is not None:
            try:
                self.ctx.__exit__(exc_type, exc, tb)
            except Exception:
                pass
        return None


def maybe_step(recorder: Any, name: str) -> MaybeStep:  # factory
    return MaybeStep(recorder, name)
