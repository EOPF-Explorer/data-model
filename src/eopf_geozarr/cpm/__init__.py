"""Integration of eopf-geozarr with the EOPF CPM (the ``eopf`` package).

This package provides the ``geozarr`` writer engine for CPM's writer
registry, plus an ``eopf convert-geozarr`` CLI command exposed through the
``eopf.cli`` entry-point group.

Importing :mod:`eopf_geozarr.cpm.writer` (or accessing ``GeoZarrWriter`` /
``register`` here) requires the ``eopf`` package; install it with the
``eopf-geozarr[cpm]`` extra. :mod:`eopf_geozarr.cpm.routing` is importable
without eopf-cpm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import click

    from eopf_geozarr.cpm.writer import GeoZarrWriter, register

#: Engine name used with ``write_datatree(..., engine=ENGINE_NAME)``.
ENGINE_NAME = "geozarr"

__all__ = ["ENGINE_NAME", "GeoZarrWriter", "get_cli_command", "register"]


def _import_writer() -> Any:
    """Import the writer module, translating a missing eopf into a clear error."""
    try:
        from eopf_geozarr.cpm import writer
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.split(".")[0] == "eopf":
            raise ImportError(
                "The eopf_geozarr.cpm integration requires the 'eopf' package "
                "(eopf-cpm), which supports Python >= 3.13 only. On Python 3.13+ "
                "install it with: pip install 'eopf-geozarr[cpm]'. Note that on "
                "older interpreters that command succeeds without installing eopf.",
            ) from exc
        raise
    return writer


def get_cli_command() -> click.Command:
    """Build the ``eopf convert-geozarr`` command (entry-point hook for CPM's CLI)."""
    return _import_writer().get_cli_command()


def __getattr__(name: str) -> Any:
    if name in {"GeoZarrWriter", "register"}:
        return getattr(_import_writer(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
