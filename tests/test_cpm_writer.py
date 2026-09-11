"""Tests for the CPM writer plugin (eopf_geozarr.cpm.writer).

These tests require the ``eopf`` package (the eopf-cpm distribution) and are
skipped when it is not installed. Run them with the ``cpm`` extra, e.g.::

    uv run --python 3.13 --extra cpm --group test pytest tests/test_cpm_writer.py
"""

# eopf is an optional dependency (the `cpm` extra, Python >= 3.13 only), so it
# is absent from the default type-checking environment.
# pyright: reportMissingImports=false

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr
import zarr

pytest.importorskip("eopf")

from eopf.exceptions.errors import EOStoreProductAlreadyExistsError
from eopf.store.writer_dispatcher import get_writer_by_name, write_datatree
from eopf.store.writer_registry import EOWriterRegistry

from eopf_geozarr.cpm.writer import ENGINE_NAME, GeoZarrWriter, get_cli_command

from .conftest import create_zarrv2_group_from_json, s2_example_json_paths
from .test_olci_integration import build_synthetic_olci


def test_engine_registered() -> None:
    """Importing the plugin registers the engine, retrievable by name only."""
    assert EOWriterRegistry.contains(ENGINE_NAME)
    assert get_writer_by_name(ENGINE_NAME) is GeoZarrWriter
    # By-target discovery must stay with cpm_zarr; the geozarr engine is name-only.
    assert not EOWriterRegistry.is_discoverable_by_target(ENGINE_NAME)


def test_get_cli_command_builds_click_command() -> None:
    """The eopf.cli entry-point hook returns a click command."""
    import click

    command = get_cli_command()
    assert isinstance(command, click.Command)
    assert command.name == "convert-geozarr"


def test_write_s2_end_to_end(tmp_path: pathlib.Path) -> None:
    """A zarr-backed S2 tree written via CPM's dispatcher yields the flat pyramid."""
    source = create_zarrv2_group_from_json(s2_example_json_paths[0], tmp_path / "source")
    dtree = xr.open_datatree(source, engine="zarr", chunks="auto")
    target = tmp_path / "output.zarr"

    write_datatree(dtree, target, engine=ENGINE_NAME)

    root = zarr.open_group(str(target), mode="r")
    reflectance = root["measurements/reflectance"]
    for resolution in ("r10m", "r20m", "r60m", "r120m", "r360m", "r720m"):
        assert resolution in reflectance, f"missing pyramid level {resolution}"
    assert "multiscales" in reflectance.attrs
    # Store-root summary footprint written by the S2 pipeline.
    assert "spatial:bbox" in root.attrs


def test_write_olci_end_to_end(tmp_path: pathlib.Path) -> None:
    """An in-memory OLCI tree (no zarr backend) is auto-detected and written.

    Uses ``build_synthetic_olci``, unbacked by any zarr store -- the same
    shape the CPM SAFE reader hands writers -- to exercise both the
    structural-fallback routing (no stac_discovery attrs are set) and the
    "no backing store" path the S2 pipeline already has to tolerate.

    1024x1024 with the default min_dimension=256 yields r0 + two overview
    levels (see test_olci_integration.py::test_convert_olci_creates_overviews);
    the default 512x480 fixture size yields only r0.
    """
    dtree = build_synthetic_olci(rows=1024, cols=1024)
    target = tmp_path / "output.zarr"

    write_datatree(dtree, target, engine=ENGINE_NAME)

    root = zarr.open_group(str(target), mode="r")
    measurements = root["measurements"]
    for level in ("r0", "r2", "r4"):
        assert level in measurements, f"missing pyramid level {level}"
    assert "multiscales" in measurements.attrs
    # Unlike the S2 pipeline, native-mode OLCI output has no projected CRS,
    # so there is no store-root spatial:bbox to assert here.


def test_write_olci_forced_pipeline(tmp_path: pathlib.Path) -> None:
    """s3_olci_optimized=True selects the OLCI pipeline explicitly."""
    dtree = build_synthetic_olci()
    target = tmp_path / "output.zarr"

    write_datatree(dtree, target, engine=ENGINE_NAME, s3_olci_optimized=True)

    root = zarr.open_group(str(target), mode="r")
    assert "r0" in root["measurements"]


def test_write_rejects_both_s2_and_olci_forced(tmp_path: pathlib.Path) -> None:
    """s2_optimized=True and s3_olci_optimized=True together is a usage error."""
    with pytest.raises(ValueError, match="cannot both be True"):
        GeoZarrWriter().write(
            xr.DataTree(),
            tmp_path / "out.zarr",
            s2_optimized=True,
            s3_olci_optimized=True,
        )


def test_resolve_forced_pipeline_olci_suppressed_without_s2_structure() -> None:
    """s3_olci_optimized=False on an OLCI-only tree falls back to generic (no S2 shape)."""
    tree = xr.DataTree()
    tree.attrs = {"stac_discovery": {"properties": {"product:type": "S03OLCEFR"}}}
    resolved = GeoZarrWriter._resolve_forced_pipeline(
        tree,
        s2_optimized=None,
        s3_olci_optimized=False,
    )
    assert resolved == "generic"


def test_resolve_forced_pipeline_olci_suppressed_with_s2_structure() -> None:
    """s3_olci_optimized=False falls back to s2-optimized when the tree also has S2 shape.

    Contrived (a real product would not carry both structures at once), but
    exercises the branch directly: with no declared product:type, structural
    detection is what select_pipeline itself would use, and S2 is checked
    ahead of OLCI there too.
    """
    tree = xr.DataTree()
    ds = xr.Dataset({"b01": (["y", "x"], np.zeros((2, 2)))})
    for resolution in ("r10m", "r20m", "r60m"):
        tree[f"measurements/reflectance/{resolution}"] = ds
    # oa01_radiance as a data variable directly on "measurements" (not a
    # child node), alongside the "reflectance" child group added above.
    tree["measurements"].dataset = xr.Dataset(
        {"oa01_radiance": (["y", "x"], np.zeros((2, 2)))},
    )
    resolved = GeoZarrWriter._resolve_forced_pipeline(
        tree,
        s2_optimized=None,
        s3_olci_optimized=False,
    )
    assert resolved == "s2-optimized"


def test_cli_convert_geozarr_end_to_end(tmp_path: pathlib.Path) -> None:
    """`eopf convert-geozarr` converts a product through CPM's convert() machinery.

    Exercises the whole operator-facing path: entry-point discovery in CPM's
    CLI, the click option wiring, convert()'s reader dispatch (cpm_zarr source),
    and the geozarr engine selected by name.
    """
    source = create_zarrv2_group_from_json(s2_example_json_paths[0], tmp_path / "source")
    target = tmp_path / "out.zarr"
    # The eopf console script installed next to the current interpreter.
    eopf_cli = pathlib.Path(sys.executable).parent / "eopf"
    assert eopf_cli.exists(), "eopf console script not found in the test environment"

    result = subprocess.run(
        [str(eopf_cli), "convert-geozarr", str(source), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    root = zarr.open_group(str(target), mode="r")
    reflectance = root["measurements/reflectance"]
    for resolution in ("r10m", "r20m", "r60m", "r120m", "r360m", "r720m"):
        assert resolution in reflectance, f"missing pyramid level {resolution}"
    assert "multiscales" in reflectance.attrs
    assert "spatial:bbox" in root.attrs


def test_write_generic_requires_groups(tmp_path: pathlib.Path) -> None:
    """The generic pipeline fails loudly when 'groups' is not provided."""
    tree = xr.DataTree()
    tree["measurements"] = xr.Dataset({"var": (["y", "x"], np.zeros((4, 4)))})
    with pytest.raises(ValueError, match="groups"):
        GeoZarrWriter().write(tree, tmp_path / "out.zarr", s2_optimized=False)


def test_write_option_errors_do_not_destroy_existing_target(tmp_path: pathlib.Path) -> None:
    """Argument-derivable failures must be raised before the mode='w' store removal."""
    target = tmp_path / "out.zarr"
    target.mkdir()
    sentinel = target / "previous-product-content"
    sentinel.touch()
    tree = xr.DataTree()
    tree["measurements"] = xr.Dataset({"var": (["y", "x"], np.zeros((4, 4)))})
    # Missing 'groups' on the generic pipeline: must fail with the old store intact.
    with pytest.raises(ValueError, match="groups"):
        GeoZarrWriter().write(tree, target, s2_optimized=False)
    assert sentinel.exists()


def test_write_accepts_compute_true(tmp_path: pathlib.Path) -> None:
    """CPM's staged-output path injects compute=True; the writer must accept it."""
    writer = GeoZarrWriter()
    writer.validate_write_options(tmp_path / "out.zarr", compute=True)
    # And through write(): compute=True with a missing-groups error means the
    # option itself was accepted (rejection would raise NotImplementedError first).
    tree = xr.DataTree()
    tree["measurements"] = xr.Dataset({"var": (["y", "x"], np.zeros((4, 4)))})
    with pytest.raises(ValueError, match="groups"):
        writer.write(tree, tmp_path / "out.zarr", s2_optimized=False, compute=True)


def test_write_rejects_compute_false(tmp_path: pathlib.Path) -> None:
    with pytest.raises(NotImplementedError, match="compute"):
        GeoZarrWriter().write(xr.DataTree(), tmp_path / "out.zarr", compute=False)


def test_write_rejects_zarr_format_2(tmp_path: pathlib.Path) -> None:
    with pytest.raises(NotImplementedError, match="Zarr format 3"):
        GeoZarrWriter().write(xr.DataTree(), tmp_path / "out.zarr", zarr_format=2)


def test_write_rejects_unconsolidated(tmp_path: pathlib.Path) -> None:
    with pytest.raises(NotImplementedError, match="consolidated"):
        GeoZarrWriter().write(xr.DataTree(), tmp_path / "out.zarr", consolidated=False)


def test_write_rejects_unsupported_mode(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="mode"):
        GeoZarrWriter().write(xr.DataTree(), tmp_path / "out.zarr", mode="a")


def test_write_rejects_unknown_options(tmp_path: pathlib.Path) -> None:
    with pytest.raises(NotImplementedError, match="storage_options"):
        GeoZarrWriter().write(xr.DataTree(), tmp_path / "out.zarr", storage_options={"anon": True})


def test_write_rejects_remote_target() -> None:
    with pytest.raises(NotImplementedError, match="stage_target"):
        GeoZarrWriter().write(xr.DataTree(), "s3://bucket/out.zarr")


def test_write_rejects_non_path_target() -> None:
    with pytest.raises(TypeError, match="str or Path"):
        GeoZarrWriter().write(xr.DataTree(), {"not": "a path"})


def test_write_mode_w_dash_refuses_existing_target(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "out.zarr"
    target.mkdir()
    with pytest.raises(EOStoreProductAlreadyExistsError):
        GeoZarrWriter().write(xr.DataTree(), target, mode="w-")


def test_validate_write_options_checks_without_writing(tmp_path: pathlib.Path) -> None:
    """validate_write_options raises the same errors as write, touching nothing."""
    writer = GeoZarrWriter()
    target = tmp_path / "out.zarr"
    with pytest.raises(NotImplementedError, match="Zarr format 3"):
        writer.validate_write_options(target, zarr_format=2)
    with pytest.raises(NotImplementedError, match="storage_options"):
        writer.validate_write_options(target, storage_options={"anon": True})
    with pytest.raises(NotImplementedError, match="stage_target"):
        writer.validate_write_options("s3://bucket/out.zarr")
    with pytest.raises(TypeError, match="str or Path"):
        writer.validate_write_options({"not": "a path"})
    writer.validate_write_options(target, mode="w", spatial_chunk=512)
    assert not target.exists()
