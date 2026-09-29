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

from .conftest import create_zarrv2_group_from_json, get_stem, s2_example_json_paths
from .test_generic_rechunker import (
    build_synthetic_tree,
    leaf_groups_with_data,
    open_capped_s1_slc_example,
    read_array,
    s1_slc_example_json_paths,
)
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
    with pytest.raises(
        ValueError,
        match="Only one of s2_optimized, s3_olci_optimized and generic_rechunker",
    ):
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
        generic_rechunker=None,
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
        generic_rechunker=None,
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


def _tree_with_product_type(product_type: str) -> xr.DataTree:
    tree = xr.DataTree()
    tree.attrs = {"stac_discovery": {"properties": {"product:type": product_type}}}
    return tree


@pytest.mark.parametrize(
    ("product_type", "expected"),
    [
        ("S02MSIL2A", "s2-optimized"),
        ("S03OLCEFR", "s3-olci-optimized"),
        ("S01SIWSLC", "generic_rechunker"),
    ],
)
def test_resolve_forced_pipeline_generic_rechunker(product_type: str, expected: str) -> None:
    """generic_rechunker=True keeps S2 and OLCI on their optimized pipelines."""
    resolved = GeoZarrWriter._resolve_forced_pipeline(
        _tree_with_product_type(product_type),
        generic_rechunker=True,
        s2_optimized=None,
        s3_olci_optimized=None,
    )
    assert resolved == expected


@pytest.mark.parametrize(
    ("generic_rechunker", "s2_optimized", "s3_olci_optimized"),
    [(True, True, None), (True, None, True), (True, True, True)],
)
def test_resolve_forced_pipeline_rejects_generic_rechunker_with_other_forced(
    generic_rechunker: bool | None, s2_optimized: bool | None, s3_olci_optimized: bool | None
) -> None:
    with pytest.raises(ValueError, match="Only one of"):
        GeoZarrWriter._resolve_forced_pipeline(
            xr.DataTree(),
            generic_rechunker=generic_rechunker,
            s2_optimized=s2_optimized,
            s3_olci_optimized=s3_olci_optimized,
        )


def test_write_generic_rechunker_end_to_end(tmp_path: pathlib.Path) -> None:
    """generic_rechunker=True converts every data group without the 'groups' option."""
    target = tmp_path / "out.zarr"

    write_datatree(
        build_synthetic_tree(),
        target,
        engine=ENGINE_NAME,
        generic_rechunker=True,
        spatial_chunk=128,
    )

    assert read_array(target, "measurements/amplitude").chunks == (128, 128)
    assert "incidence_angle" in zarr.open_group(str(target), mode="r")["conditions/geometry"]


def test_write_generic_rechunker_default_spatial_chunk(tmp_path: pathlib.Path) -> None:
    """Without spatial_chunk, the generic_rechunker pipeline chunks at 1024."""
    tree = xr.DataTree()
    tree["measurements"] = xr.Dataset({"var": (("y", "x"), np.zeros((1500, 20), "float32"))})
    target = tmp_path / "out.zarr"

    GeoZarrWriter().write(tree, target, generic_rechunker=True)

    assert read_array(target, "measurements/var").chunks == (1024, 20)


def test_write_generic_rechunker_mode_w_replaces_target(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "out.zarr"
    target.mkdir()
    stale = target / "previous-product-content"
    stale.touch()

    GeoZarrWriter().write(build_synthetic_tree(), target, generic_rechunker=True, mode="w")

    assert not stale.exists()
    assert "amplitude" in zarr.open_group(str(target), mode="r")["measurements"]


def test_write_generic_rechunker_mode_w_dash_keeps_target(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "out.zarr"
    target.mkdir()
    existing = target / "previous-product-content"
    existing.touch()

    with pytest.raises(EOStoreProductAlreadyExistsError):
        GeoZarrWriter().write(build_synthetic_tree(), target, generic_rechunker=True, mode="w-")
    assert existing.exists()


@pytest.mark.filterwarnings("ignore:.*:UserWarning")
@pytest.mark.parametrize("source_path", s1_slc_example_json_paths, ids=get_stem)
def test_write_generic_rechunker_s1_slc(source_path: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """A real S1 SLC layout converts through the writer with its default options."""
    tree = open_capped_s1_slc_example(source_path, tmp_path)
    target = tmp_path / "out.zarr"

    write_datatree(tree, target, engine=ENGINE_NAME, generic_rechunker=True, spatial_chunk=32)

    root = zarr.open_group(str(target), mode="r")
    missing = [group for group in leaf_groups_with_data(tree) if group.lstrip("/") not in root]
    assert missing == []


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
