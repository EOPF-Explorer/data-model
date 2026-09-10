"""Tests for S1 GRD RTC GeoTIFF → GeoZarr V3 ingestion pipeline."""

from __future__ import annotations

import datetime as dt
import json
import os
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr
from rasterio.transform import Affine, from_bounds
from zarr.core.metadata import ArrayV3Metadata

from eopf_geozarr.conversion.s1_ingest import (
    FLOAT32_NAN_FILL_VALUE,
    OVERVIEW_CHAIN,
    WRITER_SCHEMA,
    S1TilingMetadata,
    _create_spatial_coordinate_arrays,
    _downsample_2d,
    _normalise_s1tiling_datetime,
    consolidate_s1_store,
    create_s1_store,
    discover_s1tiling_acquisitions,
    discover_s1tiling_conditions,
    extract_geotiff_metadata,
    ingest_s1tiling_acquisition,
    ingest_s1tiling_conditions,
    parse_s1tiling_filename,
)
from eopf_geozarr.conversion.utils import calculate_aligned_chunk_size

if TYPE_CHECKING:
    from collections.abc import Mapping

# =============================================================================
# Constants
# =============================================================================

SIZE = 256
CRS = "EPSG:32633"
XMIN, YMIN, XMAX, YMAX = 500000.0, 4997440.0, 502560.0, 5000000.0
TRANSFORM = from_bounds(XMIN, YMIN, XMAX, YMAX, SIZE, SIZE)

ACQ1_TAGS = {
    "ACQUISITION_DATETIME": "2023:01:15T06:12:34Z",
    "ORBIT_NUMBER": "47001",
    "RELATIVE_ORBIT_NUMBER": "037",
    "FLYING_UNIT_CODE": "S1A",
    "CALIBRATION": "gamma_naught",
    "INPUT_S1_IMAGES": "S1A_IW_GRDH_1SDV_20230115",
}

ACQ2_TAGS = {
    "ACQUISITION_DATETIME": "2023:01:27T06:12:35Z",
    "ORBIT_NUMBER": "47177",
    "RELATIVE_ORBIT_NUMBER": "037",
    "FLYING_UNIT_CODE": "S1A",
    "CALIBRATION": "gamma_naught",
    "INPUT_S1_IMAGES": "S1A_IW_GRDH_1SDV_20230127",
}


# =============================================================================
# Helpers
# =============================================================================


def _group(node: zarr.Group, name: str) -> zarr.Group:
    """Narrow ``node[name]`` to a ``zarr.Group`` (zarr 3.x typing returns ``Array | Group``)."""
    member = node[name]
    assert isinstance(member, zarr.Group), f"{name!r} is not a group"
    return member


def _array(node: zarr.Group, name: str) -> zarr.Array:
    """Narrow ``node[name]`` to a ``zarr.Array`` (zarr 3.x typing returns ``Array | Group``)."""
    member = node[name]
    assert isinstance(member, zarr.Array), f"{name!r} is not an array"
    return member


def _dimension_names(arr: zarr.Array) -> tuple[str | None, ...] | None:
    """Read ``dimension_names`` off a zarr-format-3 array (``metadata`` is a V2/V3 union)."""
    metadata = arr.metadata
    assert isinstance(metadata, ArrayV3Metadata), "expected a zarr-format-3 array"
    return metadata.dimension_names


def _create_synthetic_geotiff(
    path: Path,
    data: np.ndarray,
    crs: str = CRS,
    transform: Affine | None = None,
    tags: dict[str, str] | None = None,
    nodata: float | None = None,
) -> None:
    """Write a single-band GeoTIFF with optional metadata tags and declared nodata."""
    if transform is None:
        transform = TRANSFORM
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        if tags:
            dst.update_tags(**tags)
        dst.write(data, 1)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def s1_geotiff_dir(tmp_path: Path) -> Path:
    """Create a directory with synthetic S1Tiling GeoTIFFs for 2 acquisitions."""
    rng = np.random.default_rng(42)

    for acq_idx, (stamp, tags) in enumerate(
        [("20230115t061234", ACQ1_TAGS), ("20230127t061235", ACQ2_TAGS)]
    ):
        vv_data = rng.uniform(0.0, 1.0, (SIZE, SIZE)).astype(np.float32) + acq_idx
        vh_data = rng.uniform(0.0, 0.5, (SIZE, SIZE)).astype(np.float32) + acq_idx
        mask_data = np.ones((SIZE, SIZE), dtype=np.uint8)
        mask_data[:10, :] = 0  # border region

        for pol, data in [("vv", vv_data), ("vh", vh_data)]:
            fname = f"s1a_32TQM_{pol}_ASC_037_{stamp}_GammaNaughtRTC.tif"
            _create_synthetic_geotiff(tmp_path / fname, data, tags=tags)

            mask_fname = f"s1a_32TQM_{pol}_ASC_037_{stamp}_GammaNaughtRTC_BorderMask.tif"
            _create_synthetic_geotiff(tmp_path / mask_fname, mask_data, tags=tags)

    return tmp_path


@pytest.fixture
def s1_store_path(tmp_path: Path) -> Path:
    """Return a clean path for Zarr store output."""
    return tmp_path / "s1-grd-rtc-test.zarr"


@pytest.fixture
def single_vv_geotiff(tmp_path: Path) -> Path:
    """Create a single VV GeoTIFF with metadata tags."""
    rng = np.random.default_rng(42)
    data = rng.uniform(0.0, 1.0, (SIZE, SIZE)).astype(np.float32)
    path = tmp_path / "test_vv.tif"
    _create_synthetic_geotiff(path, data, tags=ACQ1_TAGS)
    return path


# =============================================================================
# Step 9: Metadata extraction tests
# =============================================================================


class TestExtractGeotiffMetadata:
    def test_extracts_all_fields(self, single_vv_geotiff: Path) -> None:
        meta = extract_geotiff_metadata(single_vv_geotiff)
        assert isinstance(meta, S1TilingMetadata)
        assert meta.crs == CRS
        assert meta.shape == [SIZE, SIZE]
        assert len(meta.spatial_transform) == 6
        assert len(meta.bounds) == 4
        assert meta.absolute_orbit == 47001
        assert meta.relative_orbit == 37
        assert meta.platform == "S1A"
        assert meta.calibration == "gamma_naught"

    def test_normalises_datetime(self, single_vv_geotiff: Path) -> None:
        meta = extract_geotiff_metadata(single_vv_geotiff)
        # "2023:01:15T06:12:34Z" → "2023-01-15T06:12:34"
        assert meta.datetime == "2023-01-15T06:12:34"

    def test_raises_on_missing_tags(self, tmp_path: Path) -> None:
        data = np.zeros((SIZE, SIZE), dtype=np.float32)
        path = tmp_path / "no_tags.tif"
        _create_synthetic_geotiff(path, data, tags={})
        with pytest.raises(ValueError, match="missing required tags"):
            extract_geotiff_metadata(path)


class TestNormaliseDatetime:
    def test_s1tiling_format(self) -> None:
        assert _normalise_s1tiling_datetime("2025:02:10T06:09:20Z") == "2025-02-10T06:09:20"

    def test_already_normalised(self) -> None:
        assert _normalise_s1tiling_datetime("2023-01-15T06:12:34") == "2023-01-15T06:12:34"


class TestParseFilename:
    def test_vv_file(self) -> None:
        result = parse_s1tiling_filename("s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif")
        assert result is not None
        assert result["platform"] == "s1a"
        assert result["tile"] == "32TQM"
        assert result["pol"] == "vv"
        assert result["orbit_dir"] == "ASC"
        assert result["rel_orbit"] == "037"
        assert result["is_mask"] is False

    def test_mask_file(self) -> None:
        result = parse_s1tiling_filename(
            "s1a_32TQM_vh_ASC_037_20230115t061234_GammaNaughtRTC_BorderMask.tif"
        )
        assert result is not None
        assert result["pol"] == "vh"
        assert result["is_mask"] is True

    def test_masked_multiframe_time_stamp(self) -> None:
        """Multi-frame products carry a masked time (…txxxxxx); the parser must still match so
        the file isn't skipped (the real stamp is resolved later from the tag). See #183."""
        result = parse_s1tiling_filename("s1a_32TQM_vv_ASC_037_20230115txxxxxx_GammaNaughtRTC.tif")
        assert result is not None
        assert result["acq_stamp"] == "20230115txxxxxx"
        assert result["pol"] == "vv"

    def test_returns_none_for_unknown(self) -> None:
        assert parse_s1tiling_filename("random_file.tif") is None
        assert parse_s1tiling_filename("not_a_geotiff.txt") is None


# =============================================================================
# Step 10: Store creation tests
# =============================================================================


@pytest.fixture
def sample_metadata(single_vv_geotiff: Path) -> S1TilingMetadata:
    """Extract metadata from the single VV fixture."""
    return extract_geotiff_metadata(single_vv_geotiff)


class TestCreateStore:
    def test_structure(self, s1_store_path: Path, sample_metadata: S1TilingMetadata) -> None:
        root = create_s1_store(s1_store_path, "ascending", sample_metadata)
        assert "ascending" in root
        orbit = root["ascending"]
        for level_name, _, _ in OVERVIEW_CHAIN:
            assert level_name in orbit, f"Missing level {level_name}"

    def test_conventions(self, s1_store_path: Path, sample_metadata: S1TilingMetadata) -> None:
        root = create_s1_store(s1_store_path, "ascending", sample_metadata)
        attrs = dict(_group(root, "ascending").attrs)
        assert "zarr_conventions" in attrs
        conventions = attrs["zarr_conventions"]
        assert isinstance(conventions, list)
        conv_names = set()
        for conv in conventions:
            assert isinstance(conv, dict)
            conv_names.add(conv["name"])
        assert "multiscales" in conv_names
        assert "proj:" in conv_names
        assert "spatial:" in conv_names
        assert attrs["proj:code"] == CRS
        assert attrs["spatial:dimensions"] == ["y", "x"]
        bbox = attrs["spatial:bbox"]
        assert isinstance(bbox, list)
        assert len(bbox) == 4

    def test_array_metadata(self, s1_store_path: Path, sample_metadata: S1TilingMetadata) -> None:
        root = create_s1_store(s1_store_path, "ascending", sample_metadata)
        r10m = _group(_group(root, "ascending"), "r10m")
        for arr_name in ["vv", "vh", "border_mask"]:
            arr = _array(r10m, arr_name)
            assert _dimension_names(arr) == ("time", "y", "x")
            assert arr.shape[0] == 0  # time axis starts at 0
        assert _array(r10m, "vv").dtype == np.float32
        assert _array(r10m, "border_mask").dtype == np.uint8

    def test_float_bands_declare_cf_fill_value(
        self, s1_store_path: Path, sample_metadata: S1TilingMetadata
    ) -> None:
        """Float backscatter bands must declare a CF ``_FillValue`` attribute at
        *every* multiscale level, matching S2 (data-model #172 / xarray #11345).

        The zarr-level ``fill_value`` alone is not surfaced by xarray's encoding,
        so ``to_masked_array()`` / ``use_zarr_fill_value_as_mask=True`` cannot mask
        NaN nodata without the attribute. ``test_array_attrs`` only guards the S2
        ``/measurements/`` layout, so the S1 RTC layout was previously unchecked.
        """
        from xarray.backends.zarr import FillValueCoder

        root = create_s1_store(s1_store_path, "ascending", sample_metadata)
        orbit = _group(root, "ascending")
        expected = FillValueCoder.encode(np.nan, np.dtype("float32"))
        for level_name, _, _ in OVERVIEW_CHAIN:
            for band in ("vv", "vh"):
                attrs = dict(_array(_group(orbit, level_name), band).attrs)
                assert attrs.get("_FillValue") == expected, (
                    f"{level_name}/{band} missing/!= CF _FillValue"
                )
                assert (
                    attrs.get("standard_name")
                    == "surface_backwards_scattering_coefficient_of_radar_wave"
                )
                assert attrs.get("units") == "1"

    def test_no_tile_matrix_set(
        self, s1_store_path: Path, sample_metadata: S1TilingMetadata
    ) -> None:
        # tile_matrix_set is not part of the S1 GRD RTC data model (confirmed with the
        # data-model owner): the multiscales attribute must not carry one.
        root = create_s1_store(s1_store_path, "ascending", sample_metadata)
        ms = dict(_group(root, "ascending").attrs)["multiscales"]
        assert isinstance(ms, dict)
        assert "tile_matrix_set" not in ms
        assert "layout" in ms

    def test_cf_grid_mapping_resolves_crs(
        self, s1_store_path: Path, sample_metadata: S1TilingMetadata
    ) -> None:
        # Each resolution level carries a CF spatial_ref grid-mapping so rioxarray (and
        # TiTiler's GeoZarr reader) can resolve the CRS -- the geozarr proj:code attr
        # alone is not read by rioxarray.
        import rioxarray  # noqa: F401  -- registers the .rio accessor

        create_s1_store(s1_store_path, "ascending", sample_metadata)
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        r10m = _group(_group(root, "ascending"), "r10m")
        assert "spatial_ref" in list(r10m.array_keys())
        assert dict(_array(r10m, "vv").attrs).get("grid_mapping") == "spatial_ref"
        assert dict(_array(r10m, "vh").attrs).get("grid_mapping") == "spatial_ref"

        ds = xr.open_zarr(
            str(s1_store_path / "ascending" / "r10m"),
            consolidated=False,
            decode_coords="all",
        )
        assert ds.rio.crs is not None
        assert ds.rio.crs.to_epsg() == 32633

    def test_coordinate_variables(
        self, s1_store_path: Path, sample_metadata: S1TilingMetadata
    ) -> None:
        root = create_s1_store(s1_store_path, "ascending", sample_metadata)
        r10m = _group(_group(root, "ascending"), "r10m")
        for coord_name in ["time", "absolute_orbit", "relative_orbit", "platform"]:
            assert coord_name in r10m, f"Missing coord {coord_name}"
            assert _array(r10m, coord_name).shape == (0,)

    def test_overview_shapes(self, s1_store_path: Path, sample_metadata: S1TilingMetadata) -> None:
        root = create_s1_store(s1_store_path, "ascending", sample_metadata)
        orbit = _group(root, "ascending")
        # Verify shape chain follows ceiling division
        expected_h, expected_w = SIZE, SIZE
        for level_name, _, factor in OVERVIEW_CHAIN:
            if factor > 1:
                expected_h = ceil(expected_h / factor)
                expected_w = ceil(expected_w / factor)
            level = _group(orbit, level_name)
            arr = _array(level, "vv")
            assert arr.shape[1] == expected_h
            assert arr.shape[2] == expected_w

    def test_spatial_coordinate_arrays(
        self, s1_store_path: Path, sample_metadata: S1TilingMetadata
    ) -> None:
        """Verify x and y 1D arrays exist at every resolution level."""
        root = create_s1_store(s1_store_path, "ascending", sample_metadata)
        orbit = _group(root, "ascending")
        for level_name, _, _ in OVERVIEW_CHAIN:
            level = _group(orbit, level_name)
            for coord in ["x", "y"]:
                assert coord in level, f"Missing {coord} at {level_name}"
                arr = _array(level, coord)
                assert len(arr.shape) == 1
                attrs = dict(arr.attrs)
                assert "units" in attrs
                assert "standard_name" in attrs
                assert "_ARRAY_DIMENSIONS" in attrs

            # Verify x array shape matches level width
            level_attrs = dict(level.attrs)
            level_shape = level_attrs["spatial:shape"]
            assert isinstance(level_shape, list)
            level_h, level_w = level_shape
            assert _array(level, "x").shape[0] == level_w
            assert _array(level, "y").shape[0] == level_h


# =============================================================================
# Step 11: Ingestion tests
# =============================================================================


class TestIngestAcquisition:
    def _get_acq_paths(self, geotiff_dir: Path, stamp: str) -> tuple[Path, Path, Path]:
        """Get VV, VH, border mask paths for a given acquisition stamp."""
        vv = geotiff_dir / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC.tif"
        vh = geotiff_dir / f"s1a_32TQM_vh_ASC_037_{stamp}_GammaNaughtRTC.tif"
        mask = geotiff_dir / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC_BorderMask.tif"
        return vv, vh, mask

    def test_first_acquisition(self, s1_geotiff_dir: Path, s1_store_path: Path) -> None:
        vv, vh, mask = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        idx = ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")
        assert idx == 0
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        assert _array(_group(_group(root, "ascending"), "r10m"), "vv").shape[0] == 1

    def test_second_acquisition_appends(self, s1_geotiff_dir: Path, s1_store_path: Path) -> None:
        vv1, vh1, mask1 = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        vv2, vh2, mask2 = self._get_acq_paths(s1_geotiff_dir, "20230127t061235")
        ingest_s1tiling_acquisition(vv1, vh1, mask1, s1_store_path, "ascending")
        idx = ingest_s1tiling_acquisition(vv2, vh2, mask2, s1_store_path, "ascending")
        assert idx == 1
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        assert _array(_group(_group(root, "ascending"), "r10m"), "vv").shape[0] == 2

    def test_ingested_bands_declare_cf_fill_value(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """End-to-end (production path): vv/vh carry CF ``_FillValue``/``standard_name``/
        ``units`` at every level for BOTH the store-creating orbit (``create_s1_store``)
        and a second orbit added via the inline new-orbit path — the two paths that were
        previously inconsistent. Parity with S2 / S1 GRD (#172; xarray #11345).
        """
        from xarray.backends.zarr import FillValueCoder

        vv, vh, mask = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")  # create_s1_store
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "descending")  # inline new orbit
        expected = FillValueCoder.encode(np.nan, np.dtype("float32"))
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        for orbit in ("ascending", "descending"):
            for level_name, _, _ in OVERVIEW_CHAIN:
                for band in ("vv", "vh"):
                    attrs = dict(_array(_group(_group(root, orbit), level_name), band).attrs)
                    assert attrs.get("_FillValue") == expected, f"{orbit}/{level_name}/{band}"
                    assert (
                        attrs.get("standard_name")
                        == "surface_backwards_scattering_coefficient_of_radar_wave"
                    )
                    assert attrs.get("units") == "1"

    def test_new_orbit_level_groups_carry_proj_code(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """A second orbit added to an existing store must get the same per-level metadata
        as the store-creating orbit — incl. ``proj:code`` on every level group. The inline
        new-orbit path previously omitted it (drift vs ``create_s1_store``)."""
        vv, vh, mask = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "descending")
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        for orbit in ("ascending", "descending"):
            for level_name, _, _ in OVERVIEW_CHAIN:
                attrs = dict(_group(_group(root, orbit), level_name).attrs)
                assert attrs.get("proj:code") == CRS, f"{orbit}/{level_name} missing proj:code"

    def test_fill_value_masking_roundtrip(self, tmp_path: Path, s1_store_path: Path) -> None:
        """End-to-end: out-of-swath nodata (``border_mask == 0``) comes back masked when the cube
        is reopened with ``use_zarr_fill_value_as_mask=True`` — the behaviour the CF ``_FillValue``
        attribute exists to enable despite xarray #11345. Mirrors the S2 guarantee in
        ``tests/test_array_attrs.py::test_fill_value_masking_roundtrip``.

        The nodata region comes from ``border_mask``, not a pre-seeded NaN: s1tiling stores ``0.0``
        out of swath, and the writer is what must convert that to NaN.
        """
        stamp = "20230115t061234"
        rng = np.random.default_rng(0)
        vv_data = rng.uniform(0.1, 1.0, (SIZE, SIZE)).astype(np.float32)
        vh_data = rng.uniform(0.1, 0.5, (SIZE, SIZE)).astype(np.float32)
        mask_data = np.ones((SIZE, SIZE), dtype=np.uint8)
        mask_data[0:16, 0:16] = 0  # out-of-swath border
        vv_data[0:16, 0:16] = 0.0  # s1tiling stores 0 where there is no swath
        vh_data[0:16, 0:16] = 0.0
        vv = tmp_path / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC.tif"
        vh = tmp_path / f"s1a_32TQM_vh_ASC_037_{stamp}_GammaNaughtRTC.tif"
        mask = tmp_path / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC_BorderMask.tif"
        _create_synthetic_geotiff(vv, vv_data, tags=ACQ1_TAGS)
        _create_synthetic_geotiff(vh, vh_data, tags=ACQ1_TAGS)
        _create_synthetic_geotiff(mask, mask_data, tags=ACQ1_TAGS)
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")

        ds = xr.open_dataset(
            str(s1_store_path / "ascending" / "r10m"),
            engine="zarr",
            consolidated=False,
            decode_times=False,
            decode_coords=False,
            use_zarr_fill_value_as_mask=True,
        )
        try:
            masked = ds["vv"].to_masked_array()
            assert np.ma.is_masked(masked), "out-of-swath nodata must be masked via `_FillValue`"
            mask = masked.mask
            assert isinstance(mask, np.ndarray)
            assert mask[0, 0, 0], "nodata cell (border_mask==0) must be masked"
            assert not mask[0, -1, -1], "valid cell must not be masked"
        finally:
            ds.close()

    def test_nodata_masked_to_nan(self, tmp_path: Path, s1_store_path: Path) -> None:
        """The writer stores NaN — not 0 — wherever ``border_mask == 0``, at the native level and
        every overview, so titiler masks out-of-swath nodata transparent like the S2 reference.

        NaN must coincide exactly with ``border_mask == 0``: valid pixels stay finite. Root cause
        of the "black area" render bug: s1tiling writes 0 out of swath, and 0 is valid data to
        titiler. ``np.nanmean`` downsampling must carry the NaN to every overview level.
        """
        stamp = "20230115t061234"
        rng = np.random.default_rng(7)
        vv_data = rng.uniform(0.1, 1.0, (SIZE, SIZE)).astype(np.float32)
        vh_data = rng.uniform(0.1, 0.5, (SIZE, SIZE)).astype(np.float32)
        mask_data = np.ones((SIZE, SIZE), dtype=np.uint8)
        mask_data[0:32, :] = 0  # out-of-swath border band (whole rows)
        vv_data[0:32, :] = 0.0  # s1tiling stores 0 there — valid data to titiler today
        vh_data[0:32, :] = 0.0
        vv = tmp_path / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC.tif"
        vh = tmp_path / f"s1a_32TQM_vh_ASC_037_{stamp}_GammaNaughtRTC.tif"
        mask = tmp_path / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC_BorderMask.tif"
        _create_synthetic_geotiff(vv, vv_data, tags=ACQ1_TAGS)
        _create_synthetic_geotiff(vh, vh_data, tags=ACQ1_TAGS)
        _create_synthetic_geotiff(mask, mask_data, tags=ACQ1_TAGS)
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")

        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        asc = _group(root, "ascending")
        r10m = _group(asc, "r10m")
        nodata = mask_data == 0
        for band in ("vv", "vh"):
            native = np.asarray(_array(r10m, band)[0, :, :])
            assert np.all(np.isnan(native[nodata])), f"{band}: nodata region must be NaN, not 0"
            assert not np.any(np.isnan(native[~nodata])), f"{band}: valid region must stay finite"
            assert not np.any(native[nodata] == 0.0), f"{band}: nodata must not read back as 0"

        # NaN propagates through np.nanmean downsampling: the all-nodata top band stays NaN.
        coarse = np.asarray(_array(_group(asc, "r20m"), "vv")[0, :, :])
        assert np.isnan(coarse[0, 0]), "all-nodata block must stay NaN at the overview level"
        assert not np.any(np.isnan(coarse[-1, :])), "fully-valid bottom row must stay finite"

    def test_preserves_data_integrity(self, s1_geotiff_dir: Path, s1_store_path: Path) -> None:
        vv, vh, mask = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")

        # Read back and compare
        with rasterio.open(str(vv)) as src:
            expected_vv = src.read(1)
        with rasterio.open(str(mask)) as src:
            expected_mask = src.read(1).astype(np.uint8)
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        r10m = _group(_group(root, "ascending"), "r10m")
        actual_vv = np.asarray(_array(r10m, "vv")[0, :, :])

        # Valid pixels (border_mask == 1) are preserved exactly; out-of-swath pixels
        # (border_mask == 0) are written as NaN, not the raw 0 — the render-bug fix.
        valid = expected_mask == 1
        np.testing.assert_allclose(actual_vv[valid], expected_vv[valid], rtol=1e-6)
        assert np.all(np.isnan(actual_vv[~valid])), "out-of-swath nodata must be NaN"

        # border_mask itself is stored verbatim (uint8, never masked).
        actual_mask = _array(r10m, "border_mask")[0, :, :]
        np.testing.assert_array_equal(actual_mask, expected_mask)

    def test_coordinate_values(self, s1_geotiff_dir: Path, s1_store_path: Path) -> None:
        vv, vh, mask = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")

        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        r10m = _group(_group(root, "ascending"), "r10m")
        assert _array(r10m, "absolute_orbit")[0] == 47001
        assert _array(r10m, "relative_orbit")[0] == 37
        assert str(_array(r10m, "platform")[0]) == "S1A"

        # Verify time is a valid nanosecond timestamp (stored as int64)
        time_val = int(np.asarray(_array(r10m, "time"))[0])
        dt = np.datetime64(time_val, "ns")
        assert str(dt).startswith("2023-01-15")

    def test_overview_consistency(self, s1_geotiff_dir: Path, s1_store_path: Path) -> None:
        vv, vh, mask = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")

        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        orbit = _group(root, "ascending")
        expected_h, expected_w = SIZE, SIZE
        for level_name, _, factor in OVERVIEW_CHAIN:
            if factor > 1:
                expected_h = ceil(expected_h / factor)
                expected_w = ceil(expected_w / factor)
            arr = _array(_group(orbit, level_name), "vv")
            assert arr.shape == (1, expected_h, expected_w), (
                f"Shape mismatch at {level_name}: {arr.shape}"
            )

    def test_rejects_mismatched_crs(
        self, s1_geotiff_dir: Path, s1_store_path: Path, tmp_path: Path
    ) -> None:
        vv1, vh1, mask1 = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv1, vh1, mask1, s1_store_path, "ascending")

        # Create a GeoTIFF with different CRS
        data = np.ones((SIZE, SIZE), dtype=np.float32)
        wrong_crs_dir = tmp_path / "wrong_crs"
        wrong_crs_dir.mkdir()
        for name, d in [("vv.tif", data), ("vh.tif", data), ("mask.tif", data)]:
            _create_synthetic_geotiff(wrong_crs_dir / name, d, crs="EPSG:32632", tags=ACQ1_TAGS)

        with pytest.raises(ValueError, match="CRS mismatch"):
            ingest_s1tiling_acquisition(
                wrong_crs_dir / "vv.tif",
                wrong_crs_dir / "vh.tif",
                wrong_crs_dir / "mask.tif",
                s1_store_path,
                "ascending",
            )

    def test_rejects_mismatched_shape(
        self, s1_geotiff_dir: Path, s1_store_path: Path, tmp_path: Path
    ) -> None:
        vv1, vh1, mask1 = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv1, vh1, mask1, s1_store_path, "ascending")

        # Create GeoTIFFs with different shape
        wrong_shape_dir = tmp_path / "wrong_shape"
        wrong_shape_dir.mkdir()
        small_data = np.ones((128, 128), dtype=np.float32)
        small_transform = from_bounds(XMIN, YMIN, XMAX, YMAX, 128, 128)
        for name in ["vv.tif", "vh.tif", "mask.tif"]:
            _create_synthetic_geotiff(
                wrong_shape_dir / name,
                small_data,
                transform=small_transform,
                tags=ACQ1_TAGS,
            )

        with pytest.raises(ValueError, match="Shape mismatch"):
            ingest_s1tiling_acquisition(
                wrong_shape_dir / "vv.tif",
                wrong_shape_dir / "vh.tif",
                wrong_shape_dir / "mask.tif",
                s1_store_path,
                "ascending",
            )

    def test_xarray_roundtrip(self, s1_geotiff_dir: Path, s1_store_path: Path) -> None:
        vv1, vh1, mask1 = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        vv2, vh2, mask2 = self._get_acq_paths(s1_geotiff_dir, "20230127t061235")
        ingest_s1tiling_acquisition(vv1, vh1, mask1, s1_store_path, "ascending")
        ingest_s1tiling_acquisition(vv2, vh2, mask2, s1_store_path, "ascending")

        # Open r10m with xarray
        r10m_path = s1_store_path / "ascending" / "r10m"
        ds = xr.open_zarr(str(r10m_path))
        assert "vv" in ds
        assert ds["vv"].shape[0] == 2
        # Sort by time should work
        ds_sorted = ds.sortby("time")
        assert ds_sorted["vv"].shape[0] == 2


# =============================================================================
# Step 12b: Consolidation tests
# =============================================================================


class TestConsolidation:
    def test_consolidate_s1_store(self, s1_geotiff_dir: Path, s1_store_path: Path) -> None:
        vv, vh, mask = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")
        consolidate_s1_store(s1_store_path, "ascending")

        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        assert root.metadata.consolidated_metadata is not None
        orbit = _group(root, "ascending")
        assert orbit.metadata.consolidated_metadata is not None

    def test_consolidate_after_all_ingestions(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        vv1, vh1, mask1 = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        vv2, vh2, mask2 = self._get_acq_paths(s1_geotiff_dir, "20230127t061235")
        ingest_s1tiling_acquisition(vv1, vh1, mask1, s1_store_path, "ascending")
        ingest_s1tiling_acquisition(vv2, vh2, mask2, s1_store_path, "ascending")
        consolidate_s1_store(s1_store_path, "ascending")

        # Verify consolidated metadata reflects final shape (2 timesteps)
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        r10m = _group(_group(root, "ascending"), "r10m")
        assert _array(r10m, "vv").shape[0] == 2

    def test_consolidate_all_orbits_present(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """``consolidate_s1_store`` must leave EVERY orbit group consolidated on disk, not just the
        one passed. The pipeline ingests acquisitions one orbit at a time after stripping all
        consolidated metadata (so ``time`` can resize), so consolidating only the passed orbit left
        staging cubes asc✓/desc✗. Each orbit group is checked **standalone**: a consolidated root
        synthesises the child's view, so ``root[orbit].metadata.consolidated_metadata`` is a
        false-green (non-None even when ``<orbit>/zarr.json`` lacks it).
        """
        vv1, vh1, mask1 = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        vv2, vh2, mask2 = self._get_acq_paths(s1_geotiff_dir, "20230127t061235")
        ingest_s1tiling_acquisition(vv1, vh1, mask1, s1_store_path, "ascending")
        ingest_s1tiling_acquisition(vv2, vh2, mask2, s1_store_path, "descending")
        consolidate_s1_store(s1_store_path, "descending")  # only one orbit passed

        for orbit in ("ascending", "descending"):
            grp = zarr.open_group(str(s1_store_path / orbit), mode="r", zarr_format=3)
            assert grp.metadata.consolidated_metadata is not None, f"{orbit} orbit not consolidated"

    def _get_acq_paths(self, geotiff_dir: Path, stamp: str) -> tuple[Path, Path, Path]:
        vv = geotiff_dir / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC.tif"
        vh = geotiff_dir / f"s1a_32TQM_vh_ASC_037_{stamp}_GammaNaughtRTC.tif"
        mask = geotiff_dir / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC_BorderMask.tif"
        return vv, vh, mask


# =============================================================================
# Step 12: File discovery tests
# =============================================================================


class TestDiscoverAcquisitions:
    def test_groups_correctly(self, s1_geotiff_dir: Path) -> None:
        acqs = discover_s1tiling_acquisitions(s1_geotiff_dir)
        assert len(acqs) == 2
        # Each should have vv, vh, vv_mask, vh_mask
        for acq in acqs:
            assert "vv" in acq
            assert "vh" in acq
            assert "vv_mask" in acq
            assert "vh_mask" in acq

    def test_incomplete_is_skipped_by_default(self, tmp_path: Path) -> None:
        """A bundle the ingester cannot consume must not be returned.

        It used to be: `acquisitions.append(acq)` ran unconditionally after the warning, so a
        VV-only bundle reached `ingest_s1tiling_acquisition` and died on `KeyError: 'vh'` --
        the warning that should have prevented it having already been logged and ignored.
        """
        data = np.ones((SIZE, SIZE), dtype=np.float32)
        fname = "s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif"
        _create_synthetic_geotiff(tmp_path / fname, data, tags=ACQ1_TAGS)

        assert discover_s1tiling_acquisitions(tmp_path) == []

    def test_incomplete_is_returned_and_flagged_when_asked_for(self, tmp_path: Path) -> None:
        """`skip_incomplete=False` returns the bundle, flagged, for callers that triage."""
        data = np.ones((SIZE, SIZE), dtype=np.float32)
        fname = "s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif"
        _create_synthetic_geotiff(tmp_path / fname, data, tags=ACQ1_TAGS)

        acqs = discover_s1tiling_acquisitions(tmp_path, skip_incomplete=False)
        assert len(acqs) == 1
        assert acqs[0]["complete"] is False
        missing = [k for k in ("vh", "vv_mask", "vh_mask") if k not in acqs[0]]
        assert len(missing) == 3

    def test_missing_only_vh_mask_is_complete(self, tmp_path: Path) -> None:
        """`vh_mask` is discovered but never consumed, so requiring it condemned good bundles."""
        data = np.ones((SIZE, SIZE), dtype=np.float32)
        stem = "s1a_32TQM_{pol}_ASC_037_20230115t061234_GammaNaughtRTC{mask}.tif"
        for pol, mask in (("vv", ""), ("vh", ""), ("vv", "_BorderMask")):
            _create_synthetic_geotiff(
                tmp_path / stem.format(pol=pol, mask=mask), data, tags=ACQ1_TAGS
            )

        acqs = discover_s1tiling_acquisitions(tmp_path)
        assert len(acqs) == 1
        assert acqs[0]["complete"] is True
        assert "vh_mask" not in acqs[0]

    def test_skips_non_matching(self, tmp_path: Path) -> None:
        data = np.ones((SIZE, SIZE), dtype=np.float32)
        _create_synthetic_geotiff(tmp_path / "random_file.tif", data, tags=ACQ1_TAGS)
        acqs = discover_s1tiling_acquisitions(tmp_path)
        assert len(acqs) == 0

    def test_resolves_masked_multiframe_stamp_from_tag(self, tmp_path: Path) -> None:
        """Multi-frame products whose filename time is masked (…txxxxxx) must still be discovered
        as a complete acquisition, with acq_stamp resolved from the GeoTIFF ACQUISITION_DATETIME
        tag rather than the filename. Regression for #183 (previously returned 0 acquisitions)."""
        data = np.ones((SIZE, SIZE), dtype=np.float32)
        mask = np.ones((SIZE, SIZE), dtype=np.uint8)
        stamp = "20230115txxxxxx"  # masked multi-frame time
        for pol in ("vv", "vh"):
            base = f"s1a_32TQM_{pol}_ASC_037_{stamp}_GammaNaughtRTC"
            _create_synthetic_geotiff(tmp_path / f"{base}.tif", data, tags=ACQ1_TAGS)
            _create_synthetic_geotiff(tmp_path / f"{base}_BorderMask.tif", mask, tags=ACQ1_TAGS)

        acqs = discover_s1tiling_acquisitions(tmp_path)

        assert len(acqs) == 1
        acq = acqs[0]
        # ACQUISITION_DATETIME "2023:01:15T06:12:34Z" -> resolved stamp
        assert acq["acq_stamp"] == "20230115t061234"
        for k in ("vv", "vh", "vv_mask", "vh_mask"):
            assert k in acq

    def test_s3_uri_discovers_acquisitions(self) -> None:
        """s3:// prefix is listed via s3fs; pathlib.glob is NOT used."""
        s3_files = [
            "bucket/prefix/s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif",
            "bucket/prefix/s1a_32TQM_vh_ASC_037_20230115t061234_GammaNaughtRTC.tif",
            "bucket/prefix/s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC_BorderMask.tif",
            "bucket/prefix/s1a_32TQM_vh_ASC_037_20230115t061234_GammaNaughtRTC_BorderMask.tif",
        ]
        with patch("s3fs.S3FileSystem.glob", return_value=s3_files):
            acqs = discover_s1tiling_acquisitions("s3://bucket/prefix/")
        assert len(acqs) == 1
        expected_vv = "s3://bucket/prefix/s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif"
        assert acqs[0]["vv"] == expected_vv


# =============================================================================
# Phase 3: Conditions ingestion tests
# =============================================================================


@pytest.fixture
def s1_store_with_acquisition(s1_geotiff_dir: Path, tmp_path: Path) -> Path:
    """Create a Zarr store with one ingested acquisition (prerequisite for conditions)."""
    store_path = tmp_path / "s1-grd-rtc-cond.zarr"
    vv = s1_geotiff_dir / "s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif"
    vh = s1_geotiff_dir / "s1a_32TQM_vh_ASC_037_20230115t061234_GammaNaughtRTC.tif"
    mask = s1_geotiff_dir / "s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC_BorderMask.tif"
    ingest_s1tiling_acquisition(vv, vh, mask, store_path, "ascending")
    return store_path


@pytest.fixture
def gamma_area_geotiff(tmp_path: Path) -> Path:
    """Create a synthetic gamma_area GeoTIFF."""
    rng = np.random.default_rng(99)
    data = rng.uniform(0.5, 2.0, (SIZE, SIZE)).astype(np.float32)
    path = tmp_path / "GAMMA_AREA_32TQM_037.tif"
    _create_synthetic_geotiff(path, data)
    return path


@pytest.fixture
def lia_geotiff(tmp_path: Path) -> Path:
    """Create a synthetic LIA GeoTIFF."""
    rng = np.random.default_rng(100)
    data = rng.uniform(0.0, 1.0, (SIZE, SIZE)).astype(np.float32)
    path = tmp_path / "sin_LIA_32TQM_037.tif"
    _create_synthetic_geotiff(path, data)
    return path


class TestIngestConditions:
    def test_gamma_area_creates_conditions_group(
        self, s1_store_with_acquisition: Path, gamma_area_geotiff: Path
    ) -> None:
        ingest_s1tiling_conditions(
            store_path=s1_store_with_acquisition,
            orbit_direction="ascending",
            relative_orbit=37,
            gamma_area_path=gamma_area_geotiff,
        )
        root = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)
        orbit = _group(root, "ascending")
        assert "conditions" in orbit
        conditions = _group(orbit, "conditions")
        assert "gamma_area_037" in conditions

    def test_conditions_group_attributes(
        self, s1_store_with_acquisition: Path, gamma_area_geotiff: Path
    ) -> None:
        ingest_s1tiling_conditions(
            store_path=s1_store_with_acquisition,
            orbit_direction="ascending",
            relative_orbit=37,
            gamma_area_path=gamma_area_geotiff,
        )
        root = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)
        conditions = _group(_group(root, "ascending"), "conditions")
        attrs = dict(conditions.attrs)
        assert attrs["proj:code"] == CRS
        assert attrs["spatial:dimensions"] == ["y", "x"]
        transform = attrs["spatial:transform"]
        assert isinstance(transform, list)
        assert len(transform) == 6
        assert attrs["spatial:shape"] == [SIZE, SIZE]
        # CF grid-mapping so rioxarray can resolve the CRS of the condition arrays
        assert "spatial_ref" in list(conditions.array_keys())
        assert dict(_array(conditions, "gamma_area_037").attrs).get("grid_mapping") == "spatial_ref"

    def test_gamma_area_array_shape_and_dtype(
        self, s1_store_with_acquisition: Path, gamma_area_geotiff: Path
    ) -> None:
        ingest_s1tiling_conditions(
            store_path=s1_store_with_acquisition,
            orbit_direction="ascending",
            relative_orbit=37,
            gamma_area_path=gamma_area_geotiff,
        )
        root = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)
        arr = _array(_group(_group(root, "ascending"), "conditions"), "gamma_area_037")
        assert arr.shape == (SIZE, SIZE)
        assert arr.dtype == np.float32
        assert _dimension_names(arr) == ("y", "x")

    def test_data_integrity_roundtrip(
        self, s1_store_with_acquisition: Path, gamma_area_geotiff: Path
    ) -> None:
        ingest_s1tiling_conditions(
            store_path=s1_store_with_acquisition,
            orbit_direction="ascending",
            relative_orbit=37,
            gamma_area_path=gamma_area_geotiff,
        )
        # Read original
        with rasterio.open(str(gamma_area_geotiff)) as src:
            expected = src.read(1).astype(np.float32)
        # Read from Zarr
        root = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)
        actual = np.asarray(
            _array(_group(_group(root, "ascending"), "conditions"), "gamma_area_037")[:]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_conditions_nodata_masked_to_nan(
        self, s1_store_with_acquisition: Path, tmp_path: Path
    ) -> None:
        """A condition GeoTIFF's declared-nodata pixels read back as NaN (not the raw sentinel),
        so the auxiliary arrays mask transparent like vv/vh. border_mask is N/A for static
        conditions, so the writer relies on the GeoTIFF's declared nodata via a masked read.
        """
        data = np.full((SIZE, SIZE), 1.5, dtype=np.float32)
        data[0:20, 0:20] = 0.0  # declared-nodata region
        cond_path = tmp_path / "GAMMA_AREA_32TQM_037.tif"
        _create_synthetic_geotiff(cond_path, data, nodata=0.0)
        ingest_s1tiling_conditions(
            store_path=s1_store_with_acquisition,
            orbit_direction="ascending",
            relative_orbit=37,
            gamma_area_path=cond_path,
        )
        root = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)
        conditions = _group(_group(root, "ascending"), "conditions")
        arr = np.asarray(_array(conditions, "gamma_area_037")[:])
        assert np.all(np.isnan(arr[0:20, 0:20])), "declared-nodata region must be NaN"
        assert not np.any(np.isnan(arr[20:, 20:])), "valid region must stay finite"

    def test_float_conditions_declare_cf_fill_value(
        self,
        s1_store_with_acquisition: Path,
        gamma_area_geotiff: Path,
        lia_geotiff: Path,
    ) -> None:
        """Float condition arrays (gamma_area, lia) must declare a CF ``_FillValue`` so
        readers mask NaN nodata (xarray #11345), like vv/vh (#172)."""
        from xarray.backends.zarr import FillValueCoder

        ingest_s1tiling_conditions(
            store_path=s1_store_with_acquisition,
            orbit_direction="ascending",
            relative_orbit=37,
            gamma_area_path=gamma_area_geotiff,
            lia_path=lia_geotiff,
        )
        expected = FillValueCoder.encode(np.nan, np.dtype("float32"))
        root = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)
        conditions = _group(_group(root, "ascending"), "conditions")
        for arr_name in ("gamma_area_037", "lia_037"):
            assert dict(_array(conditions, arr_name).attrs).get("_FillValue") == expected, arr_name

    def test_gamma_area_is_sharded(
        self, s1_store_with_acquisition: Path, gamma_area_geotiff: Path
    ) -> None:
        """The condition array carries a sharding codec: one shard over the full (y, x) extent,
        512-aligned inner chunks (the same layout vv/vh already use)."""
        ingest_s1tiling_conditions(
            store_path=s1_store_with_acquisition,
            orbit_direction="ascending",
            relative_orbit=37,
            gamma_area_path=gamma_area_geotiff,
        )
        root = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)
        arr = _array(_group(_group(root, "ascending"), "conditions"), "gamma_area_037")
        # shards == full extent (None would mean unsharded — the pre-fix layout)
        assert arr.shards == (SIZE, SIZE)
        assert arr.chunks == (calculate_aligned_chunk_size(SIZE, 512),) * 2

    def test_sharding_collapses_chunk_objects_to_one(self, s1_store_with_acquisition: Path) -> None:
        """A multi-chunk condition array lands as a SINGLE on-disk shard object, not one object per
        inner chunk — the object-count collapse (real gamma_area: ~900 chunk objects → 1 shard)."""
        # 1098 sq with a 366 sq inner chunk = 3x3 = 9 inner chunks that, sharded, share one shard.
        big = 1098
        rng = np.random.default_rng(7)
        data = rng.uniform(0.5, 2.0, (big, big)).astype(np.float32)
        gpath = s1_store_with_acquisition.parent / "GAMMA_AREA_BIG_037.tif"
        _create_synthetic_geotiff(
            gpath, data, transform=from_bounds(XMIN, YMIN, XMAX, YMAX, big, big)
        )
        ingest_s1tiling_conditions(
            store_path=s1_store_with_acquisition,
            orbit_direction="ascending",
            relative_orbit=37,
            gamma_area_path=gpath,
        )
        root = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)
        arr = _array(_group(_group(root, "ascending"), "conditions"), "gamma_area_037")
        assert arr.chunks == (366, 366)
        assert arr.shards == (big, big)
        # exactly one chunk-data object on disk (the shard), regardless of the 9 inner chunks
        array_dir = s1_store_with_acquisition / "ascending" / "conditions" / "gamma_area_037"
        data_objects = [
            f for _r, _d, files in os.walk(array_dir) for f in files if f != "zarr.json"
        ]
        assert len(data_objects) == 1, data_objects
        # values still byte-identical through the shard
        np.testing.assert_allclose(np.asarray(arr[:]), data, rtol=1e-6)

    def test_multiple_conditions(
        self, s1_store_with_acquisition: Path, gamma_area_geotiff: Path, lia_geotiff: Path
    ) -> None:
        ingest_s1tiling_conditions(
            store_path=s1_store_with_acquisition,
            orbit_direction="ascending",
            relative_orbit=37,
            gamma_area_path=gamma_area_geotiff,
            lia_path=lia_geotiff,
        )
        root = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)
        conditions = _group(_group(root, "ascending"), "conditions")
        assert "gamma_area_037" in conditions
        assert "lia_037" in conditions

    def test_multiple_orbits(self, s1_store_with_acquisition: Path, tmp_path: Path) -> None:
        """Conditions for different orbits create separate arrays."""
        rng = np.random.default_rng(101)
        ga_037 = tmp_path / "GAMMA_AREA_32TQM_037.tif"
        ga_110 = tmp_path / "GAMMA_AREA_32TQM_110.tif"
        _create_synthetic_geotiff(ga_037, rng.uniform(0.5, 2.0, (SIZE, SIZE)).astype(np.float32))
        _create_synthetic_geotiff(ga_110, rng.uniform(0.5, 2.0, (SIZE, SIZE)).astype(np.float32))

        ingest_s1tiling_conditions(
            s1_store_with_acquisition, "ascending", 37, gamma_area_path=ga_037
        )
        ingest_s1tiling_conditions(
            s1_store_with_acquisition, "ascending", 110, gamma_area_path=ga_110
        )

        root = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)
        conditions = _group(_group(root, "ascending"), "conditions")
        assert "gamma_area_037" in conditions
        assert "gamma_area_110" in conditions

    def test_overwrite_existing_condition(
        self, s1_store_with_acquisition: Path, tmp_path: Path
    ) -> None:
        """Writing the same condition array twice overwrites data."""
        ga_path = tmp_path / "GAMMA_AREA_32TQM_037.tif"

        data_v1 = np.ones((SIZE, SIZE), dtype=np.float32)
        _create_synthetic_geotiff(ga_path, data_v1)
        ingest_s1tiling_conditions(
            s1_store_with_acquisition, "ascending", 37, gamma_area_path=ga_path
        )

        data_v2 = np.full((SIZE, SIZE), 2.0, dtype=np.float32)
        _create_synthetic_geotiff(ga_path, data_v2)
        ingest_s1tiling_conditions(
            s1_store_with_acquisition, "ascending", 37, gamma_area_path=ga_path
        )

        root = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)
        actual = np.asarray(
            _array(_group(_group(root, "ascending"), "conditions"), "gamma_area_037")[:]
        )
        np.testing.assert_allclose(actual, data_v2, rtol=1e-6)

    def test_raises_no_conditions_provided(self, s1_store_with_acquisition: Path) -> None:
        with pytest.raises(ValueError, match="At least one condition"):
            ingest_s1tiling_conditions(s1_store_with_acquisition, "ascending", 37)

    def test_raises_store_not_exists(self, tmp_path: Path, gamma_area_geotiff: Path) -> None:
        with pytest.raises(ValueError, match="Store does not exist"):
            ingest_s1tiling_conditions(
                tmp_path / "nonexistent.zarr",
                "ascending",
                37,
                gamma_area_path=gamma_area_geotiff,
            )

    def test_raises_orbit_not_exists(self, tmp_path: Path, gamma_area_geotiff: Path) -> None:
        """Raise if the orbit group hasn't been created yet."""
        # Create minimal empty store
        store_path = tmp_path / "empty-store.zarr"
        zarr.open_group(str(store_path), mode="w-", zarr_format=3)
        with pytest.raises(ValueError, match="not found in store"):
            ingest_s1tiling_conditions(
                store_path, "ascending", 37, gamma_area_path=gamma_area_geotiff
            )

    def test_raises_file_not_found(self, s1_store_with_acquisition: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ingest_s1tiling_conditions(
                s1_store_with_acquisition,
                "ascending",
                37,
                gamma_area_path="/nonexistent/gamma_area.tif",
            )

    def test_consolidation_includes_conditions(
        self, s1_store_with_acquisition: Path, gamma_area_geotiff: Path
    ) -> None:
        """Consolidation after conditions ingestion includes the conditions group."""
        ingest_s1tiling_conditions(
            s1_store_with_acquisition, "ascending", 37, gamma_area_path=gamma_area_geotiff
        )
        consolidate_s1_store(s1_store_with_acquisition, "ascending")

        root = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)
        assert root.metadata.consolidated_metadata is not None
        orbit = _group(root, "ascending")
        assert orbit.metadata.consolidated_metadata is not None
        # Conditions group should be accessible through consolidated metadata
        assert "conditions" in orbit
        assert "gamma_area_037" in _group(orbit, "conditions")


# =============================================================================
# Phase 3: Conditions file discovery tests
# =============================================================================


class TestDiscoverConditions:
    def test_discovers_gamma_area(self, tmp_path: Path) -> None:
        data = np.ones((SIZE, SIZE), dtype=np.float32)
        _create_synthetic_geotiff(tmp_path / "GAMMA_AREA_32TQM_037.tif", data)
        _create_synthetic_geotiff(tmp_path / "GAMMA_AREA_32TQM_110.tif", data)

        conditions = discover_s1tiling_conditions(tmp_path)
        assert len(conditions) == 2
        orbits = {c["orbit"] for c in conditions}
        assert orbits == {"037", "110"}
        for c in conditions:
            assert "gamma_area" in c
            assert c["tile"] == "32TQM"

    def test_discovers_lia(self, tmp_path: Path) -> None:
        data = np.ones((SIZE, SIZE), dtype=np.float32)
        _create_synthetic_geotiff(tmp_path / "sin_LIA_32TQM_037.tif", data)

        conditions = discover_s1tiling_conditions(tmp_path)
        assert len(conditions) == 1
        assert "lia" in conditions[0]

    def test_groups_gamma_area_and_lia(self, tmp_path: Path) -> None:
        """Gamma area and LIA for the same tile/orbit are grouped together."""
        data = np.ones((SIZE, SIZE), dtype=np.float32)
        _create_synthetic_geotiff(tmp_path / "GAMMA_AREA_32TQM_037.tif", data)
        _create_synthetic_geotiff(tmp_path / "sin_LIA_32TQM_037.tif", data)

        conditions = discover_s1tiling_conditions(tmp_path)
        assert len(conditions) == 1
        assert "gamma_area" in conditions[0]
        assert "lia" in conditions[0]
        assert conditions[0]["tile"] == "32TQM"
        assert conditions[0]["orbit"] == "037"

    def test_skips_non_matching(self, tmp_path: Path) -> None:
        data = np.ones((SIZE, SIZE), dtype=np.float32)
        _create_synthetic_geotiff(tmp_path / "random_file.tif", data)
        conditions = discover_s1tiling_conditions(tmp_path)
        assert len(conditions) == 0

    def test_empty_directory(self, tmp_path: Path) -> None:
        conditions = discover_s1tiling_conditions(tmp_path)
        assert len(conditions) == 0

    def test_s3_uri_discovers_conditions(self) -> None:
        """s3:// prefix is listed via s3fs; pathlib.glob is NOT used."""
        s3_files = ["bucket/prefix/GAMMA_AREA_32TQM_037.tif"]
        with patch("s3fs.S3FileSystem.glob", return_value=s3_files):
            conditions = discover_s1tiling_conditions("s3://bucket/prefix/")
        assert len(conditions) == 1
        assert conditions[0]["tile"] == "32TQM"
        assert conditions[0]["orbit"] == "037"


# =============================================================================
# CF datetime `time` coordinate — render-by-datetime support (data-model #192)
# =============================================================================

_LEVELS = ["r10m", "r20m", "r60m", "r120m", "r360m", "r720m"]


class TestTimeCFDatetime:
    """`time` is CF-encoded at every multiscale level so readers decode it to datetime64 and can
    select a slice by datetime (`sel=time={datetime}`) at any rendered scale, even on a non-monotonic
    axis. This replaces the fragile positional `sel=time={index}` rendering (#192)."""

    def _paths(self, geotiff_dir: Path, stamp: str) -> tuple[Path, Path, Path]:
        vv = geotiff_dir / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC.tif"
        vh = geotiff_dir / f"s1a_32TQM_vh_ASC_037_{stamp}_GammaNaughtRTC.tif"
        mask = geotiff_dir / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC_BorderMask.tif"
        return vv, vh, mask

    def test_time_has_cf_attrs_at_every_level(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        vv, vh, mask = self._paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        asc = _group(root, "ascending")
        for level in _LEVELS:
            attrs = dict(_array(_group(asc, level), "time").attrs)
            assert attrs.get("units") == "nanoseconds since 1970-01-01", level
            assert attrs.get("calendar") == "proleptic_gregorian", level
            assert attrs.get("standard_name") == "time", level

    def test_open_datatree_decodes_time_to_datetime64(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        vv, vh, mask = self._paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")
        dt = xr.open_datatree(
            str(s1_store_path), engine="zarr", decode_times=True, consolidated=False
        )
        for level in ("r10m", "r720m"):
            da = dt["ascending"][level]["vv"]
            assert "time" in da.coords, level
            assert np.issubdtype(da["time"].dtype, np.datetime64), level

    def test_exact_datetime_sel_on_nonmonotonic_axis(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """Ingest LATER acq first then EARLIER → non-monotonic axis; exact datetime `.sel` still
        returns the right physical slice at both the native and a coarse level (the 31TEH case).

        Building this cube now requires `allow_out_of_order=True` — the ingest refuses to create a
        non-monotonic axis by default. The escape hatch is exactly what recovering an already-broken
        cube like 31TEH needs, so this exercises both it and the ordering-independence of exact
        `.sel`."""
        vv2, vh2, mask2 = self._paths(s1_geotiff_dir, "20230127t061235")  # 2023-01-27 (later)
        vv1, vh1, mask1 = self._paths(s1_geotiff_dir, "20230115t061234")  # 2023-01-15 (earlier)
        ingest_s1tiling_acquisition(vv2, vh2, mask2, s1_store_path, "ascending")  # -> index 0
        ingest_s1tiling_acquisition(
            vv1, vh1, mask1, s1_store_path, "ascending", allow_out_of_order=True
        )  # -> index 1

        dt = xr.open_datatree(
            str(s1_store_path), engine="zarr", decode_times=True, consolidated=False
        )
        time_node = dt["ascending"]["r10m"]["time"]
        assert isinstance(time_node, xr.DataArray)
        times = time_node.values
        assert times[0] > times[1], "axis should be non-monotonic (later acq appended first)"

        early = np.datetime64("2023-01-15T06:12:34")  # physical index 1
        later = np.datetime64("2023-01-27T06:12:35")  # physical index 0
        for level in ("r10m", "r720m"):
            vvda = dt["ascending"][level]["vv"]
            np.testing.assert_array_equal(
                vvda.sel(time=early).values, vvda.isel(time=1).values, err_msg=f"{level} early"
            )
            np.testing.assert_array_equal(
                vvda.sel(time=later).values, vvda.isel(time=0).values, err_msg=f"{level} later"
            )

    def test_time_values_identical_across_levels(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        vv1, vh1, mask1 = self._paths(s1_geotiff_dir, "20230115t061234")
        vv2, vh2, mask2 = self._paths(s1_geotiff_dir, "20230127t061235")
        ingest_s1tiling_acquisition(vv1, vh1, mask1, s1_store_path, "ascending")
        ingest_s1tiling_acquisition(vv2, vh2, mask2, s1_store_path, "ascending")
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        asc = _group(root, "ascending")
        ref = np.asarray(_array(_group(asc, "r10m"), "time")[:])
        assert ref.shape == (2,)
        for level in _LEVELS[1:]:
            np.testing.assert_array_equal(np.asarray(_array(_group(asc, level), "time")[:]), ref)

    def test_r10m_time_still_int64_for_register(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """register_per_acquisition reads r10m/time as raw int64 ns — CF attrs must not change that."""
        vv, vh, mask = self._paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        arr = _array(_group(_group(root, "ascending"), "r10m"), "time")
        assert arr.dtype == np.dtype("int64")
        assert str(np.datetime64(int(np.asarray(arr)[0]), "ns")).startswith("2023-01-15")


# =============================================================================
# Per-level `time` self-heal on append (robust to a pre-#192 / half-built cube)
# =============================================================================


class TestPerLevelTimeHeal:
    """The append recreates a multiscale level's missing `time` from r10m/time instead of raising
    `KeyError: 'time'` (a cube built before #192, or left half-built by an interrupted append), and
    refuses to mis-heal a genuinely inconsistent cube."""

    def _paths(self, d: Path, stamp: str) -> tuple[Path, Path, Path]:
        return (
            d / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC.tif",
            d / f"s1a_32TQM_vh_ASC_037_{stamp}_GammaNaughtRTC.tif",
            d / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC_BorderMask.tif",
        )

    def _coarse_levels(self, store_path: Path) -> list[str]:
        root = zarr.open_group(str(store_path), mode="r", zarr_format=3)
        return [n for n, _ in _group(root, "ascending").groups() if n not in ("r10m", "conditions")]

    def _drop_time(self, store_path: Path, level: str) -> None:
        """Simulate a pre-#192 cube by removing a level's `time` array. `ingest_s1tiling_acquisition`
        does not consolidate, so a filesystem removal is enough for the group to no longer see it."""
        import shutil

        shutil.rmtree(Path(store_path) / "ascending" / level / "time")

    def test_append_heals_levels_missing_time(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """A coarser level lacking `time` is recreated from r10m/time (prior slices preserved); the
        append that previously crashed now succeeds."""
        a1 = self._paths(s1_geotiff_dir, "20230115t061234")
        a2 = self._paths(s1_geotiff_dir, "20230127t061235")
        ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending")
        coarse = self._coarse_levels(s1_store_path)
        assert coarse  # sanity: there ARE coarser levels
        for lvl in coarse:
            self._drop_time(s1_store_path, lvl)

        idx = ingest_s1tiling_acquisition(*a2, s1_store_path, "ascending")  # was KeyError: 'time'

        assert idx == 1
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        asc = _group(root, "ascending")
        ref = list(np.asarray(_array(_group(asc, "r10m"), "time")[:]))
        assert len(ref) == 2
        for lvl in coarse:
            t = _array(_group(asc, lvl), "time")
            assert t.dtype == np.dtype("int64")
            assert list(np.asarray(t[:])) == ref  # backfilled prior + appended new, matching r10m

    def test_append_noop_when_all_levels_have_time(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """A healthy cube: the heal is a no-op and the append works normally."""
        a1 = self._paths(s1_geotiff_dir, "20230115t061234")
        a2 = self._paths(s1_geotiff_dir, "20230127t061235")
        ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending")
        idx = ingest_s1tiling_acquisition(*a2, s1_store_path, "ascending")
        assert idx == 1
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        asc = _group(root, "ascending")
        for lvl in self._coarse_levels(s1_store_path):
            assert _array(_group(asc, lvl), "time").shape[0] == 2

    def test_append_raises_on_half_built_cube(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """A level whose data length disagrees with r10m/time (and lacks `time`) is unhealable -> raise
        rather than write a wrong-length coordinate."""
        a1 = self._paths(s1_geotiff_dir, "20230115t061234")
        a2 = self._paths(s1_geotiff_dir, "20230127t061235")
        ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending")
        ingest_s1tiling_acquisition(*a2, s1_store_path, "ascending")  # 2 slices

        root = zarr.open_group(str(s1_store_path), mode="r+", zarr_format=3)
        r20m = _group(_group(root, "ascending"), "r20m")
        vv_arr = _array(r20m, "vv")
        _, h, w = vv_arr.shape
        vv_arr.resize((1, h, w))  # half-built: r20m has 1 slice, r10m has 2
        self._drop_time(s1_store_path, "r20m")

        with pytest.raises(ValueError, match="half-built"):
            ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending")

    def test_append_raises_when_r10m_time_missing(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """r10m holds slices but no `time` -> no backfill source -> raise (not a silent invention)."""
        a1 = self._paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending")
        self._drop_time(s1_store_path, "r10m")
        with pytest.raises(ValueError, match="no backfill source"):
            ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending")


# =============================================================================
# Post-consolidation append, grid identity, store URIs and the writer stamp
# =============================================================================


class TestAppendAfterConsolidation:
    """`consolidate_s1_store` writes a block on every orbit group, and `use_consolidated`
    applies only to the group actually opened. Reading array shapes through a consolidated
    root therefore returned the pre-consolidation length, so every post-consolidation append
    targeted the same time index."""

    @staticmethod
    def _paths(d: Path, stamp: str) -> tuple[Path, Path, Path]:
        return (
            d / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC.tif",
            d / f"s1a_32TQM_vh_ASC_037_{stamp}_GammaNaughtRTC.tif",
            d / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC_BorderMask.tif",
        )

    def test_append_after_consolidate_does_not_overwrite_slice(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """append, consolidate, append -> 2 distinct slices, not 1 overwritten in place."""
        a1 = self._paths(s1_geotiff_dir, "20230115t061234")
        a2 = self._paths(s1_geotiff_dir, "20230127t061235")

        assert ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending") == 0
        consolidate_s1_store(s1_store_path, "ascending")
        assert ingest_s1tiling_acquisition(*a2, s1_store_path, "ascending") == 1

        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3, use_consolidated=False)
        r10m = _group(_group(root, "ascending"), "r10m")
        assert _array(r10m, "vv").shape[0] == 2
        times = np.asarray(_array(r10m, "time")[:])
        assert len(set(times.tolist())) == 2, f"expected 2 distinct timestamps, got {times}"

    def test_append_strips_stale_consolidated_metadata(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """A stale block is worse than none: it reports pre-append shapes as authoritative."""
        a1 = self._paths(s1_geotiff_dir, "20230115t061234")
        a2 = self._paths(s1_geotiff_dir, "20230127t061235")
        ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending")
        consolidate_s1_store(s1_store_path, "ascending")
        ingest_s1tiling_acquisition(*a2, s1_store_path, "ascending")

        for rel in ("zarr.json", "ascending/zarr.json"):
            meta = json.loads((s1_store_path / rel).read_text())
            assert "consolidated_metadata" not in meta, f"stale block left in {rel}"

    def test_consolidate_covers_orbit_created_after_last_consolidation(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """Enumerating orbits through a stale root block silently skipped the newest orbit."""
        a1 = self._paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending")
        consolidate_s1_store(s1_store_path, "ascending")
        ingest_s1tiling_acquisition(*a1, s1_store_path, "descending")
        consolidate_s1_store(s1_store_path, "descending")

        for orbit in ("ascending", "descending"):
            meta = json.loads((s1_store_path / orbit / "zarr.json").read_text())
            assert "consolidated_metadata" in meta, f"{orbit} was not consolidated"

    def test_ragged_cube_raises_instead_of_writing_1970(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """A crash inside the per-level write loop leaves r10m longer than a coarser level. The
        next append then wrote NaN fill at the gap with `time == 0` (1970) and exited 0."""
        a1 = self._paths(s1_geotiff_dir, "20230115t061234")
        a2 = self._paths(s1_geotiff_dir, "20230127t061235")
        ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending")
        ingest_s1tiling_acquisition(*a2, s1_store_path, "ascending")

        # Simulate the interrupted append: advance r10m only, leaving r20m one slice short.
        root = zarr.open_group(str(s1_store_path), mode="r+", zarr_format=3, use_consolidated=False)
        r10m = _group(_group(root, "ascending"), "r10m")
        for name in ("vv", "vh", "border_mask"):
            arr = _array(r10m, name)
            n, h, w = arr.shape
            arr.resize((n + 1, h, w))
        _array(r10m, "time").resize((3,))

        with pytest.raises(ValueError, match="half-built"):
            ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending")


class TestGridIdentityOnAppend:
    """Two adjacent MGRS tiles in one UTM zone share a CRS and a shape and differ only in
    origin, so a CRS+shape check passed both and served slice 1's pixels 100 km away."""

    @staticmethod
    def _shifted_tile(d: Path, stamp: str, tags: dict[str, str]) -> tuple[Path, Path, Path]:
        """Write an acquisition on the same CRS and shape but a 100 km-shifted origin."""
        rng = np.random.default_rng(7)
        shifted = from_bounds(XMIN + 100000.0, YMIN, XMAX + 100000.0, YMAX, SIZE, SIZE)
        out = []
        for pol in ("vv", "vh"):
            data = rng.uniform(0.0, 1.0, (SIZE, SIZE)).astype(np.float32)
            p = d / f"s1a_33TQM_{pol}_ASC_037_{stamp}_GammaNaughtRTC.tif"
            _create_synthetic_geotiff(p, data, transform=shifted, tags=tags)
            out.append(p)
        mask = np.ones((SIZE, SIZE), dtype=np.uint8)
        m = d / f"s1a_33TQM_vv_ASC_037_{stamp}_GammaNaughtRTC_BorderMask.tif"
        _create_synthetic_geotiff(m, mask, transform=shifted, tags=tags)
        return out[0], out[1], m

    def test_shifted_origin_rejected_on_same_orbit_append(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        a1 = TestAppendAfterConsolidation._paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending")

        foreign = self._shifted_tile(s1_geotiff_dir, "20230127t061235", ACQ2_TAGS)
        with pytest.raises(ValueError, match="different grid"):
            ingest_s1tiling_acquisition(*foreign, s1_store_path, "ascending")

    def test_shifted_origin_rejected_on_new_orbit_branch(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """The new-orbit branch built the group straight from the incoming metadata with no
        comparison at all, poisoning the whole orbit group rather than one slice."""
        a1 = TestAppendAfterConsolidation._paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending")

        foreign = self._shifted_tile(s1_geotiff_dir, "20230127t061235", ACQ2_TAGS)
        with pytest.raises(ValueError, match="different grid"):
            ingest_s1tiling_acquisition(*foreign, s1_store_path, "descending")

        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3, use_consolidated=False)
        assert "descending" not in root, "the rejected tile still created an orbit group"


class TestStoreUriHandling:
    """`Path("s3://bucket/x.zarr")` collapses to `PosixPath("s3:/bucket/x.zarr")`, whose
    `.exists()` is always False, so the create branch wrote a LocalStore under `./s3:/...`,
    logged success and exited 0 with nothing in S3."""

    def test_s3_uri_reaches_the_store_layer_intact(
        self, s1_geotiff_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The URI must arrive at the store layer as `s3://...`, never collapsed to `s3:/...`.

        Intercepts at `create_s1_store` so the assertion is on the path handling alone, with no
        S3 credentials, no network and no dependence on how a real S3 failure surfaces.
        """
        a1 = TestAppendAfterConsolidation._paths(s1_geotiff_dir, "20230115t061234")
        monkeypatch.chdir(tmp_path)

        seen: list[str] = []

        class _Stop(Exception):
            pass

        def _capture(store_path: str | Path, orbit_direction: str, metadata: object) -> None:
            seen.append(str(store_path))
            raise _Stop

        monkeypatch.setattr(
            "eopf_geozarr.conversion.s1_ingest.create_s1_store", _capture, raising=True
        )
        monkeypatch.setattr(
            "eopf_geozarr.conversion.s1_ingest.fs_utils.path_exists",
            lambda *_a, **_k: False,
            raising=True,
        )

        with pytest.raises(_Stop):
            ingest_s1tiling_acquisition(*a1, "s3://no-such-bucket/cube.zarr", "ascending")

        assert seen == ["s3://no-such-bucket/cube.zarr"], seen
        assert not (tmp_path / "s3:").exists(), "wrote a LocalStore under ./s3:/"


class TestWriterSchemaStamp:
    """The stamp lets a later release tell layout generations apart and refuse to mix them."""

    def test_new_store_is_stamped(self, s1_geotiff_dir: Path, s1_store_path: Path) -> None:
        a1 = TestAppendAfterConsolidation._paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending")
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3, use_consolidated=False)
        assert root.attrs["eopf:writer_schema"] == WRITER_SCHEMA

    def test_root_is_complete_at_creation(self, s1_geotiff_dir: Path, s1_store_path: Path) -> None:
        """`spatial:bbox` has no default in the root model, so a never-consolidated store
        failed root validation outright."""
        a1 = TestAppendAfterConsolidation._paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending")
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3, use_consolidated=False)
        attrs = dict(root.attrs)
        assert "zarr_conventions" in attrs
        assert attrs["proj:code"] == CRS
        bbox = attrs["spatial:bbox"]
        assert isinstance(bbox, list)
        assert len(bbox) == 4

    def test_unstamped_store_still_accepts_appends(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """Generation 2 is layout-compatible with earlier stores, so an append to a store
        written by <=0.10.2 must warn, not refuse -- refusing would brick the existing archive."""
        a1 = TestAppendAfterConsolidation._paths(s1_geotiff_dir, "20230115t061234")
        a2 = TestAppendAfterConsolidation._paths(s1_geotiff_dir, "20230127t061235")
        ingest_s1tiling_acquisition(*a1, s1_store_path, "ascending")

        # Strip the stamp on disk, the way a store written by <=0.10.2 actually looks.
        meta_path = s1_store_path / "zarr.json"
        meta = json.loads(meta_path.read_text())
        del meta["attributes"]["eopf:writer_schema"]
        meta_path.write_text(json.dumps(meta, indent=2))

        assert ingest_s1tiling_acquisition(*a2, s1_store_path, "ascending") == 1


class TestIntegerConditionGeotiff:
    """`.filled(np.nan)` on an integer-dtype masked array raises, so an integer condition
    GeoTIFF aborted the whole ingest. NaN is only a legal fill once the array is float."""

    def test_integer_dtype_condition_is_ingested(
        self, s1_store_with_acquisition: Path, tmp_path: Path
    ) -> None:
        rng = np.random.default_rng(101)
        data = rng.integers(0, 90, (SIZE, SIZE), dtype=np.int16)
        path = tmp_path / "GAMMA_AREA_32TQM_037.tif"
        _create_synthetic_geotiff(path, data, nodata=0)

        ingest_s1tiling_conditions(
            store_path=s1_store_with_acquisition,
            orbit_direction="ascending",
            relative_orbit=37,
            gamma_area_path=path,
        )

        root = zarr.open_group(
            str(s1_store_with_acquisition), mode="r", zarr_format=3, use_consolidated=False
        )
        arr = _array(_group(_group(root, "ascending"), "conditions"), "gamma_area_037")
        assert arr.dtype == np.float32
        # The declared nodata (0) must have become NaN, not stayed 0.
        values = np.asarray(arr[:])
        assert np.isnan(values).any(), "declared nodata did not become NaN"


# =============================================================================
# GeoZarr minispec conformance (WS2a)
# =============================================================================


class TestMinispecConformance:
    """The writer must produce stores the project's own validator accepts.

    Live cubes report 86 issues per store today. The causes are structural, not incidental:
    a subgroup never inherits `zarr_conventions` (the validator passes inherited convention
    UUIDs only to a group's direct child *arrays*), so every level group and the conditions
    group has to declare its own; and the conditions group had no 1-D coordinate arrays at all.
    """

    @staticmethod
    def _declared_conventions(attrs: Mapping[str, Any]) -> set[str]:
        """Collect the `name` of every declared convention (attrs are untyped JSON)."""
        conventions = attrs["zarr_conventions"]
        assert isinstance(conventions, list)
        names = set()
        for conv in conventions:
            assert isinstance(conv, dict)
            names.add(conv["name"])
        return names

    @staticmethod
    def _build(geotiff_dir: Path, store_path: Path) -> None:
        vv = geotiff_dir / "s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif"
        vh = geotiff_dir / "s1a_32TQM_vh_ASC_037_20230115t061234_GammaNaughtRTC.tif"
        mask = geotiff_dir / "s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC_BorderMask.tif"
        ingest_s1tiling_acquisition(vv, vh, mask, store_path, "ascending")

        rng = np.random.default_rng(99)
        gamma = geotiff_dir / "GAMMA_AREA_32TQM_037.tif"
        _create_synthetic_geotiff(gamma, rng.uniform(0.5, 2.0, (SIZE, SIZE)).astype(np.float32))
        ingest_s1tiling_conditions(
            store_path=store_path,
            orbit_direction="ascending",
            relative_orbit=37,
            gamma_area_path=gamma,
        )
        consolidate_s1_store(store_path, "ascending")

    def test_store_root_is_compliant(self, s1_geotiff_dir: Path, s1_store_path: Path) -> None:
        from eopf_geozarr.data_api.geozarr.validation import validate_store

        self._build(s1_geotiff_dir, s1_store_path)
        report = validate_store(str(s1_store_path))
        assert report.compliant, "\n".join(str(i) for i in report.issues)

    def test_orbit_group_href_is_compliant(self, s1_geotiff_dir: Path, s1_store_path: Path) -> None:
        """STAC data assets point at the orbit groups, so each must validate standalone."""
        from eopf_geozarr.data_api.geozarr.validation import validate_store

        self._build(s1_geotiff_dir, s1_store_path)
        report = validate_store(f"{s1_store_path}/ascending")
        assert report.compliant, "\n".join(str(i) for i in report.issues)

    def test_level_groups_declare_their_own_conventions(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        self._build(s1_geotiff_dir, s1_store_path)
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3, use_consolidated=False)
        orbit = _group(root, "ascending")
        for level_name, _, _ in OVERVIEW_CHAIN:
            attrs = dict(_group(orbit, level_name).attrs)
            declared = self._declared_conventions(attrs)
            assert "spatial:" in declared, f"{level_name} does not declare the spatial convention"
            assert "proj:" in declared, f"{level_name} does not declare the geo-proj convention"
            assert attrs["spatial:dimensions"] == ["y", "x"], level_name

    def test_conditions_group_has_coordinate_arrays(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """Without 1-D x/y the group opens with no coordinates and rioxarray infers an
        identity transform — the georeferencing is absent, not merely undeclared."""
        self._build(s1_geotiff_dir, s1_store_path)
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3, use_consolidated=False)
        conditions = _group(_group(root, "ascending"), "conditions")

        arrays = set(conditions.array_keys())
        assert {"x", "y"} <= arrays, f"conditions group has no coordinate arrays: {sorted(arrays)}"

        declared = self._declared_conventions(dict(conditions.attrs))
        assert "spatial:" in declared
        assert "proj:" in declared

        # x/y must describe the grid the group's own attrs record.
        transform = dict(conditions.attrs)["spatial:transform"]
        assert isinstance(transform, list)
        assert float(np.asarray(_array(conditions, "x")[:])[0]) == transform[2]
        assert float(np.asarray(_array(conditions, "y")[:])[0]) == transform[5]

    def test_conditions_rejects_a_raster_on_a_different_grid(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """Every condition raster in one orbit shares a single x/y pair, so a raster on
        another grid would be stored under coordinates that do not describe it."""
        self._build(s1_geotiff_dir, s1_store_path)

        rng = np.random.default_rng(7)
        shifted = from_bounds(XMIN + 100000.0, YMIN, XMAX + 100000.0, YMAX, SIZE, SIZE)
        other = s1_geotiff_dir / "GAMMA_AREA_32TQM_110.tif"
        _create_synthetic_geotiff(
            other, rng.uniform(0.5, 2.0, (SIZE, SIZE)).astype(np.float32), transform=shifted
        )

        with pytest.raises(ValueError, match="grid mismatch"):
            ingest_s1tiling_conditions(
                store_path=s1_store_path,
                orbit_direction="ascending",
                relative_orbit=110,
                gamma_area_path=other,
            )


class TestTimeAxisOrdering:
    """A non-monotonic time axis makes `.sel(time=slice(...))` raise, so a cube that accepts an
    out-of-order append is silently unqueryable by time afterwards."""

    @staticmethod
    def _paths(d: Path, stamp: str) -> tuple[Path, Path, Path]:
        return (
            d / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC.tif",
            d / f"s1a_32TQM_vh_ASC_037_{stamp}_GammaNaughtRTC.tif",
            d / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC_BorderMask.tif",
        )

    def test_out_of_order_append_raises(self, s1_geotiff_dir: Path, s1_store_path: Path) -> None:
        later = self._paths(s1_geotiff_dir, "20230127t061235")
        earlier = self._paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(*later, s1_store_path, "ascending")

        with pytest.raises(ValueError, match="Out-of-order append"):
            ingest_s1tiling_acquisition(*earlier, s1_store_path, "ascending")

    def test_out_of_order_append_writes_nothing(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """The refusal must come before any array is resized, or it leaves a ragged cube."""
        later = self._paths(s1_geotiff_dir, "20230127t061235")
        earlier = self._paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(*later, s1_store_path, "ascending")

        with pytest.raises(ValueError, match="Out-of-order append"):
            ingest_s1tiling_acquisition(*earlier, s1_store_path, "ascending")

        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3, use_consolidated=False)
        r10m = _group(_group(root, "ascending"), "r10m")
        assert _array(r10m, "vv").shape[0] == 1
        assert _array(r10m, "time").shape[0] == 1

    def test_escape_hatch_allows_the_append(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """`allow_out_of_order=True` is how an already-broken cube gets recovered."""
        later = self._paths(s1_geotiff_dir, "20230127t061235")
        earlier = self._paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(*later, s1_store_path, "ascending")
        assert (
            ingest_s1tiling_acquisition(
                *earlier, s1_store_path, "ascending", allow_out_of_order=True
            )
            == 1
        )

    def test_duplicate_timestamp_is_rejected(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """The append is positional: re-ingesting the same acquisition would add a second slice
        carrying the same instant rather than replacing the first."""
        acq = self._paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(*acq, s1_store_path, "ascending")

        with pytest.raises(ValueError, match="duplicates"):
            ingest_s1tiling_acquisition(*acq, s1_store_path, "ascending")

    def test_discovery_is_chronological_across_platforms(self, tmp_path: Path) -> None:
        """The group key is (platform, tile, orbit_dir, rel_orbit, acq_stamp), so sorting it
        verbatim let `platform` dominate the timestamp: a mixed S1A/S1C archive came back as all
        the S1A dates followed by all the S1C dates."""
        rng = np.random.default_rng(3)
        stamps = [
            ("s1a", "20250210t061234"),
            ("s1c", "20250115t061234"),
            ("s1a", "20250305t061234"),
            ("s1c", "20250127t061234"),
        ]
        for platform, stamp in stamps:
            tags = dict(ACQ1_TAGS)
            tags["FLYING_UNIT_CODE"] = platform.upper()
            tags["ACQUISITION_DATETIME"] = f"{stamp[0:4]}:{stamp[4:6]}:{stamp[6:8]}T06:12:34Z"
            for pol in ("vv", "vh"):
                data = rng.uniform(0.0, 1.0, (SIZE, SIZE)).astype(np.float32)
                _create_synthetic_geotiff(
                    tmp_path / f"{platform}_32TQM_{pol}_ASC_037_{stamp}_GammaNaughtRTC.tif",
                    data,
                    tags=tags,
                )
                _create_synthetic_geotiff(
                    tmp_path
                    / f"{platform}_32TQM_{pol}_ASC_037_{stamp}_GammaNaughtRTC_BorderMask.tif",
                    np.ones((SIZE, SIZE), dtype=np.uint8),
                    tags=tags,
                )

        discovered = [a["acq_stamp"] for a in discover_s1tiling_acquisitions(tmp_path)]
        assert discovered == sorted(discovered), f"not chronological: {discovered}"
        assert discovered == [
            "20250115t061234",
            "20250127t061234",
            "20250210t061234",
            "20250305t061234",
        ]


class TestStoreRootGeographicMetadata:
    """After consolidation the root is published in the same geographic form as S2 and OLCI."""

    def test_root_is_geographic_after_consolidation(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        vv = s1_geotiff_dir / "s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif"
        vh = s1_geotiff_dir / "s1a_32TQM_vh_ASC_037_20230115t061234_GammaNaughtRTC.tif"
        mask = s1_geotiff_dir / "s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC_BorderMask.tif"
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")

        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3, use_consolidated=False)
        assert dict(root.attrs)["proj:code"] == CRS, "native CRS expected before consolidation"

        consolidate_s1_store(s1_store_path, "ascending")

        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3, use_consolidated=False)
        attrs = dict(root.attrs)
        assert attrs["proj:code"] == "EPSG:4326"
        bbox = attrs["spatial:bbox"]
        assert isinstance(bbox, list)
        lon, lat = float(cast("float", bbox[0])), float(cast("float", bbox[1]))
        assert -180.0 <= lon <= 180.0, bbox
        assert -90.0 <= lat <= 90.0, bbox
        # The writer stamp must survive the rewrite (it is an update, not a replace).
        assert attrs["eopf:writer_schema"] == WRITER_SCHEMA


# =============================================================================
# F11 — the overview border mask must agree with the averaged backscatter
# =============================================================================


class TestOverviewBorderMask:
    """The mask is documented as the authoritative valid-data mask and STAC advertises it with
    `nodata: 0`, so a mask pixel that says "invalid" over finite backscatter erases real data at
    preview zoom. Subsampling the mask while block-averaging vv/vh broke exactly that.
    """

    def test_max_branch_matches_an_explicit_block_reference(self) -> None:
        """Block-max over the edge-padded grid, verified against a hand-computed reference.

        Pinned directly because the end-to-end invariant below cannot distinguish block-max from
        subsampling whenever the factor is 1 -- both then return the same pixels.
        """
        data = np.array(
            [[0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            dtype=np.uint8,
        )
        # 2x2 blocks: [0,0/0,0]=0  [1,0/0,0]=1  [0,1/0,0]=1  [0,0/0,0]=0
        assert np.array_equal(
            _downsample_2d(data, 2, "max"), np.array([[0, 1], [1, 0]], dtype=np.uint8)
        )
        # Subsampling takes data[::2, ::2], which misses the valid pixel at [2][1] entirely:
        # its block is valid but the sampled corner is not. That lower-left 0 over finite
        # averaged backscatter is precisely the defect.
        assert np.array_equal(
            _downsample_2d(data, 2, "nearest"), np.array([[0, 1], [0, 0]], dtype=np.uint8)
        )

    def test_max_edge_pads_on_a_non_divisible_size(self) -> None:
        """Non-divisible sizes use the same ceil grid and edge padding as `average`, so mask and
        backscatter levels stay aligned pixel for pixel."""
        data = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.uint8)
        out = _downsample_2d(data, 2, "max")
        assert out.shape == _downsample_2d(data.astype(np.float32), 2, "average").shape
        # Edge padding replicates the real border column/row -- it cannot invent validity.
        assert np.array_equal(out, np.array([[0, 1], [1, 0]], dtype=np.uint8))

    def test_mask_agrees_with_backscatter_at_every_level(self, tmp_path: Path) -> None:
        """End-to-end on a diagonal swath edge, the case that reproduced the defect.

        60 pixels at r20m (20 at r60m, 10 at r120m) previously had `border_mask == 0` over
        finite `vv`.
        """
        rows, cols = np.mgrid[0:SIZE, 0:SIZE]
        inside = rows + cols > SIZE // 2  # diagonal swath edge
        vv = np.where(inside, 0.1, np.nan).astype(np.float32)
        vh = np.where(inside, 0.05, np.nan).astype(np.float32)
        mask = inside.astype(np.uint8)

        store = tmp_path / "s1-rtc-32TQM.zarr"
        for name, arr in (("vv", vv), ("vh", vh), ("mask", mask)):
            _create_synthetic_geotiff(tmp_path / f"{name}.tif", arr, tags=ACQ1_TAGS)
        ingest_s1tiling_acquisition(
            tmp_path / "vv.tif", tmp_path / "vh.tif", tmp_path / "mask.tif", store, "ascending"
        )
        consolidate_s1_store(store, "ascending")

        root = zarr.open_group(str(store), mode="r", zarr_format=3)
        for level_name, _, _ in OVERVIEW_CHAIN:
            level = _group(_group(root, "ascending"), level_name)
            level_vv = np.asarray(_array(level, "vv"))[0]
            level_mask = np.asarray(_array(level, "border_mask"))[0]
            assert level_mask.shape == level_vv.shape, level_name
            # One-directional: `isfinite(vv) => mask != 0`. Equality holds on THIS fixture, which
            # ties vv and mask together by construction, but it is not what the code guarantees --
            # real γ⁰ can be NaN inside the swath. Asserting equality here would be asserting a
            # false statement that happens to hold on synthetic data; see
            # `test_nan_inside_the_swath_keeps_the_invariant_one_directional`.
            erased = np.isfinite(level_vv) & (level_mask == 0)
            assert not erased.any(), (
                f"{level_name}: {int(erased.sum())} pixel(s) of real backscatter would be erased "
                "by a mask that calls them invalid"
            )
            # This fixture DOES satisfy the r10m precondition (vv is NaN exactly where mask is 0),
            # and given that precondition the two are identical at every level -- block-max and
            # nanmean agree because a block is valid iff any source pixel was. That equivalence is
            # what makes `slice_coverages` meaningful: it reads the mask at r720m and calls the
            # result "the fraction of the preview image that renders as data". Pinning it here is
            # what stops that claim silently becoming false.
            assert np.array_equal(level_mask != 0, np.isfinite(level_vv)), (
                f"{level_name}: mask and averaged backscatter diverged despite agreeing at r10m; "
                "slice_coverages' preview-fill claim depends on them matching"
            )


# =============================================================================
# F15 — discovery and ingest must agree on the orbit value
# =============================================================================


class TestOrbitDirectionContract:
    def test_discovery_emits_both_the_short_and_long_orbit_forms(self, tmp_path: Path) -> None:
        """`orbit_direction` is the group name the ingester takes; `orbit_dir` stays the short
        filename form, which filename reconstruction still needs."""
        data = np.ones((SIZE, SIZE), dtype=np.float32)
        for short, expected in (("ASC", "ascending"), ("DES", "descending")):
            case_dir = tmp_path / short
            case_dir.mkdir()
            stem = "s1a_32TQM_{pol}_" + short + "_037_20230115t061234_GammaNaughtRTC{mask}.tif"
            for pol, mask in (("vv", ""), ("vh", ""), ("vv", "_BorderMask")):
                _create_synthetic_geotiff(
                    case_dir / stem.format(pol=pol, mask=mask), data, tags=ACQ1_TAGS
                )
            (acq,) = discover_s1tiling_acquisitions(case_dir)
            assert acq["orbit_dir"] == short
            assert acq["orbit_direction"] == expected

    def test_short_form_raises_before_any_geotiff_is_opened(self, tmp_path: Path) -> None:
        """The paths do not exist, so reaching I/O would raise FileNotFoundError instead.

        This ordering is the whole point: the old code only failed once STAC ran, after a
        multi-hour ingest had already written `<store>/ASC/`.
        """
        missing = tmp_path / "nope.tif"
        with pytest.raises(ValueError, match="orbit_direction must be one of"):
            ingest_s1tiling_acquisition(missing, missing, missing, tmp_path / "s.zarr", "ASC")

    def test_discovered_orbit_direction_is_accepted_by_the_ingester(self, tmp_path: Path) -> None:
        """The contract, end to end: what discovery emits is what ingest takes."""
        data = np.ones((SIZE, SIZE), dtype=np.float32)
        stem = "s1a_32TQM_{pol}_DES_037_20230115t061234_GammaNaughtRTC{mask}.tif"
        for pol, mask in (("vv", ""), ("vh", ""), ("vv", "_BorderMask")):
            _create_synthetic_geotiff(
                tmp_path / stem.format(pol=pol, mask=mask), data, tags=ACQ1_TAGS
            )
        (acq,) = discover_s1tiling_acquisitions(tmp_path)

        store = tmp_path / "s1-rtc-32TQM.zarr"
        ingest_s1tiling_acquisition(
            acq["vv"], acq["vh"], acq["vv_mask"], store, acq["orbit_direction"]
        )
        assert (store / "descending").exists()


# =============================================================================
# B3 / B7 / B8 — below-cap findings
# =============================================================================


def test_untagged_geotiff_does_not_abort_the_whole_discovery(tmp_path: Path) -> None:
    """B3: one bad file used to raise out of discovery, losing every other bundle in the archive
    -- while an unparseable *filename* was silently skipped."""
    data = np.ones((SIZE, SIZE), dtype=np.float32)
    # A masked multi-frame stamp forces the tag lookup, and the tags are absent.
    for pol, mask in (("vv", ""), ("vh", ""), ("vv", "_BorderMask")):
        _create_synthetic_geotiff(
            tmp_path / f"s1a_32TQM_{pol}_ASC_037_20230115txxxxxx_GammaNaughtRTC{mask}.tif", data
        )
    # A complete, well-tagged bundle that must survive the bad one.
    for pol, mask in (("vv", ""), ("vh", ""), ("vv", "_BorderMask")):
        _create_synthetic_geotiff(
            tmp_path / f"s1a_32TQM_{pol}_ASC_037_20230115t061234_GammaNaughtRTC{mask}.tif",
            data,
            tags=ACQ1_TAGS,
        )

    acqs = discover_s1tiling_acquisitions(tmp_path)

    assert [a["acq_stamp"] for a in acqs] == ["20230115t061234"]
    flagged = discover_s1tiling_acquisitions(tmp_path, skip_incomplete=False)
    assert [a["complete"] for a in flagged].count(False) == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2023:01:15T06:12:34Z", "2023-01-15T06:12:34"),
        ("2023:01:15T06:12:34", "2023-01-15T06:12:34"),
        # The space-separated form real S1Tiling output emits via GDAL's TIFF DateTime.
        ("2023:01:15 06:12:34", "2023-01-15T06:12:34"),
    ],
)
def test_normalise_datetime_accepts_the_real_forms(raw: str, expected: str) -> None:
    assert _normalise_s1tiling_datetime(raw) == expected


@pytest.mark.parametrize("raw", ["", "not-a-date", "2023:13:45T99:99:99Z", "2023:01:15T25:00:00Z"])
def test_normalise_datetime_raises_naming_the_tag_and_file(raw: str) -> None:
    """B7: an unrecognised form used to pass through untouched and fail ~560 lines later inside
    `np.datetime64`, from a traceback naming neither the tag nor the GeoTIFF."""
    with pytest.raises(ValueError, match="Unparseable ACQUISITION_DATETIME") as excinfo:
        _normalise_s1tiling_datetime(raw, source="/archive/bad.tif")
    assert repr(raw) in str(excinfo.value)
    assert "/archive/bad.tif" in str(excinfo.value)


def test_float32_nan_fill_value_matches_xarray() -> None:
    """B8: the stdlib-encoded `_FillValue` must stay equal to what xarray's encoder produces.

    `FillValueCoder` lives in `xarray.backends.zarr` and is not public API. It used to be
    imported at module scope, so any xarray reorganisation raised from `import eopf_geozarr`
    itself -- taking down the S2 and OLCI paths, which never touch S1. The value is now eight
    bytes of `struct`, which cannot break; this test is the drift alarm that keeps the two
    pinned together, so if xarray ever encodes it differently CI says so rather than every
    store we write carrying a `_FillValue` xarray no longer recognises.
    """
    from xarray.backends.zarr import FillValueCoder

    assert FillValueCoder.encode(np.nan, np.dtype("float32")) == FLOAT32_NAN_FILL_VALUE
    # Pin the literal too: a change to either side must be a deliberate, visible edit.
    assert FLOAT32_NAN_FILL_VALUE == "AAAAAAAA+H8="


class TestConditionsRepairOnAConsolidatedStore:
    """Every shipped store is consolidated, so that is the shape the repair must handle.

    The first version of this repair passed a test that never consolidated and read the
    conditions group *directly*. Both choices hid the bug: membership was answered from the
    orbit group's stale consolidated block, so the repair wrote `x` to disk while every reader
    that goes through the consolidated view still saw it missing -- and the next ordinary
    conditions call read "x is absent" from that same block and raised `ContainsArrayError`
    over the array already on disk.
    """

    @staticmethod
    def _store_with_legacy_conditions(tmp_path: Path) -> Path:
        """A consolidated store whose conditions group has no x/y -- the pre-#216 shape."""
        data = np.ones((SIZE, SIZE), dtype=np.float32)
        store = tmp_path / "s1-rtc-32TQM.zarr"
        for name in ("vv", "vh", "mask"):
            _create_synthetic_geotiff(tmp_path / f"{name}.tif", data, tags=ACQ1_TAGS)
        ingest_s1tiling_acquisition(
            tmp_path / "vv.tif", tmp_path / "vh.tif", tmp_path / "mask.tif", store, "ascending"
        )
        gamma = tmp_path / "GAMMA_AREA_32TQM_037.tif"
        _create_synthetic_geotiff(gamma, data)
        ingest_s1tiling_conditions(
            store_path=store,
            orbit_direction="ascending",
            relative_orbit=37,
            gamma_area_path=gamma,
        )
        orbit = zarr.open_group(str(store / "ascending"), mode="r+", zarr_format=3)
        conditions = _group(orbit, "conditions")
        for coord in ("x", "y"):
            conditions.__delitem__(coord)
        # THE POINT: consolidate after the deletion, so the block is what a reader sees.
        consolidate_s1_store(store, "ascending")
        return store

    def test_repair_is_visible_through_the_consolidated_view(self, tmp_path: Path) -> None:
        """The repair must reach the view STAC, xarray and TiTiler actually read."""
        store = self._store_with_legacy_conditions(tmp_path)
        gamma = tmp_path / "GAMMA_AREA_32TQM_037.tif"

        ingest_s1tiling_conditions(
            store_path=store,
            orbit_direction="ascending",
            relative_orbit=8,
            gamma_area_path=gamma,
        )

        # Read the way every consumer does: through the root, consolidated block honoured.
        root = zarr.open_group(str(store), mode="r", zarr_format=3)
        conditions = _group(_group(root, "ascending"), "conditions")
        members = set(conditions)
        for coord in ("x", "y", "spatial_ref"):
            assert coord in members, f"{coord} is missing from the consolidated view: {members}"
        assert "gamma_area_008" in members, "the newly written raster is absent from the view"

    def test_repaired_store_accepts_a_further_conditions_call(self, tmp_path: Path) -> None:
        """The repair must not brick the next ordinary call.

        Membership resolved from a stale block said `x` was absent while it existed on disk, so
        recreating it raised `ContainsArrayError` -- reaching an unrepairable state by the
        ordinary path, not a torn-write edge case.
        """
        store = self._store_with_legacy_conditions(tmp_path)
        gamma = tmp_path / "GAMMA_AREA_32TQM_037.tif"

        for relative_orbit in (8, 9):
            ingest_s1tiling_conditions(
                store_path=store,
                orbit_direction="ascending",
                relative_orbit=relative_orbit,
                gamma_area_path=gamma,
            )

        root = zarr.open_group(str(store), mode="r", zarr_format=3)
        conditions = _group(_group(root, "ascending"), "conditions")
        assert {"gamma_area_008", "gamma_area_009", "x", "y"} <= set(conditions)

    def test_every_condition_array_gets_its_grid_mapping(self, tmp_path: Path) -> None:
        """A raster created during the same call must not miss `grid_mapping`.

        `_add_grid_mapping` enumerates `group.arrays()`; resolved through a stale block that
        enumeration omitted the array just written, so `gamma_area_008` shipped with no CRS
        while its sibling on the identical grid had one -- silently, with no error or warning.
        rioxarray and TiTiler resolve the CRS through `grid_mapping` -> `spatial_ref`.
        """
        store = self._store_with_legacy_conditions(tmp_path)
        gamma = tmp_path / "GAMMA_AREA_32TQM_037.tif"

        ingest_s1tiling_conditions(
            store_path=store,
            orbit_direction="ascending",
            relative_orbit=8,
            gamma_area_path=gamma,
        )

        root = zarr.open_group(str(store), mode="r", zarr_format=3)
        conditions = _group(_group(root, "ascending"), "conditions")
        for name in ("gamma_area_037", "gamma_area_008"):
            attrs = dict(_array(conditions, name).attrs)
            assert attrs.get("grid_mapping") == "spatial_ref", (
                f"{name} has no grid_mapping, so it opens with no CRS: {sorted(attrs)}"
            )

    def test_half_repaired_group_is_completed_not_rejected(self, tmp_path: Path) -> None:
        """Only the missing coordinate is created.

        A run interrupted between the two writes leaves `x` present and `y` absent; recreating
        the pair would raise `ContainsArrayError` on `x`, turning a half-repaired store into an
        unrepairable one.
        """
        store = self._store_with_legacy_conditions(tmp_path)
        gamma = tmp_path / "GAMMA_AREA_32TQM_037.tif"
        orbit = zarr.open_group(
            str(store / "ascending"), mode="r+", zarr_format=3, use_consolidated=False
        )
        conditions = _group(orbit, "conditions")
        _create_spatial_coordinate_arrays(
            conditions, SIZE, SIZE, [10.0, 0.0, XMIN, 0.0, -10.0, YMAX], only=["x"]
        )
        assert "x" in conditions
        assert "y" not in conditions

        ingest_s1tiling_conditions(
            store_path=store,
            orbit_direction="ascending",
            relative_orbit=8,
            gamma_area_path=gamma,
        )

        root = zarr.open_group(str(store), mode="r", zarr_format=3)
        healed = _group(_group(root, "ascending"), "conditions")
        assert {"x", "y"} <= set(healed)
        # The backfilled coordinate must describe the real grid, not merely exist.
        assert np.asarray(_array(healed, "y"))[0] == pytest.approx(YMAX)
        assert np.asarray(_array(healed, "x"))[0] == pytest.approx(XMIN)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The gate (`fromisoformat`) and the consumer (`np.datetime64`) are different parsers.
        # Validating with one while passing the other the RAW text blessed forms they disagree
        # about, so the normaliser must return the parser's own canonical spelling.
        #
        # `np.datetime64("20230115")` is dtype datetime64[Y] -- the YEAR 20230115 -- which lands
        # in the store as 2206-09-06 with no error, and then makes the out-of-order guard reject
        # every later append to that orbit. Permanently poisoned, silently.
        ("20230115", "2023-01-15T00:00:00"),
        # `np.datetime64` raises on these, after the rasters have been read -- the exact late
        # failure this function exists to prevent.
        ("20230115T061234", "2023-01-15T06:12:34"),
        ("2023-W03-1", "2023-01-16T00:00:00"),
    ],
)
def test_normalise_datetime_returns_a_form_numpy_reads_identically(raw: str, expected: str) -> None:
    """Whatever the gate accepts must mean the same thing to `np.datetime64`."""
    normalised = _normalise_s1tiling_datetime(raw)
    assert normalised == expected
    # The real invariant: the stored instant equals what the gate parsed.
    assert np.datetime64(normalised) == np.datetime64(dt.datetime.fromisoformat(raw).isoformat())
    assert np.datetime64(normalised).dtype != np.dtype("datetime64[Y]")


def test_downsample_rejects_an_unknown_method() -> None:
    """An unrecognised method used to fall through to the block mean, which for the uint8 mask
    silently reproduces the defect F11 fixes: a block of [1,0,0,0] means 0.25 and truncates to 0.
    """
    data = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    assert _downsample_2d(data, 2, "max")[0, 0] == 1
    with pytest.raises(ValueError, match="Unknown downsample method"):
        _downsample_2d(data, 2, "Max")


def test_masked_multiframe_bundle_survives_an_untagged_sibling(tmp_path: Path) -> None:
    """Derived `_BorderMask` products commonly carry no ACQUISITION_DATETIME.

    Resolving the stamp per FILE keyed vv/vh under the resolved stamp and the masks under the
    masked one, splitting one acquisition into two half-bundles that `skip_incomplete` then
    discarded -- so discovery returned `[]` with every file present on disk, and an operator's
    ingest loop reported success having written nothing.
    """
    data = np.ones((SIZE, SIZE), dtype=np.float32)
    stem = "s1a_32TQM_{pol}_ASC_037_20230115txxxxxx_GammaNaughtRTC{mask}.tif"
    # vv/vh carry the tag; both masks do not, exactly as S1Tiling emits them.
    for pol, mask, tags in (
        ("vv", "", ACQ1_TAGS),
        ("vh", "", ACQ1_TAGS),
        ("vv", "_BorderMask", None),
        ("vh", "_BorderMask", None),
    ):
        _create_synthetic_geotiff(tmp_path / stem.format(pol=pol, mask=mask), data, tags=tags)

    acqs = discover_s1tiling_acquisitions(tmp_path)

    assert len(acqs) == 1, f"the bundle was split: {[a['acq_stamp'] for a in acqs]}"
    acq = acqs[0]
    assert acq["complete"] is True
    assert acq["acq_stamp"] == "20230115t061234", "the sibling's tag must stamp the whole bundle"
    assert {"vv", "vh", "vv_mask", "vh_mask"} <= set(acq)


def test_bundle_with_no_usable_tag_anywhere_is_reported_once(tmp_path: Path) -> None:
    """When NO file carries the tag the bundle is unstampable -- flagged, not split."""
    data = np.ones((SIZE, SIZE), dtype=np.float32)
    stem = "s1a_32TQM_{pol}_ASC_037_20230115txxxxxx_GammaNaughtRTC{mask}.tif"
    for pol, mask in (("vv", ""), ("vh", ""), ("vv", "_BorderMask")):
        _create_synthetic_geotiff(tmp_path / stem.format(pol=pol, mask=mask), data)

    assert discover_s1tiling_acquisitions(tmp_path) == []
    flagged = discover_s1tiling_acquisitions(tmp_path, skip_incomplete=False)
    assert len(flagged) == 1, "an unstampable bundle must stay one bundle"
    assert flagged[0]["complete"] is False
