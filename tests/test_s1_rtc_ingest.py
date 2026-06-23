"""Tests for S1 GRD RTC GeoTIFF → GeoZarr V3 ingestion pipeline."""

from __future__ import annotations

import os
from math import ceil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import xarray as xr
import zarr
from rasterio.transform import from_bounds

from eopf_geozarr.conversion.s1_ingest import (
    OVERVIEW_CHAIN,
    S1TilingMetadata,
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


def _create_synthetic_geotiff(
    path: Path,
    data: np.ndarray,
    crs: str = CRS,
    transform: rasterio.transform.Affine | None = None,
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
        result = parse_s1tiling_filename(
            "s1a_32TQM_vv_ASC_037_20230115txxxxxx_GammaNaughtRTC.tif"
        )
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
        attrs = dict(root["ascending"].attrs)
        assert "zarr_conventions" in attrs
        conv_names = {c["name"] for c in attrs["zarr_conventions"]}
        assert "multiscales" in conv_names
        assert "proj:" in conv_names
        assert "spatial:" in conv_names
        assert attrs["proj:code"] == CRS
        assert attrs["spatial:dimensions"] == ["y", "x"]
        assert len(attrs["spatial:bbox"]) == 4

    def test_array_metadata(self, s1_store_path: Path, sample_metadata: S1TilingMetadata) -> None:
        root = create_s1_store(s1_store_path, "ascending", sample_metadata)
        r10m = root["ascending"]["r10m"]
        for arr_name in ["vv", "vh", "border_mask"]:
            arr = r10m[arr_name]
            assert arr.metadata.dimension_names == ("time", "y", "x")
            assert arr.shape[0] == 0  # time axis starts at 0
        assert r10m["vv"].dtype == np.float32
        assert r10m["border_mask"].dtype == np.uint8

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
        orbit = root["ascending"]
        expected = FillValueCoder.encode(np.nan, np.dtype("float32"))
        for level_name, _, _ in OVERVIEW_CHAIN:
            for band in ("vv", "vh"):
                attrs = dict(orbit[level_name][band].attrs)
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
        ms = dict(root["ascending"].attrs)["multiscales"]
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
        r10m = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)["ascending"]["r10m"]
        assert "spatial_ref" in list(r10m.array_keys())
        assert dict(r10m["vv"].attrs).get("grid_mapping") == "spatial_ref"
        assert dict(r10m["vh"].attrs).get("grid_mapping") == "spatial_ref"

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
        r10m = root["ascending"]["r10m"]
        for coord_name in ["time", "absolute_orbit", "relative_orbit", "platform"]:
            assert coord_name in r10m, f"Missing coord {coord_name}"
            assert r10m[coord_name].shape == (0,)

    def test_overview_shapes(self, s1_store_path: Path, sample_metadata: S1TilingMetadata) -> None:
        root = create_s1_store(s1_store_path, "ascending", sample_metadata)
        orbit = root["ascending"]
        # Verify shape chain follows ceiling division
        expected_h, expected_w = SIZE, SIZE
        for level_name, _, factor in OVERVIEW_CHAIN:
            if factor > 1:
                expected_h = ceil(expected_h / factor)
                expected_w = ceil(expected_w / factor)
            level = orbit[level_name]
            arr = level["vv"]
            assert arr.shape[1] == expected_h
            assert arr.shape[2] == expected_w

    def test_spatial_coordinate_arrays(
        self, s1_store_path: Path, sample_metadata: S1TilingMetadata
    ) -> None:
        """Verify x and y 1D arrays exist at every resolution level."""
        root = create_s1_store(s1_store_path, "ascending", sample_metadata)
        orbit = root["ascending"]
        for level_name, _, _ in OVERVIEW_CHAIN:
            level = orbit[level_name]
            for coord in ["x", "y"]:
                assert coord in level, f"Missing {coord} at {level_name}"
                arr = level[coord]
                assert len(arr.shape) == 1
                attrs = dict(arr.attrs)
                assert "units" in attrs
                assert "standard_name" in attrs
                assert "_ARRAY_DIMENSIONS" in attrs

            # Verify x array shape matches level width
            level_attrs = dict(level.attrs)
            level_h, level_w = level_attrs["spatial:shape"]
            assert level["x"].shape[0] == level_w
            assert level["y"].shape[0] == level_h


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
        assert root["ascending"]["r10m"]["vv"].shape[0] == 1

    def test_second_acquisition_appends(self, s1_geotiff_dir: Path, s1_store_path: Path) -> None:
        vv1, vh1, mask1 = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        vv2, vh2, mask2 = self._get_acq_paths(s1_geotiff_dir, "20230127t061235")
        ingest_s1tiling_acquisition(vv1, vh1, mask1, s1_store_path, "ascending")
        idx = ingest_s1tiling_acquisition(vv2, vh2, mask2, s1_store_path, "ascending")
        assert idx == 1
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        assert root["ascending"]["r10m"]["vv"].shape[0] == 2

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
                    attrs = dict(root[orbit][level_name][band].attrs)
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
                attrs = dict(root[orbit][level_name].attrs)
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
            assert masked.mask[0, 0, 0], "nodata cell (border_mask==0) must be masked"
            assert not masked.mask[0, -1, -1], "valid cell must not be masked"
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
        nodata = mask_data == 0
        for band in ("vv", "vh"):
            native = root["ascending"]["r10m"][band][0, :, :]
            assert np.all(np.isnan(native[nodata])), f"{band}: nodata region must be NaN, not 0"
            assert not np.any(np.isnan(native[~nodata])), f"{band}: valid region must stay finite"
            assert not np.any(native[nodata] == 0.0), f"{band}: nodata must not read back as 0"

        # NaN propagates through np.nanmean downsampling: the all-nodata top band stays NaN.
        coarse = root["ascending"]["r20m"]["vv"][0, :, :]
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
        actual_vv = root["ascending"]["r10m"]["vv"][0, :, :]

        # Valid pixels (border_mask == 1) are preserved exactly; out-of-swath pixels
        # (border_mask == 0) are written as NaN, not the raw 0 — the render-bug fix.
        valid = expected_mask == 1
        np.testing.assert_allclose(actual_vv[valid], expected_vv[valid], rtol=1e-6)
        assert np.all(np.isnan(actual_vv[~valid])), "out-of-swath nodata must be NaN"

        # border_mask itself is stored verbatim (uint8, never masked).
        actual_mask = root["ascending"]["r10m"]["border_mask"][0, :, :]
        np.testing.assert_array_equal(actual_mask, expected_mask)

    def test_coordinate_values(self, s1_geotiff_dir: Path, s1_store_path: Path) -> None:
        vv, vh, mask = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")

        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        r10m = root["ascending"]["r10m"]
        assert r10m["absolute_orbit"][0] == 47001
        assert r10m["relative_orbit"][0] == 37
        assert str(r10m["platform"][0]) == "S1A"

        # Verify time is a valid nanosecond timestamp (stored as int64)
        time_val = int(r10m["time"][0])
        dt = np.datetime64(time_val, "ns")
        assert str(dt).startswith("2023-01-15")

    def test_overview_consistency(self, s1_geotiff_dir: Path, s1_store_path: Path) -> None:
        vv, vh, mask = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")

        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        orbit = root["ascending"]
        expected_h, expected_w = SIZE, SIZE
        for level_name, _, factor in OVERVIEW_CHAIN:
            if factor > 1:
                expected_h = ceil(expected_h / factor)
                expected_w = ceil(expected_w / factor)
            arr = orbit[level_name]["vv"]
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
        orbit = root["ascending"]
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
        r10m = root["ascending"]["r10m"]
        assert r10m["vv"].shape[0] == 2

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

    def test_warns_on_incomplete(self, tmp_path: Path) -> None:
        # Create only VV (no VH, no masks)
        data = np.ones((SIZE, SIZE), dtype=np.float32)
        fname = "s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif"
        _create_synthetic_geotiff(tmp_path / fname, data, tags=ACQ1_TAGS)

        acqs = discover_s1tiling_acquisitions(tmp_path)
        assert len(acqs) == 1
        # Should be missing vh, vv_mask, vh_mask
        missing = [k for k in ("vh", "vv_mask", "vh_mask") if k not in acqs[0]]
        assert len(missing) == 3

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
        orbit = root["ascending"]
        assert "conditions" in orbit
        conditions = orbit["conditions"]
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
        conditions = root["ascending"]["conditions"]
        attrs = dict(conditions.attrs)
        assert attrs["proj:code"] == CRS
        assert attrs["spatial:dimensions"] == ["y", "x"]
        assert len(attrs["spatial:transform"]) == 6
        assert attrs["spatial:shape"] == [SIZE, SIZE]
        # CF grid-mapping so rioxarray can resolve the CRS of the condition arrays
        assert "spatial_ref" in list(conditions.array_keys())
        assert dict(conditions["gamma_area_037"].attrs).get("grid_mapping") == "spatial_ref"

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
        arr = root["ascending"]["conditions"]["gamma_area_037"]
        assert arr.shape == (SIZE, SIZE)
        assert arr.dtype == np.float32
        assert arr.metadata.dimension_names == ("y", "x")

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
        actual = root["ascending"]["conditions"]["gamma_area_037"][:]
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
        conditions = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)[
            "ascending"
        ]["conditions"]
        arr = conditions["gamma_area_037"][:]
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
        conditions = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)[
            "ascending"
        ]["conditions"]
        for arr_name in ("gamma_area_037", "lia_037"):
            assert dict(conditions[arr_name].attrs).get("_FillValue") == expected, arr_name

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
        arr = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)[
            "ascending"
        ]["conditions"]["gamma_area_037"]
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
        arr = zarr.open_group(str(s1_store_with_acquisition), mode="r", zarr_format=3)[
            "ascending"
        ]["conditions"]["gamma_area_037"]
        assert arr.chunks == (366, 366)
        assert arr.shards == (big, big)
        # exactly one chunk-data object on disk (the shard), regardless of the 9 inner chunks
        array_dir = s1_store_with_acquisition / "ascending" / "conditions" / "gamma_area_037"
        data_objects = [
            f for _r, _d, files in os.walk(array_dir) for f in files if f != "zarr.json"
        ]
        assert len(data_objects) == 1, data_objects
        # values still byte-identical through the shard
        np.testing.assert_allclose(arr[:], data, rtol=1e-6)

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
        conditions = root["ascending"]["conditions"]
        assert "gamma_area_037" in conditions
        assert "lia_037" in conditions

    def test_multiple_orbits(
        self, s1_store_with_acquisition: Path, tmp_path: Path
    ) -> None:
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
        conditions = root["ascending"]["conditions"]
        assert "gamma_area_037" in conditions
        assert "gamma_area_110" in conditions

    def test_overwrite_existing_condition(
        self, s1_store_with_acquisition: Path, tmp_path: Path
    ) -> None:
        """Writing the same condition array twice overwrites data."""
        rng = np.random.default_rng(102)
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
        actual = root["ascending"]["conditions"]["gamma_area_037"][:]
        np.testing.assert_allclose(actual, data_v2, rtol=1e-6)

    def test_raises_no_conditions_provided(
        self, s1_store_with_acquisition: Path
    ) -> None:
        with pytest.raises(ValueError, match="At least one condition"):
            ingest_s1tiling_conditions(
                s1_store_with_acquisition, "ascending", 37
            )

    def test_raises_store_not_exists(self, tmp_path: Path, gamma_area_geotiff: Path) -> None:
        with pytest.raises(ValueError, match="Store does not exist"):
            ingest_s1tiling_conditions(
                tmp_path / "nonexistent.zarr", "ascending", 37,
                gamma_area_path=gamma_area_geotiff,
            )

    def test_raises_orbit_not_exists(
        self, tmp_path: Path, gamma_area_geotiff: Path
    ) -> None:
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
                s1_store_with_acquisition, "ascending", 37,
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
        orbit = root["ascending"]
        assert orbit.metadata.consolidated_metadata is not None
        # Conditions group should be accessible through consolidated metadata
        assert "conditions" in orbit
        assert "gamma_area_037" in orbit["conditions"]


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
        for level in _LEVELS:
            attrs = dict(root["ascending"][level]["time"].attrs)
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
        returns the right physical slice at both the native and a coarse level (the 31TEH case)."""
        vv2, vh2, mask2 = self._paths(s1_geotiff_dir, "20230127t061235")  # 2023-01-27 (later)
        vv1, vh1, mask1 = self._paths(s1_geotiff_dir, "20230115t061234")  # 2023-01-15 (earlier)
        ingest_s1tiling_acquisition(vv2, vh2, mask2, s1_store_path, "ascending")  # -> index 0
        ingest_s1tiling_acquisition(vv1, vh1, mask1, s1_store_path, "ascending")  # -> index 1

        dt = xr.open_datatree(
            str(s1_store_path), engine="zarr", decode_times=True, consolidated=False
        )
        times = dt["ascending"]["r10m"]["time"].values
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
        ref = np.asarray(root["ascending"]["r10m"]["time"][:])
        assert ref.shape == (2,)
        for level in _LEVELS[1:]:
            np.testing.assert_array_equal(np.asarray(root["ascending"][level]["time"][:]), ref)

    def test_r10m_time_still_int64_for_register(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """register_per_acquisition reads r10m/time as raw int64 ns — CF attrs must not change that."""
        vv, vh, mask = self._paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        arr = root["ascending"]["r10m"]["time"]
        assert arr.dtype == np.dtype("int64")
        assert str(np.datetime64(int(arr[0]), "ns")).startswith("2023-01-15")


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
        return [n for n, _ in root["ascending"].groups() if n not in ("r10m", "conditions")]

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
        ref = list(root["ascending"]["r10m"]["time"][:])
        assert len(ref) == 2
        for lvl in coarse:
            t = root["ascending"][lvl]["time"]
            assert t.dtype == np.dtype("int64")
            assert list(t[:]) == ref  # backfilled prior slice + appended new slice, matching r10m

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
        for lvl in self._coarse_levels(s1_store_path):
            assert root["ascending"][lvl]["time"].shape[0] == 2

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
        r20m = root["ascending"]["r20m"]
        _, h, w = r20m["vv"].shape
        r20m["vv"].resize((1, h, w))  # half-built: r20m has 1 slice, r10m has 2
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
