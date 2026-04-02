"""Tests for S1 GRD RTC GeoTIFF → GeoZarr V3 ingestion pipeline."""

from __future__ import annotations

from math import ceil
from pathlib import Path

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
    extract_geotiff_metadata,
    ingest_s1tiling_acquisition,
    parse_s1tiling_filename,
)
from eopf_geozarr.data_api.s1_rtc import S1RtcRoot
from pydantic_zarr.v3 import GroupSpec

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
) -> None:
    """Write a single-band GeoTIFF with optional metadata tags."""
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
    ) as dst:
        if tags:
            dst.update_tags(**tags)
        dst.write(data, 1)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
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


@pytest.fixture()
def s1_store_path(tmp_path: Path) -> Path:
    """Return a clean path for Zarr store output."""
    return tmp_path / "s1-grd-rtc-test.zarr"


@pytest.fixture()
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
        result = parse_s1tiling_filename(
            "s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif"
        )
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

    def test_returns_none_for_unknown(self) -> None:
        assert parse_s1tiling_filename("random_file.tif") is None
        assert parse_s1tiling_filename("not_a_geotiff.txt") is None


# =============================================================================
# Step 10: Store creation tests
# =============================================================================


@pytest.fixture()
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

    def test_array_metadata(
        self, s1_store_path: Path, sample_metadata: S1TilingMetadata
    ) -> None:
        root = create_s1_store(s1_store_path, "ascending", sample_metadata)
        r10m = root["ascending"]["r10m"]
        for arr_name in ["vv", "vh", "border_mask"]:
            arr = r10m[arr_name]
            assert arr.metadata.dimension_names == ("time", "y", "x")
            assert arr.shape[0] == 0  # time axis starts at 0
        assert r10m["vv"].dtype == np.float32
        assert r10m["border_mask"].dtype == np.uint8

    def test_coordinate_variables(
        self, s1_store_path: Path, sample_metadata: S1TilingMetadata
    ) -> None:
        root = create_s1_store(s1_store_path, "ascending", sample_metadata)
        r10m = root["ascending"]["r10m"]
        for coord_name in ["time", "absolute_orbit", "relative_orbit", "platform"]:
            assert coord_name in r10m, f"Missing coord {coord_name}"
            assert r10m[coord_name].shape == (0,)

    def test_overview_shapes(
        self, s1_store_path: Path, sample_metadata: S1TilingMetadata
    ) -> None:
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

    def test_first_acquisition(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        vv, vh, mask = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        idx = ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")
        assert idx == 0
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        assert root["ascending"]["r10m"]["vv"].shape[0] == 1

    def test_second_acquisition_appends(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        vv1, vh1, mask1 = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        vv2, vh2, mask2 = self._get_acq_paths(s1_geotiff_dir, "20230127t061235")
        ingest_s1tiling_acquisition(vv1, vh1, mask1, s1_store_path, "ascending")
        idx = ingest_s1tiling_acquisition(vv2, vh2, mask2, s1_store_path, "ascending")
        assert idx == 1
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        assert root["ascending"]["r10m"]["vv"].shape[0] == 2

    def test_preserves_data_integrity(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        vv, vh, mask = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")

        # Read back and compare
        with rasterio.open(str(vv)) as src:
            expected_vv = src.read(1)
        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        actual_vv = root["ascending"]["r10m"]["vv"][0, :, :]
        np.testing.assert_allclose(actual_vv, expected_vv, rtol=1e-6)

        # Mask should be exact
        with rasterio.open(str(mask)) as src:
            expected_mask = src.read(1).astype(np.uint8)
        actual_mask = root["ascending"]["r10m"]["border_mask"][0, :, :]
        np.testing.assert_array_equal(actual_mask, expected_mask)

    def test_coordinate_values(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
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

    def test_overview_consistency(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
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
            _create_synthetic_geotiff(
                wrong_crs_dir / name, d, crs="EPSG:32632", tags=ACQ1_TAGS
            )

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

    def test_xarray_roundtrip(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
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
    def test_consolidate_s1_store(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
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


# =============================================================================
# Schema validation tests
# =============================================================================


class TestSchemaValidation:
    def _get_acq_paths(self, geotiff_dir: Path, stamp: str) -> tuple[Path, Path, Path]:
        vv = geotiff_dir / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC.tif"
        vh = geotiff_dir / f"s1a_32TQM_vh_ASC_037_{stamp}_GammaNaughtRTC.tif"
        mask = geotiff_dir / f"s1a_32TQM_vv_ASC_037_{stamp}_GammaNaughtRTC_BorderMask.tif"
        return vv, vh, mask

    def test_produced_store_conforms_to_schema(
        self, s1_geotiff_dir: Path, s1_store_path: Path
    ) -> None:
        """Verify that an ingested store validates against S1RtcRoot."""
        vv, vh, mask = self._get_acq_paths(s1_geotiff_dir, "20230115t061234")
        ingest_s1tiling_acquisition(vv, vh, mask, s1_store_path, "ascending")

        root = zarr.open_group(str(s1_store_path), mode="r", zarr_format=3)
        untyped = GroupSpec.from_zarr(root).model_dump()
        model = S1RtcRoot(**untyped)
        assert model.ascending is not None
        assert "vv" in model.ascending.r10m.members
        assert "x" in model.ascending.r10m.members
        assert "y" in model.ascending.r10m.members
