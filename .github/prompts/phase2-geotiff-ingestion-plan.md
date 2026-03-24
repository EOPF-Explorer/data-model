# Phase 2 — GeoTIFF Ingestion: Detailed Implementation Plan

## Objective

Productionise the prototype GeoTIFF → Zarr V3 ingestion pipeline (`analysis/s1_grd_rtc_prototype.py` and `analysis/s1_real_geotiff_to_zarr.py`) into a production module at `src/eopf_geozarr/conversion/s1_ingest.py` with proper logging, error handling, and test coverage.

Phase 2 covers **data ingestion only** (not conditions, overviews, CLI, or S3) — those are Phases 3–4. However, the internal design should cleanly accommodate them.

---

## Prior Art — What Exists

| Asset | Path | Relevance |
|-------|------|-----------|
| Phase 0 synthetic prototype | `analysis/s1_grd_rtc_prototype.py` | 600 lines. Creates store, appends 2 acquisitions with overviews, validates. All synthetic 256×256 data. |
| Phase 0 real-data script | `analysis/s1_real_geotiff_to_zarr.py` | 700 lines. Handles real S1Tiling filenames, datetime normalisation (`2025:02:10T06:09:20Z`), gamma_area conditions, validation report. Validated on 3 real 10980×10980 acquisitions. |
| Phase 1 data model | `src/eopf_geozarr/data_api/s1_rtc.py` | 316 lines. Pydantic-zarr V3 models: `S1RtcRoot`, `S1RtcOrbitGroup`, `S1RtcNativeResolutionDataset`, etc. Strict validation via `@model_validator`. |
| Phase 1 test fixture | `tests/_test_data/s1_rtc_examples/s1-grd-rtc-31TCH.json` | Full JSON metadata for a 3-acquisition store with conditions. |
| Existing utilities | `src/eopf_geozarr/conversion/utils.py` | `calculate_aligned_chunk_size()`, `downsample_2d_array()` — **reuse directly** |
| Existing FS abstraction | `src/eopf_geozarr/conversion/fs_utils.py` | S3/local filesystem helpers — **reuse for S3 path support in Phase 4** |
| Real GeoTIFF metadata | `analysis/s1tiling_output_metadata.json` | Captures all 32 tags from actual S1Tiling 1.4.0 outputs |
| Zarr conventions library | `zarr_cm` (geo_proj, multiscales, spatial) | UUIDs and schema URLs used in Phase 1 models — **reuse for constants** |

### Key decisions already validated in Phase 0

- `zarr.create_array(shards=, compressors=)` works (NOT `codecs=`)
- `array.resize()` works on sharded arrays, preserves existing data
- `rasterio src.transform` returns Affine ordering natively (no GDAL conversion needed)
- Inner chunk size must evenly divide shard: `calculate_aligned_chunk_size(10980, 512)` → 366
- S1Tiling datetime format `"2025:02:10T06:09:20Z"` needs normalisation (colons in date part)
- border_mask is per-polarisation; prototype uses VV mask as primary
- Overview ceiling division: `ceil(10980/2) = 5490`, `ceil(5490/3) = 1830`, etc.

---

## Architecture

### Module: `src/eopf_geozarr/conversion/s1_ingest.py`

One file, ~500 lines estimated. Internal organisation:

```
s1_ingest.py
├── Constants (conventions, overview chain)
├── GeoTIFF metadata extraction
│   ├── S1TilingMetadata (dataclass)
│   ├── S1TILING_FILENAME_PATTERN (regex)
│   ├── extract_geotiff_metadata()
│   └── parse_s1tiling_filename()
├── Store creation
│   ├── compute_multiscales_layout()
│   └── create_s1_store()
├── Acquisition ingestion
│   ├── ingest_s1tiling_acquisition()    ← PUBLIC API
│   └── _normalise_s1tiling_datetime()
├── File discovery
│   └── discover_s1tiling_acquisitions() ← PUBLIC API
└── (future hooks for conditions and overview levels)
```

### Tests: `tests/test_s1_rtc_ingest.py`

~300 lines estimated. Uses synthetic GeoTIFFs created via `rasterio` in fixtures.

---

## Step-by-step Implementation

### Step 1: Constants and dataclass

**What**: Define `S1TilingMetadata` dataclass and shared constants.

**Details**:
- `S1TilingMetadata`: A frozen dataclass holding all fields extracted from a GeoTIFF (crs, spatial_transform, shape, bounds, datetime, absolute_orbit, relative_orbit, platform, calibration, input_s1_images). NOT a Pydantic model — this is simple data transfer, not validation.
- `OVERVIEW_CHAIN`: same `[("r10m", None, 1), ("r20m", "r10m", 2), ...]` tuple list
- Zarr convention constants: import from `zarr_cm` (already used in Phase 1 models) rather than hardcoding UUIDs. Specifically:
  ```python
  from zarr_cm import geo_proj, multiscales as multiscales_cm, spatial as spatial_cm
  MULTISCALES_UUID = multiscales_cm.UUID
  GEO_PROJ_UUID = geo_proj.UUID
  SPATIAL_UUID = spatial_cm.UUID
  ```
- `ZARR_CONVENTIONS` list: build from `zarr_cm` library attributes (schema_url, spec_url, name, description, uuid) rather than hardcoding the full dicts. Check if `zarr_cm` exposes these as structured objects; if not, hardcode as in prototype.
- `S1TILING_FILENAME_PATTERN`: the regex from `s1_real_geotiff_to_zarr.py`

**Reuse**: Constants pattern from `s1_rtc.py` (Phase 1). `calculate_aligned_chunk_size` from `utils.py`.

---

### Step 2: Metadata extraction

**What**: `extract_geotiff_metadata(path) -> S1TilingMetadata` and `parse_s1tiling_filename(filename) -> dict`.

**Details**:

`extract_geotiff_metadata`:
- Opens GeoTIFF with `rasterio.open()` (read-only)
- Reads CRS (`str(src.crs)`), transform (Affine → list of 6 floats `[a, b, c, d, e, f]`), bounds, shape
- Reads custom tags: `ACQUISITION_DATETIME`, `ORBIT_NUMBER`, `RELATIVE_ORBIT_NUMBER`, `FLYING_UNIT_CODE`, `CALIBRATION`, `INPUT_S1_IMAGES`
- Normalises datetime: `_normalise_s1tiling_datetime("2025:02:10T06:09:20Z")` → `"2025-02-10T06:09:20"`
- Returns `S1TilingMetadata` instance
- Raises `ValueError` if critical tags are missing (ACQUISITION_DATETIME, ORBIT_NUMBER, RELATIVE_ORBIT_NUMBER, FLYING_UNIT_CODE)
- structlog for info-level logging of extracted metadata

`parse_s1tiling_filename`:
- Applies `S1TILING_FILENAME_PATTERN` regex to filename string
- Returns dict with keys: platform, tile, pol, orbit_dir, rel_orbit, acq_stamp, is_mask
- Returns `None` if filename doesn't match (not an error — allows callers to skip)

`_normalise_s1tiling_datetime`:
- Input: `"2025:02:10T06:09:20Z"` (S1Tiling format with colons in date)
- Output: `"2025-02-10T06:09:20"` (ISO 8601 for `np.datetime64`)
- Strip trailing Z, split on T, replace colons with hyphens in date part only

**Source to lift from**: `s1_real_geotiff_to_zarr.py` lines 107–132 (extract_geotiff_metadata) and lines 91–105 (S1TILING_PATTERN + parse logic) and lines 392–399 (datetime normalisation).

---

### Step 3: Multiscales layout computation

**What**: `compute_multiscales_layout(native_shape, native_transform) -> list[dict]`

**Details**:
- Takes native shape `[H, W]` and transform `[a, b, c, d, e, f]` (Affine ordering)
- Iterates OVERVIEW_CHAIN, producing one layout entry per level
- Each entry: `{"asset": name, "spatial:shape": [h, w], "spatial:transform": [a,b,c,d,e,f], "transform": {"scale": [f, f]}, ...}`
- Native level has `"transform": {"scale": [1.0, 1.0]}`; deeper levels have `"derived_from": parent_name`
- Shape computation: `ceil(parent_h / factor)`, `ceil(parent_w / factor)`
- Transform: scale pixel size by factor (`transform[0] *= factor`, `transform[4] *= factor`), keep origin

**Source to lift from**: `s1_grd_rtc_prototype.py` lines 129–173 (`compute_multiscales_layout`). Unchanged logic, production-ready.

---

### Step 4: Store creation

**What**: `create_s1_store(store_path, orbit_direction, metadata) -> zarr.Group`

**Details**:
- Creates Zarr V3 store at `store_path` (local path string or Path)
- `zarr.open_group(str(store_path), mode="w", zarr_format=3)`
- Creates orbit direction group with full conventions attributes:
  - `zarr_conventions`: ZARR_CONVENTIONS list
  - `multiscales`: `{"layout": [...], "resampling_method": "average"}`
  - `proj:code`: from metadata CRS
  - `spatial:dimensions`: `["Y", "X"]`
  - `spatial:bbox`: from metadata bounds
- For each level in layout:
  - Create level group with `spatial:shape` and `spatial:transform` attributes
  - Create `vv`, `vh`, `border_mask` arrays with:
    - `shape=(0, level_h, level_w)` — time axis starts at 0
    - `dtype`: float32 for vv/vh, uint8 for border_mask
    - `fill_value`: NaN for float32, 0 for uint8
    - `chunks`: `(1, best_chunk(level_h), best_chunk(level_w))` using `calculate_aligned_chunk_size()`
    - `shards`: `(1, level_h, level_w)` — one shard per timestep
    - `compressors`: `BloscCodec(cname="zstd", clevel=5)`
    - `dimension_names`: `["time", "Y", "X"]`
- Create coordinate variables at r10m only:
  - `time`: int64, shape (0,), chunks (512,), dim_names ["time"]
  - `absolute_orbit`: int32, same
  - `relative_orbit`: int32, same
  - `platform`: `<U4`, same (accepts Zarr V3 FixedLengthUTF32 warning)
- Returns the root group

**Open question for implementer**: The prototype uses `mode="w"` which overwrites. Production code should check if store already exists and raise an error (or use `mode="w-"`). The caller (`ingest_s1tiling_acquisition`) handles the create-or-append branching.

**Source to lift from**: `s1_real_geotiff_to_zarr.py` lines 239–309 (`create_s1_store`). Change `min(512, dim)` to `calculate_aligned_chunk_size(dim, 512)`.

---

### Step 5: Acquisition ingestion (core)

**What**: `ingest_s1tiling_acquisition(vv_path, vh_path, border_mask_path, store_path, orbit_direction) -> int`

This is the **main public API** for Phase 2.

**Details**:

1. **Extract metadata** from vv_path via `extract_geotiff_metadata()`
2. **Create-or-open store**:
   - If store doesn't exist → call `create_s1_store()`
   - If store exists → open in `mode="r+"`
   - If store exists but orbit direction group doesn't exist → create it
3. **Read GeoTIFF pixel data**:
   - `rasterio.open(vv_path).read(1)` → vv_data (float32)
   - `rasterio.open(vh_path).read(1)` → vh_data (float32)
   - `rasterio.open(border_mask_path).read(1).astype(np.uint8)` → mask_data
4. **Determine time index**: `current_size = r10m["vv"].shape[0]`, `new_size = current_size + 1`
5. **Generate overviews** from native data:
   - For each level in `OVERVIEW_CHAIN[1:]`: downsample from previous level
   - vv/vh: `downsample_2d` with `"average"` method
   - border_mask: `downsample_2d` with `"nearest"` method
   - Use existing `downsample_2d_array()` from `utils.py` (different signature — takes target_height/target_width instead of factor, so compute target sizes first using ceiling division)
   - **NOTE**: The existing `downsample_2d_array()` in `utils.py` has a subtly different interface from the prototype's `downsample_2d()`. The prototype takes a `factor` and computes target dims internally with ceiling division. The production util takes target dims directly. Need to compute `target_h = ceil(h / factor)`, `target_w = ceil(w / factor)` before calling.
   - **ALTERNATIVE**: The prototype's `downsample_2d()` is simpler and handles edge padding correctly for non-divisible sizes. Consider adding it as a private helper in `s1_ingest.py` or adding a `factor`-based variant to `utils.py`. Decision: **Use the prototype's `downsample_2d()` as a private helper** — it's purpose-built for power-of-2/3 downsampling with padding, which is the S1 use case. The existing `downsample_2d_array()` was designed for arbitrary target dimensions (S2 overview path).
6. **Write data at all levels**:
   - For each `(level_name, data_tuple)`:
     - `level["vv"].resize((new_size, h, w))`
     - `level["vv"][current_size, :, :] = vv_lev`
     - Same for vh, border_mask
7. **Append coordinate variables**:
   - Resize all 1-D arrays to `new_size`
   - Write `time[current_size] = np.datetime64(dt).astype("datetime64[ns]").astype(np.int64)`
   - Write `absolute_orbit[current_size] = meta.absolute_orbit`
   - Write `relative_orbit[current_size] = meta.relative_orbit`
   - Write `platform[current_size] = meta.platform`
8. **Return** the time index (`current_size`)

**Logging** (structlog):
- INFO: "Ingesting S1 acquisition", vv_path, orbit_direction, time_index
- INFO: "GeoTIFF read complete", read_time_s, vv_min, vv_max, mask_coverage_pct
- INFO: "Overviews generated", overview_time_s
- INFO: "Zarr write complete", write_time_s, levels_written

**Error handling** (at system boundary — GeoTIFF files are external input):
- `FileNotFoundError`: if vv/vh/mask paths don't exist
- `ValueError`: if GeoTIFF CRS doesn't match store's `proj:code` (on append)
- `ValueError`: if GeoTIFF shape doesn't match store's native shape (on append)
- Let zarr exceptions propagate naturally for I/O errors

**Source to lift from**: `s1_real_geotiff_to_zarr.py` lines 352–432 (`ingest_acquisition`). Add create-or-open logic, CRS/shape consistency checks, structlog.

---

### Step 6: File discovery utility

**What**: `discover_s1tiling_acquisitions(input_dir) -> list[dict]`

**Details**:
- Glob `input_dir/*.tif`, apply `parse_s1tiling_filename()` to each
- Group by `(platform, tile, orbit_dir, rel_orbit, acq_stamp)` key
- Validate each group has vv, vh, vv_mask, vh_mask
- Return list of dicts with keys: `platform, tile, orbit_dir, rel_orbit, acq_stamp, vv, vh, vv_mask, vh_mask` (paths as `Path` objects)
- Log warnings for incomplete acquisitions

This is mainly useful for the CLI batch ingest (Phase 4) but is simple to implement and test now.

**Source to lift from**: `s1_real_geotiff_to_zarr.py` lines 142–178 (`discover_acquisitions`).

---

### Step 7: Module exports and wiring

**What**: Register public API in `__init__.py`.

**Details**:
- Add to `src/eopf_geozarr/conversion/__init__.py`:
  ```python
  from .s1_ingest import (
      ingest_s1tiling_acquisition,
      discover_s1tiling_acquisitions,
      extract_geotiff_metadata,
  )
  ```
- Add to `__all__`

---

### Step 8: Test — synthetic GeoTIFF fixtures

**What**: Create pytest fixtures that produce synthetic S1Tiling GeoTIFFs.

**File**: `tests/test_s1_rtc_ingest.py`

**Fixtures**:
- `s1_geotiff_dir(tmp_path)`: Creates a temp directory with synthetic GeoTIFFs for 2 acquisitions:
  - `s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif`
  - `s1a_32TQM_vh_ASC_037_20230115t061234_GammaNaughtRTC.tif`
  - `s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC_BorderMask.tif`
  - `s1a_32TQM_vh_ASC_037_20230115t061234_GammaNaughtRTC_BorderMask.tif`
  - (second acquisition: `20230127t061235`)
  - All 256×256, EPSG:32633, with proper tags (`ACQUISITION_DATETIME`, `ORBIT_NUMBER`, etc.)
- `s1_store_path(tmp_path)`: Returns a clean path for Zarr store output

**Helper**: `_create_synthetic_geotiff(path, data, crs, transform, tags)` — same as prototype's `create_test_geotiff()`.

---

### Step 9: Test — metadata extraction

**Tests**:
- `test_extract_geotiff_metadata()`: Verify all fields populated from tags
- `test_extract_geotiff_metadata_normalises_datetime()`: `"2025:02:10T06:09:20Z"` → `"2025-02-10T06:09:20"`
- `test_extract_geotiff_metadata_raises_on_missing_tags()`: Missing ORBIT_NUMBER → ValueError
- `test_parse_s1tiling_filename()`: Correct field extraction for VV, VH, mask variants
- `test_parse_s1tiling_filename_returns_none_for_unknown()`: Non-matching filename

---

### Step 10: Test — store creation

**Tests**:
- `test_create_s1_store_structure()`: Creates store, verifies group hierarchy (root → ascending → r10m…r720m)
- `test_create_s1_store_conventions()`: Verifies `zarr_conventions`, `proj:code`, `spatial:dimensions`, `spatial:bbox` on orbit group
- `test_create_s1_store_array_metadata()`: Verifies `dimension_names`, dtype, fill_value, shape=(0,H,W) for data arrays
- `test_create_s1_store_coordinate_arrays()`: Verifies time, absolute_orbit, relative_orbit, platform at r10m only
- `test_create_s1_store_overview_shapes()`: Verifies consistent shape chain (ceiling division)

---

### Step 11: Test — ingestion (create + append)

**Tests**:
- `test_ingest_first_acquisition()`: Ingest into non-existing store → store created, time_index=0, data readable
- `test_ingest_second_acquisition_appends()`: Ingest twice → shape[0]=2, both timesteps have data
- `test_ingest_preserves_data_integrity()`: Write known data, read back, compare values (within float32 tolerance for vv/vh, exact for mask)
- `test_ingest_coordinate_values()`: Verify time, orbit, platform values written correctly
- `test_ingest_overview_consistency()`: Verify overview shapes follow ceiling division chain
- `test_ingest_rejects_mismatched_crs()`: Second acquisition with different CRS → ValueError
- `test_ingest_rejects_mismatched_shape()`: Second acquisition with different shape → ValueError
- `test_ingest_xarray_roundtrip()`: After 2 ingestions, `xr.open_zarr(r10m_path)` succeeds, `ds.sortby("time")` works

---

### Step 12: Test — file discovery

**Tests**:
- `test_discover_acquisitions_groups_correctly()`: 2 acquisitions × 4 files each → 2 groups
- `test_discover_acquisitions_warns_on_incomplete()`: Missing VH file → warning logged, still returns partial
- `test_discover_acquisitions_skips_non_matching()`: Random .tif files in directory → ignored

---

## What is NOT in Phase 2

These are explicitly deferred:

| Item | Phase | Reason |
|------|-------|--------|
| Conditions ingestion (gamma_area, LIA) | Phase 3 | Separate data path, no time dimension |
| Conditions group with own conventions | Phase 3 | Depends on conditions ingestion |
| CLI subcommands (ingest-s1, etc.) | Phase 4 | Needs stable public API first |
| S3 output support | Phase 4 | Requires fs_utils integration; local-first |
| Multiscale generation as separate concern | Phase 3 | Currently embedded in ingest; may refactor |
| STAC item creation/update | Phase 5 | Needs STAC collection definition first |
| Validation via Phase 1 Pydantic models | Phase 3+ | Optional; store structure is validated by tests |
| border_mask combining (VV ∪ VH) | Future | Prototype uses VV-only; acceptable for now |

---

## Dependency/Import Map

```
s1_ingest.py imports:
  ├── from __future__ import annotations
  ├── dataclasses (dataclass, frozen)
  ├── math (ceil)
  ├── pathlib (Path)
  ├── re
  ├── numpy as np
  ├── rasterio
  ├── structlog
  ├── zarr
  ├── zarr.codecs (BloscCodec)
  ├── zarr_cm (geo_proj, multiscales, spatial)  # UUIDs and schema URLs
  └── eopf_geozarr.conversion.utils (calculate_aligned_chunk_size)
```

No new external dependencies. All imports already in the project.

---

## Estimated Scope

| Component | Lines (est.) | Complexity |
|-----------|-------------|------------|
| Constants + S1TilingMetadata | ~50 | Low |
| extract_geotiff_metadata + helpers | ~70 | Low |
| compute_multiscales_layout | ~50 | Low (direct lift) |
| create_s1_store | ~80 | Medium |
| ingest_s1tiling_acquisition | ~120 | Medium-High |
| discover_s1tiling_acquisitions | ~50 | Low |
| _downsample_2d (private helper) | ~30 | Low (direct lift) |
| **Total s1_ingest.py** | **~450** | |
| Test fixtures | ~80 | Low |
| Test cases (12 tests) | ~250 | Medium |
| **Total test_s1_rtc_ingest.py** | **~330** | |

---

## Implementation Order for the Executing Agent

The steps above are ordered for incremental, testable progress. A reasonable execution sequence:

1. **Steps 1–3**: Constants, metadata extraction, multiscales layout — pure functions, no I/O beyond rasterio reads
2. **Step 8**: Create test fixtures (needed for all subsequent testing)
3. **Steps 9**: Test metadata extraction
4. **Step 4**: Store creation
5. **Step 10**: Test store creation
6. **Step 5**: Acquisition ingestion
7. **Steps 6**: File discovery
8. **Steps 11–12**: Integration tests
9. **Step 7**: Wire up exports

Each step can be committed independently. The agent should run `pytest tests/test_s1_rtc_ingest.py -v` after each test step to verify green.
