# Sentinel-1 GRD γ0T RTC — Implementation Plan for data-model codebase

## Context

EOPF Explorer needs Sentinel-1 GRD data in its catalog. ESA has not provided NRB ARCO products. We use **S1Tiling** (CNES/OTB) to produce γ0T RTC GeoTIFFs on the Sentinel-2 MGRS grid, then convert those GeoTIFFs into GeoZarr stores following the EOPF hierarchy.

This plan is scoped to the `EOPF-Explorer/data-model` repository. It extends the existing codebase (which handles S2 EOPF Zarr → GeoZarr conversion) with a new ingestion path for S1Tiling GeoTIFF outputs.

Repository: https://github.com/EOPF-Explorer/data-model
Current version: v0.8.0
GeoZarr spec: **GeoZarr 1.0** — modular Zarr Conventions framework (post-Rome Summit Dec 2025)
Mini-spec: https://eopf-explorer.github.io/data-model/geozarr-minispec/
Key dependency: S1Tiling 1.4.0 (https://s1-tiling.pages.orfeo-toolbox.org/s1tiling/latest/)

### Architecture decision: two-step pipeline, GeoTIFF as handoff

S1Tiling is treated as an **external black box** that produces GeoTIFFs. It is NOT
integrated as a Python library dependency in this repository.

Rationale: S1Tiling orchestrates OTB C++ applications that fundamentally produce
GeoTIFF on disk (SAFE → OTB → GeoTIFF). Even calling `s1_process()` from Python,
the pixel data still materialises as GeoTIFF intermediates. Embedding OTB (a ~1GB
C++ binary with its own GDAL, proj, and LD_LIBRARY_PATH) into our image would add
dependency risk with no performance benefit — the GeoTIFF intermediate exists
regardless.

The pipeline is therefore two Argo workflow steps:

```
Step 1: cnes/s1tiling Docker    →  GeoTIFFs on shared volume / S3
Step 2: eopf-geozarr Docker     →  reads GeoTIFFs, writes Zarr + STAC
```

This repository owns Step 2 only. S1Tiling configuration and Docker image
management are pipeline concerns (Argo workflow YAML), not data-model concerns.

---

## 1. GeoZarr Conventions Used

Three core Zarr conventions declared in `zarr_conventions` at each relevant group level:

| Convention | Namespace | UUID | Purpose |
|------------|-----------|------|---------|
| multiscales | `multiscales` | `d35379db-88df-4056-af3a-620245f8e347` | Pyramid layout |
| geo-proj | `proj:` | `f17cb550-5864-4468-aeb7-f3180cfb622f` | CRS encoding |
| spatial | `spatial:` | `689b58e2-cf7b-45e0-9fff-9cfc0883d6b4` | Array index → spatial coordinate |

Key principles from the mini-spec:
- **No `grid_mapping` 0D arrays** — CRS via `proj:code` at group level, inherited by child arrays
- **No mandatory CF conventions** — CF metadata is compatible but not required
- **`dimension_names`** in Zarr V3 array metadata (not `_ARRAY_DIMENSIONS` attribute)
- **`spatial:transform`** in Rasterio/Affine ordering `[a, b, c, d, e, f]` (NOT GDAL ordering)
- **Multiscales** use `layout` array with `asset`, `derived_from`, `transform.scale`
- **1D coordinate arrays** (`x`, `y`) must exist at every resolution level for GeoZarr reader compatibility (e.g. titiler-eopf)
- **Consolidated metadata** at root and orbit direction group levels for performant reads

---

## 2. Zarr Store Structure

One Zarr V3 store per S2 MGRS tile. Store name: `s1-grd-rtc-{tile_id}.zarr`

```
s1-grd-rtc-32TQM.zarr/
├── zarr.json                           # Root group (no conventions at root)
│
├── ascending/
│   ├── zarr.json                       # Group: zarr_conventions [multiscales, proj:, spatial:]
│   │                                   #   proj:code: "EPSG:32633"
│   │                                   #   spatial:dimensions: ["y", "x"]
│   │                                   #   spatial:bbox: [xmin, ymin, xmax, ymax]
│   │                                   #   multiscales: { layout: [...] }
│   │
│   ├── r10m/                           # Native resolution dataset (asset: "r10m")
│   │   ├── zarr.json                   # Group: spatial:shape, spatial:transform
│   │   ├── vv/                         # (time, y, x) float32
│   │   │   ├── zarr.json              # dimension_names: ["time", "y", "x"]
│   │   │   └── c/{t}/0/0             # One shard per timestep
│   │   ├── vh/                         # (time, y, x) float32
│   │   │   └── c/{t}/0/0
│   │   ├── border_mask/               # (time, y, x) uint8
│   │   │   └── c/{t}/0/0
│   │   ├── x/                          # (x,) float64 — projection x coordinates
│   │   ├── y/                          # (y,) float64 — projection y coordinates
│   │   ├── time/                       # (time,) datetime64[ns]
│   │   ├── absolute_orbit/            # (time,) int32
│   │   ├── relative_orbit/            # (time,) int32
│   │   └── platform/                  # (time,) str
│   │
│   ├── r20m/                           # Overview level 1 (2x from r10m)
│   │   ├── zarr.json                   # spatial:shape: [5490, 5490]
│   │   │                               # spatial:transform: [20.0, 0.0, ...]
│   │   ├── vv/
│   │   ├── vh/
│   │   ├── border_mask/
│   │   ├── x/                          # (x,) float64 — projection x at this resolution
│   │   └── y/                          # (y,) float64 — projection y at this resolution
│   │
│   ├── r60m/                           # Overview level 2 (3x from r20m)
│   ├── r120m/                          # Overview level 3 (2x from r60m)
│   ├── r360m/                          # Overview level 4 (3x from r120m)
│   ├── r720m/                          # Overview level 5 (2x from r360m)
│   │
│   └── conditions/                     # Time-invariant, NOT in multiscales layout
│       ├── zarr.json                   # proj:code, spatial:dimensions, spatial:transform
│       ├── gamma_area_{orbit}/        # (Y, X) float32 — one per relative orbit number
│       ├── lia_{orbit}/               # (Y, X) float32 — sin(LIA) [if available]
│       └── incidence_angle_{orbit}/   # (Y, X) float32 [if available]
│
└── descending/
    └── (same structure)
```

### Key design decisions

**Multiscale levels are named by resolution** (`r10m`, `r20m`, `r60m`, ...) to match the S2 convention used in the existing codebase. Each level is a Dataset containing the same set of variables.

**Coordinate variables** (`time`, `absolute_orbit`, `relative_orbit`, `platform`) live inside the native resolution dataset (`r10m/`) because they follow the Dataset rule: for each dimension name in a data variable, there must be a matching 1D coordinate variable. Overview levels share the same time dimension but don't need separate coordinate copies (they reference the same time axis).

**1D spatial coordinate arrays** (`x`, `y`) are required at **every resolution level** (r10m through r720m). They are computed from the level's `spatial:transform` using `np.linspace` and carry CF-standard attributes (`units: "m"`, `standard_name: "projection_x_coordinate"` / `"projection_y_coordinate"`). Without these arrays, GeoZarr readers like titiler-eopf cannot resolve spatial coordinates from the data.

**Consolidated metadata** is written at the root and orbit direction group levels after all ingestions are complete. This embeds the full hierarchy metadata in the group-level `zarr.json`, enabling single-request metadata reads. Important: consolidation must happen **after** all resize/append operations — consolidating mid-ingestion caches stale array shapes and breaks subsequent writes.

**Conditions** sit outside the multiscales layout as a separate group. They are (Y, X) only — no time dimension — and are per orbit, not per acquisition. They carry their own `proj:` and `spatial:` conventions. Each array is named with the relative orbit number suffix (e.g. `gamma_area_008`). Only `gamma_area` is confirmed as a direct S1Tiling output; `lia` and `incidence_angle` may require additional S1Tiling configuration or post-processing to produce as separate files.

**border_mask** is included as a variable alongside vv/vh in each resolution level. It shares the (time, Y, X) shape and gets downsampled with the overviews (using `nearest` resampling for masks, not `average`).

### Chunking and sharding

- **Chunks**: `(1, C, C)` where C = largest divisor of tile dimension ≤ 512 (e.g. 366 for 10980, since 10980/30=366). Zarr sharding requires inner chunks to **evenly divide** the shard; 512 does NOT divide 10980.
- **Shards**: `(1, H, W)` — one shard file = one timestep = all spatial chunks
- **Physical files**: `vv/c/{time_index}/0/0` — each new acquisition → one new shard file per array
- **chunk_key_encoding**: `{"name": "default", "configuration": {"separator": "/"}}`
- **Overview shards**: same strategy, naturally smaller per level

### Time dimension — append model

- Append-order integer index as dimension axis
- Actual datetime stored in `time` coordinate variable
- Non-monotonic time is expected and by design
- Consumers use `ds.sortby('time')` — no performance penalty since each timestep is an independent shard

### Multiscales metadata (ascending/zarr.json)

```json
{
    "zarr_conventions": [
        {"uuid": "d35379db-...", "name": "multiscales", ...},
        {"uuid": "f17cb550-...", "name": "proj:", ...},
        {"uuid": "689b58e2-...", "name": "spatial:", ...}
    ],
    "multiscales": {
        "layout": [
            {
                "asset": "r10m",
                "transform": {"scale": [1.0, 1.0]},
                "spatial:shape": [10980, 10980],
                "spatial:transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0]
            },
            {
                "asset": "r20m",
                "derived_from": "r10m",
                "transform": {"scale": [2.0, 2.0], "translation": [0.0, 0.0]},
                "spatial:shape": [5490, 5490],
                "spatial:transform": [20.0, 0.0, 500000.0, 0.0, -20.0, 5000000.0]
            },
            {
                "asset": "r60m",
                "derived_from": "r20m",
                "transform": {"scale": [3.0, 3.0], "translation": [0.0, 0.0]},
                "spatial:shape": [1830, 1830],
                "spatial:transform": [60.0, 0.0, 500000.0, 0.0, -60.0, 5000000.0]
            },
            {
                "asset": "r120m",
                "derived_from": "r60m",
                "transform": {"scale": [2.0, 2.0], "translation": [0.0, 0.0]},
                "spatial:shape": [915, 915],
                "spatial:transform": [120.0, 0.0, 500000.0, 0.0, -120.0, 5000000.0]
            },
            {
                "asset": "r360m",
                "derived_from": "r120m",
                "transform": {"scale": [3.0, 3.0], "translation": [0.0, 0.0]},
                "spatial:shape": [305, 305],
                "spatial:transform": [360.0, 0.0, 500000.0, 0.0, -360.0, 5000000.0]
            },
            {
                "asset": "r720m",
                "derived_from": "r360m",
                "transform": {"scale": [2.0, 2.0], "translation": [0.0, 0.0]},
                "spatial:shape": [153, 153],
                "spatial:transform": [720.0, 0.0, 500000.0, 0.0, -720.0, 5000000.0]
            }
        ],
        "resampling_method": "average"
    },
    "proj:code": "EPSG:32633",
    "spatial:dimensions": ["y", "x"],
    "spatial:bbox": [500000.0, 4890200.0, 609800.0, 5000000.0]
}
```

### Array metadata example (r10m/vv/zarr.json)

This is the actual on-disk format produced by zarr-python 3.1.1. Sharding is
represented as a codec wrapping the inner codecs. The `chunk_grid` holds the
**shard shape** (one shard = one timestep of the full spatial extent). The
**inner chunk shape** (`[1, 366, 366]`) lives inside the sharding codec config.

```json
{
    "zarr_format": 3,
    "node_type": "array",
    "shape": [0, 10980, 10980],
    "data_type": "float32",
    "chunk_grid": {
        "name": "regular",
        "configuration": {"chunk_shape": [1, 10980, 10980]}
    },
    "chunk_key_encoding": {
        "name": "default",
        "configuration": {"separator": "/"}
    },
    "codecs": [
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": [1, 366, 366],
                "codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5}}
                ],
                "index_codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "crc32c"}
                ],
                "index_location": "end"
            }
        }
    ],
    "dimension_names": ["time", "y", "x"],
    "fill_value": "NaN"
}
```

**Python API** (zarr-python 3.1.1):
```python
group.create_array(
    "vv", shape=(0, 10980, 10980), dtype="float32",
    chunks=(1, 366, 366),          # inner chunk shape
    shards=(1, 10980, 10980),      # shard shape
    compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=5),
    fill_value=float("nan"),
    dimension_names=["time", "y", "x"],
)
```

Note: `shape[0]` starts at 0 and grows with each append. `dimension_names` is Zarr V3 native — no `_ARRAY_DIMENSIONS` attribute needed. Inner chunk size 366 = `best_chunk_size(10980)` = largest divisor of 10980 ≤ 512.

---

## 3. New Modules

### 3.1 Pydantic Models — `src/eopf_geozarr/models/sentinel1.py`

This is the 404 page at https://eopf-explorer.github.io/data-model/models/sentinel1.md — it needs to be implemented.

Extend existing base classes (inspect `models/sentinel2.py` for the pattern). The models should validate:

- Store structure (ascending/descending groups)
- Zarr conventions declarations at each group level
- Array dimension_names consistency
- Coordinate variable existence for each dimension name in data variables
- Multiscales layout structure matching the mini-spec
- `proj:code`, `spatial:dimensions`, `spatial:transform` presence

```python
# Core model sketch — refine against existing sentinel2.py base classes

class S1GrdMeasurementsDataset(BaseModel):
    """A single resolution level dataset (r10m, r20m, etc.)."""
    vv: ArraySpec        # (time, Y, X) float32
    vh: ArraySpec        # (time, Y, X) float32
    border_mask: ArraySpec  # (time, Y, X) uint8
    # Coordinate variables (at native resolution only)
    time: Optional[ArraySpec] = None         # (time,) datetime64
    absolute_orbit: Optional[ArraySpec] = None  # (time,) int32
    relative_orbit: Optional[ArraySpec] = None  # (time,) int32
    platform: Optional[ArraySpec] = None     # (time,) str

class S1GrdConditionsDataset(BaseModel):
    """Time-invariant conditions per orbit. Arrays named with orbit suffix (e.g. gamma_area_008)."""
    gamma_area: Dict[str, ArraySpec]                    # gamma_area_{orbit}: (Y, X) float32 — always present
    lia: Optional[Dict[str, ArraySpec]] = None          # lia_{orbit}: (Y, X) float32 — if available
    incidence_angle: Optional[Dict[str, ArraySpec]] = None  # incidence_angle_{orbit}: (Y, X) float32 — if available

class S1GrdOrbitGroup(BaseModel):
    """One orbit direction — carries multiscales, proj:, spatial: conventions."""
    multiscales: MultiscalesMetadata
    proj_code: str                    # e.g., "EPSG:32633"
    spatial_dimensions: List[str]     # ["y", "x"]
    spatial_bbox: List[float]
    conditions: S1GrdConditionsDataset

class S1GrdStore(BaseModel):
    """Root store for one S2 MGRS tile."""
    tile_id: str
    ascending: Optional[S1GrdOrbitGroup] = None
    descending: Optional[S1GrdOrbitGroup] = None
```

### 3.2 GeoTIFF Ingestion — `src/eopf_geozarr/conversion/s1_ingest.py`

**Public API:**

```python
def ingest_s1tiling_acquisition(
    vv_path: str,
    vh_path: str,
    mask_path: str,
    zarr_store: str,
    tile_id: str,
    orbit_direction: str,   # "ascending" or "descending"
) -> int:
    """
    Append one S1Tiling acquisition to the Zarr store.
    Creates the store with full zarr_conventions metadata if first acquisition.
    Returns the time index of the appended acquisition.

    Metadata extraction from GeoTIFF tags:
    - ACQUISITION_DATETIME → time coordinate
    - ORBIT_NUMBER → absolute_orbit coordinate
    - RELATIVE_ORBIT_NUMBER → relative_orbit coordinate
    - FLYING_UNIT_CODE → platform coordinate
    - CRS + GeoTransform → proj:code + spatial:transform (Rasterio ordering)
    """

def ingest_s1tiling_conditions(
    lia_path: str,
    incidence_angle_path: str,
    gamma_area_path: str,
    zarr_store: str,
    tile_id: str,
    orbit_direction: str,
) -> None:
    """
    Write time-invariant condition arrays.
    Conditions group gets its own proj: and spatial: conventions.
    """
```

**Store creation** (first acquisition):
1. Write root `zarr.json` (minimal, no conventions at root)
2. Write `ascending/zarr.json` with full `zarr_conventions` array, `multiscales` layout, `proj:code`, `spatial:dimensions`, `spatial:bbox`
3. Write `ascending/r10m/zarr.json` with `spatial:shape` and `spatial:transform`
4. Create arrays: `vv/zarr.json`, `vh/zarr.json`, `border_mask/zarr.json` with `dimension_names: ["time", "y", "x"]`
5. Create coordinate arrays: `time/zarr.json` with `dimension_names: ["time"]`, etc.
5b. Create 1D spatial coordinate arrays (`x`, `y`) at every resolution level
6. Write first data shard: `vv/c/0/0/0`
7. Generate overviews for this timestep → write to r20m, r60m, ..., r720m

**Append** (subsequent acquisitions):
1. Read current time dimension size
2. Resize time axis (increment shape[0])
3. Write new shard: `vv/c/{new_index}/0/0`
4. Append coordinate values
5. Generate overviews for new timestep at all levels

**Consolidation** (after all ingestions for a batch):
1. `zarr.consolidate_metadata(store_path, path=orbit_direction, zarr_format=3)` — orbit level
2. `zarr.consolidate_metadata(store_path, zarr_format=3)` — root level

Note: Do NOT consolidate between individual ingestions — consolidated metadata caches array shapes and breaks subsequent resize+write operations.

**CRITICAL — spatial:transform ordering**: S1Tiling GeoTIFFs use GDAL GeoTransform ordering `[c, a, b, f, d, e]`. The mini-spec requires Rasterio/Affine ordering `[a, b, c, d, e, f]`. The converter MUST apply the mapping: `spatial_transform = [GT(1), GT(2), GT(0), GT(4), GT(5), GT(3)]`.

### 3.3 CLI Extension

```bash
# Ingest a single acquisition
eopf-geozarr ingest-s1 \
  --vv /path/to/s1a_32TQM_vv_ASC_037_..._GammaNaughtRTC.tif \
  --vh /path/to/s1a_32TQM_vh_ASC_037_..._GammaNaughtRTC.tif \
  --mask /path/to/..._BorderMask.tif \
  --store s3://eopf-explorer/s1-grd-rtc-32TQM.zarr \
  --tile 32TQM \
  --orbit-dir ascending

# Ingest conditions (once per tile/orbit)
eopf-geozarr ingest-s1-conditions \
  --lia /path/to/sin_LIA_32TQM_037.tif \
  --ia /path/to/IA_32TQM_037.tif \
  --gamma-area /path/to/GAMMA_AREA_32TQM_037.tif \
  --store s3://eopf-explorer/s1-grd-rtc-32TQM.zarr \
  --tile 32TQM \
  --orbit-dir ascending

# Validate S1 store against mini-spec
eopf-geozarr validate-s1 s3://eopf-explorer/s1-grd-rtc-32TQM.zarr
```

---

## 4. Overview Generation

Reuse existing downsampling logic with the same variable factor chain as S2:
- r10m → r20m (2×) → r60m (3×) → r120m (2×) → r360m (3×) → r720m (2×)

Per-timestep generation: when a new acquisition is appended to r10m, generate overviews for that single timestep at all levels. Overviews are spatial only — no temporal resampling.

Resampling methods:
- `vv`, `vh`: `"average"` (default in multiscales metadata)
- `border_mask`: `"nearest"` (binary mask, must not interpolate)

---

## 5. STAC Registration

New collection: `sentinel-1-grd-rtc-geozarr`

Extensions: `sar`, `sat`, `zarr`, `render`

Each store = one STAC item per tile. Temporal extent derived from sorted `time` coordinate. Updated on each ingest.

---

## 6. Risks

| Risk | Severity | Mitigation | Phase 0 Status |
|------|----------|------------|----------------|
| Zarr V3 append + sharding maturity | HIGH | Pin zarr-python, test heavily, serialize per tile | **MITIGATED** — resize+sharding verified with real 10980×10980 data, 3 timesteps appended successfully |
| Partial tile coverage per timestep | HIGH | border_mask in quality, coverage % in STAC | **MITIGATED** — real data shows 0.4%–99.8% coverage; border_mask ingestion validated |
| spatial:transform GDAL↔Rasterio ordering | MEDIUM | Explicit conversion in ingestion code, validation | **MITIGATED** — rasterio `src.transform` returns Affine ordering directly |
| Zarr V3 string dtype instability | MEDIUM | Use `<U4` with accepted warning, or integer enum | **ACCEPTED** — `FixedLengthUTF32` triggers warning but works; revisit when Zarr V3 string spec stabilises |
| Inner chunk divisibility | MEDIUM | `best_chunk_size()` finds largest divisor ≤ 512 | **MITIGATED** — discovered and fixed during real-data testing (10980 % 512 ≠ 0) |
| Processing cost at continental scale | MEDIUM | Priority tiles first, spot instances, cost tracking | Unchanged |
| S1A/S1C platform transition | MEDIUM | platform coordinate per timestep | Unchanged |
| S1Tiling GeoTIFF format changes | MEDIUM | Pin S1Tiling 1.4.0, validate GeoTIFF tag presence on ingest | **MITIGATED** — 32 tags documented, all ingestion-critical tags confirmed present |
| GeoTIFF handoff between Argo steps | LOW | Well-defined file naming convention, validation on read | **MITIGATED** — filename regex pattern validated on real outputs |
| Time coordinate non-monotonicity | LOW | By design, documented | **CONFIRMED** — 3 acquisitions ingested in discovery order (not time order), xarray `sortby` works |

---

## 7. Implementation Phases

## Phase 0 — Design and prototyping

- [x] Finalise data model design (this document)
- [x] Define a consolidated implementation plan with clear phases and milestones from this design document
- [x] Test S1Tiling end-to-end: SAFE → GeoTIFF → read with Rasterio, inspect tags
- [x] Prototype GeoTIFF → Zarr conversion for one acquisition (proof of concept)
- [x] **Real-data validation**: Convert 3 real S1Tiling γ0T RTC acquisitions to GeoZarr V3 store
- [x] Use the feedback from prototyping (previous points) to refine the data model and implementation plan before starting full development
- [x] **Post-validation fix**: Add 1D spatial coordinate arrays (`x`, `y`) at every resolution level
- [x] **Post-validation fix**: Use lowercase dimension names (`y`, `x`) throughout
- [x] **Post-validation fix**: Add consolidated metadata at root and orbit group levels

**Phase 0 prototype**: `analysis/s1_grd_rtc_prototype.py` — runnable end-to-end proof of concept.
See [Phase 0 Findings](#phase-0-findings) below for detailed results.

### Phase 1 — Models (extends existing sentinel2 pattern)
- [x] Inspect existing `data_api/s2.py` for base classes (`pyz.v2` GroupSpec/ArraySpec, TypedDict members)
- [x] Implement `data_api/s1_rtc.py` with S1 RTC-specific structure (aligned with S2 pattern, using `pyz.v3`)
- [x] Implement zarr_conventions validation (multiscales, geo_proj, spatial UUIDs via `zarr_cm`)
- [ ] Wire up `sentinel1.md` docs page (currently 404) — deferred, not blocking
- [x] Unit tests for model validation (11 tests: round-trip, structure, 5 negative cases)

**Phase 1 code**: `src/eopf_geozarr/data_api/s1_rtc.py` — 316 lines.
**Phase 1 PR**: https://github.com/EOPF-Explorer/data-model/pull/138 (draft, for review by pydantic-zarr schema maintainers).
See [Phase 1 Findings](#phase-1-findings) below for detailed results.

### Phase 2 — GeoTIFF ingestion
- [x] Metadata extraction from S1Tiling GeoTIFF tags (rasterio)
- [x] Store creation with full zarr_conventions metadata
- [x] Single-acquisition ingest (create path)
- [x] Append path (resize + new shard)
- [x] Coordinate variable append (time, absolute_orbit, relative_orbit, platform)
- [x] Rasterio/Affine transform — direct from `src.transform` (no GDAL conversion needed)
- [x] Overview generation at all resolution levels (average for data, nearest for masks)
- [x] File discovery and grouping (`discover_s1tiling_acquisitions()`)
- [x] Consolidation function (`consolidate_s1_store()`)
- [x] CRS and shape consistency validation on append
- [x] Integration tests with synthetic GeoTIFFs (27 tests)

**Phase 2 code**: `src/eopf_geozarr/conversion/s1_ingest.py` — ~500 lines.
**Phase 2 tests**: `tests/test_s1_rtc_ingest.py` — 27 tests (metadata extraction, store creation, ingestion, consolidation, file discovery).
**Phase 2 PR**: https://github.com/EOPF-Explorer/data-model/pull/TBD (for review).
See [Phase 2 Findings](#phase-2-findings) below for detailed results.

### Phase 3 — Conditions and overviews
- [ ] Conditions ingest (lia, incidence_angle, gamma_area)
- [ ] Conditions group with own proj: and spatial: conventions
- [ ] Wire up existing multiscale generation for S1 data
- [ ] border_mask with nearest resampling at overview levels
- [ ] Test overview generation per timestep

### Phase 4 — CLI and S3
- [ ] CLI subcommands (ingest-s1, ingest-s1-conditions, validate-s1)
- [ ] S3 output support (reuse existing)
- [ ] End-to-end test: CLI → S3 → read back with xarray

### Phase 5 — STAC and Argo
- [ ] STAC collection definition
- [ ] STAC item update logic per ingest
- [ ] Argo workflow template YAML

---

## 8. Open Questions for Refinement

### Data model questions
1. **Existing base classes**: Inspect `models/sentinel2.py` to determine inheritance pattern.
2. **Zarr-python V3 append/resize API**: Verify exact API for growing arrays along time axis.
3. **Coordinate variables at overview levels**: Mini-spec says same variable names at all levels. Do overview levels need copies of time/orbit coordinate arrays?
4. **Shard size**: ~460MB per shard at 10980×10980 float32. Acceptable?
5. **border_mask in overviews**: Useful at r720m or skip?

### Pipeline questions (Argo workflow, outside this repo)
6. **S1Tiling .cfg template**: What configuration parameters to set for γ0T RTC with CDSE as data source? This is Argo workflow configuration, not data-model code, but needs documenting.
7. **GeoTIFF handoff location**: Shared volume mount between the two Argo steps, or S3 intermediate bucket? Shared volume is simpler but couples the steps; S3 is decoupled but adds transfer time.
8. **S1Tiling output path conventions**: The file naming pattern (`s1{a|c}_{tile}_{pol}_{dir}_{orbit}_{date}_GammaNaughtRTC.tif`) is well-documented. The ingestion module should validate filenames match this pattern and extract metadata from the name as a fallback if GeoTIFF tags are missing.

---

## 9. Phase 0 Findings

Phase 0 prototyping was completed on 2026-03-22. The prototype (`analysis/s1_grd_rtc_prototype.py`) validates the full pipeline from synthetic GeoTIFFs through Zarr V3 store creation, two-acquisition append, overview generation, and xarray readback.

### Verified Technical Assumptions

| Assumption | Result | Notes |
|------------|--------|-------|
| zarr-python 3.1.1 `resize()` on sharded arrays | **WORKS** | `shape[0]` starts at 0, `resize()` grows time axis, data integrity preserved |
| Zarr V3 `create_array` API | **WORKS — plan updated** | Uses `shards=` and `compressors=` params, NOT `codecs=` or `storage_transformers`. Section 2 example now reflects actual on-disk format |
| Rasterio metadata extraction | **WORKS** | `src.transform` returns Affine ordering natively — no GDAL conversion needed when reading with rasterio |
| GeoTIFF custom tags | **WORKS** | `dst.update_tags()` / `src.tags()` round-trips `ACQUISITION_DATETIME`, `ORBIT_NUMBER`, etc. |
| xarray reads Zarr V3 store | **WORKS** | `xr.open_zarr(path, zarr_format=3, consolidated=False)` reads all arrays correctly |
| Overview pyramid chain | **WORKS** | 2x/3x chain: 256→128→43→22→8→4 (matches ceil-based division) |
| Zarr V3 string dtype | **WARNING** | `FixedLengthUTF32` triggers `UnstableSpecificationWarning` — consider using `bytes` or `vlen-utf8` for `platform` |

### Key API Corrections vs. Design Document

**1. Array metadata format (Section 2):** The `zarr.json` example in Section 2 has been updated to show the actual on-disk format from zarr-python 3.1.1. Sharding is represented as a codec (not `storage_transformers`), with the shard shape in `chunk_grid` and inner chunk shape inside the `sharding_indexed` codec config. The Python API uses `shards=` and `compressors=` params on `create_array()`.

```python
# Correct API (zarr-python 3.1.1):
group.create_array(
    name, shape=(...), dtype='float32',
    chunks=(1, 512, 512),          # inner chunk shape
    shards=(1, 10980, 10980),      # shard shape
    compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=5),
    fill_value=float('nan'),
    dimension_names=['time', 'y', 'x'],
)
```

**2. GDAL → Rasterio transform (Section 3.2):** When reading GeoTIFFs with rasterio, `src.transform` already returns Affine ordering `[a, b, c, d, e, f]`. The GDAL-to-Rasterio conversion (`GT(1), GT(2), GT(0), GT(4), GT(5), GT(3)`) is only needed if reading raw GDAL GeoTransform (e.g., from `src.get_transform()`). The ingestion code should use `src.transform` directly.

**3. String coordinate dtype:** Zarr V3 does not have a stable string data type specification. The `platform` coordinate triggers a warning:
> `UnstableSpecificationWarning: The data type (FixedLengthUTF32) does not have a Zarr V3 specification.`

**Decision needed:** Use `<U4` and accept the warning (platform codes are 3-4 chars), or encode platforms as integer enum (0=S1A, 1=S1B, 2=S1C) with a lookup table in attributes.

**4. Overview shape calculation:** The design document specifies `r720m: [153, 153]` for the full 10980 tile. This comes from `ceil(305/2) = 153`, not `floor(305/2) = 152`. The downsample function must use ceiling division. Verified in prototype.

### Answers to Open Questions

| # | Question | Answer from Prototyping |
|---|----------|------------------------|
| 1 | Existing base classes | S1 models exist in `data_api/s1.py` (EOPF L1 GRD structure). The new S1 RTC models are a *different data product* — they should be a new file, not extensions of the existing L1 GRD models. The existing `data_api/geozarr/common.py` base classes (`ProjAttrs`, `BaseDataArrayAttrs`) are reusable. |
| 2 | Zarr V3 append/resize API | `array.resize((new_time_size, H, W))` — works on sharded arrays, preserves existing data. No separate "append" method; resize + write-at-index is the pattern. |
| 3 | Coordinate variables at overview levels | **Decision: NO.** Overview levels contain only data variables (vv, vh, border_mask). Coordinate variables (time, absolute_orbit, etc.) live only at r10m. Consumers reference r10m coordinates across all levels. The mini-spec says "same variable names" but coordinate arrays matching the time dimension would be redundant copies. |
| 4 | Shard size | 10980×10980 float32 = ~460MB uncompressed per shard. With Blosc/zstd clevel=5, expect ~100-200MB. This is acceptable for cloud storage (S3 multipart upload). The alternative (multiple shards per timestep) complicates the append model. |
| 5 | border_mask in overviews | **YES, include at all levels.** At r720m (~150 pixels), the border mask is still useful for quick coverage assessment. Uses `nearest` resampling (verified in prototype). |

### Codebase Integration Notes

The existing codebase has two separate S1 code paths:
1. **`data_api/s1.py`** — Pydantic models for EOPF Sentinel-1 L1 GRD products (radar geometry, VV/VH polarization groups, conditions with antenna_pattern, attitude, etc.)
2. **`conversion/sentinel1_reprojection.py`** — GCP-based reprojection from radar geometry to EPSG:4326

Neither is directly usable for S1 RTC GeoZarr. The new S1 RTC pipeline is fundamentally different:
- Input: GeoTIFFs already on the S2 MGRS grid (UTM projection), not EOPF Zarr in radar geometry
- Output: Time-series Zarr store with append model, not single-scene conversion
- Conventions: New Zarr Conventions (multiscales, proj:, spatial:), not CF grid_mapping

**The new code should live in:**
- `src/eopf_geozarr/conversion/s1_ingest.py` — GeoTIFF → Zarr ingestion
- `src/eopf_geozarr/data_api/s1_rtc.py` — Pydantic models for S1 RTC GeoZarr stores (new file, separate from existing `s1.py`)
- `tests/test_s1_rtc_ingest.py` — Integration tests

### Real-Data Validation (2026-03-23)

S1Tiling γ0T RTC ran to completion on 3 S1A GRD products (orbits 008, 037, 110) over MGRS tile 31TCH. The real GeoTIFF → GeoZarr V3 conversion was validated with `analysis/s1_real_geotiff_to_zarr.py`.

**S1Tiling output characteristics (real data):**

| Property | Value |
|----------|-------|
| CRS | EPSG:32631 (UTM 31N) |
| Shape | 10980 × 10980 |
| Pixel size | 10.0 m |
| Origin | (299999.9999974121, 4799999.99999915) |
| Data dtype | float32 (backscatter), uint8 (border mask) |
| Nodata | 0.0 (both backscatter and masks) |
| Compression | DEFLATE with PREDICTOR=3 |
| Calibration tag | `GammaNaughtRTC` (not `gamma_naught_rtc` as in config) |
| Datetime format | `2025:02:10T06:09:20Z` (colon-separated date, not ISO dashes) |
| Tags (32 total) | All ingestion-critical tags present |
| Border mask values | Binary: 0 (no data) and 1 (valid) |

**Key discoveries from real data:**

1. **Chunk divisibility constraint**: Zarr sharding requires inner chunks to evenly divide the shard. 10980 is NOT divisible by 512. Solution: `best_chunk_size()` finds largest divisor ≤ 512. For 10980 → 366 (10980/30=366). This must be in the production code.

2. **Border mask coverage varies dramatically**: Orbit 037 covers only 0.4% of the tile, orbit 008 covers 30.5%, orbit 110 covers 99.8%. This validates the design decision to include border_mask at all levels.

3. **VV backscatter ranges**: Linear values mean ~0.07–0.20, dB mean ~-13 to -9 dB. Some extreme outliers (min -2033, max 4723 linear) which are likely edge artifacts or noise in low-coverage areas. The `nodata=0.0` from S1Tiling means 0-valued pixels are no-data, not actual backscatter. The Zarr fill_value should be NaN for float32 arrays.

4. **Datetime parsing**: S1Tiling uses `YYYY:MM:DDThh:mm:ssZ` format (colon-separated date). Must normalise to ISO 8601 before converting to numpy datetime64.

5. **Gamma area maps**: 3 maps (one per orbit), same 10980×10980 grid. Values range widely (min ~-47000, max ~431000). These are per-orbit, not per-acquisition — correct for the conditions group design.

6. **Write performance**: First shard write is slow (~40s), subsequent shards ~1-2s (OS page cache effect). Overview generation ~3.4s per acquisition (in-memory numpy). Total 3-acquisition conversion: 154s for 1.8 GB output store.

7. **Store size**: r10m=1021 MB, r20m=256 MB, r60m=28 MB, r120m=7 MB, r360m=0.8 MB, r720m=0.2 MB, conditions=488 MB. Total 1.8 GB for 3 timesteps. At ~340 MB/timestep for r10m alone, a full year (26 acquisitions per orbit) would be ~9 GB per orbit direction.

8. **xarray readback**: All levels readable with `xr.open_zarr(path, zarr_format=3, consolidated=False)`. Dimension names correctly detected. Coordinate variables (time, absolute_orbit, relative_orbit, platform) all round-trip correctly.

**Design refinements from real-data testing:**

| Design Item | Original | Refined |
|-------------|----------|---------|
| Inner chunk shape | `(1, 512, 512)` | `(1, best_chunk_size(H), best_chunk_size(W))` → `(1, 366, 366)` for 10980 |
| Backscatter fill_value | NaN | **NaN** (confirmed — S1Tiling uses 0.0 as nodata but we use NaN in Zarr) |
| Border mask fill_value | 0 | **0** (confirmed — matches S1Tiling convention) |
| Datetime parsing | ISO 8601 assumed | Must handle `YYYY:MM:DD` S1Tiling format |
| Calibration tag value | `gamma_naught_rtc` | Actual: `GammaNaughtRTC` (mixed case) |
| Conditions per orbit | Single `gamma_area/` array | Per-orbit naming: `gamma_area_{orbit_num}` (e.g. `gamma_area_008`) |
| Conditions scope | LIA + incidence_angle + gamma_area | Only `gamma_area` confirmed as S1Tiling output; LIA/incidence_angle aspirational |
| Array metadata format | `storage_transformers` with `sharding_indexed` | Sharding is a codec in the `codecs` array; `chunk_grid` holds shard shape |

### Operational Findings (S1Tiling Pipeline)

These findings are specific to running S1Tiling 1.4.0 via Docker and are
relevant for the Argo workflow (Step 1) rather than the data-model code
(Step 2), but are documented here for completeness.

**EODAG 4.0.0 breaking changes:** S1Tiling 1.4.0 ships EODAG 4.0.0, which has
five breaking changes that prevent `cop_dataspace` from working correctly. A
monkey-patch script (`analysis/s1tiling_eodag4_patch.py`) fixes all five:

1. `productType` kwarg renamed to `collection` — having both causes silent fallback to peps
2. Product properties use STAC names (`sat:orbit_state`) instead of legacy (`orbitDirection`)
3. `cop_dataspace` OData v4 API rejects `polarizationChannels` and `sensorMode`
4. Orbit direction must be UPPERCASE (`"DESCENDING"` not `"descending"`)
5. `relativeOrbitNumber` search param silently returns 0 results

An upstream issue has been prepared: `issues/s1tiling-eodag-collection-bug.md`.

**DEM tile extent:** The Gamma Area computation requires SRTM DEM tiles covering
the full S1 swath geometry, which extends far beyond the target MGRS tile. For
31TCH (42–43°N, 0–2°E), 20 SRTM tiles were needed (41°N–44°N, 3°W–5°E) instead
of the expected 4–6. This is because multiple overlapping descending orbits have
wide swaths. The DEM tile list for each MGRS tile must be pre-computed or
discovered during a dry-run.

**S1Tiling output filename format:** The GAMMA_AREA filename includes the flying
unit code and orbit direction (`GAMMA_AREA_s1a_31TCH_DES_008.tif`), which differs
from the simpler pattern in the S1Tiling docs (`GAMMA_AREA_{tile}_{orbit}.tif`).
The ingestion code must handle the actual naming pattern.

See `analysis/s1tiling_docker_instructions.md` for complete Docker setup and
troubleshooting guide.

### Post-Validation Fixes (2026-03-25)

After testing the Phase 0 output store with titiler-eopf (GeoZarr reader), three issues were discovered and fixed:

**1. Missing 1D spatial coordinate arrays (`x`, `y`):**
The prototype created data arrays with `dimension_names: ["time", "y", "x"]` but never created actual 1D coordinate arrays for the spatial dimensions. GeoZarr readers require these to resolve pixel coordinates. Fix: added `np.linspace`-based coordinate array creation at every resolution level, using pixel size and origin from `spatial:transform`. Each array carries CF-standard attributes (`units: "m"`, `standard_name: "projection_x_coordinate"` / `"projection_y_coordinate"`, `_ARRAY_DIMENSIONS: ["x"]` / `["y"]`).

**2. Lowercase dimension names (`y`, `x` not `Y`, `X`):**
The original design used uppercase `Y`, `X` dimension names. titiler-eopf expects lowercase `y`, `x` (matching CF/GeoZarr convention). All `dimension_names`, `spatial:dimensions`, and coordinate array names were updated to lowercase throughout the prototype, test store, models, and fixtures.

**3. Consolidated metadata:**
Without consolidated metadata, readers must make separate HTTP requests for every group and array `zarr.json`. Fix: added `consolidate_store()` function calling `zarr.consolidate_metadata()` at both the orbit direction and root group levels. **Critical constraint**: consolidation must happen *after* all ingestions complete — consolidating mid-flow caches stale array shapes in `zarr.json`, causing `BoundsCheckError` on subsequent `resize()` + write operations.

**zarr-python 3.1.1 API note:** `create_array(data=...)` cannot be combined with `dtype=` — providing both raises `ValueError`. When passing `data=`, omit `dtype=` (it is inferred).

---

## 10. Phase 1 Findings

Phase 1 was completed on 2026-03-23. The Pydantic-zarr V3 models for S1 GRD RTC GeoZarr stores are implemented in `src/eopf_geozarr/data_api/s1_rtc.py` and validated with 11 tests.

### Pattern Alignment with S2

The S1 RTC model follows the exact same structural pattern as the S2 model (`data_api/s2.py`), with one key difference: S2 uses `pyz.v2` (Zarr V2 products), while S1 RTC uses `pyz.v3` (Zarr V3 products with sharding).

| Pattern Element | S2 (`s2.py`) | S1 RTC (`s1_rtc.py`) |
|----------------|-------------|---------------------|
| GroupSpec/ArraySpec wrapper | `pyz.v2` | `pyz.v3` |
| TypedDict members | `closed=True, total=False` | Same |
| Attrs models | `BaseModel` with `populate_by_name` | Same, plus `extra="allow"` and `serialize_by_alias` |
| Convention validation | zarr_cm UUIDs | Same (multiscales, geo_proj, spatial) |
| Hierarchy depth | Root → Tile → Resolution → Arrays | Root → OrbitDirection → Resolution → Arrays |

### Model Hierarchy

```
S1RtcRoot
├── ascending: S1RtcOrbitGroup (optional)
│   ├── attrs: S1RtcOrbitGroupAttrs (zarr_conventions, multiscales, proj:code, spatial:dimensions, spatial:bbox)
│   ├── r10m: S1RtcNativeResolutionDataset (vv, vh, border_mask, time, absolute_orbit, relative_orbit, platform)
│   ├── r20m..r720m: S1RtcOverviewResolutionDataset (vv, vh, border_mask only)
│   └── conditions: S1RtcConditionsGroup (gamma_area_{orbit} arrays)
└── descending: S1RtcOrbitGroup (optional)
    └── (same structure)
```

### Design Decisions

1. **`extra="allow"` on attrs models**: S1 RTC attrs models use `extra="allow"` (moved into `model_config` per mypy requirement) because the orbit group JSON may carry additional metadata not yet modelled (e.g. future extensions). S2 does not use `extra="allow"`.

2. **Conditions group uses `dict[str, ArraySpec]`** instead of a closed TypedDict because condition array names are dynamic (per-orbit: `gamma_area_008`, `gamma_area_037`, etc.). A `model_validator` enforces that at least one `gamma_area_*` key exists.

3. **Both `ascending` and `descending` are optional** in the root TypedDict (`total=False`), but a `model_validator` on `S1RtcRoot` ensures at least one is present. This matches real-world usage where a tile may only have ascending or descending orbits.

4. **Resolution levels `r10m` is required**, all others (`r20m`–`r720m`) are optional. This allows progressive population: ingest first, generate overviews later.

### Pre-commit Compliance

All Phase 1 code passes pre-commit hooks (ruff check, ruff format, mypy). Notable fixes applied:
- **RUF002**: Replaced ambiguous Unicode characters (Greek gamma, en-dash) with ASCII equivalents
- **mypy `misc`**: Moved `extra="allow"` from class kwargs into `model_config` dict to avoid "config in two places" error
- **mypy `unused-ignore`**: Removed unnecessary `# type: ignore[type-var]` on `S1RtcConditionsGroup` (only needed on TypedDict-based members, not `dict`)

### Test Coverage

| Test | What it validates |
|------|------------------|
| `test_s1_rtc_roundtrip` | JSON → model → JSON round-trip fidelity |
| `test_s1_rtc_descending_present` | Fixture has descending orbit group |
| `test_s1_rtc_r10m_has_data_arrays` | r10m contains vv, vh, border_mask, time, platform, orbits |
| `test_s1_rtc_overview_levels` | r20m-r720m exist with vv/vh/border_mask |
| `test_s1_rtc_conditions` | Conditions group has gamma_area_* arrays |
| `test_s1_rtc_orbit_attrs` | zarr_conventions UUIDs, proj:code, spatial:dimensions validated |
| `test_s1_rtc_rejects_no_orbit` | Rejects empty root (no ascending/descending) |
| `test_s1_rtc_rejects_missing_r10m` | Rejects orbit group without r10m |
| `test_s1_rtc_rejects_missing_convention_uuid` | Rejects missing zarr_conventions UUID |
| `test_s1_rtc_rejects_bad_spatial_dimensions` | Rejects spatial:dimensions != ["y", "x"] |
| `test_s1_rtc_rejects_conditions_without_gamma_area` | Rejects conditions group without gamma_area_* keys |

### Open Questions for Phase 1 Review (PR #138)

1. Should `S1RtcOrbitGroupAttrs` inherit from a shared base with S2's resolution attrs, or keep them independent?
2. Is `extra="allow"` the right choice for attrs models, or should we lock them down and enumerate all known fields?
3. The `multiscales` field is validated structurally (must have `layout` array with `asset` keys) but not via the `zarr_cm.multiscales` Pydantic model. Should it use the typed model instead of `dict[str, Any]`?
4. Should the `S1RtcConditionsGroup` enforce a specific naming pattern (regex on keys) beyond the `gamma_area_` prefix check?

---

## 11. Phase 2 Findings

Phase 2 was completed on 2026-03-24. The GeoTIFF ingestion pipeline is implemented in `src/eopf_geozarr/conversion/s1_ingest.py` and validated with 27 tests in `tests/test_s1_rtc_ingest.py`.

### Delivered Components

| Component | Lines | Description |
|-----------|-------|-------------|
| `S1TilingMetadata` dataclass | ~15 | Frozen dataclass for metadata transfer (not Pydantic — simple data only) |
| `extract_geotiff_metadata()` | ~40 | Rasterio-based extraction with tag validation and datetime normalisation |
| `parse_s1tiling_filename()` | ~15 | Regex-based filename parsing (returns `None` for non-matching) |
| `compute_multiscales_layout()` | ~40 | Pure function building layout for all 6 resolution levels |
| `create_s1_store()` | ~80 | Full store creation with conventions, spatial coords, coordinate vars |
| `ingest_s1tiling_acquisition()` | ~120 | Main API: create-or-open, read, overview, write, append coords |
| `consolidate_s1_store()` | ~10 | Post-batch consolidation at orbit + root levels |
| `discover_s1tiling_acquisitions()` | ~40 | File discovery with grouping and completeness warnings |
| `_downsample_2d()` | ~20 | Private helper: factor-based downsampling with edge padding |
| `_create_spatial_coordinate_arrays()` | ~35 | 1D x/y arrays from spatial:transform at every level |

### Design Decisions

1. **Convention UUIDs from `zarr_cm`**: Instead of hardcoding UUID strings, imports `CMO` dicts directly from `zarr_cm.multiscales`, `zarr_cm.geo_proj`, `zarr_cm.spatial`. Uses `ZARR_CONVENTIONS = [multiscales_cm.CMO, geo_proj.CMO, spatial_cm.CMO]`.

2. **Private `_downsample_2d()` instead of reusing `downsample_2d_array()` from `utils.py`**: The existing utility takes `target_height`/`target_width` and does floor division for block sizes; the S1 pipeline needs factor-based downsampling with ceiling division and edge padding for non-divisible sizes. A purpose-built private helper is simpler than adapting the existing function.

3. **`mode="w-"` for store creation**: Uses exclusive creation mode to prevent accidental overwrites. The caller handles create-or-open branching.

4. **Consistency validation on append**: When appending to an existing store, validates CRS and native shape match. Mismatches raise `ValueError` at the system boundary (external GeoTIFF input).

5. **Phase 1 model discrepancy noted**: `S1RtcNativeResolutionMembers` and `S1RtcOverviewResolutionMembers` TypedDicts do not declare `x` and `y` members. The store structure requires 1D spatial coordinate arrays at every level. This should be fixed in a follow-up PR to Phase 1.

### Test Coverage (27 tests)

| Test Class | Tests | What it validates |
|-----------|-------|------------------|
| `TestExtractGeotiffMetadata` | 3 | Field extraction, datetime normalisation, missing tag rejection |
| `TestNormaliseDatetime` | 2 | S1Tiling colon-date format, already-normalised passthrough |
| `TestParseFilename` | 3 | VV file, mask file, non-matching returns None |
| `TestCreateStore` | 6 | Structure, conventions, array metadata, coord vars, overview shapes, spatial coords |
| `TestIngestAcquisition` | 8 | Create, append, data integrity, coord values, overview consistency, CRS/shape mismatch rejection, xarray roundtrip |
| `TestConsolidation` | 2 | Post-ingestion consolidation, correct array shapes after 2 ingestions |
| `TestDiscoverAcquisitions` | 3 | Correct grouping, incomplete acquisition warning, non-matching file skipping |

### Known Warnings (Expected)

- `UnstableSpecificationWarning` for `FixedLengthUTF32` on `platform` coordinate — known from Phase 0, accepted
- `UserWarning` about consolidated metadata not being part of Zarr V3 spec — expected, consolidation still works
- xarray `RuntimeWarning` about non-consolidated reads — only in tests that skip consolidation

## Additional instructions
- Keep a devlog of implementation progress, challenges, and decisions in the GitHub issue linked to this design document: https://github.com/EOPF-Explorer/data-model/issues/139
- Regularly update this design document with any refinements or changes to the plan as development progresses
- Make sure to be able to resume work after interruptions without losing context by keeping detailed notes and documentation in the issue and this design document.
