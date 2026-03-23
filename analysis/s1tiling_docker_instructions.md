# S1Tiling γ0T RTC Processing – Docker Instructions

Run S1Tiling 1.4.0 via Docker to produce γ0T RTC-calibrated GeoTIFFs from a
Sentinel-1 GRD product downloaded from CDSE, orthorectified onto an S2 MGRS tile.

> **Goal:** obtain *real* S1Tiling outputs to validate metadata/format
> assumptions for the GeoZarr ingestion pipeline.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker | `docker --version` ≥ 20 |
| Disk space | ~20 GB (S1 GRD ≈ 1.7 GB, DEM tiles ≈ 300 MB, outputs + tmp ≈ 15 GB) |
| RAM | ≥ 16 GB recommended (γ area estimation is RAM-greedy) |
| CDSE account | Register at <https://dataspace.copernicus.eu> |
| Internet | For S1 product download and DEM fetching |

---

## 1 – Create working directory structure

```bash
export S1T_WORKDIR=$HOME/Downloads/s1tiling_test
mkdir -p "$S1T_WORKDIR"/{data_out,data_raw,data_gamma_area,tmp,eof,config}
mkdir -p "$S1T_WORKDIR"/DEM/SRTM_30_hgt
```

---

## 2 – Configure EODAG for CDSE

Create `$S1T_WORKDIR/config/eodag.yml`:

```yaml
cop_dataspace:
  priority: 1
  auth:
    credentials:
      username: "YOUR_CDSE_EMAIL"
      password: "YOUR_CDSE_PASSWORD"
```

Replace the credentials with your CDSE (Copernicus Data Space Ecosystem)
account. EODAG will use `cop_dataspace` as the preferred provider to search
and download Sentinel-1 GRD products. The patch script fixes all the EODAG
4.0.0 incompatibilities so no other provider configuration is needed.

> **Tip:** You can test your credentials at
> <https://dataspace.copernicus.eu/browser/> before running S1Tiling.

---

## 3 – (Optional) Pre-download SRTM DEM tiles

S1Tiling needs SRTM 30 m DEM tiles in `$S1T_WORKDIR/DEM/SRTM_30_hgt/`.

If you already have them, copy/symlink. Otherwise S1Tiling can sometimes
auto-download, but it's more reliable to pre-stage them.

For MGRS tile **31TCH** (south of France), 20 SRTM tiles are needed because
the S1 GRD swaths extend well beyond the MGRS tile footprint — the Gamma Area
computation (for RTC) needs DEM covering the full S1 acquisition geometry:

```bash
cd "$S1T_WORKDIR/DEM/SRTM_30_hgt"
for tile in N41E002 N41E003 \
            N42E000 N42E001 N42E002 N42E003 N42W001 N42W002 N42W003 \
            N43E000 N43E001 N43E002 N43E003 N43E004 N43E005 N43W001 N43W002 N43W003 \
            N44W001 N44W002; do
  lat="${tile:0:3}"
  curl -sS -o "${tile}.hgt.gz" \
    "https://s3.amazonaws.com/elevation-tiles-prod/skadi/${lat}/${tile}.hgt.gz"
  gunzip -f "${tile}.hgt.gz"
done
# Verify: should be 20 .hgt files (~25 MB each, ~500 MB total)
ls -1 *.hgt | wc -l
```

> **Why so many tiles?** The MGRS tile 31TCH only covers lat 42–43°N,
> lon 0–2°E, but the 3 overlapping S1 descending orbits (008, 037, 110)
> have swaths extending from ~41°N to ~44°N and from ~3°W to ~5°E.
> The AgglomerateDEM step in S1Tiling needs DEM for the full swath.

> **Alternative tile:** Use any MGRS tile you prefer. Adjust DEM tiles and
> the config accordingly. Just pick a tile with Sentinel-1 coverage in the
> chosen date range.

---

## 4 – Create the S1Tiling configuration file

Create `$S1T_WORKDIR/config/S1GRD_RTC.cfg`:

```ini
[Paths]
# Final γ0T RTC calibrated products
output : /data/data_out

# Gamma Area maps directory
gamma_area : /data/data_gamma_area

# Raw S1 products (downloaded here)
s1_images : /data/data_raw

# Precise Orbit files (downloaded automatically)
eof_dir : /data/eof

# Temporary files (can be large ~15 GB)
tmp : /tmp/s1tiling

# DEM information
dem_dir : /MNT/SRTM_30_hgt
dem_info : SRTM 30m

# Geoid (shipped inside the docker image — use the image's install path, NOT /data)
geoid_file : /opt/S1TilingEnv/lib/python3.10/site-packages/s1tiling/resources/Geoid/egm96.grd

[DataSource]
# EODAG config (mounted in docker)
eodag_config : /eo_config/eodag.yml

# Enable downloading from CDSE
download : True
nb_parallel_downloads : 2

# Region of interest: S2 MGRS tile(s)
# S1Tiling will download S1 GRD products overlapping this tile
roi_by_tiles : 31TCH

# Platform filter (leave empty for both S1A and S1B)
platform_list : S1A

# Polarisation
polarisation : VV VH

# Orbit direction filter (optional — DES = descending)
orbit_direction : DES

# Date range — keep it VERY SHORT (12 days = 1 revisit cycle)
# This minimises download volume. One date pair is enough for our test.
first_date : 2025-02-01
last_date : 2025-02-14

[Processing]
# --- γ0T RTC calibration ---
calibration : gamma_naught_rtc

# Noise removal
remove_thermal_noise : True
lower_signal_value : 1e-7

# Output resolution (meters)
output_spatial_resolution : 10.

# Tiles to process
tiles : 31TCH

# Orthorectification interpolation
orthorectification_interpolation_method : bco
orthorectification_gridspacing : 40

# DEM cache strategy
cache_dem_by : copy

# γ area RTC specific
distribute_area : False
min_gamma_area : 1.0
calibration_factor : 1.0
output_nodata : False

# Do NOT use resampled DEM (simpler, less RAM issue)
use_resampled_dem : False

# Streaming: disable for gamma_area to avoid artefacts
disable_streaming.apply_gamma_area : True

# Parallelism — for a single test, keep low
nb_parallel_processes : 1
ram_per_process : 8192
nb_otb_threads : 4

# Logging
mode : debug logging

# Generate border masks (useful for our pipeline)
[Mask]
generate_border_mask : True

[Quicklook]
generate : False

[Metadata]
phase0_test : s1-grd-rtc-validation
producer : eopf-geozarr-phase0
```

### Key options explained

| Option | Value | Why |
|---|---|---|
| `calibration` | `gamma_naught_rtc` | This is the γ0T RTC pipeline |
| `roi_by_tiles` | `31TCH` | S2 MGRS tile (matches S1Tiling docs examples) |
| `first/last_date` | 12-day window | Minimises downloads — 1 revisit cycle |
| `platform_list` | `S1A` | Single platform to reduce data volume |
| `orbit_direction` | `DES` | Single direction to further reduce volume |
| `generate_border_mask` | `True` | Produces mask files we need for pipeline |

---

## 5 – Pull the Docker image

```bash
docker pull registry.orfeo-toolbox.org/s1-tiling/s1tiling:1.4.0-ubuntu-otb9.1.1
```

Alternative registry:
```bash
docker pull cnes/s1tiling:1.4.0-ubuntu-otb9.1.1
```

---

## 6 – Run S1Tiling (γ0T RTC)

S1Tiling with `gamma_naught_rtc` calibration automatically computes the Gamma
Area maps and applies them. A single `S1Processor` invocation handles
everything.

```bash
docker run --rm \
  -v "$S1T_WORKDIR"/DEM:/MNT \
  -v "$S1T_WORKDIR":/data \
  -v "$S1T_WORKDIR"/config:/eo_config \
  -v "$(dirname "$0")"/../analysis:/patch \
  --entrypoint bash \
  registry.orfeo-toolbox.org/s1-tiling/s1tiling:1.4.0-ubuntu-otb9.1.1 \
  -c 'python3 /patch/s1tiling_eodag4_patch.py && S1Processor /data/config/S1GRD_RTC.cfg'
```

If running from a different directory, replace the `-v` mount for `/patch`
with the absolute path to the `analysis/` folder:

```bash
docker run --rm \
  -v "$S1T_WORKDIR"/DEM:/MNT \
  -v "$S1T_WORKDIR":/data \
  -v "$S1T_WORKDIR"/config:/eo_config \
  -v /home/emathot/Workspace/eopf-explorer/data-model/analysis:/patch \
  --entrypoint bash \
  registry.orfeo-toolbox.org/s1-tiling/s1tiling:1.4.0-ubuntu-otb9.1.1 \
  -c 'python3 /patch/s1tiling_eodag4_patch.py && S1Processor /data/config/S1GRD_RTC.cfg'
```

> **Bug workaround:** S1Tiling 1.4.0 ships EODAG 4.0.0 which has five
> breaking changes that prevent it from working with `cop_dataspace`:
>
> 1. `productType` kwarg to `dag.search()` was renamed to `collection`;
>    having both causes cop_dataspace to fail silently → falls back to peps
> 2. Product properties now use STAC names (`sat:orbit_state`, `platform`, etc.)
>    instead of legacy names (`orbitDirection`, `platformSerialIdentifier`, etc.)
> 3. `cop_dataspace` OData v4 API rejects `polarizationChannels` and `sensorMode`
> 4. `cop_dataspace` requires UPPERCASE orbit direction (`"DESCENDING"` not `"descending"`)
> 5. `relativeOrbitNumber` search param silently returns 0 results on cop_dataspace
>
> The patch script `s1tiling_eodag4_patch.py` fixes all five issues.
> The container is `--rm` so nothing persists.

**What this does:**
1. Downloads S1A GRD products from CDSE (via EODAG) for the date range
2. Downloads precise orbit (EOF) files
3. Calibrates (σ0 internally), cuts, and orthorectifies onto the 31TCH MGRS grid
4. Computes the Gamma Area map for the orbit
5. Applies the γ0T RTC correction
6. Concatenates half-tiles into final products
7. Generates border masks

**Expected runtime:** 30 min – 2 hours depending on network speed and host CPU.

### Troubleshooting

- **Download failures:** CDSE can be slow or rate-limited. Re-run the same
  command — S1Tiling caches intermediary results. Already-completed steps are
  skipped.
- **RAM issues:** If the gamma area computation fails with OOM, try setting
  `use_resampled_dem : True` with `resample_dem_factor_x : 2.0` /
  `resample_dem_factor_y : 2.0` in the config, or increase `ram_per_process`.
- **Misleading warnings at the end:** S1Tiling may report "download failures"
  for redundant S1 products that weren't strictly needed. Check if the actual
  output files were produced.

---

## 7 – Expected output files

After a successful run, you should find:

### Final products: `$S1T_WORKDIR/data_out/31TCH/`

| Pattern | Description |
|---|---|
| `s1a_31TCH_vv_DES_{orbit}_{date}_GammaNaughtRTC.tif` | γ0T VV backscatter |
| `s1a_31TCH_vh_DES_{orbit}_{date}_GammaNaughtRTC.tif` | γ0T VH backscatter |
| `s1a_31TCH_vv_DES_{orbit}_{date}_GammaNaughtRTC_BorderMask.tif` | VV border mask |
| `s1a_31TCH_vh_DES_{orbit}_{date}_GammaNaughtRTC_BorderMask.tif` | VH border mask |

### Gamma Area maps: `$S1T_WORKDIR/data_gamma_area/`

| Pattern | Description |
|---|---|
| `GAMMA_AREA_s1a_31TCH_DES_{orbit}.tif` | γ area map for this S2 tile + orbit |

### GeoTIFF metadata (from S1Tiling docs)

Each final product should contain these GeoTIFF tags:

| Tag | Example value |
|---|---|
| `CALIBRATION` | `gamma_naught_rtc` |
| `IMAGE_TYPE` | `BACKSCATTERING` |
| `FLYING_UNIT_CODE` | `s1a` |
| `POLARIZATION` | `vv` or `vh` |
| `ORBIT_DIRECTION` | `DES` |
| `RELATIVE_ORBIT_NUMBER` | e.g. `110` |
| `S2_TILE_CORRESPONDING_CODE` | `31TCH` |
| `SPATIAL_RESOLUTION` | `10.0` |
| `ORTHORECTIFIED` | `true` |
| `GAMMA_AREA_FILE` | name of the gamma area map used |
| `TIFFTAG_SOFTWARE` | `S1 Tiling v1.4.0` |
| `ACQUISITION_DATETIME` | UTC datetime |

Product encoding: **Float32 GeoTIFF, deflate compressed**, CRS matching the
MGRS tile UTM zone (e.g., EPSG:32631 for 31TCH).

---

## 8 – Inspect the outputs

Run the inspector script on the results:

```bash
cd /home/emathot/Workspace/eopf-explorer/data-model

# Inspect all final products with validation
python analysis/inspect_s1tiling_geotiff.py --validate "$S1T_WORKDIR/data_out/"

# Inspect gamma area maps
python analysis/inspect_s1tiling_geotiff.py --validate "$S1T_WORKDIR/data_gamma_area/"

# JSON output for programmatic analysis
python analysis/inspect_s1tiling_geotiff.py --json "$S1T_WORKDIR/data_out/" > analysis/s1tiling_output_metadata.json
```

### What to report back

Please share the following after the run:

1. **Console output** of the docker run (especially the final execution report)
2. **File listing:**
   ```bash
   find "$S1T_WORKDIR"/data_out -name "*.tif" -ls
   find "$S1T_WORKDIR"/data_gamma_area -name "*.tif" -ls
   ```
3. **Inspector output:**
   ```bash
   python analysis/inspect_s1tiling_geotiff.py --validate "$S1T_WORKDIR/data_out/"
   python analysis/inspect_s1tiling_geotiff.py --validate "$S1T_WORKDIR/data_gamma_area/"
   ```
4. **JSON dump** (for detailed programmatic review):
   ```bash
   python analysis/inspect_s1tiling_geotiff.py --json "$S1T_WORKDIR/data_out/" \
     "$S1T_WORKDIR/data_gamma_area/" > analysis/s1tiling_output_metadata.json
   ```

---

## 9 – Alternative: different MGRS tile / shorter run

If 31TCH doesn't work (DEM unavailable, no S1 coverage in the window, etc.),
pick another tile. Good candidates with frequent S1 coverage:

| MGRS Tile | Location | UTM Zone |
|---|---|---|
| `31TCH` | South France (Toulouse area) | 31N |
| `32TQM` | North Italy | 32N |
| `33UUP` | Germany | 33N |
| `10SEG` | California coast | 10N |

Adjust `tiles`, `roi_by_tiles`, and DEM tiles accordingly.

To further reduce processing time, try `calibration: gamma` (simple γ0 without
RTC). This skips the Gamma Area map computation but won't produce the RTC
product we actually need.

---

## 10 – Filename pattern for the GAMMA_AREA file

Note from the docs: the Gamma Area filename uses the format
`GAMMA_AREA_s1a_{tile}_{direction}_{orbit}.tif` (example:
`GAMMA_AREA_s1a_31TCH_DES_110.tif`). This is different from the documented
template `GAMMA_AREA_{tile}_{orbit}.tif` — the actual filename includes the
flying unit code and orbit direction. The inspector script handles both
patterns.
