# Plan: Deprecate v0 Layout + Clean Attrs + Fix `_FillValue` (issue #171)

## TL;DR
Three coordinated cleanups: (1) drop the legacy GeoZarr v0.4 layout (TMS + `r10m/0,/1,/2`) from the general `convert` CLI and the entire package, aligning it on the S2-optimized layout, (2) strip the source-only `_eopf_attrs` block and other misleading raw-encoding attrs from every array written by every converter (S2 + general + S1), (3) set CF `_FillValue` (via `xarray.backends.zarr.FillValueCoder.encode`) on every decoded float measurement array, alongside zarr-level `fill_value: NaN`, so xarray's `to_masked_array()` works (issue #171). Regenerate fixtures and add guardrail tests.

## Phase 1 — Remove v0 layout artefacts

1. **Drop TMS generation** in `src/eopf_geozarr/s2_optimization/s2_multiscale.py`:
   - Remove `multiscales_flavor` branching; always emit `experimental_multiscales_convention` only.
   - Delete the `ogc_tms` codepath: `tile_matrix_set`, `tile_matrix_limits`, calls to `create_native_crs_tile_matrix_set` and `_create_tile_matrix_limits`.
   - Drop the `MultiscalesFlavor` literal.
2. **Delete v0 layout in `src/eopf_geozarr/conversion/geozarr.py`**:
   - Remove the `{group_name}/0` native subgroup write and the per-level `/N` overview write loop (around lines 485 and `create_geozarr_compliant_multiscales`).
   - Rewrite `create_geozarr_dataset` (and `iterative_copy`, `write_geozarr_group`, `create_geozarr_compliant_multiscales`) so each group writes data arrays directly at the group root with a sibling `multiscales` convention entry — mirroring the structure produced by `convert_s2_optimized` for non-S2-specific cases.
   - Remove `tile_width` parameter from the public + internal signatures.
3. **Remove TMS types/utilities** that become unreachable: `TileMatrixLimitJSON`, `TileMatrixJSON`, `TileMatrixSetJSON` in `src/eopf_geozarr/types.py`; the `tms` import and `create_native_crs_tile_matrix_set` / `_create_tile_matrix_limits` helpers (locate via grep), if not referenced elsewhere.
4. **CLI**: drop `--tile-width` in `src/eopf_geozarr/cli.py` (`convert` parser) and any TMS-flavor flags. Keep both `convert` and `convert-s2-optimized` commands. *(parallel with steps 1-3 once signatures are settled)*
5. **Docs**: rewrite `docs/converter.md` "V0 vs V1" section to describe only the current single layout; update `docs/quickstart.md`, `docs/architecture.md`, `docs/api-reference.md`, `docs/faq.md`, `docs/index.md`, and `CHANGELOG.md` accordingly. *(parallel with step 4)*

## Phase 2 — Strip misleading / source-only attrs

6. **Add a shared attr-sanitizer** under `src/eopf_geozarr/conversion/utils.py`, e.g. `sanitize_array_attrs(var)` that:
   - Pops `_eopf_attrs` unconditionally.
   - For decoded float measurement vars (CF mask_and_scale applied): pops raw-encoding leftovers `dtype`, `fill_value`, `valid_min`, `valid_max`; rewrites `units` to `"1"` when source value was `"digital_counts"`.
   - Leaves CF `scale_factor` / `add_offset` intact (xarray needs them on read when CF decoding is disabled by the user).
7. **Apply sanitizer** in all write paths (depends on step 6):
   - `s2_optimization/s2_multiscale.py` `create_measurements_encoding` / before `stream_write_dataset` calls (native + downsampled levels).
   - `conversion/geozarr.py` before each `to_zarr` call inside `write_geozarr_group` and overview writes.
   - `conversion/sentinel1_reprojection.py` before its `to_zarr` (S1 path keeps current layout per decision).
8. **Drop the dead helper** `extract_scale_offset_encoding` in `s2_optimization/s2_multiscale.py` (lines ~1250–1259) — only references `_eopf_attrs` and is unused.

## Phase 3 — Correct `_FillValue` (issue #171)

9. **Centralise fill encoding** in `conversion/utils.py`: new helper `encode_cf_fill_value(dtype, fill)` wrapping `xarray.backends.zarr.FillValueCoder.encode`. Used by every converter.
10. **Float measurement arrays** (S2 + general): in encoding setup (next to `var_encoding["fill_value"] = "NaN"` blocks at `s2_multiscale.py:531` and the new equivalent in `geozarr.py`), also inject `attrs["_FillValue"] = encode_cf_fill_value(dtype, np.nan)`. Required on both native and downsampled levels.
11. **Integer mask / classification arrays** (S2): when the source declares a raw nodata via `_eopf_attrs.fill_value`, propagate to `attrs["_FillValue"] = encode_cf_fill_value(dtype, fill)` *and* zarr `fill_value=fill`. Skip arrays without a declared nodata.
12. **S1 measurement arrays**: in `sentinel1_reprojection.py`, ensure `_FillValue` attribute is set via the same helper (replace the raw float assignment at line 262).
13. **Coord / angle arrays** keep their existing handling — the fixture shows it already produces correct base64 `_FillValue`; only verify after sanitizer pass doesn't strip it.

## Phase 4 — Fixtures, tests, validation

14. **Regenerate JSON fixtures** under `tests/_test_data/geozarr_examples/`, `tests/_test_data/optimized_geozarr_examples/`, `tests/_test_data/s1_examples/`, `tests/_test_data/s2_examples/` via the existing fixture-build scripts (locate via `analysis/` or per-fixture conftest). *(depends on Phases 1-3)*
15. **Add guardrail tests** in `tests/test_conversion.py` (or a new `tests/test_array_attrs.py`):
    - Walk every array in a converted Zarr store, assert `"_eopf_attrs" not in attrs`.
    - Assert no array has `units == "digital_counts"`.
    - For every float array under `/measurements/`, assert `"_FillValue" in attrs` and its value round-trips via `FillValueCoder.decode` to `NaN`.
    - Assert no array group contains `tile_matrix_set` / `tile_matrix_limits` metadata.
16. **Issue #171 regression test**: open a converted store with `xarray.open_dataset(..., use_zarr_fill_value_as_mask=True)`, call `.to_masked_array()` on a reflectance band, assert `.mask.any()` over a region known to contain fill pixels.
17. **Run** `pytest -k "conversion or s2 or s1 or projjson or array_attrs"`, then full `pytest`, and `mkdocs build --strict`.

## Relevant files

- `src/eopf_geozarr/conversion/geozarr.py` — rewrite layout, remove `tile_width`, apply sanitizer + `_FillValue`.
- `src/eopf_geozarr/conversion/utils.py` — add `sanitize_array_attrs` + `encode_cf_fill_value` helpers (sits next to existing `_FillValue` selector at line 18).
- `src/eopf_geozarr/conversion/sentinel1_reprojection.py` — sanitize attrs + use `encode_cf_fill_value` at line ~262.
- `src/eopf_geozarr/s2_optimization/s2_multiscale.py` — drop TMS branch, remove `extract_scale_offset_encoding`, set `_FillValue` attr alongside `fill_value: "NaN"` at lines ~531 and ~590; apply sanitizer in `create_measurements_encoding`.
- `src/eopf_geozarr/s2_optimization/s2_converter.py` — verify no TMS coupling remains.
- `src/eopf_geozarr/cli.py` — drop `--tile-width` and any TMS flags from `convert_parser`.
- `src/eopf_geozarr/types.py` — remove TMS TypedDicts.
- `src/eopf_geozarr/data_api/geozarr/multiscales/tms.py` — keep models for downstream consumers (data_api still parses TMS produced elsewhere) **unless** grep confirms no external dependency; decide during step 3.
- `docs/converter.md`, `docs/quickstart.md`, `docs/architecture.md`, `docs/api-reference.md`, `docs/faq.md`, `docs/index.md`, `CHANGELOG.md` — doc updates.
- `tests/_test_data/**/*.json` — regenerated fixtures.
- `tests/test_array_attrs.py` (new) or extension to `tests/test_conversion.py` — guardrail tests.

## Verification

1. `pytest tests/ -x` passes, including new attr/`_FillValue` assertions and issue #171 regression.
2. Manual check on a freshly converted store: `python -c "import xarray as xr; ds=xr.open_dataset('out.zarr/measurements/reflectance/r10m', engine='zarr', consolidated=True, use_zarr_fill_value_as_mask=True); print(ds.b02.to_masked_array())"` shows masked fill pixels.
3. `grep -r "_eopf_attrs" tests/_test_data/` returns no match in regenerated fixtures.
4. `grep -r "tile_matrix" tests/_test_data/` returns no match.
5. `mkdocs build --strict` passes (no broken anchors after V0/V1 section removal).
6. `eopf-geozarr convert --help` no longer lists `--tile-width`.

## Decisions (from user)

- Keep two CLI commands: `convert` (general, rewritten to S2-style layout) and `convert-s2-optimized`.
- Remove TMS generation from the entire package (not just from general converter).
- Defer S1 layout migration to a follow-up issue — this PR only fixes attrs + `_FillValue` on the S1 codepath.
- Strip `_eopf_attrs` on every data/coord array in every converter.
- Strip raw-encoding attrs (`dtype`, raw `fill_value`, `valid_min/max`, misleading `units`) on decoded float measurements; keep CF `scale_factor`/`add_offset`.
- Set `_FillValue = FillValueCoder.encode(NaN, dtype)` and zarr `fill_value: NaN` on every float measurement array.
- Regenerate fixtures and add guardrail tests.

## Out of scope

- S1 RTC / S1 GRD multiscale layout migration to the new S2-style layout (follow-up issue).
- Removing the `tms` pydantic models in `data_api/` if they are still useful for parsing external TMS metadata (defer to step 3 decision).
- Bumping major version / writing the migration guide for downstream consumers (separate PR if needed).
