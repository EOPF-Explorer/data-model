# Pipeline Integration Recommendations

This document lists enhancements observed while integrating `eopf-geozarr` into an Argo-based batch conversion pipeline (data-model-pipeline). They are candidates for upstream inclusion or API refinement.

## 1. Output Prefix Expansion
**Current (pipeline)**: Wrapper detects `output_zarr` ending with `/` and appends `<item_id>_geozarr.zarr` derived from the input STAC/Zarr URL.
**Recommendation**: Support `--output-prefix` OR accept a trailing slash on positional `output_path` and perform the expansion internally (emitting the resolved final path). Add a log line: `Resolved output store: s3://.../S2A_..._geozarr.zarr`.

## 2. Group Existence Validation (Pre-flight)
**Current (pipeline)**: `validate_params_groups.py` inspects filesystem structure (presence of `.zarray` / `.zgroup`).
**Recommendation**: Native CLI flag `--validate-groups` to prune or fail fast when groups don’t exist. Modes:
- `--validate-groups=warn` (default): drop missing, report.
- `--validate-groups=error`: abort if any missing.
Emit JSON or structured summary when `--verbose`.

## 3. Profiles (WOZ Profiles)
**Current (pipeline)**: External JSON profile expansion before calling CLI.
**Recommendation**: Provide `--profile <name>` in CLI mapping to preset groups + chunk params. Add `eopf-geozarr profile list` / `profile show <name>` subcommands. Keep external mechanism as fallback.

## 4. Compressor Handling
**Current**: Template attempted to pass `--compressor`; CLI does not expose codec choice.
**Recommendation**: If codec selection is desired, add `--compressor <name>` now (zstd, lz4, blosc) with validation; else document fixed default explicitly to avoid confusion.

## 5. CRS Groups Convenience
**Current**: `--crs-groups` optional list.
**Recommendation**: Discover candidate groups automatically (search for geometry-like datasets) unless `--crs-groups` provided (override). Provide `eopf-geozarr info --crs-scan` to preview.

## 6. Dask Cluster Ergonomics
**Current**: `--dask-cluster` toggles local cluster with no feedback.
**Recommendation**: Print cluster dashboard URL (if available) and add `--dask-workers N` for quick scaling.

## 7. Structured Logging / Run Metadata
**Current**: Plain prints; pipeline scrapes logs.
**Recommendation**: Optional `--run-metadata <path.json>` to write machine-readable summary: inputs, resolved groups, timings, warnings. Eases automation and reproducibility.

## 8. Validation Command Enhancements
**Current**: `validate` skeleton present but incomplete.
**Recommendation**:
- Implement spec checks (multiscales, attributes, chunk shape policy).
- Exit code non-zero on *hard* failures, zero with warnings for soft issues.
- `--format json` for programmatic consumption.

## 9. HTML Tree Generation
**Current**: `_generate_html_output` scaffold incomplete.
**Recommendation**: Finish implementation; integrate with `info --html-output`. Provide minimal inline CSS (already drafted) and optional `--open-browser` flag.

## 10. Progress Reporting
**Recommendation**: Emit periodic progress per group: `group=/measurements/r10m scale=2 written=...MB elapsed=...s` to assist monitoring in batch workflows.

## 11. Retry / Resumability
**Recommendation**: Add `--resume` to skip already existing multiscale levels if output store partially present.

## 12. Exit Codes (Contract)
Document exit code meanings:
- 0 success
- 2 validation (input) error
- 3 group resolution failure
- 4 conversion runtime error

## 13. Environment Variable Overrides
Allow `EOZ_DEFAULT_PROFILE`, `EOZ_OUTPUT_PREFIX` env vars as implicit defaults (still overridden by flags).

## 14. Example Invocation Block in README
Provide ready-to-copy examples for Sentinel-2 & Sentinel-1 including polarization groups when Phase 1 logic is public.

---
**Next Steps (Suggested Order)**
1. Implement output prefix expansion (low risk, high UX win)
2. Group validation flag (prevents silent empty writes)
3. Finish validate + info HTML features
4. Add structured run metadata output
5. Introduce profiles subcommands then deprecate external expansion path over time

Feedback welcome—pipeline experience will continue surfacing actionable deltas.
