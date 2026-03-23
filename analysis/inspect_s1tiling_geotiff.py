"""
S1Tiling GeoTIFF Output Inspector

Inspects real S1Tiling GeoTIFF outputs to validate metadata assumptions
for the S1 GRD RTC → GeoZarr V3 ingestion pipeline.

Usage:
    # Inspect a single file
    python analysis/inspect_s1tiling_geotiff.py /path/to/s1a_32TQM_vv_ASC_037_20240101t120000_GammaNaughtRTC.tif

    # Inspect all GeoTIFFs in a directory (recursive)
    python analysis/inspect_s1tiling_geotiff.py /path/to/s1tiling_output/

    # Inspect and validate against expected schema
    python analysis/inspect_s1tiling_geotiff.py --validate /path/to/s1tiling_output/

Reference: https://s1-tiling.pages.orfeo-toolbox.org/s1tiling/latest/files.html
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import rasterio

# =============================================================================
# S1Tiling documented tag schema (from files.html, S1Tiling 1.4.0)
# =============================================================================

# Tags present on Orthorectified S2 tiles (final products)
EXPECTED_TAGS_ORTHO = {
    # --- Always present ---
    "ACQUISITION_DATETIME": "time of the first S1 image (UTC)",
    "ACQUISITION_DATETIME_1": "time of the first S1 image (UTC)",
    "ACQUISITION_DATETIME_2": "time of the second S1 image (UTC, when two S1 images concatenated)",
    "CALIBRATION": "calibration option (e.g. 'gamma')",
    "DEM_INFO": "DEM identifier",
    "FLYING_UNIT_CODE": "s1a | s1b | s1c",
    "IMAGE_TYPE": "BACKSCATTERING",
    "INPUT_S1_IMAGES": "list of input S1 images",
    "NOISE_REMOVED": "noise removal option",
    "ORBIT_NUMBER": "absolute orbit number",
    "ORBIT_DIRECTION": "ASC | DES",
    "ORTHORECTIFICATION_INTERPOLATOR": "interpolation method",
    "ORTHORECTIFIED": "true",
    "POLARIZATION": "vv | vh",
    "RELATIVE_ORBIT_NUMBER": "relative orbit number",
    "S2_TILE_CORRESPONDING_CODE": "MGRS tile code",
    "SPATIAL_RESOLUTION": "output spatial resolution",
    # --- RTC-specific ---
    "GAMMA_AREA_FILE": "γ area map file used for RTC (when applies)",
    # --- Standard TIFF tags ---
    "TIFFTAG_DATETIME": "generation time",
    "TIFFTAG_IMAGEDESCRIPTION": "product description",
    "TIFFTAG_SOFTWARE": "S1 Tiling version",
}

# Tags present on BorderMask files
EXPECTED_TAGS_MASK = {
    **EXPECTED_TAGS_ORTHO,
    "IMAGE_TYPE": "MASK",  # overrides BACKSCATTERING
}

# Tags present on LIA map files
EXPECTED_TAGS_LIA = {
    "DATA_TYPE": "100 * degrees(LIA) / sin(LIA)",
    "DEM_INFO": "DEM identifier",
    "DEM_LIST": "DEM tiles used",
    "DEM_RESAMPLING_METHOD": "DEM resampling method",
    "EOF_FILE": "precise orbit file",
    "GEOID_ORTHORECTIFICATION_INTERPOLATOR": "interpolation method",
    "IMAGE_TYPE": "LIA",
    "ORBIT_DIRECTION": "orbit direction",
    "ORTHORECTIFIED": "true",
    "RELATIVE_ORBIT_NUMBER": "relative orbit number",
    "S2_TILE_CORRESPONDING_CODE": "MGRS tile code",
    "SPATIAL_RESOLUTION": "output spatial resolution",
}

# Tags present on GAMMA_AREA map files
EXPECTED_TAGS_GAMMA_AREA = {
    "ACQUISITION_DATETIME": "time of first S1 image (UTC)",
    "ACQUISITION_DATETIME_1": "time of first S1 image (UTC)",
    "ACQUISITION_DATETIME_2": "time of second S1 image (UTC)",
    "DATA_TYPE": "GAMMA_AREA",
    "IMAGE_TYPE": "GRD",
    "INPUT_S1_IMAGES": "list of input S1 images",
    "ORBIT_DIRECTION": "orbit direction",
    "ORTHORECTIFIED": "true",
    "RELATIVE_ORBIT_NUMBER": "relative orbit number",
    "S2_TILE_CORRESPONDING_CODE": "MGRS tile code",
    "SPATIAL_RESOLUTION": "output spatial resolution",
}

# File name patterns
FNAME_PATTERN_ORTHO = re.compile(
    r"^s1([abc])_([A-Z0-9]+)_(vv|vh)_(ASC|DES)_(\d+)_(\d{8}t\d{6})"
    r"(?:_GammaNaughtRTC|_NormLim)?\.tif$",
    re.IGNORECASE,
)
FNAME_PATTERN_MASK = re.compile(
    r"^s1([abc])_([A-Z0-9]+)_(vv|vh)_(ASC|DES)_(\d+)_(\d{8}t\d{6})"
    r"(?:_GammaNaughtRTC|_NormLim)?_BorderMask\.tif$",
    re.IGNORECASE,
)
FNAME_PATTERN_LIA = re.compile(
    r"^(sin_LIA|LIA)_([A-Z0-9]+)_(\d+)\.tif$",
    re.IGNORECASE,
)
FNAME_PATTERN_GAMMA_AREA = re.compile(
    r"^GAMMA_AREA_(?:s1[abc]_)?([A-Z0-9]+)_(?:(?:ASC|DES)_)?(\d+)\.tif$",
    re.IGNORECASE,
)


def classify_file(path: Path) -> str:
    """Classify an S1Tiling output file by its filename pattern."""
    name = path.name
    if FNAME_PATTERN_MASK.match(name):
        return "border_mask"
    if FNAME_PATTERN_ORTHO.match(name):
        if "_GammaNaughtRTC" in name:
            return "gamma_naught_rtc"
        if "_NormLim" in name:
            return "sigma_normlim"
        return "ortho"
    if FNAME_PATTERN_LIA.match(name):
        return "lia"
    if FNAME_PATTERN_GAMMA_AREA.match(name):
        return "gamma_area"
    return "unknown"


def parse_filename_metadata(path: Path) -> dict:
    """Extract metadata fields encoded in the filename."""
    name = path.name
    result = {}

    m = FNAME_PATTERN_ORTHO.match(name) or FNAME_PATTERN_MASK.match(name)
    if m:
        result["flying_unit_code"] = f"s1{m.group(1)}"
        result["tile_name"] = m.group(2)
        result["polarisation"] = m.group(3).lower()
        result["orbit_direction"] = m.group(4)
        result["orbit"] = m.group(5)
        result["acquisition_stamp"] = m.group(6)
        return result

    m = FNAME_PATTERN_LIA.match(name)
    if m:
        result["lia_kind"] = m.group(1)
        result["tile_name"] = m.group(2)
        result["orbit"] = m.group(3)
        return result

    m = FNAME_PATTERN_GAMMA_AREA.match(name)
    if m:
        result["tile_name"] = m.group(1)
        result["orbit"] = m.group(2)
        return result

    return result


def inspect_geotiff(path: Path) -> dict:
    """Read all metadata from a GeoTIFF using rasterio."""
    with rasterio.open(str(path)) as src:
        t = src.transform
        info = {
            "file": str(path),
            "filename": path.name,
            "file_type": classify_file(path),
            "filename_metadata": parse_filename_metadata(path),
            "rasterio_profile": {
                "driver": src.driver,
                "dtype": str(src.dtypes[0]) if src.dtypes else None,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "crs": str(src.crs) if src.crs else None,
                "nodata": src.nodata,
            },
            "transform": {
                "affine": [t.a, t.b, t.c, t.d, t.e, t.f],
                "pixel_size_x": abs(t.a),
                "pixel_size_y": abs(t.e),
            },
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "tags": dict(src.tags()),
            "band_descriptions": list(src.descriptions) if src.descriptions else [],
            "band_tags": {},
        }
        # Per-band tags
        for i in range(1, src.count + 1):
            band_tags = src.tags(i)
            if band_tags:
                info["band_tags"][f"band_{i}"] = dict(band_tags)

    return info


def validate_tags(info: dict) -> list[str]:
    """Validate that expected tags are present based on file type."""
    file_type = info["file_type"]
    tags = info["tags"]
    issues = []

    if file_type == "unknown":
        issues.append(f"Could not classify file: {info['filename']}")
        return issues

    expected = {
        "gamma_naught_rtc": EXPECTED_TAGS_ORTHO,
        "sigma_normlim": EXPECTED_TAGS_ORTHO,
        "ortho": EXPECTED_TAGS_ORTHO,
        "border_mask": EXPECTED_TAGS_MASK,
        "lia": EXPECTED_TAGS_LIA,
        "gamma_area": EXPECTED_TAGS_GAMMA_AREA,
    }.get(file_type, {})

    # Check which expected tags are present/missing
    for tag_name in expected:
        if tag_name not in tags:
            # Some tags are conditional — don't flag as errors
            if tag_name in ("GAMMA_AREA_FILE", "ACQUISITION_DATETIME_2", "LIA_FILE"):
                continue
            issues.append(f"MISSING expected tag: {tag_name}")

    # Report unexpected tags (informational)
    expected_names = set(expected.keys())
    for tag_name in tags:
        if tag_name not in expected_names:
            issues.append(f"EXTRA tag found: {tag_name} = {tags[tag_name]!r}")

    # Cross-validate filename metadata against tags
    fname_meta = info["filename_metadata"]
    if "flying_unit_code" in fname_meta and "FLYING_UNIT_CODE" in tags:
        if fname_meta["flying_unit_code"] != tags["FLYING_UNIT_CODE"]:
            issues.append(
                f"MISMATCH flying_unit_code: filename={fname_meta['flying_unit_code']!r} "
                f"vs tag={tags['FLYING_UNIT_CODE']!r}"
            )
    if "orbit_direction" in fname_meta and "ORBIT_DIRECTION" in tags:
        # Filename may use ASC/DES, tag may use ASCENDING/DESCENDING or vice versa
        fname_dir = fname_meta["orbit_direction"].upper()
        tag_dir = tags["ORBIT_DIRECTION"].upper()
        if fname_dir not in tag_dir and tag_dir not in fname_dir:
            issues.append(f"MISMATCH orbit_direction: filename={fname_dir!r} vs tag={tag_dir!r}")

    return issues


def print_report(info: dict, validate: bool = False) -> None:
    """Print a human-readable report for one file."""
    print(f"\n{'=' * 80}")
    print(f"FILE: {info['filename']}")
    print(f"TYPE: {info['file_type']}")
    print(f"PATH: {info['file']}")
    print(f"{'=' * 80}")

    print("\n--- Rasterio Profile ---")
    for k, v in info["rasterio_profile"].items():
        print(f"  {k}: {v}")

    print("\n--- Transform ---")
    print(f"  Affine [a,b,c,d,e,f]: {info['transform']['affine']}")
    print(
        f"  Pixel size: {info['transform']['pixel_size_x']}m x {info['transform']['pixel_size_y']}m"
    )

    print("\n--- Bounds ---")
    for k, v in info["bounds"].items():
        print(f"  {k}: {v}")

    print("\n--- Filename Metadata ---")
    for k, v in info["filename_metadata"].items():
        print(f"  {k}: {v}")

    print(f"\n--- GeoTIFF Tags ({len(info['tags'])} total) ---")
    for k in sorted(info["tags"].keys()):
        print(f"  {k}: {info['tags'][k]!r}")

    if info["band_descriptions"]:
        print("\n--- Band Descriptions ---")
        for i, desc in enumerate(info["band_descriptions"], 1):
            print(f"  Band {i}: {desc}")

    if info["band_tags"]:
        print("\n--- Band Tags ---")
        for band, tags in info["band_tags"].items():
            print(f"  {band}:")
            for k, v in tags.items():
                print(f"    {k}: {v!r}")

    if validate:
        issues = validate_tags(info)
        print(f"\n--- Validation ({len(issues)} issues) ---")
        if not issues:
            print("  ALL OK: all expected tags present, no mismatches")
        else:
            for issue in issues:
                print(f"  {'WARNING' if 'EXTRA' in issue else 'ERROR'}: {issue}")

    # Critical tags for our ingestion pipeline
    print("\n--- Ingestion-Critical Tags ---")
    critical = [
        "ACQUISITION_DATETIME",
        "ORBIT_NUMBER",
        "RELATIVE_ORBIT_NUMBER",
        "FLYING_UNIT_CODE",
        "ORBIT_DIRECTION",
        "POLARIZATION",
        "S2_TILE_CORRESPONDING_CODE",
        "CALIBRATION",
        "IMAGE_TYPE",
    ]
    for tag in critical:
        value = info["tags"].get(tag)
        status = "PRESENT" if value is not None else "MISSING"
        print(f"  [{status:>7}] {tag}: {value!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect S1Tiling GeoTIFF outputs for metadata validation"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a GeoTIFF file or directory containing GeoTIFFs",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate tags against expected S1Tiling schema",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable format",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: path does not exist: {args.path}", file=sys.stderr)
        sys.exit(1)

    # Collect files
    if args.path.is_file():
        files = [args.path]
    else:
        files = sorted(args.path.rglob("*.tif"))
        if not files:
            print(f"No .tif files found in {args.path}", file=sys.stderr)
            sys.exit(1)

    results = []
    for f in files:
        info = inspect_geotiff(f)
        results.append(info)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(f"Found {len(results)} GeoTIFF file(s)")
        for info in results:
            print_report(info, validate=args.validate)

        # Summary table
        if len(results) > 1:
            print(f"\n{'=' * 80}")
            print("SUMMARY")
            print(f"{'=' * 80}")
            print(f"  Files inspected: {len(results)}")
            types = {}
            for info in results:
                ft = info["file_type"]
                types[ft] = types.get(ft, 0) + 1
            for ft, count in sorted(types.items()):
                print(f"  {ft}: {count}")

            # Union of all tag keys seen
            all_tags = set()
            for info in results:
                all_tags.update(info["tags"].keys())
            print(f"\n  All tag keys across files ({len(all_tags)}):")
            for tag in sorted(all_tags):
                present_in = sum(1 for info in results if tag in info["tags"])
                print(f"    {tag}: present in {present_in}/{len(results)} files")


if __name__ == "__main__":
    main()
