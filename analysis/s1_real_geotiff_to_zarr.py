"""
Phase 0 — Real GeoTIFF → GeoZarr V3 Conversion Test

Reads actual S1Tiling γ0T RTC GeoTIFFs produced by the Docker pipeline
and converts them into a Zarr V3 store following the implementation plan.

Three acquisitions (orbits 008, 037, 110) × two polarisations (VV, VH)
+ border masks.  Full 10980×10980 at 10m resolution, EPSG:32631.

Usage:
    python analysis/s1_real_geotiff_to_zarr.py \
        --input-dir ~/Downloads/s1tiling_test/data_out/31TCH \
        --gamma-area-dir ~/Downloads/s1tiling_test/data_gamma_area \
        --output-dir ~/Downloads/s1tiling_test/zarr_test
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
import zarr

# =============================================================================
# Constants: Zarr Conventions (same as prototype)
# =============================================================================

MULTISCALES_UUID = "d35379db-88df-4056-af3a-620245f8e347"
GEO_PROJ_UUID = "f17cb550-5864-4468-aeb7-f3180cfb622f"
SPATIAL_UUID = "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4"

ZARR_CONVENTIONS = [
    {
        "uuid": MULTISCALES_UUID,
        "schema_url": "https://raw.githubusercontent.com/zarr-conventions/multiscales/refs/tags/v1/schema.json",
        "spec_url": "https://github.com/zarr-conventions/multiscales/blob/v1/README.md",
        "name": "multiscales",
        "description": "Multiscale layout of zarr datasets",
    },
    {
        "uuid": GEO_PROJ_UUID,
        "schema_url": "https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v1/schema.json",
        "spec_url": "https://github.com/zarr-experimental/geo-proj/blob/v1/README.md",
        "name": "proj:",
        "description": "Coordinate reference system information for geospatial data",
    },
    {
        "uuid": SPATIAL_UUID,
        "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
        "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
        "name": "spatial:",
        "description": "Spatial coordinate information",
    },
]

# Overview chain: (level_name, parent_name, downsample_factor)
OVERVIEW_CHAIN = [
    ("r10m", None, 1),
    ("r20m", "r10m", 2),
    ("r60m", "r20m", 3),
    ("r120m", "r60m", 2),
    ("r360m", "r120m", 3),
    ("r720m", "r360m", 2),
]


def best_chunk_size(dim: int, max_chunk: int = 512) -> int:
    """Find the largest divisor of *dim* that is <= *max_chunk*.

    Zarr sharding requires inner chunks to divide the shard evenly.
    """
    if dim <= max_chunk:
        return dim
    for d in range(max_chunk, 0, -1):
        if dim % d == 0:
            return d
    return 1


# Filename pattern for S1Tiling outputs
# e.g. s1a_31TCH_vv_DES_110_20250205t060110_GammaNaughtRTC.tif
S1TILING_PATTERN = re.compile(
    r"(?P<platform>s1[abc])_"
    r"(?P<tile>[0-9]{2}[A-Z]{3})_"
    r"(?P<pol>vv|vh)_"
    r"(?P<orbit_dir>ASC|DES)_"
    r"(?P<rel_orbit>\d{3})_"
    r"(?P<acq_stamp>\d{8}t\d{6})_"
    r"(?P<product>GammaNaughtRTC)"
    r"(?P<mask>_BorderMask)?\.tif$"
)


# =============================================================================
# GeoTIFF metadata extraction
# =============================================================================


def extract_geotiff_metadata(path: Path) -> dict:
    """Extract CRS, transform, bounds, and custom tags from a real S1Tiling GeoTIFF."""
    with rasterio.open(str(path)) as src:
        tags = src.tags()
        t = src.transform
        spatial_transform = [t.a, t.b, t.c, t.d, t.e, t.f]
        return {
            "crs": str(src.crs),
            "spatial_transform": spatial_transform,
            "shape": [src.height, src.width],
            "bounds": [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
            "datetime": tags.get("ACQUISITION_DATETIME", ""),
            "absolute_orbit": int(tags.get("ORBIT_NUMBER", "0")),
            "relative_orbit": int(tags.get("RELATIVE_ORBIT_NUMBER", "0")),
            "platform": tags.get("FLYING_UNIT_CODE", ""),
            "calibration": tags.get("CALIBRATION", ""),
            "input_s1_images": tags.get("INPUT_S1_IMAGES", ""),
            "tags": tags,
        }


# =============================================================================
# Group S1Tiling output files into acquisitions
# =============================================================================


def discover_acquisitions(input_dir: Path) -> list[dict]:
    """Parse filenames and group into acquisition bundles (vv, vh, masks)."""
    files = sorted(input_dir.glob("*.tif"))
    # Group by (platform, tile, orbit_dir, rel_orbit, acq_stamp)
    groups: dict[tuple, dict] = {}

    for f in files:
        m = S1TILING_PATTERN.match(f.name)
        if not m:
            print(f"  SKIP: {f.name} (doesn't match pattern)")
            continue

        key = (
            m.group("platform"),
            m.group("tile"),
            m.group("orbit_dir"),
            m.group("rel_orbit"),
            m.group("acq_stamp"),
        )

        if key not in groups:
            groups[key] = {
                "platform": m.group("platform"),
                "tile": m.group("tile"),
                "orbit_dir": m.group("orbit_dir"),
                "rel_orbit": m.group("rel_orbit"),
                "acq_stamp": m.group("acq_stamp"),
            }

        pol = m.group("pol")
        is_mask = m.group("mask") is not None

        if is_mask:
            groups[key][f"{pol}_mask"] = f
        else:
            groups[key][pol] = f

    # Validate completeness
    acquisitions = []
    for key, acq in sorted(groups.items()):
        missing = [k for k in ("vv", "vh", "vv_mask", "vh_mask") if k not in acq]
        if missing:
            print(f"  WARNING: Acquisition {key} missing: {missing}")
        acquisitions.append(acq)

    return acquisitions


# =============================================================================
# Multiscales layout
# =============================================================================


def compute_multiscales_layout(
    native_shape: list[int],
    native_transform: list[float],
) -> list[dict]:
    """Build the multiscales layout array for all resolution levels."""
    layout: list[dict] = []
    current_shape = native_shape[:]
    current_transform = native_transform[:]

    for level_name, parent_name, factor in OVERVIEW_CHAIN:
        if parent_name is not None:
            current_shape = [
                int(np.ceil(current_shape[0] / factor)),
                int(np.ceil(current_shape[1] / factor)),
            ]
            current_transform = [
                current_transform[0] * factor,
                current_transform[1],
                current_transform[2],
                current_transform[3],
                current_transform[4] * factor,
                current_transform[5],
            ]

        entry: dict = {
            "asset": level_name,
            "spatial:shape": current_shape[:],
            "spatial:transform": current_transform[:],
        }
        if parent_name is None:
            entry["transform"] = {"scale": [1.0, 1.0]}
        else:
            entry["derived_from"] = parent_name
            entry["transform"] = {
                "scale": [float(factor), float(factor)],
                "translation": [0.0, 0.0],
            }
        layout.append(entry)

    return layout


# =============================================================================
# Store creation
# =============================================================================


def create_s1_store(
    store_path: Path,
    orbit_direction: str,
    meta: dict,
) -> None:
    """Create a new S1 GRD RTC Zarr V3 store with full conventions metadata."""
    height, width = meta["shape"]

    root = zarr.open_group(str(store_path), mode="w", zarr_format=3)
    orbit_group = root.create_group(orbit_direction)

    layout = compute_multiscales_layout(meta["shape"], meta["spatial_transform"])

    orbit_group.attrs.update(
        {
            "zarr_conventions": ZARR_CONVENTIONS,
            "multiscales": {
                "layout": layout,
                "resampling_method": "average",
            },
            "proj:code": meta["crs"],
            "spatial:dimensions": ["Y", "X"],
            "spatial:bbox": meta["bounds"],
        }
    )

    # Create each resolution level
    for level_entry in layout:
        level_name = level_entry["asset"]
        level_h, level_w = level_entry["spatial:shape"]

        level_group = orbit_group.create_group(level_name)
        level_group.attrs.update(
            {
                "spatial:shape": [level_h, level_w],
                "spatial:transform": level_entry["spatial:transform"],
            }
        )

        inner_chunks = (1, best_chunk_size(level_h), best_chunk_size(level_w))
        shard_shape = (1, level_h, level_w)

        for name, dtype, fill in [
            ("vv", "float32", float("nan")),
            ("vh", "float32", float("nan")),
            ("border_mask", "uint8", 0),
        ]:
            level_group.create_array(
                name,
                shape=(0, level_h, level_w),
                dtype=dtype,
                chunks=inner_chunks,
                shards=shard_shape,
                compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=5),
                fill_value=fill,
                dimension_names=["time", "Y", "X"],
            )

    # Coordinate variables at native resolution only
    r10m = orbit_group["r10m"]
    for name, dtype, fill in [
        ("time", "int64", 0),
        ("absolute_orbit", "int32", 0),
        ("relative_orbit", "int32", 0),
    ]:
        r10m.create_array(
            name,
            shape=(0,),
            dtype=dtype,
            chunks=(512,),
            fill_value=fill,
            dimension_names=["time"],
        )
    r10m.create_array(
        "platform",
        shape=(0,),
        dtype="<U4",
        chunks=(512,),
        fill_value="",
        dimension_names=["time"],
    )


# =============================================================================
# Downsampling
# =============================================================================


def downsample_2d(data: np.ndarray, factor: int, method: str = "average") -> np.ndarray:
    """Downsample a 2D array by the given factor."""
    h, w = data.shape
    new_h = int(np.ceil(h / factor))
    new_w = int(np.ceil(w / factor))

    if method == "nearest":
        return data[::factor, ::factor][:new_h, :new_w]

    # Average with edge padding
    pad_h = new_h * factor - h
    pad_w = new_w * factor - w
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(data, ((0, pad_h), (0, pad_w)), mode="edge")
    else:
        padded = data

    reshaped = padded.reshape(new_h, factor, new_w, factor)
    if np.issubdtype(data.dtype, np.floating):
        return np.nanmean(reshaped, axis=(1, 3)).astype(data.dtype)
    return reshaped.mean(axis=(1, 3)).astype(data.dtype)


# =============================================================================
# Ingestion: append one acquisition
# =============================================================================


def ingest_acquisition(
    store_path: Path,
    orbit_direction: str,
    vv_path: Path,
    vh_path: Path,
    vv_mask_path: Path,
    vh_mask_path: Path,
    meta: dict,
) -> int:
    """Append one acquisition to the store, including overviews.

    Uses the VV border mask (S1Tiling produces per-pol masks; they differ
    slightly because each pol band has different no-data fringe patterns).
    For our pipeline, we use the VV mask as the primary border mask.
    """
    root = zarr.open_group(str(store_path), mode="r+", zarr_format=3)
    orbit = root[orbit_direction]

    # Read GeoTIFF data
    t0 = time.time()
    with rasterio.open(str(vv_path)) as src:
        vv_data = src.read(1)
    with rasterio.open(str(vh_path)) as src:
        vh_data = src.read(1)
    with rasterio.open(str(vv_mask_path)) as src:
        mask_data = src.read(1).astype(np.uint8)
    read_time = time.time() - t0

    print(
        f"    Read GeoTIFFs: {read_time:.1f}s  "
        f"(vv: {vv_data.dtype} min={np.nanmin(vv_data):.4f} max={np.nanmax(vv_data):.4f}, "
        f"mask: unique={np.unique(mask_data).tolist()})"
    )

    # Determine new time index
    r10m = orbit["r10m"]
    current_size = r10m["vv"].shape[0]
    new_size = current_size + 1

    # Build data at all levels
    data_by_level = {"r10m": (vv_data, vh_data, mask_data)}

    t0 = time.time()
    prev_vv, prev_vh, prev_mask = vv_data, vh_data, mask_data
    for level_name, _, factor in OVERVIEW_CHAIN[1:]:
        prev_vv = downsample_2d(prev_vv, factor, "average")
        prev_vh = downsample_2d(prev_vh, factor, "average")
        prev_mask = downsample_2d(prev_mask, factor, "nearest")
        data_by_level[level_name] = (prev_vv, prev_vh, prev_mask)
    overview_time = time.time() - t0
    print(f"    Overviews generated: {overview_time:.1f}s")

    # Write to each level
    t0 = time.time()
    for level_name, (vv_lev, vh_lev, mask_lev) in data_by_level.items():
        level = orbit[level_name]
        h, w = vv_lev.shape

        level["vv"].resize((new_size, h, w))
        level["vh"].resize((new_size, h, w))
        level["border_mask"].resize((new_size, h, w))

        level["vv"][current_size, :, :] = vv_lev
        level["vh"][current_size, :, :] = vh_lev
        level["border_mask"][current_size, :, :] = mask_lev
    write_time = time.time() - t0
    print(f"    Zarr write (all levels): {write_time:.1f}s")

    # Append coordinate variables
    for coord_name in ["time", "absolute_orbit", "relative_orbit", "platform"]:
        r10m[coord_name].resize((new_size,))

    # Parse datetime — S1Tiling uses format: "2025:02:10T06:09:20Z"
    dt_str = meta["datetime"]
    # Normalise separators: "2025:02:10T06:09:20Z" → "2025-02-10T06:09:20"
    dt_normalised = dt_str.replace("Z", "")
    # Handle "2025:02:10" date format from S1Tiling
    parts = dt_normalised.split("T")
    if len(parts) == 2:
        date_part = parts[0].replace(":", "-")
        dt_normalised = f"{date_part}T{parts[1]}"

    dt_ns = np.datetime64(dt_normalised).astype("datetime64[ns]").astype(np.int64)
    r10m["time"][current_size] = dt_ns
    r10m["absolute_orbit"][current_size] = meta["absolute_orbit"]
    r10m["relative_orbit"][current_size] = meta["relative_orbit"]
    r10m["platform"][current_size] = meta["platform"]

    return current_size


# =============================================================================
# Ingestion: gamma_area conditions
# =============================================================================


def ingest_gamma_area(
    store_path: Path,
    orbit_direction: str,
    gamma_area_files: list[Path],
    meta: dict,
) -> None:
    """Write gamma_area condition arrays (time-invariant, per orbit)."""
    root = zarr.open_group(str(store_path), mode="r+", zarr_format=3)
    orbit = root[orbit_direction]

    # Create conditions group if it doesn't exist
    if "conditions" not in orbit:
        conditions = orbit.create_group("conditions")
        conditions.attrs.update(
            {
                "proj:code": meta["crs"],
                "spatial:dimensions": ["Y", "X"],
                "spatial:transform": meta["spatial_transform"],
            }
        )
    else:
        conditions = orbit["conditions"]

    for ga_path in gamma_area_files:
        # Extract orbit number from filename: GAMMA_AREA_31TCH_008.tif
        m = re.search(r"GAMMA_AREA_\w+_(\d{3})\.tif$", ga_path.name)
        if not m:
            print(f"  SKIP gamma area: {ga_path.name}")
            continue

        orbit_num = m.group(1)
        array_name = f"gamma_area_{orbit_num}"

        with rasterio.open(str(ga_path)) as src:
            data = src.read(1)

        print(
            f"    Writing {array_name}: shape={data.shape}, "
            f"dtype={data.dtype}, min={np.nanmin(data):.4f}, max={np.nanmax(data):.4f}"
        )

        h, w = data.shape
        if array_name in conditions:
            conditions[array_name][:, :] = data
        else:
            arr = conditions.create_array(
                array_name,
                shape=(h, w),
                dtype="float32",
                chunks=(min(512, h), min(512, w)),
                compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=5),
                fill_value=float("nan"),
                dimension_names=["Y", "X"],
            )
            arr[:, :] = data


# =============================================================================
# Validation
# =============================================================================


def validate_and_report(store_path: Path, orbit_direction: str) -> None:
    """Validate the store and print a comprehensive report."""
    print("\n" + "=" * 72)
    print("VALIDATION REPORT")
    print("=" * 72)

    root = zarr.open_group(str(store_path), mode="r", zarr_format=3)
    orbit = root[orbit_direction]
    attrs = dict(orbit.attrs)

    # 1. Zarr conventions
    convs = attrs.get("zarr_conventions", [])
    conv_names = {c["name"] for c in convs}
    for required in ["multiscales", "proj:", "spatial:"]:
        status = "OK" if required in conv_names else "MISSING"
        print(f"  Convention '{required}': {status}")

    # 2. proj:code
    print(f"  proj:code: {attrs.get('proj:code', 'MISSING')}")
    print(f"  spatial:dimensions: {attrs.get('spatial:dimensions', 'MISSING')}")
    print(f"  spatial:bbox: {attrs.get('spatial:bbox', 'MISSING')}")

    # 3. Multiscales layout
    layout = attrs.get("multiscales", {}).get("layout", [])
    print(f"\n  Multiscales layout ({len(layout)} levels):")
    for entry in layout:
        name = entry["asset"]
        shape = entry["spatial:shape"]
        res = entry["spatial:transform"][0]
        derived = entry.get("derived_from", "—")
        scale = entry["transform"].get("scale", [1, 1])
        print(f"    {name}: {shape[0]}×{shape[1]} @ {res}m  (from {derived}, scale {scale})")

    # 4. Array shapes and time dimension
    print("\n  Data arrays at r10m:")
    r10m = orbit["r10m"]
    for arr_name in ["vv", "vh", "border_mask"]:
        arr = r10m[arr_name]
        print(
            f"    {arr_name}: shape={arr.shape}, dtype={arr.dtype}, "
            f"dim_names={arr.metadata.dimension_names}"
        )

    # 5. Coordinate variables
    print("\n  Coordinate variables:")
    for coord in ["time", "absolute_orbit", "relative_orbit", "platform"]:
        arr = r10m[coord]
        vals = arr[:]
        print(f"    {coord}: shape={arr.shape}, dtype={arr.dtype}, values={vals}")

    # Decode time values
    time_arr = r10m["time"][:]
    if len(time_arr) > 0:
        datetimes = time_arr.astype("datetime64[ns]")
        print(f"    time (decoded): {[str(dt) for dt in datetimes]}")

    # 6. Overview consistency
    print("\n  Overview shapes:")
    for level_name, _, _ in OVERVIEW_CHAIN:
        if level_name in orbit:
            vv = orbit[level_name]["vv"]
            print(f"    {level_name}: vv.shape={vv.shape}")

    # 7. Conditions
    if "conditions" in orbit:
        conditions = orbit["conditions"]
        print("\n  Conditions:")
        cond_attrs = dict(conditions.attrs)
        print(f"    proj:code: {cond_attrs.get('proj:code', 'MISSING')}")
        for name in conditions.keys():
            item = conditions[name]
            if hasattr(item, "shape"):
                print(f"    {name}: shape={item.shape}, dtype={item.dtype}")

    # 8. xarray readback
    print("\n  xarray readback:")
    try:
        ds = xr.open_zarr(
            str(store_path / orbit_direction / "r10m"), zarr_format=3, consolidated=False
        )
        print(f"    Dataset: {dict(ds.dims)}")
        print(f"    Variables: {list(ds.data_vars)}")
        print(f"    Coords: {list(ds.coords)}")
        # Read a small sample to verify data integrity
        sample = ds["vv"].isel(time=0, Y=slice(0, 3), X=slice(0, 3)).values
        print(f"    vv[0, :3, :3] = {sample}")
        ds.close()
    except Exception as e:
        print(f"    ERROR: {e}")

    # 9. Store size on disk
    store_size = sum(f.stat().st_size for f in store_path.rglob("*") if f.is_file())
    print(f"\n  Total store size: {store_size / 1e9:.2f} GB")

    print("\n" + "=" * 72)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert real S1Tiling GeoTIFFs to GeoZarr V3 store"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Path to S1Tiling output directory (e.g. data_out/31TCH)"
    )
    parser.add_argument("--gamma-area-dir", default=None, help="Path to gamma area maps directory")
    parser.add_argument(
        "--output-dir", required=True, help="Directory where Zarr store will be created"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("S1 GRD RTC — Real GeoTIFF → GeoZarr V3 Conversion")
    print("=" * 72)

    # 1. Discover acquisitions
    print(f"\nDiscovering acquisitions in {input_dir}...")
    acquisitions = discover_acquisitions(input_dir)
    print(f"  Found {len(acquisitions)} acquisitions:")
    for acq in acquisitions:
        print(
            f"    {acq['platform']} orbit {acq['rel_orbit']} @ {acq['acq_stamp']} ({acq['orbit_dir']})"
        )

    if not acquisitions:
        print("ERROR: No acquisitions found!")
        sys.exit(1)

    # 2. Extract metadata from first VV file to initialise the store
    first_acq = acquisitions[0]
    meta = extract_geotiff_metadata(first_acq["vv"])
    print("\n  Reference metadata:")
    print(f"    CRS: {meta['crs']}")
    print(f"    Shape: {meta['shape']}")
    print(f"    Transform: {meta['spatial_transform']}")
    print(f"    Bounds: {meta['bounds']}")
    print(f"    Calibration: {meta['calibration']}")

    # Determine orbit direction
    orbit_dir_code = first_acq["orbit_dir"]
    orbit_direction = "descending" if orbit_dir_code == "DES" else "ascending"

    # Tile ID from first acquisition
    tile_id = first_acq["tile"]
    store_path = output_dir / f"s1-grd-rtc-{tile_id}.zarr"

    # 3. Create the store
    print(f"\nCreating store: {store_path}")
    total_t0 = time.time()
    create_s1_store(store_path, orbit_direction, meta)
    print("  Store structure created.")

    # 4. Ingest each acquisition
    for i, acq in enumerate(acquisitions):
        print(
            f"\n--- Ingesting acquisition {i + 1}/{len(acquisitions)}: "
            f"{acq['platform']} orbit {acq['rel_orbit']} @ {acq['acq_stamp']} ---"
        )

        acq_meta = extract_geotiff_metadata(acq["vv"])
        t0 = time.time()
        idx = ingest_acquisition(
            store_path,
            orbit_direction,
            vv_path=acq["vv"],
            vh_path=acq["vh"],
            vv_mask_path=acq["vv_mask"],
            vh_mask_path=acq["vh_mask"],
            meta=acq_meta,
        )
        acq_time = time.time() - t0
        print(f"    → Time index {idx}, acquisition total: {acq_time:.1f}s")

    # 5. Ingest gamma_area conditions
    if args.gamma_area_dir:
        gamma_dir = Path(args.gamma_area_dir).expanduser()
        gamma_files = sorted(gamma_dir.glob("GAMMA_AREA_*.tif"))
        if gamma_files:
            print(f"\n--- Ingesting {len(gamma_files)} gamma area condition(s) ---")
            ingest_gamma_area(store_path, orbit_direction, gamma_files, meta)

    total_time = time.time() - total_t0
    print(f"\n  Total conversion time: {total_time:.1f}s")

    # 6. Validate
    validate_and_report(store_path, orbit_direction)


if __name__ == "__main__":
    main()
