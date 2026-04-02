"""S1 GRD RTC GeoTIFF → GeoZarr V3 ingestion pipeline.

Converts S1Tiling γ0T RTC GeoTIFF outputs into a sharded Zarr V3 store
with multiscale overviews, spatial coordinate arrays, and full GeoZarr
convention metadata.

Public API:
    - extract_geotiff_metadata(path) -> S1TilingMetadata
    - ingest_s1tiling_acquisition(vv_path, vh_path, border_mask_path, store_path, orbit_direction) -> int
    - consolidate_s1_store(store_path, orbit_direction) -> None
    - discover_s1tiling_acquisitions(input_dir) -> list[dict]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import numpy as np
import rasterio
import structlog
import zarr
import zarr.codecs
from zarr_cm import geo_proj, multiscales as multiscales_cm, spatial as spatial_cm

from eopf_geozarr.conversion.utils import calculate_aligned_chunk_size, downsample_2d_array

log = structlog.get_logger()

# =============================================================================
# Constants
# =============================================================================

MULTISCALES_UUID = multiscales_cm.UUID
GEO_PROJ_UUID = geo_proj.UUID
SPATIAL_UUID = spatial_cm.UUID

ZARR_CONVENTIONS = [multiscales_cm.CMO, geo_proj.CMO, spatial_cm.CMO]

# Overview chain: (level_name, parent_name, downsample_factor)
OVERVIEW_CHAIN = [
    ("r10m", None, 1),
    ("r20m", "r10m", 2),
    ("r60m", "r20m", 3),
    ("r120m", "r60m", 2),
    ("r360m", "r120m", 3),
    ("r720m", "r360m", 2),
]

# S1Tiling filename pattern
# e.g. s1a_32TQM_vv_ASC_037_20230115t061234_GammaNaughtRTC.tif
S1TILING_FILENAME_PATTERN = re.compile(
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
# Data Transfer Object
# =============================================================================


@dataclass(frozen=True)
class S1TilingMetadata:
    """Metadata extracted from an S1Tiling GeoTIFF."""

    crs: str
    spatial_transform: list[float]
    shape: list[int]
    bounds: list[float]
    datetime: str
    absolute_orbit: int
    relative_orbit: int
    platform: str
    calibration: str
    input_s1_images: str


# =============================================================================
# Metadata Extraction
# =============================================================================


def _normalise_s1tiling_datetime(dt_str: str) -> str:
    """Normalise S1Tiling datetime format to ISO 8601.

    Input:  "2025:02:10T06:09:20Z" (S1Tiling uses colons in date part)
    Output: "2025-02-10T06:09:20"
    """
    dt_normalised = dt_str.replace("Z", "")
    parts = dt_normalised.split("T")
    if len(parts) == 2:
        date_part = parts[0].replace(":", "-")
        dt_normalised = f"{date_part}T{parts[1]}"
    return dt_normalised


def extract_geotiff_metadata(path: str | Path) -> S1TilingMetadata:
    """Extract CRS, transform, bounds, and custom tags from an S1Tiling GeoTIFF.

    Raises
    ------
    ValueError
        If critical tags (ACQUISITION_DATETIME, ORBIT_NUMBER,
        RELATIVE_ORBIT_NUMBER, FLYING_UNIT_CODE) are missing.
    """
    with rasterio.open(str(path)) as src:
        tags = src.tags()
        t = src.transform
        spatial_transform = [t.a, t.b, t.c, t.d, t.e, t.f]

        # Validate critical tags
        required_tags = [
            "ACQUISITION_DATETIME",
            "ORBIT_NUMBER",
            "RELATIVE_ORBIT_NUMBER",
            "FLYING_UNIT_CODE",
        ]
        missing = [tag for tag in required_tags if tag not in tags]
        if missing:
            raise ValueError(f"GeoTIFF {path} missing required tags: {missing}")

        dt_raw = tags["ACQUISITION_DATETIME"]
        dt_normalised = _normalise_s1tiling_datetime(dt_raw)

        metadata = S1TilingMetadata(
            crs=str(src.crs),
            spatial_transform=spatial_transform,
            shape=[src.height, src.width],
            bounds=[src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top],
            datetime=dt_normalised,
            absolute_orbit=int(tags["ORBIT_NUMBER"]),
            relative_orbit=int(tags["RELATIVE_ORBIT_NUMBER"]),
            platform=tags["FLYING_UNIT_CODE"],
            calibration=tags.get("CALIBRATION", ""),
            input_s1_images=tags.get("INPUT_S1_IMAGES", ""),
        )

    log.info(
        "Extracted GeoTIFF metadata",
        path=str(path),
        crs=metadata.crs,
        shape=metadata.shape,
        datetime=metadata.datetime,
    )
    return metadata


def parse_s1tiling_filename(filename: str) -> dict | None:
    """Parse an S1Tiling filename into component fields.

    Returns None if the filename does not match the expected pattern.
    """
    m = S1TILING_FILENAME_PATTERN.match(filename)
    if not m:
        return None
    return {
        "platform": m.group("platform"),
        "tile": m.group("tile"),
        "pol": m.group("pol"),
        "orbit_dir": m.group("orbit_dir"),
        "rel_orbit": m.group("rel_orbit"),
        "acq_stamp": m.group("acq_stamp"),
        "is_mask": m.group("mask") is not None,
    }


# =============================================================================
# Multiscales Layout
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
                ceil(current_shape[0] / factor),
                ceil(current_shape[1] / factor),
            ]
            current_transform = [
                current_transform[0] * factor,  # a: pixel width
                current_transform[1],  # b: rotation (0)
                current_transform[2],  # c: x origin
                current_transform[3],  # d: rotation (0)
                current_transform[4] * factor,  # e: pixel height (negative)
                current_transform[5],  # f: y origin
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
# Store Creation
# =============================================================================


def _create_spatial_coordinate_arrays(
    level_group: zarr.Group,
    level_h: int,
    level_w: int,
    level_transform: list[float],
) -> None:
    """Create 1D x and y spatial coordinate arrays at a resolution level."""
    pixel_w = level_transform[0]  # a: pixel width
    x_origin = level_transform[2]  # c: x origin (left edge)
    pixel_h = level_transform[4]  # e: pixel height (negative)
    y_origin = level_transform[5]  # f: y origin (top edge)

    # Pixel-center convention: offset by half a pixel from the edge origin
    x_coords = x_origin + (np.arange(level_w, dtype="float64") + 0.5) * pixel_w
    y_coords = y_origin + (np.arange(level_h, dtype="float64") + 0.5) * pixel_h

    x_arr = level_group.create_array(
        "x",
        data=x_coords,
        chunks=(level_w,),
        fill_value=float("nan"),
        dimension_names=["x"],
    )
    x_arr.attrs.update(
        {
            "units": "m",
            "long_name": "x coordinate of projection",
            "standard_name": "projection_x_coordinate",
            "_ARRAY_DIMENSIONS": ["x"],
        }
    )

    y_arr = level_group.create_array(
        "y",
        data=y_coords,
        chunks=(level_h,),
        fill_value=float("nan"),
        dimension_names=["y"],
    )
    y_arr.attrs.update(
        {
            "units": "m",
            "long_name": "y coordinate of projection",
            "standard_name": "projection_y_coordinate",
            "_ARRAY_DIMENSIONS": ["y"],
        }
    )


def create_s1_store(
    store_path: str | Path,
    orbit_direction: str,
    metadata: S1TilingMetadata,
) -> zarr.Group:
    """Create a new S1 GRD RTC Zarr V3 store with full conventions metadata.

    Returns the root group.
    """
    layout = compute_multiscales_layout(metadata.shape, metadata.spatial_transform)

    root = zarr.open_group(str(store_path), mode="w-", zarr_format=3)
    orbit_group = root.create_group(orbit_direction)

    orbit_group.attrs.update(
        {
            "zarr_conventions": ZARR_CONVENTIONS,
            "multiscales": {
                "layout": layout,
                "resampling_method": "average",
            },
            "proj:code": metadata.crs,
            "spatial:dimensions": ["y", "x"],
            "spatial:bbox": metadata.bounds,
        }
    )

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

        inner_chunks = (
            1,
            calculate_aligned_chunk_size(level_h, 512),
            calculate_aligned_chunk_size(level_w, 512),
        )
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
                dimension_names=["time", "y", "x"],
            )

        _create_spatial_coordinate_arrays(
            level_group, level_h, level_w, level_entry["spatial:transform"]
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

    log.info(
        "Created S1 store",
        store_path=str(store_path),
        orbit_direction=orbit_direction,
        crs=metadata.crs,
        native_shape=metadata.shape,
    )
    return root


# =============================================================================
# Acquisition Ingestion
# =============================================================================


def ingest_s1tiling_acquisition(
    vv_path: str | Path,
    vh_path: str | Path,
    border_mask_path: str | Path,
    store_path: str | Path,
    orbit_direction: str,
) -> int:
    """Ingest one S1Tiling acquisition into a GeoZarr V3 store.

    Creates the store if it does not exist, or appends to an existing store.
    Returns the time index of the ingested acquisition.

    Parameters
    ----------
    vv_path : str or Path
        Path to the VV polarisation GeoTIFF.
    vh_path : str or Path
        Path to the VH polarisation GeoTIFF.
    border_mask_path : str or Path
        Path to the VV border mask GeoTIFF.
    store_path : str or Path
        Path to the output Zarr V3 store.
    orbit_direction : str
        Orbit direction group name (e.g. "ascending", "descending").

    Returns
    -------
    int
        The time index (0-based) of the newly ingested acquisition.

    Raises
    ------
    FileNotFoundError
        If any of the input GeoTIFF paths do not exist.
    ValueError
        If the GeoTIFF CRS or shape does not match the existing store.
    """
    vv_path = Path(vv_path)
    vh_path = Path(vh_path)
    border_mask_path = Path(border_mask_path)
    store_path = Path(store_path)

    for p in [vv_path, vh_path, border_mask_path]:
        if not p.exists():
            raise FileNotFoundError(f"GeoTIFF not found: {p}")

    # Extract metadata from VV file
    meta = extract_geotiff_metadata(vv_path)

    log.info(
        "Ingesting S1 acquisition",
        vv_path=str(vv_path),
        orbit_direction=orbit_direction,
    )

    # Create-or-open store
    if not store_path.exists():
        root = create_s1_store(store_path, orbit_direction, meta)
    else:
        root = zarr.open_group(str(store_path), mode="r+", zarr_format=3)
        if orbit_direction not in root:
            # Create orbit direction group in existing store
            orbit_group = root.create_group(orbit_direction)
            layout = compute_multiscales_layout(meta.shape, meta.spatial_transform)
            orbit_group.attrs.update(
                {
                    "zarr_conventions": ZARR_CONVENTIONS,
                    "multiscales": {
                        "layout": layout,
                        "resampling_method": "average",
                    },
                    "proj:code": meta.crs,
                    "spatial:dimensions": ["y", "x"],
                    "spatial:bbox": meta.bounds,
                }
            )
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
                inner_chunks = (
                    1,
                    calculate_aligned_chunk_size(level_h, 512),
                    calculate_aligned_chunk_size(level_w, 512),
                )
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
                        dimension_names=["time", "y", "x"],
                    )
                _create_spatial_coordinate_arrays(
                    level_group, level_h, level_w, level_entry["spatial:transform"]
                )
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
        else:
            # Validate consistency on append
            orbit_group = root[orbit_direction]
            store_crs = dict(orbit_group.attrs).get("proj:code")
            if store_crs != meta.crs:
                raise ValueError(
                    f"CRS mismatch: store has {store_crs}, GeoTIFF has {meta.crs}"
                )
            store_layout = dict(orbit_group.attrs).get("multiscales", {}).get("layout", [])
            if store_layout:
                native_entry = store_layout[0]
                store_shape = native_entry.get("spatial:shape")
                if store_shape != meta.shape:
                    raise ValueError(
                        f"Shape mismatch: store has {store_shape}, GeoTIFF has {meta.shape}"
                    )

    orbit = root[orbit_direction]

    # Read GeoTIFF pixel data
    with rasterio.open(str(vv_path)) as src:
        vv_data = src.read(1)
    with rasterio.open(str(vh_path)) as src:
        vh_data = src.read(1)
    with rasterio.open(str(border_mask_path)) as src:
        mask_data = src.read(1).astype(np.uint8)

    log.info(
        "GeoTIFF read complete",
        vv_min=float(np.nanmin(vv_data)),
        vv_max=float(np.nanmax(vv_data)),
    )

    # Determine time index
    r10m = orbit["r10m"]
    current_size = r10m["vv"].shape[0]
    new_size = current_size + 1

    # Generate overviews
    data_by_level: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "r10m": (vv_data, vh_data, mask_data)
    }
    prev_vv, prev_vh, prev_mask = vv_data, vh_data, mask_data
    for level_name, _, factor in OVERVIEW_CHAIN[1:]:
        h, w = prev_vv.shape
        target_h, target_w = ceil(h / factor), ceil(w / factor)
        prev_vv = downsample_2d_array(prev_vv, target_h, target_w, nodata_value=float("nan"))
        prev_vh = downsample_2d_array(prev_vh, target_h, target_w, nodata_value=float("nan"))
        prev_mask = downsample_2d_array(prev_mask, target_h, target_w, method="nearest")
        data_by_level[level_name] = (prev_vv, prev_vh, prev_mask)

    log.info("Overviews generated", levels=len(data_by_level))

    # Write data at all levels
    for level_name, (vv_lev, vh_lev, mask_lev) in data_by_level.items():
        level = orbit[level_name]
        h, w = vv_lev.shape

        level["vv"].resize((new_size, h, w))
        level["vh"].resize((new_size, h, w))
        level["border_mask"].resize((new_size, h, w))

        level["vv"][current_size, :, :] = vv_lev
        level["vh"][current_size, :, :] = vh_lev
        level["border_mask"][current_size, :, :] = mask_lev

    # Append coordinate variables
    for coord_name in ["time", "absolute_orbit", "relative_orbit", "platform"]:
        r10m[coord_name].resize((new_size,))

    dt_ns = np.datetime64(meta.datetime).astype("datetime64[ns]").astype(np.int64)
    r10m["time"][current_size] = dt_ns
    r10m["absolute_orbit"][current_size] = meta.absolute_orbit
    r10m["relative_orbit"][current_size] = meta.relative_orbit
    r10m["platform"][current_size] = meta.platform

    log.info(
        "Zarr write complete",
        time_index=current_size,
        levels_written=len(data_by_level),
    )
    return current_size


# =============================================================================
# Consolidation
# =============================================================================


def consolidate_s1_store(store_path: str | Path, orbit_direction: str) -> None:
    """Consolidate metadata at orbit direction and root levels.

    Must be called AFTER all ingestions complete — consolidated metadata
    caches array shapes and will become stale if called mid-ingestion.
    """
    zarr.consolidate_metadata(str(store_path), path=orbit_direction, zarr_format=3)
    zarr.consolidate_metadata(str(store_path), zarr_format=3)
    log.info(
        "Metadata consolidated",
        store_path=str(store_path),
        orbit_direction=orbit_direction,
    )


# =============================================================================
# File Discovery
# =============================================================================


def discover_s1tiling_acquisitions(input_dir: str | Path) -> list[dict]:
    """Discover and group S1Tiling GeoTIFF files into acquisition bundles.

    Returns a list of dicts, each with keys:
        platform, tile, orbit_dir, rel_orbit, acq_stamp, vv, vh, vv_mask, vh_mask

    Logs warnings for incomplete acquisitions (missing polarisation or mask files).
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.tif"))
    groups: dict[tuple, dict] = {}

    for f in files:
        parsed = parse_s1tiling_filename(f.name)
        if parsed is None:
            continue

        key = (
            parsed["platform"],
            parsed["tile"],
            parsed["orbit_dir"],
            parsed["rel_orbit"],
            parsed["acq_stamp"],
        )

        if key not in groups:
            groups[key] = {
                "platform": parsed["platform"],
                "tile": parsed["tile"],
                "orbit_dir": parsed["orbit_dir"],
                "rel_orbit": parsed["rel_orbit"],
                "acq_stamp": parsed["acq_stamp"],
            }

        pol = parsed["pol"]
        is_mask = parsed["is_mask"]

        if is_mask:
            groups[key][f"{pol}_mask"] = f
        else:
            groups[key][pol] = f

    acquisitions = []
    for key, acq in sorted(groups.items()):
        missing = [k for k in ("vv", "vh", "vv_mask", "vh_mask") if k not in acq]
        if missing:
            log.warning(
                "Incomplete acquisition",
                key=key,
                missing=missing,
            )
        acquisitions.append(acq)

    log.info("Discovered acquisitions", count=len(acquisitions), input_dir=str(input_dir))
    return acquisitions
