import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import fsspec

from ..conversion.fs_utils import get_storage_options, is_s3_path


def _bbox_to_geometry(bbox: List[float]) -> Dict[str, Any]:
    minx, miny, maxx, maxy = bbox
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny],
            ]
        ],
    }


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    out_path = path + "stac.json" if path.endswith("/") else path
    if not is_s3_path(out_path) and not out_path.startswith(("http://", "https://")):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    storage_options = get_storage_options(out_path) or {}
    open_opts = {"compression": None}
    open_opts.update(storage_options)
    with fsspec.open(out_path, "w", **open_opts) as f:
        f.write(json.dumps(obj, indent=2))
    print(f"STAC written to: {out_path}")


def stac_command(args: argparse.Namespace) -> None:
    ds_id = args.id or Path(args.input_path).stem
    link_href = args.input_path
    now_iso = (
        datetime.now(tz=__import__("datetime").timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    bbox: Optional[List[float]] = None
    if args.bbox and len(args.bbox) == 4:
        bbox = [float(x) for x in args.bbox]
    if args.type == "item":
        item: Dict[str, Any] = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": ds_id,
            "properties": {"datetime": args.datetime or now_iso},
            "links": [],
            "assets": {
                "geozarr": {
                    "href": link_href,
                    "type": "application/vnd+zarr",
                    "roles": ["data"],
                    "title": args.title or "GeoZarr dataset",
                }
            },
        }
        if bbox:
            item["bbox"] = bbox
            item["geometry"] = _bbox_to_geometry(bbox)
        if args.collection_id:
            item["collection"] = args.collection_id
        _write_json(args.output, item)
    else:
        if not bbox:
            print("Warning: Collection without bbox provided; extent will be minimal placeholder.")
            bbox = [0.0, 0.0, 0.0, 0.0]
        collection: Dict[str, Any] = {
            "type": "Collection",
            "stac_version": "1.0.0",
            "id": ds_id,
            "description": args.title or ds_id,
            "license": args.license or "proprietary",
            "extent": {
                "spatial": {"bbox": [bbox]},
                "temporal": {
                    "interval": [[args.start_datetime or now_iso, args.end_datetime or None]]
                },
            },
            "links": [],
        }
        _write_json(args.output, collection)
