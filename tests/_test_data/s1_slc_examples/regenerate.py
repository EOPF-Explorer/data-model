"""Regenerate the S1 SLC example JSON from a real ``.SAFE`` product.

Run from the repository root, in an environment with the ``cpm`` extra::

    uv run --extra cpm python tests/_test_data/s1_slc_examples/regenerate.py \\
        /path/to/S1C_IW_SLC__1SDV_....SAFE [more .SAFE ...]

Each product is converted with CPM's own ``cpm_zarr`` engine (the layout the CPM
SAFE reader hands to writers), and the resulting Zarr V3 hierarchy is dumped with the
generic ``GroupSpec``. Only metadata reaches the JSON, and
``create_zarrv3_group_from_json`` rebuilds it as a store of fill-valued arrays.

An IW SLC product holds one subtree per burst (13 here), all with the same layout, so
only the first burst of the first ``KEEP_SWATHS`` swaths is kept. That keeps the
fixture small while still giving the converter more than one burst subtree. The root
``stac_discovery`` attributes are left untouched, so they still list every burst.
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import zarr
from eopf.store.convert import convert
from pydantic_zarr.v3 import GroupSpec

OUTPUT_DIR = Path(__file__).parent

#: Number of swaths (IW1, IW2, ...) to keep one burst subtree from.
KEEP_SWATHS = 2


def trim_to_one_burst_per_swath(spec: dict[str, Any], keep_swaths: int = KEEP_SWATHS) -> None:
    """Keep only the first burst subtree of the first ``keep_swaths`` swaths, in place.

    Burst subtrees are named ``S01SIWSLC_<...>_<swath>_<burst id>``.
    """
    first_burst_per_swath: dict[str, str] = {}
    for name in sorted(spec["members"]):
        swath = name.rsplit("_", 2)[-2]
        first_burst_per_swath.setdefault(swath, name)
    keep = {first_burst_per_swath[swath] for swath in sorted(first_burst_per_swath)[:keep_swaths]}
    spec["members"] = {name: member for name, member in spec["members"].items() if name in keep}


def safe_to_json(safe_path: Path, output_dir: Path = OUTPUT_DIR) -> Path:
    """Convert one ``.SAFE`` to a CPM Zarr store and write its GroupSpec JSON."""
    output = output_dir / f"{safe_path.stem}.json"
    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / f"{safe_path.stem}.zarr"
        convert(
            str(safe_path),
            str(store_path),
            source_store_kwargs={"engine": "safe", "mode": "r"},
            target_store_kwargs={"engine": "cpm_zarr", "mode": "w"},
        )
        spec = GroupSpec.from_zarr(zarr.open_group(str(store_path), mode="r"))
    # mode="json" makes the dump JSON-safe (NaN fill values -> "NaN", tuples -> lists).
    data = spec.model_dump(mode="json")
    trim_to_one_burst_per_swath(data)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return output


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(__doc__)
    for arg in argv:
        print(f"Wrote {safe_to_json(Path(arg))}")


if __name__ == "__main__":
    main(sys.argv[1:])
