"""Regenerate the S1 GRDH example JSON from a real ``.SAFE`` product.

Run from the repository root, in an environment with the ``cpm`` extra::

    uv run --extra cpm python tests/_test_data/s1_grdh_examples/regenerate.py \\
        /path/to/S1A_IW_GRDH_1SDV_....SAFE [more .SAFE ...]
        #home/samuel/data/samples/cpm_v300rc4a/safe_products

Each product is converted with CPM's own ``cpm_zarr`` engine (the layout the CPM
SAFE reader hands to writers), and the resulting Zarr V3 hierarchy is dumped with the
generic ``GroupSpec``. Only metadata reaches the JSON, and
``create_zarrv3_group_from_json`` rebuilds it as a store of fill-valued arrays.
"""

# eopf is an optional dependency (the `cpm` extra, Python >= 3.13 only), so it
# is absent from the default type-checking environment.
# pyright: reportMissingImports=false

import json
import sys
import tempfile
from pathlib import Path

import zarr
from eopf.store.convert import convert
from pydantic_zarr.v3 import GroupSpec

OUTPUT_DIR = Path(__file__).parent


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
    output.write_text(json.dumps(spec.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    return output


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(__doc__)
    for arg in argv:
        print(f"Wrote {safe_to_json(Path(arg))}")


if __name__ == "__main__":
    main(sys.argv[1:])
