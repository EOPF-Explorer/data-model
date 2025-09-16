import logging
import os
import shutil
import time
from typing import Optional

from . import fs_utils

log = logging.getLogger(__name__)


def safe_rmtree(path: str, retries: int = 3, delay: float = 0.2) -> None:
    last_err: Optional[Exception] = None
    for _ in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            return
        except Exception as e:  # pragma: no cover
            last_err = e
            time.sleep(delay)
    if last_err:
        raise last_err


def ensure_group_hierarchy(output_path: str, group_name: str) -> None:
    try:
        base = fs_utils.normalize_path(f"{output_path}/{group_name.lstrip('/')}")
        lvl0 = fs_utils.normalize_path(f"{base}/0")
        zpath = fs_utils.normalize_path(f"{base}/zarr.json")
        zpath0 = fs_utils.normalize_path(f"{lvl0}/zarr.json")
        if not fs_utils.path_exists(zpath):
            log.debug(f"Ensuring base group: {zpath}")
            fs_utils.write_json_metadata(
                zpath, {"zarr_format": 3, "node_type": "group", "attributes": {}}
            )
        if not fs_utils.path_exists(zpath0):
            log.debug(f"Ensuring level-0 group: {zpath0}")
            fs_utils.write_json_metadata(
                zpath0, {"zarr_format": 3, "node_type": "group", "attributes": {}}
            )
    except Exception as e:
        log.warning(f"Could not ensure group hierarchy for {group_name}: {e}")
