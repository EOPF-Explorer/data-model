import sys
from typing import Any, Optional


def setup_dask_cluster(enable_dask: bool, verbose: bool = False) -> Optional[Any]:
    if not enable_dask:
        return None
    try:
        from dask.distributed import Client

        client = Client()
        if verbose:
            print(f"Dask cluster started: {client}")
            print(f"   Dashboard: {client.dashboard_link}")
            print(f"   Workers: {len(client.scheduler_info()['workers'])}")
        else:
            print("Dask cluster started for parallel processing")
        return client
    except ImportError:
        print(
            "Error: dask.distributed not available. Install with: pip install 'dask[distributed]'"
        )
        sys.exit(1)
    except Exception as e:  # pragma: no cover
        print(f"Error starting dask cluster: {e}")
        sys.exit(1)
