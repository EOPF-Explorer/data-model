import xarray
import rioxarray
import obstore
import zarr
from zarr.storage import ObjectStore

import os

os.environ["AWS_ACCESS_KEY_ID"] = "f175f89808a84ba9a443aef60c65c4cc"
os.environ["AWS_SECRET_ACCESS_KEY"] = "cb759c4151814d0f998be784a904d42c"
os.environ["AWS_DEFAULT_REGION"] = "de"
os.environ["AWS_ENDPOINT_URL"] = "https://s3.de.io.cloud.ovh.net/"

src_path = "s3://esa-zarr-sentinel-explorer-fra/tests-output/eopf_geozarr/S2A_MSIL1C_20250725T091041_N0511_R050_T34SGE_20250725T101000.zarr"

store = obstore.store.from_url(src_path)
zarr_store = ObjectStore(store=store)
ds = xarray.open_datatree(zarr_store, decode_times=True, decode_coords="all", consolidated=True, engine="zarr")