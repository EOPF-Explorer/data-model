# Sentinel-1 Pydantic Models

This document demonstrates the basic usage of models for Sentinel-1 L2 data products. These models only describe the Zarr hierarchy; they do not support reading and writing array values.

## Sentinel-1A

### Basic Usage

#### Loading a Sentinel-1A Product

To load the schema for a Sentinel-1A product from a Zarr store:

```python
from eopf_geozarr.data_api.s1 import Sentinel1Root
import zarr

# Open the Zarr group
group = zarr.open_group("https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:notebook-data/tutorial_data/cpm_v260/S1A_IW_GRDH_1SDV_20241124T180254_20241124T180319_056700_06F516_BA27.zarr", mode="r")

# Create a Sentinel1Root model instance from the group
model = Sentinel1Root.from_zarr(group)

print(model.members.keys())
#> dict_keys(['S01SIWGRD_20241124T180254_0025_A324_BA27_06F516_VH', 'S01SIWGRD_20241124T180254_0025_A324_BA27_06F516_VV'])

print(model.members['S01SIWGRD_20241124T180254_0025_A324_BA27_06F516_VH'].members.keys())
#> dict_keys(['conditions', 'measurements', 'quality'])
```

## Sentinel-1C

### Basic Usage

#### Loading a Sentinel-1C Product

To load the schema for Sentinel-1C product from a Zarr store:

```python
from eopf_geozarr.data_api.s1 import Sentinel1Root
import zarr

# Open the Zarr group
group = zarr.open_group("https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202509-s01siwgrh/12/products/cpm_v256/S1C_IW_GRDH_1SDV_20250912T053648_20250912T053713_004087_0081FD_5AA4.zarr", mode="r")

# Create a Sentinel1Root model instance from the group
model = Sentinel1Root.from_zarr(group)

print(model.members.keys())
#> dict_keys(['S01SIWGRD_20250912T053648_0025_C023_5AA4_0081FD_VH', 'S01SIWGRD_20250912T053648_0025_C023_5AA4_0081FD_VV'])

print(model.members['S01SIWGRD_20250912T053648_0025_C023_5AA4_0081FD_VH'].members.keys())
#> dict_keys(['conditions', 'measurements', 'quality'])
```

