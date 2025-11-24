# Sentinel-2 Pydantic Models

This document demonstrates the basic usage of models for Sentinel-2 L2 data products. These models only describe the Zarr hierarchy; they do not support reading and writing array values.

## Sentinel-2A

### Basic Usage

#### Loading a Sentinel-2A Product

To load the schema for Sentinel-2A product from a Zarr store:

```python
from eopf_geozarr.data_api.s2 import Sentinel2Root
import zarr

# Open the Zarr group from remote S3 or local path
store = zarr.open_group(
    "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202510-s02msil2a-eu/25/products/cpm_v256/S2A_MSIL2A_20251025T095111_N0511_R079_T33SVB_20251025T121315.zarr",
    mode="r"
)

# Create a Sentinel2Root model instance from the group
model = Sentinel2Root.from_zarr(store)

print(model.members.keys())
#> dict_keys(['measurements', 'quality', 'conditions'])

print(model.conditions.members)
#> {'geometry': Sentinel2GeometryGroup(members={angle, band, detector, sun_angles, viewing_incidence_angles, +4 more}), 'mask': Sentinel2ConditionsMaskGroup(members={detector_footprint, l1c_classification, l2a_classification}), 'meteorology': Sentinel2MeteorologyGroup(members={cams, ecmwf})}
```

## Sentinel-2B

### Basic Usage

#### Loading a Sentinel-2B Product

To load the schema for a Sentinel-2B product from a Zarr store:

```python
from eopf_geozarr.data_api.s2 import Sentinel2Root
import zarr

# Open the Zarr group from remote S3 or local path
store = zarr.open_group(
    "https://objectstore.eodc.eu:2222/e05ab01a9d56408d82ac32d69a5aae2a:sample-data/tutorial_data/cpm_v253/S2B_MSIL1C_20250113T103309_N0511_R108_T32TLQ_20250113T122458.zarr",
    mode="r"
)

# Create a Sentinel2Root model instance from the group
model = Sentinel2Root.from_zarr(store)

print(model.members.keys())
#> dict_keys(['measurements', 'quality', 'conditions'])

print(model.conditions.members)
#> {'geometry': Sentinel2GeometryGroup(members={angle, band, detector, sun_angles, viewing_incidence_angles, +4 more}), 'mask': Sentinel2ConditionsMaskGroup(members={detector_footprint, l1c_classification}), 'meteorology': Sentinel2MeteorologyGroup(members={cams, ecmwf})}
```

## Sentinel-2C

### Basic Usage

#### Loading a Sentinel-2C Product

To load the schema for a Sentinel-2C product from a Zarr store:

```python
from eopf_geozarr.data_api.s2 import Sentinel2Root
import zarr

# Open the Zarr group from remote S3 or local path
store = zarr.open_group(
    "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202508-s02msil2a/11/products/cpm_v256/S2C_MSIL2A_20250811T112131_N0511_R037_T29TPF_20250811T152216.zarr",
    mode="r"
)

# Create a Sentinel2Root model instance from the group
model = Sentinel2Root.from_zarr(store)
print(model.members.keys())
#> dict_keys(['measurements', 'quality', 'conditions'])

print(model.conditions.members)
#> {'geometry': Sentinel2GeometryGroup(members={angle, band, detector, sun_angles, viewing_incidence_angles, +4 more}), 'mask': Sentinel2ConditionsMaskGroup(members={detector_footprint, l1c_classification, l2a_classification}), 'meteorology': Sentinel2MeteorologyGroup(members={cams, ecmwf})}
```
