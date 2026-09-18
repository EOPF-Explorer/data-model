# S1Tiling EODAG 4.0 compatibility: `productType` deprecated, `collection` now required

## Title

S1 product search fails with EODAG ≥ 4.0: `productType` is deprecated, `collection` parameter required

## Description

Starting with EODAG 4.0.0, the `productType` keyword argument to `dag.search()` was replaced by `collection`. The old `productType` parameter is no longer recognized and `collection` is now mandatory. This affects S1Tiling 1.4.0 (Docker image `registry.orfeo-toolbox.org/s1-tiling/s1tiling:1.4.0-ubuntu-otb9.1.1`) which ships EODAG 4.0.0, and the bug persists on the current `develop` branch (version 2.0.0).

In `s1tiling/libs/s1/file_manager.py`, the `as_eodag_parameters()` method (line 195) builds a search dict with the key `"productType"`. This dict is unpacked via `**search_args` in the `dag.search()` call (line 443). EODAG 4.0.0 raises `ValidationError("Field required: collection")` because the deprecated `productType` kwarg is not recognized and the required `collection` parameter is missing.

> **Note:** The orbit provider (`s1tiling/libs/orbit/_providers.py`, line 199–202) already uses `collection="SENTINEL-1"` alongside `productType="S1_AUX_POEORB"` in `search_all()`, so the orbit file search is already partially adapted. Only the main S1 product search is broken.

### Verified fix

Tested inside the 1.4.0 Docker container:
```python
from eodag import EODataAccessGateway
dag = EODataAccessGateway()

# FAILS — productType no longer recognized:
dag.search(productType="S1_SAR_GRD", limit=1, box=(0, 42, 2, 44))
# → ValidationError: Field required: collection

# WORKS — collection is the new parameter name:
dag.search(collection="S1_SAR_GRD", limit=1, box=(0, 42, 2, 44))
# → 1 result from cop_dataspace
```

## Steps to Reproduce

1. Pull the 1.4.0 Docker image:
   ```bash
   docker pull registry.orfeo-toolbox.org/s1-tiling/s1tiling:1.4.0-ubuntu-otb9.1.1
   ```

2. Configure `eodag.yml` with valid `cop_dataspace` credentials.

3. Run `S1Processor` with `download: True` and any tile configuration:
   ```
   WARNING - Cannot download S1 images associated to 31TCH:
   Cannot request products for tile 31TCH on data provider: Field required: collection
   ```

## Root Cause

In `s1tiling/libs/s1/file_manager.py`, the `SearchCriteria.as_eodag_parameters()` method returns:

```python
# s1tiling/libs/s1/file_manager.py, line 193–205
res = {
    "productType"             : self.__product_type,     # ← deprecated in EODAG 4.0
    "start"                   : self.__first_date,
    "end"                     : self.__last_date,
    "sensorMode"              : self.__sensor_mode,
    "polarizationChannels"    : dag_polarization_param,
    "orbitDirection"          : dag_orbit_dir_param,
    "relativeOrbitNumber"     : dag_orbit_list_param,
    "platformSerialIdentifier": dag_platform_list_param,
}
```

This dict is then unpacked into the search call:

```python
# s1tiling/libs/s1/file_manager.py, line 443–446
page_products = dag.search(
    page=page, items_per_page=self.__searched_items_per_page,
    raise_errors=True,
    geom=footprint,
    **search_args,       # ← includes productType, sensorMode, polarizationChannels
)
```

EODAG 4.0 renamed the `productType` parameter to `collection` (the value `"S1_SAR_GRD"` stays the same).

## Suggested Fix

Two issues need fixing in `as_eodag_parameters()`:

### 1. Replace `"productType"` key with `"collection"`

In `s1tiling/libs/s1/file_manager.py`, line 195, change:
```python
"productType"             : self.__product_type,
```
to:
```python
"collection"              : self.__product_type,
```

### 2. Remove `"polarizationChannels"` and `"sensorMode"` entries

When using `cop_dataspace`, the OData v4 API rejects `polarizationChannels` and `sensorMode` as invalid fields (HTTP 400: `"Invalid field: polarizationChannels"`). These parameters are not mapped in EODAG's `cop_dataspace` provider config and are passed through raw to the OData endpoint, which rejects them.

`orbitDirection`, `relativeOrbitNumber`, and `platformSerialIdentifier` are fine — they have proper mappings in EODAG's provider config.

The `filter_eodag_search_results()` method already handles post-search filtering for polarization and platform, but `sensorMode` is currently only enforced at the search level. It should be moved to post-search filtering as well (or guarded conditionally per provider).

Remove both entries from the returned dict:
```python
res = {
    "collection"              : self.__product_type,
    "start"                   : self.__first_date,
    "end"                     : self.__last_date,
    # "sensorMode"            : self.__sensor_mode,           # removed: rejected by cop_dataspace OData
    # "polarizationChannels"  : dag_polarization_param,       # removed: rejected by cop_dataspace OData
    "orbitDirection"          : dag_orbit_dir_param,
    "relativeOrbitNumber"     : dag_orbit_list_param,
    "platformSerialIdentifier": dag_platform_list_param,
}
```

> **Note:** Removing `sensorMode` from the search parameters means the S1 query will no longer filter by acquisition mode server-side. Since S1Tiling only targets IW-mode GRD products and `productType="S1_SAR_GRD"` already constrains results to GRD, this is unlikely to cause issues in practice, but a post-search filter on `sensorMode` should be added to `filter_eodag_search_results()` for correctness.

## Workaround

Patch the file inside the Docker container before running:

```bash
docker exec -it <container> python3 -c "
import pathlib
p = pathlib.Path('/opt/S1TilingEnv/lib/python3.10/site-packages/s1tiling/libs/s1/file_manager.py')
s = p.read_text()
s = s.replace('\"productType\"', '\"collection\"', 1)
s = s.replace('\"sensorMode\"              : self.__sensor_mode,\n', '')
s = s.replace('\"polarizationChannels\"    : dag_polarization_param,\n', '')
p.write_text(s)
"
```

## Environment

- **S1Tiling:** 1.4.0 (Docker) / 2.0.0 (`develop` — bug persists)
- **EODAG:** ≥ 4.0.0
- **Docker image:** `registry.orfeo-toolbox.org/s1-tiling/s1tiling:1.4.0-ubuntu-otb9.1.1`
- **Provider:** `cop_dataspace` (Copernicus Data Space Ecosystem)

## Labels

`bug`, `eodag`
