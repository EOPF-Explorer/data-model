#! /usr/bin/env bash

# Convert to GeoZarr stored on the local file system

eopf-geozarr convert \
https://objectstore.eodc.eu:2222/e05ab01a9d56408d82ac32d69a5aae2a:sample-data/tutorial_data/cpm_v253/S2B_MSIL1C_20250113T103309_N0511_R108_T32TLQ_20250113T122458.zarr \
./tests-output/eopf_geozarr/s2b_test.zarr \
--groups /measurements/reflectance/r10m /measurements/reflectance/r20m /measurements/reflectance/r60m /quality/l1c_quicklook/r10m \
--crs-groups /conditions/geometry \
--spatial-chunk 4096 \
--min-dimension 256 \
--tile-width 256 \
--max-retries 2 \
--verbose
