import warnings

# Filter noisy Zarr v3 consolidated metadata warning in tests
warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part in the Zarr format 3 specification",
    category=UserWarning,
)

# Future-proof xarray dims warning until upstream change lands
warnings.filterwarnings(
    "ignore",
    message="The return type of `Dataset.dims` will be changed",
    category=FutureWarning,
)
