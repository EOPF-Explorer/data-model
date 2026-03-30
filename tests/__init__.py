"""Tests for the eopf-geozarr package."""

import pathlib

import xarray as xr


def _verify_basic_structure(output_path: pathlib.Path, groups: list[str]) -> None:
    """Verify the basic Zarr store structure."""
    print("Verifying basic structure...")

    # Check that the main zarr store exists
    assert (output_path / "zarr.json").exists()

    # Check that each group has been created
    for group in groups:
        group_path = output_path / group.lstrip("/")
        assert group_path.exists(), f"Group {group} not found"
        assert (group_path / "zarr.json").exists(), f"Group {group} missing zarr.json"

        # Check that level 0 (native resolution) exists
        level_0_path = group_path / "0"
        assert level_0_path.exists(), f"Level 0 not found for {group}"
        assert (level_0_path / "zarr.json").exists(), f"Level 0 missing zarr.json for {group}"


def _verify_geozarr_spec_compliance(output_path: pathlib.Path, group: str) -> None:
    """
    Verify GeoZarr specification compliance following the notebook verification.

    This replicates the compliance checks from the notebook:
    - _ARRAY_DIMENSIONS attributes on all arrays
    - CF standard names properly set
    - Grid mapping attributes reference correct CRS variables
    - GeoTransform attributes in grid_mapping variables
    - Native CRS preservation
    """
    print(f"Verifying GeoZarr-spec compliance for {group}...")

    # Open the native resolution dataset (level 0)
    group_path = str(output_path / group.lstrip("/") / "0")
    ds = xr.open_dataset(group_path, engine="zarr", zarr_format=3)

    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Coordinates: {list(ds.coords)}")

    # Check 1: _ARRAY_DIMENSIONS attributes (required by GeoZarr spec)
    for var_name in ds.data_vars:
        if var_name != "spatial_ref":  # Skip grid_mapping variable
            assert "_ARRAY_DIMENSIONS" in ds[var_name].attrs, (
                f"Missing _ARRAY_DIMENSIONS for {var_name} in {group}"
            )
            assert ds[var_name].attrs["_ARRAY_DIMENSIONS"] == list(ds[var_name].dims), (
                f"Incorrect _ARRAY_DIMENSIONS for {var_name} in {group}"
            )
            print(f"    ✅ _ARRAY_DIMENSIONS: {ds[var_name].attrs['_ARRAY_DIMENSIONS']}")

    # Check coordinates
    for coord_name in ds.coords:
        if coord_name not in ["spatial_ref"]:  # Skip CRS coordinate
            assert "_ARRAY_DIMENSIONS" in ds[coord_name].attrs, (
                f"Missing _ARRAY_DIMENSIONS for coordinate {coord_name} in {group}"
            )
            print(
                f"    ✅ {coord_name} _ARRAY_DIMENSIONS: {ds[coord_name].attrs['_ARRAY_DIMENSIONS']}"
            )

    # Check 2: CF standard names (required by GeoZarr spec)
    for var_name in ds.data_vars:
        if var_name != "spatial_ref":
            assert "standard_name" in ds[var_name].attrs, (
                f"Missing standard_name for {var_name} in {group}"
            )
            assert ds[var_name].attrs["standard_name"] == "toa_bidirectional_reflectance", (
                f"Incorrect standard_name for {var_name} in {group}"
            )
            print(f"    ✅ standard_name: {ds[var_name].attrs['standard_name']}")

    # Check 3: Grid mapping attributes (required by GeoZarr spec)
    for var_name in ds.data_vars:
        if var_name != "spatial_ref":
            assert "grid_mapping" in ds[var_name].attrs, (
                f"Missing grid_mapping for {var_name} in {group}"
            )
            assert ds[var_name].attrs["grid_mapping"] == "spatial_ref", (
                f"Incorrect grid_mapping for {var_name} in {group}"
            )
            print(f"    ✅ grid_mapping: {ds[var_name].attrs['grid_mapping']}")

    # Check 4: Spatial reference variable (as in notebook)
    assert "spatial_ref" in ds, f"Missing spatial_ref variable in {group}"
    assert "_ARRAY_DIMENSIONS" in ds["spatial_ref"].attrs, (
        f"Missing _ARRAY_DIMENSIONS for spatial_ref in {group}"
    )
    assert ds["spatial_ref"].attrs["_ARRAY_DIMENSIONS"] == [], (
        f"Incorrect _ARRAY_DIMENSIONS for spatial_ref in {group}"
    )
    print(f"    ✅ spatial_ref _ARRAY_DIMENSIONS: {ds['spatial_ref'].attrs['_ARRAY_DIMENSIONS']}")

    # Check 5: GeoTransform attribute (from notebook verification)
    if "GeoTransform" in ds["spatial_ref"].attrs:
        print(f"    ✅ GeoTransform: {ds['spatial_ref'].attrs['GeoTransform']}")
    else:
        print("    ⚠️  Missing GeoTransform attribute")

    # Check 6: CRS information (from notebook verification)
    if "crs_wkt" in ds["spatial_ref"].attrs:
        print("    ✅ CRS WKT present")
    else:
        print("    ⚠️  Missing CRS WKT")

    # Check 7: Coordinate standard names (from notebook verification)
    for coord in ["x", "y"]:
        if coord in ds.coords and "standard_name" in ds[coord].attrs:
            expected_name = "projection_x_coordinate" if coord == "x" else "projection_y_coordinate"
            assert ds[coord].attrs["standard_name"] == expected_name, (
                f"Incorrect standard_name for {coord} coordinate in {group}"
            )
            print(f"    ✅ {coord} standard_name: {ds[coord].attrs['standard_name']}")

    ds.close()


def _verify_multiscale_structure(output_path: pathlib.Path, group: str) -> None:
    """Verify multiscale structure following notebook patterns."""
    print(f"Verifying multiscale structure for {group}...")

    group_path = output_path / group.lstrip("/")

    # Check that at least one level exists (level 0 is always created)
    level_dirs = [d for d in group_path.iterdir() if d.is_dir() and d.name.isdigit()]
    assert len(level_dirs) >= 1, (
        f"Expected at least 1 overview level for {group}, found {len(level_dirs)}"
    )
    print(f"    Found {len(level_dirs)} overview levels: {sorted([d.name for d in level_dirs])}")

    # For larger datasets, expect multiple levels
    level_0_path = str(group_path / "0")
    ds_0 = xr.open_dataset(level_0_path, engine="zarr", zarr_format=3)
    native_size = min(ds_0.sizes["y"], ds_0.sizes["x"])
    ds_0.close()

    if native_size >= 512:  # Larger datasets should have multiple levels
        assert len(level_dirs) >= 2, (
            f"Expected multiple overview levels for large dataset {group} (size {native_size}), found {len(level_dirs)}"
        )
    else:
        print(f"    Small dataset (size {native_size}), single level is acceptable")

    # Verify level 0 (native resolution) exists
    assert (group_path / "0").exists(), f"Level 0 missing for {group}"

    # Check that each level contains valid data
    level_shapes = {}
    for level_dir in sorted(level_dirs, key=lambda x: int(x.name)):
        level_num = int(level_dir.name)
        level_path = str(level_dir)
        ds = xr.open_dataset(level_path, engine="zarr", zarr_format=3)

        # Verify that the dataset has data variables
        assert len(ds.data_vars) > 0, f"No data variables in {level_path}"

        # Verify that spatial dimensions exist
        assert "x" in ds.dims, f"Missing 'x' dimension in {level_path}"
        assert "y" in ds.dims, f"Missing 'y' dimension in {level_path}"

        # Store shape for progression verification
        level_shapes[level_num] = (ds.sizes["y"], ds.sizes["x"])
        print(f"    Level {level_num}: {level_shapes[level_num]} pixels")

        ds.close()

    # Verify that overview levels have progressively smaller dimensions (COG-style /2 downsampling)
    if len(level_shapes) > 1:
        for level in sorted(level_shapes.keys())[1:]:
            prev_level = level - 1
            if prev_level in level_shapes:
                prev_height, prev_width = level_shapes[prev_level]
                curr_height, curr_width = level_shapes[level]

                # Check that dimensions are roughly half (allowing for rounding)
                height_ratio = prev_height / curr_height
                width_ratio = prev_width / curr_width

                assert 1.8 <= height_ratio <= 2.2, (
                    f"Height ratio between level {prev_level} and {level} should be ~2, got {height_ratio:.2f}"
                )
                assert 1.8 <= width_ratio <= 2.2, (
                    f"Width ratio between level {prev_level} and {level} should be ~2, got {width_ratio:.2f}"
                )

                print(
                    f"    Level {prev_level}→{level} downsampling ratio: {height_ratio:.2f}x{width_ratio:.2f}"
                )


def _verify_rgb_data_access(output_path: pathlib.Path, groups: list[str]) -> None:
    """Verify RGB data access patterns from the notebook."""
    print("Verifying RGB data access patterns...")

    # Find groups with RGB bands (following notebook logic)
    rgb_groups = []
    for group in groups:
        group_path_str = str(output_path / group.lstrip("/") / "0")
        ds = xr.open_dataset(group_path_str, engine="zarr", zarr_format=3)

        # Check for RGB bands (b04=red, b03=green, b02=blue for Sentinel-2)
        has_rgb = all(band in ds.data_vars for band in ["b04", "b03", "b02"])
        if has_rgb:
            rgb_groups.append(group)
            print(f"    Found RGB bands in {group}")

        ds.close()

    # Test data access for RGB groups (following notebook access patterns)
    for group in rgb_groups:
        print(f"    Testing data access for {group}...")

        # Test access to different overview levels (as in notebook)
        group_path = output_path / group.lstrip("/")
        level_dirs = [d for d in group_path.iterdir() if d.is_dir() and d.name.isdigit()]

        for level_dir in sorted(level_dirs, key=lambda x: int(x.name))[:3]:  # Test first 3 levels
            level_num = int(level_dir.name)
            level_path = str(level_dir)

            # Open dataset and access RGB bands (following notebook pattern)
            ds = xr.open_dataset(level_path, engine="zarr", zarr_format=3)

            # Access RGB data (as in notebook)
            red_data = ds["b04"].values
            green_data = ds["b03"].values
            blue_data = ds["b02"].values

            # Verify data shapes match
            assert red_data.shape == green_data.shape == blue_data.shape, (
                f"RGB band shapes don't match in {group} level {level_num}"
            )

            # Verify data is not empty
            assert red_data.size > 0, f"Empty red data in {group} level {level_num}"
            assert green_data.size > 0, f"Empty green data in {group} level {level_num}"
            assert blue_data.size > 0, f"Empty blue data in {group} level {level_num}"

            print(f"      Level {level_num}: RGB access successful, shape {red_data.shape}")

            ds.close()
