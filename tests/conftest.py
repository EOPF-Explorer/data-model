"""Tests for the eopf-geozarr package."""

import itertools
import json
import pathlib

import pytest
import xarray as xr
import zarr
from pydantic_zarr.v2 import GroupSpec as GroupSpecV2
from pydantic_zarr.v3 import GroupSpec as GroupSpecV3

# Paths to example data / metadata
s1_example_json_paths = tuple(pathlib.Path("tests/_test_data/s1_examples").glob("*.json"))
s2_example_json_paths = tuple(pathlib.Path("tests/_test_data/s2_examples").glob("*.json"))
projjson_example_paths = tuple(pathlib.Path("tests/_test_data/projjson_examples").glob("*.json"))
geoproj_example_paths = tuple(pathlib.Path("tests/_test_data/geoproj_examples").glob("*.json"))
geozarr_example_paths = tuple(pathlib.Path("tests/_test_data/geozarr_examples").glob("*.json"))
zcm_multiscales_example_paths = tuple(
    pathlib.Path("tests/_test_data/zcm_multiscales_examples").glob("*.json")
)
optimized_geozarr_example_paths = tuple(
    pathlib.Path("tests/_test_data/optimized_geozarr_examples").glob("*.json")
)


def read_json(path: pathlib.Path) -> dict[str, object]:
    """
    Read the contents of path as JSON
    """
    return json.loads(path.read_text())


def get_stem(p: pathlib.Path) -> str:
    return p.stem


def create_group_from_json(source_path: pathlib.Path, out_path: pathlib.Path) -> pathlib.Path:
    """
    Create a Zarr V2 group from a JSON model
    """
    out_dir = out_path / (source_path.stem + ".zarr")
    g = GroupSpecV2(**read_json(source_path))
    g.to_zarr(out_dir, path="")
    return out_dir


@pytest.fixture(params=s1_example_json_paths, ids=get_stem)
def s1_group_example(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> pathlib.Path:
    """
    A fixture that returns the path to a Zarr group with the same layout as a sentinel 1
    product
    """
    return create_group_from_json(request.param, tmp_path)


@pytest.fixture(params=s2_example_json_paths, ids=get_stem)
def s2_group_example(request: pytest.FixtureRequest, tmp_path: pathlib.Path) -> pathlib.Path:
    """
    A fixture that returns the path to a Zarr group with the same layout as a sentinel 2
    product
    """
    return create_group_from_json(request.param, tmp_path)


@pytest.fixture(params=s1_example_json_paths, ids=get_stem)
def s1_json_example(request: pytest.FixtureRequest) -> dict[str, object]:
    """
    A fixture that returns the JSON model of a Sentinel 1 Zarr group
    """
    source_path: pathlib.Path = request.param
    return read_json(source_path)


@pytest.fixture(params=s2_example_json_paths, ids=get_stem)
def s2_json_example(request: pytest.FixtureRequest) -> dict[str, object]:
    """
    A fixture that returns the JSON model of a Sentinel 2 Zarr group
    """
    source_path: pathlib.Path = request.param
    return read_json(source_path)


@pytest.fixture(params=geozarr_example_paths, ids=get_stem)
def s2_geozarr_group_example(request: pytest.FixtureRequest) -> zarr.Group:
    """
    Return a memory-backed Zarr V3 Group based on a sentinel 2 product converted to geozarr
    """
    source_path: pathlib.Path = request.param
    store = {}
    return GroupSpecV3(**read_json(source_path)).to_zarr(store, path="")


@pytest.fixture(params=optimized_geozarr_example_paths, ids=get_stem)
def s2_optimized_geozarr_group_example(request: pytest.FixtureRequest) -> zarr.Group:
    """
    Return a memory-backed Zarr V3 Group based on a sentinel 2 product converted to geozarr
    """
    source_path: pathlib.Path = request.param
    store = {}
    return GroupSpecV3(**read_json(source_path)).to_zarr(store, path="")


@pytest.fixture(params=zcm_multiscales_example_paths, ids=get_stem)
def zcm_multiscales_example(request: pytest.FixtureRequest) -> dict[str, object]:
    source_path: pathlib.Path = request.param
    return read_json(source_path)


@pytest.fixture(params=projjson_example_paths, ids=get_stem)
def projjson_example(request: pytest.FixtureRequest) -> dict[str, object]:
    source_path: pathlib.Path = request.param
    return read_json(source_path)


@pytest.fixture
def bound_crs_json() -> dict[str, object]:
    return read_json(pathlib.Path("tests/_test_data/projjson_examples/bound_crs.json"))


@pytest.fixture
def compound_crs_json() -> dict[str, object]:
    return read_json(pathlib.Path("tests/_test_data/projjson_examples/compound_crs.json"))


@pytest.fixture
def datum_ensemble_json() -> dict[str, object]:
    return read_json(pathlib.Path("tests/_test_data/projjson_examples/datum_ensemble.json"))


@pytest.fixture
def explicit_prime_meridian_json() -> dict[str, object]:
    return read_json(
        pathlib.Path("tests/_test_data/projjson_examples/explicit_prime_meridian.json")
    )


@pytest.fixture
def implicit_prime_meridian_json() -> dict[str, object]:
    return read_json(
        pathlib.Path("tests/_test_data/projjson_examples/implicit_prime_meridian.json")
    )


@pytest.fixture
def projected_crs_json() -> dict[str, object]:
    return read_json(pathlib.Path("tests/_test_data/projjson_examples/projected_crs.json"))


@pytest.fixture
def transformation_json() -> dict[str, object]:
    return read_json(pathlib.Path("tests/_test_data/projjson_examples/transformation.json"))


@pytest.fixture(params=geoproj_example_paths, ids=get_stem)
def geoproj_example(request: pytest.FixtureRequest) -> dict[str, object]:
    source_path: pathlib.Path = request.param
    return read_json(source_path)


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
        # Native-resolution arrays are written directly at the group root
        # (new S2-aligned layout — no /0 subdirectory).


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

    # Open the native resolution dataset (written at the group root)
    group_path = str(output_path / group.lstrip("/"))
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
    """Verify multiscale structure (new S2-aligned layout).

    Native-resolution arrays are written directly at the group root; downsampled
    overviews live under sibling subgroups named ``r2``, ``r4``, ``r8``, …
    """
    print(f"Verifying multiscale structure for {group}...")

    group_path = output_path / group.lstrip("/")

    # Discover overview subgroups (e.g. r2, r4, r8) — pure rN names.
    overview_dirs = [
        d
        for d in group_path.iterdir()
        if d.is_dir() and d.name.startswith("r") and d.name[1:].isdigit()
    ]
    print(
        f"    Found {len(overview_dirs)} overview levels: {sorted(d.name for d in overview_dirs)}"
    )

    # Native resolution dataset lives at the group root.
    ds_native = xr.open_dataset(str(group_path), engine="zarr", zarr_format=3)
    native_size = min(ds_native.sizes["y"], ds_native.sizes["x"])
    ds_native.close()

    if native_size >= 512:
        assert len(overview_dirs) >= 1, (
            f"Expected at least 1 overview for large dataset {group} (size {native_size}),"
            f" found {len(overview_dirs)}"
        )
    else:
        print(f"    Small dataset (size {native_size}), no overviews is acceptable")

    # Walk levels in factor order: native (1), r2, r4, …
    level_shapes: dict[int, tuple[int, int]] = {}
    ds_native = xr.open_dataset(str(group_path), engine="zarr", zarr_format=3)
    assert len(ds_native.data_vars) > 0, f"No data variables in {group_path}"
    assert "x" in ds_native.dims, f"Missing 'x' dimension in {group_path}"
    assert "y" in ds_native.dims, f"Missing 'y' dimension in {group_path}"
    level_shapes[1] = (ds_native.sizes["y"], ds_native.sizes["x"])
    print(f"    Native (r1): {level_shapes[1]} pixels")
    ds_native.close()

    for ov_dir in sorted(overview_dirs, key=lambda x: int(x.name[1:])):
        factor = int(ov_dir.name[1:])
        ds = xr.open_dataset(str(ov_dir), engine="zarr", zarr_format=3)
        assert len(ds.data_vars) > 0, f"No data variables in {ov_dir}"
        assert "x" in ds.dims, f"Missing 'x' dimension in {ov_dir}"
        assert "y" in ds.dims, f"Missing 'y' dimension in {ov_dir}"
        level_shapes[factor] = (ds.sizes["y"], ds.sizes["x"])
        print(f"    Overview r{factor}: {level_shapes[factor]} pixels")
        ds.close()

    # Verify overviews halve dimensions at each successive factor of 2.
    factors_sorted = sorted(level_shapes.keys())
    for prev, curr in itertools.pairwise(factors_sorted):
        if curr != prev * 2:
            continue  # gaps allowed (e.g. r1 → r4 if r2 absent)
        prev_h, prev_w = level_shapes[prev]
        curr_h, curr_w = level_shapes[curr]
        height_ratio = prev_h / curr_h
        width_ratio = prev_w / curr_w
        assert 1.8 <= height_ratio <= 2.2, (
            f"Height ratio between r{prev} and r{curr} should be ~2, got {height_ratio:.2f}"
        )
        assert 1.8 <= width_ratio <= 2.2, (
            f"Width ratio between r{prev} and r{curr} should be ~2, got {width_ratio:.2f}"
        )


def _verify_rgb_data_access(output_path: pathlib.Path, groups: list[str]) -> None:
    """Verify RGB data access patterns from the notebook."""
    print("Verifying RGB data access patterns...")

    # Find groups with RGB bands (following notebook logic)
    rgb_groups = []
    for group in groups:
        group_path_str = str(output_path / group.lstrip("/"))
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

        # Test access at native and a few overview levels.
        group_path = output_path / group.lstrip("/")
        overview_dirs = [
            d
            for d in group_path.iterdir()
            if d.is_dir() and d.name.startswith("r") and d.name[1:].isdigit()
        ]
        # Include native (group root) first, then up to two overviews.
        targets: list[tuple[str, pathlib.Path]] = [("native", group_path)]
        targets.extend(
            (ov.name, ov) for ov in sorted(overview_dirs, key=lambda x: int(x.name[1:]))[:2]
        )

        for label, level_path in targets:
            ds = xr.open_dataset(str(level_path), engine="zarr", zarr_format=3)

            red_data = ds["b04"].values
            green_data = ds["b03"].values
            blue_data = ds["b02"].values

            assert red_data.shape == green_data.shape == blue_data.shape, (
                f"RGB band shapes don't match in {group} level {label}"
            )

            assert red_data.size > 0, f"Empty red data in {group} level {label}"
            assert green_data.size > 0, f"Empty green data in {group} level {label}"
            assert blue_data.size > 0, f"Empty blue data in {group} level {label}"

            print(f"      Level {label}: RGB access successful, shape {red_data.shape}")

            ds.close()
