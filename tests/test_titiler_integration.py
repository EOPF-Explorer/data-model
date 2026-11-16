"""
Integration tests verifying that S2 converter output can be consumed by titiler-xarray.

These tests run the actual S2 converter on real data, then verify that titiler's
APIs (info, tiles, point queries, bbox crops) work correctly against the output.

Requires:
    - /tmp/s2_source.zarr to exist (real S2 product)
    - titiler-xarray, httpx installed
"""

# titiler/fastapi/starlette are optional (the `downstream-titiler` group); this
# module is skipped at runtime when they're absent, so don't flag their imports.
# pyright: reportMissingImports=false

from __future__ import annotations

import pathlib
import tempfile

import pytest
import xarray as xr

pytest.importorskip("titiler.xarray", reason="titiler-xarray not installed")

from fastapi import FastAPI
from starlette.testclient import TestClient
from titiler.xarray.factory import TilerFactory

s2_source_path = pathlib.Path("/tmp/s2_source.zarr")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not s2_source_path.exists(),
        reason="S2 source data not available at /tmp/s2_source.zarr",
    ),
]


@pytest.fixture(scope="module")
def converted_s2_path() -> pathlib.Path:
    """Run the real S2 converter and return the output path.

    This fixture is module-scoped so the conversion only runs once for all tests.
    """
    from eopf_geozarr.s2_optimization.s2_converter import convert_s2

    tmpdir = tempfile.mkdtemp(prefix="titiler_test_")
    output_path = str(pathlib.Path(tmpdir) / "s2_optimized.zarr")

    dt_input = xr.open_datatree(
        str(s2_source_path),
        engine="zarr",
        chunks="auto",
    )

    convert_s2(
        dt_input,
        output_path=output_path,
        validate_output=False,
        enable_sharding=False,
        spatial_chunk=512,
    )

    return pathlib.Path(output_path)


@pytest.fixture(scope="module")
def titiler_client() -> TestClient:
    """Create a titiler TestClient."""
    tiler = TilerFactory(router_prefix="/xarray")
    app = FastAPI()
    app.include_router(tiler.router, prefix="/xarray")
    return TestClient(app)


@pytest.fixture(scope="module")
def reflectance_groups(converted_s2_path: pathlib.Path) -> list[str]:
    """Discover reflectance resolution groups (e.g. r10m, r20m, r60m) in the output."""
    refl_path = converted_s2_path / "measurements" / "reflectance"
    if not refl_path.exists():
        pytest.skip("No measurements/reflectance in converted output")
    groups = sorted(d.name for d in refl_path.iterdir() if d.is_dir() and d.name.startswith("r"))
    assert len(groups) > 0, "No resolution groups found under measurements/reflectance"
    return groups


def _open_group(path: pathlib.Path, group: str) -> xr.Dataset:
    """Open a zarr group as xarray Dataset."""
    return xr.open_dataset(str(path), engine="zarr", group=group, zarr_format=3, consolidated=False)


def _band_vars(ds: xr.Dataset) -> list[str]:
    """Get band variable names, excluding spatial_ref."""
    return [str(v) for v in ds.data_vars if v != "spatial_ref"]


class TestTitilerInfo:
    """Test that titiler /info endpoint works for each resolution group."""

    def test_info_all_groups(
        self,
        converted_s2_path: pathlib.Path,
        titiler_client: TestClient,
        reflectance_groups: list[str],
    ) -> None:
        """Verify /info returns valid metadata for every resolution group."""
        for group_name in reflectance_groups:
            zarr_group = f"measurements/reflectance/{group_name}"

            ds = _open_group(converted_s2_path, zarr_group)
            bands = _band_vars(ds)
            ds.close()
            assert len(bands) > 0, f"No band variables in {zarr_group}"

            variable = bands[0]
            resp = titiler_client.get(
                "/xarray/info",
                params={
                    "url": str(converted_s2_path),
                    "variable": variable,
                    "group": zarr_group,
                },
            )
            assert resp.status_code == 200, (
                f"Info failed for {zarr_group}/{variable}: {resp.text[:300]}"
            )
            info = resp.json()
            assert "bounds" in info, f"No bounds in info for {zarr_group}"
            assert len(info["bounds"]) == 4
            assert info["bounds"][0] < info["bounds"][2], "Invalid x bounds"
            assert info["bounds"][1] < info["bounds"][3], "Invalid y bounds"

    def test_info_all_bands(
        self,
        converted_s2_path: pathlib.Path,
        titiler_client: TestClient,
    ) -> None:
        """Verify /info works for every band variable in the r10m group."""
        zarr_group = "measurements/reflectance/r10m"
        ds = _open_group(converted_s2_path, zarr_group)
        bands = _band_vars(ds)
        ds.close()

        for variable in bands:
            resp = titiler_client.get(
                "/xarray/info",
                params={
                    "url": str(converted_s2_path),
                    "variable": variable,
                    "group": zarr_group,
                },
            )
            assert resp.status_code == 200, (
                f"Info failed for {zarr_group}/{variable}: {resp.text[:300]}"
            )


class TestTitilerTiles:
    """Test that titiler can generate map tiles from the converted data."""

    def test_tile_generation(
        self,
        converted_s2_path: pathlib.Path,
        titiler_client: TestClient,
        reflectance_groups: list[str],
    ) -> None:
        """Verify tiles can be generated for each resolution group."""
        for group_name in reflectance_groups:
            zarr_group = f"measurements/reflectance/{group_name}"
            ds = _open_group(converted_s2_path, zarr_group)
            bands = _band_vars(ds)
            ds.close()
            if not bands:
                continue

            resp = titiler_client.get(
                "/xarray/tiles/WebMercatorQuad/0/0/0.png",
                params={
                    "url": str(converted_s2_path),
                    "variable": bands[0],
                    "group": zarr_group,
                    "rescale": "0,10000",
                },
            )
            assert resp.status_code == 200, (
                f"Tile failed for {zarr_group}/{bands[0]}: {resp.text[:300]}"
            )
            assert resp.headers["content-type"] == "image/png"
            assert len(resp.content) > 0

    def test_tilejson(
        self,
        converted_s2_path: pathlib.Path,
        titiler_client: TestClient,
    ) -> None:
        """Verify TileJSON metadata endpoint works."""
        zarr_group = "measurements/reflectance/r10m"
        ds = _open_group(converted_s2_path, zarr_group)
        bands = _band_vars(ds)
        ds.close()

        resp = titiler_client.get(
            "/xarray/WebMercatorQuad/tilejson.json",
            params={
                "url": str(converted_s2_path),
                "variable": bands[0],
                "group": zarr_group,
                "rescale": "0,10000",
            },
        )
        assert resp.status_code == 200
        tj = resp.json()
        assert "tiles" in tj
        assert "bounds" in tj
        assert len(tj["bounds"]) == 4


class TestTitilerPointQuery:
    """Test that titiler point queries return valid pixel values."""

    def test_point_query(
        self,
        converted_s2_path: pathlib.Path,
        titiler_client: TestClient,
    ) -> None:
        """Query a point within the data extent and verify values are returned."""
        from pyproj import CRS, Transformer

        zarr_group = "measurements/reflectance/r10m"
        ds = _open_group(converted_s2_path, zarr_group)
        bands = _band_vars(ds)

        x_center = float(ds.x.values[len(ds.x) // 2])
        y_center = float(ds.y.values[len(ds.y) // 2])

        # Get CRS from spatial_ref attributes
        crs_wkt = ds["spatial_ref"].attrs["crs_wkt"]
        ds.close()

        crs = CRS.from_wkt(crs_wkt)
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x_center, y_center)

        resp = titiler_client.get(
            f"/xarray/point/{lon},{lat}",
            params={
                "url": str(converted_s2_path),
                "variable": bands[0],
                "group": zarr_group,
            },
        )
        assert resp.status_code == 200, f"Point query failed at ({lon}, {lat}): {resp.text[:300]}"
        result = resp.json()
        assert "values" in result
        assert len(result["values"]) > 0


class TestTitilerBbox:
    """Test that titiler bbox (part) endpoint returns cropped images."""

    def test_bbox_crop(
        self,
        converted_s2_path: pathlib.Path,
        titiler_client: TestClient,
    ) -> None:
        """Request a bbox crop of the data and verify an image is returned."""
        from pyproj import CRS, Transformer

        zarr_group = "measurements/reflectance/r10m"
        ds = _open_group(converted_s2_path, zarr_group)
        bands = _band_vars(ds)

        # Get native CRS bounds and convert to WGS84 for the bbox endpoint
        x_min, x_max = float(ds.x.min()), float(ds.x.max())
        y_min, y_max = float(ds.y.min()), float(ds.y.max())
        crs_wkt = ds["spatial_ref"].attrs["crs_wkt"]
        ds.close()

        crs = CRS.from_wkt(crs_wkt)
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon_min, lat_min = transformer.transform(x_min, y_min)
        lon_max, lat_max = transformer.transform(x_max, y_max)

        # Use the center 10% of the geographic extent
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        lon_center = (lon_min + lon_max) / 2
        lat_center = (lat_min + lat_max) / 2
        small_bounds = [
            lon_center - lon_range * 0.05,
            lat_center - lat_range * 0.05,
            lon_center + lon_range * 0.05,
            lat_center + lat_range * 0.05,
        ]
        bbox_str = ",".join(f"{v:.6f}" for v in small_bounds)

        resp = titiler_client.get(
            f"/xarray/bbox/{bbox_str}.png",
            params={
                "url": str(converted_s2_path),
                "variable": bands[0],
                "group": zarr_group,
                "rescale": "0,10000",
            },
        )
        assert resp.status_code == 200, f"Bbox crop failed: {resp.text[:300]}"
        assert resp.headers["content-type"] == "image/png"
        assert len(resp.content) > 100


class TestTitilerMultiscaleConsistency:
    """Verify that resolution groups report consistent geographic bounds."""

    def test_bounds_consistent_across_resolutions(
        self,
        converted_s2_path: pathlib.Path,
        titiler_client: TestClient,
        reflectance_groups: list[str],
    ) -> None:
        """All resolution groups should report approximately the same geographic bounds.

        The native groups (r10m, r20m, r60m) and derived groups (r120m, r360m, r720m)
        should all cover the same spatial extent.
        """
        bounds_per_group: dict[str, list[float]] = {}
        for group_name in reflectance_groups:
            zarr_group = f"measurements/reflectance/{group_name}"
            ds = _open_group(converted_s2_path, zarr_group)
            bands = _band_vars(ds)
            ds.close()
            if not bands:
                continue

            resp = titiler_client.get(
                "/xarray/info",
                params={
                    "url": str(converted_s2_path),
                    "variable": bands[0],
                    "group": zarr_group,
                },
            )
            if resp.status_code == 200:
                bounds_per_group[group_name] = resp.json()["bounds"]

        assert len(bounds_per_group) >= 2, "Need at least 2 groups to compare bounds"

        # Use r10m as reference
        ref_group = next(g for g in reflectance_groups if g in bounds_per_group)
        ref_bounds = bounds_per_group[ref_group]
        extent = max(
            abs(ref_bounds[2] - ref_bounds[0]),
            abs(ref_bounds[3] - ref_bounds[1]),
        )
        # 2% tolerance — derived levels may have slightly different extents due to rounding
        tolerance = extent * 0.02

        for group_name, bounds in bounds_per_group.items():
            if group_name == ref_group:
                continue
            for i in range(4):
                assert abs(bounds[i] - ref_bounds[i]) < tolerance, (
                    f"Bounds mismatch {group_name} vs {ref_group}: "
                    f"index {i}: {bounds[i]} vs {ref_bounds[i]} (tolerance {tolerance})"
                )
