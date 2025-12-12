import pathlib

import zarr

from eopf_geozarr.s2_optimization.s2_converter import is_sentinel2_dataset


def test_is_sentinel2_dataset_true(s2_group_example: pathlib.Path) -> None:
    group = zarr.open_group(str(s2_group_example), mode="r")
    assert is_sentinel2_dataset(group)


def test_is_sentinel2_dataset_false(tmp_path: pathlib.Path) -> None:
    group = zarr.open_group(str(tmp_path / "empty.zarr"), mode="w")
    assert not is_sentinel2_dataset(group)
