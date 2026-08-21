"""Tests for the Sentinel-3 OLCI data-api model."""

import pytest
from pydantic import ValidationError

from eopf_geozarr.data_api.s3_olci import Sentinel3OlciRoot
from eopf_geozarr.pyz.v2 import ArraySpec, GroupSpec  # noqa: F401


def _olci_arr() -> dict[str, object]:
    """Return a minimal v2 ArraySpec-shaped dict for a 2-D uint16 array."""
    return ArraySpec(
        shape=(4, 5),
        chunks=(4, 5),
        dtype="<u2",
        fill_value=0,
        attributes={"_ARRAY_DIMENSIONS": ["rows", "columns"]},
    ).model_dump()


def test_validates_minimal_olci_product() -> None:
    """A dict with 21 radiance bands + geolocation validates as Sentinel3OlciRoot."""
    radiance = {f"oa{i:02d}_radiance": _olci_arr() for i in range(1, 22)}
    coords = {c: _olci_arr() for c in ("latitude", "longitude", "altitude")}
    root = {
        "zarr_format": 2,
        "attributes": {"other_metadata": {}, "stac_discovery": {}},
        "members": {
            "measurements": {
                "zarr_format": 2,
                "attributes": {},
                "members": {**radiance, **coords},
            },
        },
    }
    model = Sentinel3OlciRoot.model_validate(root)
    assert "oa01_radiance" in model.measurements.members


def test_rejects_non_olci_product() -> None:
    """A dict with an unexpected key in measurements is rejected by closed=True."""
    not_olci = {
        "zarr_format": 2,
        "attributes": {"other_metadata": {}, "stac_discovery": {}},
        "members": {
            "measurements": {
                "zarr_format": 2,
                "attributes": {},
                "members": {
                    "reflectance": {
                        "zarr_format": 2,
                        "attributes": {},
                        "members": {},
                    }
                },
            }
        },
    }
    with pytest.raises(ValidationError):
        Sentinel3OlciRoot.model_validate(not_olci)


def test_detector_accepts_olci_zarr(tmp_path: object) -> None:
    import pathlib

    import zarr

    from eopf_geozarr.s3_olci_optimization.olci_converter import (
        is_sentinel3_olci_dataset,
    )

    # build a minimal OLCI zarr v2 store from the model dict used above
    radiance = {f"oa{i:02d}_radiance": _olci_arr() for i in range(1, 22)}
    coords = {c: _olci_arr() for c in ("latitude", "longitude", "altitude")}
    root_dict = {
        "zarr_format": 2,
        "attributes": {"other_metadata": {}, "stac_discovery": {}},
        "members": {
            "measurements": {
                "zarr_format": 2,
                "attributes": {},
                "members": {**radiance, **coords},
            }
        },
    }
    assert isinstance(tmp_path, pathlib.Path)
    out = tmp_path / "olci.zarr"
    Sentinel3OlciRoot.model_validate(root_dict).to_zarr(out, path="")  # type: ignore[arg-type]
    group = zarr.open_group(str(out), mode="r")
    assert is_sentinel3_olci_dataset(group) is True


def test_detector_rejects_s2_zarr(s2_group_example: object) -> None:
    import pathlib

    import zarr

    from eopf_geozarr.s3_olci_optimization.olci_converter import (
        is_sentinel3_olci_dataset,
    )

    assert isinstance(s2_group_example, pathlib.Path)
    group = zarr.open_group(str(s2_group_example), mode="r")
    assert is_sentinel3_olci_dataset(group) is False


def test_real_olci_product_is_detected(s3_olci_group_example: object) -> None:
    import pathlib

    import zarr

    from eopf_geozarr.s3_olci_optimization.olci_converter import (
        is_sentinel3_olci_dataset,
    )

    assert isinstance(s3_olci_group_example, pathlib.Path)
    group = zarr.open_group(str(s3_olci_group_example), mode="r")
    assert is_sentinel3_olci_dataset(group) is True


def test_real_olci_product_validates_model(s3_olci_group_example: object) -> None:
    import pathlib

    import zarr

    from eopf_geozarr.data_api.s3_olci import Sentinel3OlciRoot
    from eopf_geozarr.pyz.v2 import GroupSpec as PyzGroupSpec

    assert isinstance(s3_olci_group_example, pathlib.Path)
    group = zarr.open_group(str(s3_olci_group_example), mode="r")
    model = Sentinel3OlciRoot.model_validate(PyzGroupSpec.from_zarr(group).model_dump())
    assert "oa01_radiance" in model.measurements.members
