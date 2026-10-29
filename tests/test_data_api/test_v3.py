from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import zarr
from pydantic import ValidationError
from pydantic_zarr.core import tuplify_json
from pydantic_zarr.v3 import ArraySpec, GroupSpec

from eopf_geozarr.data_api.geozarr.common import DatasetAttrs
from eopf_geozarr.data_api.geozarr.v3 import (
    DataArray,
    Dataset,
    MultiscaleGroup,
    check_valid_coordinates,
)

if TYPE_CHECKING:
    from eopf_geozarr.data_api.geozarr.common import GroupLike


class TestCheckValidCoordinates:
    @staticmethod
    @pytest.mark.parametrize("data_shape", [(10,), (10, 12)])
    def test_valid(data_shape: tuple[int, ...]) -> None:
        """
        Test the check_valid_coordinates function to ensure it validates coordinates correctly.
        """

        base_array = DataArray.from_array(
            np.zeros((data_shape), dtype="uint8"),
            dimension_names=[f"dim_{s}" for s in range(len(data_shape))],
        )
        coords_arrays = {
            f"dim_{idx}": DataArray.from_array(np.arange(s), dimension_names=(f"dim_{idx}",))
            for idx, s in enumerate(data_shape)
        }
        group = GroupSpec[Mapping[str, object], DataArray](
            attributes={}, members={"base": base_array, **coords_arrays}
        )
        # ``group`` structurally satisfies ``GroupLike``, but mypy cannot bind the
        # invariant ``Mapping`` value type, mirroring the cast used in the source.
        group_like = cast("GroupLike", group)
        assert check_valid_coordinates(group_like) == group_like

    @staticmethod
    @pytest.mark.parametrize("data_shape", [(10,), (10, 12)])
    def test_invalid_coordinates(
        data_shape: tuple[int, ...],
    ) -> None:
        """
        Test the check_valid_coordinates function to ensure it validates coordinates correctly.

        This test checks that the function raises a ValueError when the dimensions of the data variable
        do not match the dimensions of the coordinate arrays.
        """
        base_array = DataArray.from_array(
            np.zeros((data_shape), dtype="uint8"),
            dimension_names=[f"dim_{s}" for s in range(len(data_shape))],
        )
        coords_arrays = {
            f"dim_{idx}": DataArray.from_array(np.arange(s + 1), dimension_names=(f"dim_{idx}",))
            for idx, s in enumerate(data_shape)
        }
        group = GroupSpec[Mapping[str, object], DataArray](
            attributes={}, members={"base": base_array, **coords_arrays}
        )
        msg = "Dimension .* for array 'base' has a shape mismatch:"
        with pytest.raises(ValueError, match=msg):
            check_valid_coordinates(cast("GroupLike", group))


def test_dataarray_round_trip(s2_geozarr_group_example: zarr.Group) -> None:
    """
    Ensure that we can round-trip dataarray attributes through the `Multiscales` model.
    """
    source_untyped: GroupSpec = GroupSpec.from_zarr(s2_geozarr_group_example)
    flat = source_untyped.to_flat()
    for val in flat.values():
        if isinstance(val, ArraySpec) and val.dimension_names is not None:
            model_json = val.model_dump()
            assert DataArray(**model_json).model_dump() == model_json


def test_multiscale_attrs_round_trip(s2_geozarr_group_example: zarr.Group) -> None:
    """
    Test that multiscale datasets round-trip through the `Multiscales` model
    """
    source_group_members = dict(s2_geozarr_group_example.members(max_depth=None))
    for val in source_group_members.values():
        if isinstance(val, zarr.Group) and "multiscales" in val.attrs.asdict():
            model_json = MultiscaleGroup.from_zarr(val).model_dump()
            # tuplify both sides: level groups may carry list-valued attributes
            # (e.g. ``spatial:bbox``) that are JSON-equivalent to tuples but not
            # normalised by ``model_dump`` alone.
            assert tuplify_json(MultiscaleGroup(**model_json).model_dump()) == tuplify_json(
                model_json
            )


class TestDataset:
    @staticmethod
    def _members() -> dict[str, DataArray]:
        base_array = DataArray.from_array(
            np.zeros((4, 4), dtype="uint8"), dimension_names=["y", "x"]
        )
        coords_arrays = {
            name: DataArray.from_array(np.arange(4), dimension_names=(name,)) for name in ("y", "x")
        }
        return {"band": base_array, **coords_arrays}

    def test_valid(self) -> None:
        """A dataset with consistent coordinates and no grid mapping validates."""
        ds = Dataset(attributes=DatasetAttrs(), members=self._members())
        assert isinstance(ds, Dataset)

    def test_missing_grid_mapping_variable(self) -> None:
        """A member declaring a grid_mapping that is not in the dataset fails validation."""
        members = self._members()
        band = members["band"]
        members["band"] = band.model_copy(
            update={"attributes": band.attributes.model_copy(update={"grid_mapping": "nope"})}
        )
        with pytest.raises(ValidationError, match="Grid mapping variable 'nope'"):
            Dataset(attributes=DatasetAttrs(), members=members)


class TestStrictScope:
    """DataArray sets `zarr_scope_strict`: extension-point names must be claimed."""

    @staticmethod
    def _document_with_codec(codec: dict[str, object]) -> dict[str, object]:
        valid = DataArray.from_array(np.zeros((4,), dtype="uint8"), dimension_names=("x",))
        document = valid.model_dump()
        document["codecs"] = (*document["codecs"], codec)
        return document

    def test_claimed_names_pass(self) -> None:
        """A document built from names the scope claims validates, zstd included."""
        document = self._document_with_codec(
            {"name": "zstd", "configuration": {"level": 1, "checksum": False}}
        )
        spec = DataArray.model_validate(document)
        last = spec.codecs[-1]
        assert isinstance(last, Mapping)
        assert last["name"] == "zstd"

    def test_unclaimed_codec_name_refused(self) -> None:
        """A typo'd codec name is refused, where the lenient parent leaves it unjudged."""
        document = self._document_with_codec({"name": "zstdd", "configuration": {"level": 1}})
        with pytest.raises(ValidationError, match="'zstdd' names no codec in the scope"):
            DataArray.model_validate(document)
        # the unscoped parent class accepts the same document, the name unjudged
        last = ArraySpec.model_validate(document).codecs[-1]
        assert isinstance(last, Mapping)
        assert last["name"] == "zstdd"
