"""Tests for the Spatial Zarr Convention models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eopf_geozarr.data_api.geozarr.spatial import Spatial, SpatialConventionMetadata


class TestSpatialConventionMetadata:
    """Test the SpatialConventionMetadata class."""

    def test_default_values(self) -> None:
        """Test that default values are correctly set."""
        metadata = SpatialConventionMetadata()

        assert metadata.uuid == "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4"
        assert metadata.name == "spatial:"
        assert (
            metadata.schema_url
            == "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json"
        )
        assert metadata.spec_url == "https://github.com/zarr-conventions/spatial/blob/v1/README.md"
        assert metadata.description == "Spatial coordinate and transformation information"

    def test_serialization(self) -> None:
        """Test that metadata can be serialized correctly."""
        metadata = SpatialConventionMetadata()
        result = metadata.model_dump()

        expected = {
            "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
            "name": "spatial:",
            "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
            "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
            "description": "Spatial coordinate and transformation information",
        }

        assert result == expected


class TestSpatial:
    """Test the Spatial model class."""

    def test_minimal_required_fields(self) -> None:
        """Test creation with only required fields."""
        spatial = Spatial(**{"spatial:dimensions": ["y", "x"]})

        assert spatial.dimensions == ["y", "x"]
        assert spatial.bbox is None
        assert spatial.transform_type == "affine"  # Default value
        assert spatial.transform is None
        assert spatial.shape is None
        assert spatial.registration == "pixel"  # Default value

    def test_missing_required_dimensions(self) -> None:
        """Test that missing dimensions field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Spatial()

        assert "spatial:dimensions" in str(exc_info.value)

    def test_full_spatial_metadata(self) -> None:
        """Test creation with all fields populated."""
        data = {
            "spatial:dimensions": ["y", "x"],
            "spatial:bbox": [500000.0, 4900000.0, 600000.0, 5000000.0],
            "spatial:transform_type": "affine",
            "spatial:transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0],
            "spatial:shape": [1000, 1000],
            "spatial:registration": "pixel",
        }

        spatial = Spatial(**data)

        assert spatial.dimensions == ["y", "x"]
        assert spatial.bbox == [500000.0, 4900000.0, 600000.0, 5000000.0]
        assert spatial.transform_type == "affine"
        assert spatial.transform == [10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0]
        assert spatial.shape == [1000, 1000]
        assert spatial.registration == "pixel"

    def test_3d_spatial_data(self) -> None:
        """Test spatial model with 3D data."""
        data = {
            "spatial:dimensions": ["z", "y", "x"],
            "spatial:bbox": [500000.0, 4900000.0, 0.0, 600000.0, 5000000.0, 100.0],
            "spatial:shape": [10, 1000, 1000],
        }

        spatial = Spatial(**data)

        assert spatial.dimensions == ["z", "y", "x"]
        assert spatial.bbox == [500000.0, 4900000.0, 0.0, 600000.0, 5000000.0, 100.0]
        assert spatial.shape == [10, 1000, 1000]

    def test_serialization_by_alias(self) -> None:
        """Test that serialization uses aliases (spatial: prefixes)."""
        data = {
            "spatial:dimensions": ["y", "x"],
            "spatial:bbox": [0.0, 0.0, 100.0, 100.0],
            "spatial:transform": [1.0, 0.0, 0.0, 0.0, -1.0, 100.0],
            "spatial:shape": [100, 100],
        }

        spatial = Spatial(**data)
        result = spatial.model_dump()

        # Should serialize with spatial: prefixes
        assert "spatial:dimensions" in result
        assert "spatial:bbox" in result
        assert "spatial:transform" in result
        assert "spatial:shape" in result
        assert "spatial:transform_type" in result
        assert "spatial:registration" in result

        # Should not have unprefixed versions
        assert "dimensions" not in result
        assert "bbox" not in result
        assert "transform" not in result
        assert "shape" not in result

    def test_none_fields_excluded(self) -> None:
        """Test that None fields are excluded from serialization."""
        spatial = Spatial(**{"spatial:dimensions": ["y", "x"]})
        result = spatial.model_dump()

        # None fields should be excluded
        assert "spatial:bbox" not in result
        assert "spatial:transform" not in result
        assert "spatial:shape" not in result

        # Default values should be included
        assert result["spatial:transform_type"] == "affine"
        assert result["spatial:registration"] == "pixel"

    def test_node_registration(self) -> None:
        """Test node registration type."""
        data = {"spatial:dimensions": ["y", "x"], "spatial:registration": "node"}

        spatial = Spatial(**data)
        assert spatial.registration == "node"

    def test_non_affine_transform_type(self) -> None:
        """Test non-affine transform types."""
        data = {"spatial:dimensions": ["y", "x"], "spatial:transform_type": "rpc"}

        spatial = Spatial(**data)
        assert spatial.transform_type == "rpc"

    def test_extra_fields_allowed(self) -> None:
        """Test that extra fields are allowed."""
        data = {
            "spatial:dimensions": ["y", "x"],
            "custom_field": "custom_value",
            "spatial:custom": "also_allowed",
        }

        spatial = Spatial(**data)
        result = spatial.model_dump()

        assert result["custom_field"] == "custom_value"
        assert result["spatial:custom"] == "also_allowed"

    def test_roundtrip_serialization(self) -> None:
        """Test that serialization and deserialization preserves data."""
        original_data = {
            "spatial:dimensions": ["y", "x"],
            "spatial:bbox": [500000.0, 4900000.0, 600000.0, 5000000.0],
            "spatial:transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0],
            "spatial:shape": [1000, 1000],
            "spatial:registration": "node",
            "spatial:transform_type": "affine",
        }

        # Create model, serialize, then recreate
        spatial1 = Spatial(**original_data)
        serialized = spatial1.model_dump()
        spatial2 = Spatial(**serialized)

        # Should be equivalent
        assert spatial1.dimensions == spatial2.dimensions
        assert spatial1.bbox == spatial2.bbox
        assert spatial1.transform == spatial2.transform
        assert spatial1.shape == spatial2.shape
        assert spatial1.registration == spatial2.registration
        assert spatial1.transform_type == spatial2.transform_type

    def test_invalid_dimensions_none(self) -> None:
        """Test that None dimensions raise ValidationError."""
        with pytest.raises(ValidationError):
            Spatial(**{"spatial:dimensions": None})

    def test_empty_dimensions_not_allowed(self) -> None:
        """Test that empty dimensions raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Spatial(**{"spatial:dimensions": []})

        assert "spatial:dimensions must contain at least one dimension" in str(exc_info.value)
        data = {
            "spatial:dimensions": ["y", "x"],
            "spatial:transform_type": "affine",
            "spatial:transform": [10.0, 0.0, 500000.0, 0.0, -10.0],  # Only 5 elements
        }

        # Currently this will pass, but in the future we might want validation
        spatial = Spatial(**data)
        assert len(spatial.transform) == 5  # Current behavior

        # Future: might want to validate for exactly 6 elements for affine
        # with pytest.raises(ValidationError):
        #     Spatial(**data)
