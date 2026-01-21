"""Tests for filesystem utilities."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr

from eopf_geozarr.conversion.fs_utils import (
    get_s3_credentials_info,
    get_s3_storage_options,
    get_storage_options,
    is_s3_path,
    normalize_path,
    parse_s3_path,
    path_exists,
    read_json_metadata,
    replace_json_invalid_floats,
    sanitize_dataset_attributes,
    validate_s3_access,
    write_json_metadata,
)


def test_is_s3_path() -> None:
    """Test S3 path detection."""
    assert is_s3_path("s3://bucket/path")
    assert is_s3_path("s3://my-bucket/data/file.zarr")
    assert not is_s3_path("/local/path")
    assert not is_s3_path("https://example.com")
    assert not is_s3_path("gs://bucket/path")


def test_parse_s3_path() -> None:
    """Test S3 path parsing."""
    bucket, key = parse_s3_path("s3://my-bucket/data/file.zarr")
    assert bucket == "my-bucket"
    assert key == "data/file.zarr"

    bucket, key = parse_s3_path("s3://bucket/")
    assert bucket == "bucket"
    assert key == ""

    bucket, key = parse_s3_path("s3://bucket/single-file")
    assert bucket == "bucket"
    assert key == "single-file"

    with pytest.raises(ValueError, match=r"Invalid S3 path"):
        parse_s3_path("https://example.com")


def test_get_s3_credentials_info() -> None:
    """Test S3 credentials info retrieval."""
    with patch.dict(
        "os.environ",
        {
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "AWS_DEFAULT_REGION": "us-west-2",
        },
    ):
        creds = get_s3_credentials_info()
        assert creds["aws_access_key_id"] == "***"
        assert creds["aws_secret_access_key"] == "***"
        assert creds["aws_default_region"] == "us-west-2"


@patch("eopf_geozarr.conversion.fs_utils.s3fs.S3FileSystem")
def test_validate_s3_access_success(mock_s3fs) -> None:
    """Test successful S3 access validation."""
    mock_fs = Mock()
    mock_fs.ls.return_value = ["file1", "file2"]
    mock_s3fs.return_value = mock_fs

    success, error = validate_s3_access("s3://test-bucket/path")
    assert success is True
    assert error is None
    mock_fs.ls.assert_called_once_with("s3://test-bucket", detail=False)


@patch("eopf_geozarr.conversion.fs_utils.s3fs.S3FileSystem")
def test_validate_s3_access_failure(mock_s3fs) -> None:
    """Test failed S3 access validation."""
    mock_fs = Mock()
    mock_fs.ls.side_effect = Exception("Access denied")
    mock_s3fs.return_value = mock_fs

    success, error = validate_s3_access("s3://test-bucket/path")
    assert success is False
    assert "Access denied" in error


def test_get_s3_storage_options() -> None:
    """Test that get_s3_storage_options returns correct configuration."""
    with patch.dict(
        "os.environ",
        {
            "AWS_DEFAULT_REGION": "us-west-2",
            "AWS_ENDPOINT_URL": "https://s3.example.com",
        },
    ):
        options = get_s3_storage_options("s3://test-bucket/path")

        assert options["anon"] is False
        assert options["use_ssl"] is True
        assert options["client_kwargs"]["region_name"] == "us-west-2"
        assert options["endpoint_url"] == "https://s3.example.com"
        assert options["client_kwargs"]["endpoint_url"] == "https://s3.example.com"


def test_get_storage_options() -> None:
    """Test unified storage options function."""
    # Test S3 path
    with patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"}):
        options = get_storage_options("s3://test-bucket/path")
        assert options is not None
        assert options["anon"] is False
        assert options["use_ssl"] is True
        assert options["client_kwargs"]["region_name"] == "us-west-2"

    # Test local path
    options = get_storage_options("/local/path")
    assert options is None


def test_normalize_path() -> None:
    """Test path normalization for different path types."""
    # Test S3 path with double slashes
    s3_path = "s3://bucket/path//to//file.zarr"
    normalized = normalize_path(s3_path)
    assert normalized == "s3://bucket/path/to/file.zarr"

    # Test local path with double slashes
    local_path = "/local/path//to//file"
    normalized = normalize_path(local_path)
    # os.path.normpath behavior may vary, but should remove double slashes
    assert "//" not in normalized

    # Test normal paths
    assert normalize_path("s3://bucket/normal/path") == "s3://bucket/normal/path"
    assert normalize_path("/normal/local/path") == "/normal/local/path"


@patch("eopf_geozarr.conversion.fs_utils.get_filesystem")
def test_path_exists(mock_get_filesystem) -> None:
    """Test unified path existence check."""
    mock_fs = Mock()
    mock_fs.exists.return_value = True
    mock_get_filesystem.return_value = mock_fs

    # Test path exists
    result = path_exists("s3://bucket/path")
    assert result is True
    mock_fs.exists.assert_called_once_with("s3://bucket/path")

    # Test path doesn't exist
    mock_fs.exists.return_value = False
    result = path_exists("/local/path")
    assert result is False


@patch("eopf_geozarr.conversion.fs_utils.get_filesystem")
def test_write_json_metadata(mock_get_filesystem) -> None:
    """Test unified JSON metadata writing."""
    from unittest.mock import MagicMock, mock_open

    mock_fs = Mock()
    # Create a proper context manager mock
    mock_file = mock_open()
    mock_fs.open = MagicMock(return_value=mock_file.return_value)
    mock_get_filesystem.return_value = mock_fs

    test_data = {"key": "value", "number": 42}
    write_json_metadata("s3://bucket/metadata.json", test_data)

    mock_fs.open.assert_called_once_with("s3://bucket/metadata.json", "w")
    # Check that write was called
    mock_file.return_value.write.assert_called_once()
    # Check that JSON was written
    written_content = mock_file.return_value.write.call_args[0][0]
    assert "key" in written_content
    assert "value" in written_content


@patch("eopf_geozarr.conversion.fs_utils.get_filesystem")
def test_read_json_metadata(mock_get_filesystem) -> None:
    """Test unified JSON metadata reading."""
    from unittest.mock import MagicMock, mock_open

    mock_fs = Mock()
    # Create a proper context manager mock
    mock_file = mock_open(read_data='{"key": "value", "number": 42}')
    mock_fs.open = MagicMock(return_value=mock_file.return_value)
    mock_get_filesystem.return_value = mock_fs

    result = read_json_metadata("s3://bucket/metadata.json")

    mock_fs.open.assert_called_once_with("s3://bucket/metadata.json", "r")
    assert result == {"key": "value", "number": 42}


def test_replace_json_invalid_floats() -> None:
    data: dict[str, object] = {
        "nan": float("nan"),
        "inf": float("inf"),
        "-inf": float("-inf"),
        "nested": {
            "nan": float("nan"),
            "inf": float("inf"),
            "-inf": float("-inf"),
        },
        "in_list": [float("nan"), float("inf"), float("-inf")],
    }
    expected: dict[str, object] = {
        "nan": "NaN",
        "nested": {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"},
        "in_list": ["NaN", "Infinity", "-Infinity"],
        "inf": "Infinity",
        "-inf": "-Infinity",
    }
    observed = replace_json_invalid_floats(data)
    assert observed == expected


def test_sanitize_dataset_attributes() -> None:
    """
    Check that sanitize_dataset_attributes removes invalid floats from the attributes of an
    xarray Dataset, as well as the attributes of the data variables and the coordinate variables.
    """

    # Create a dataset with NaN and Infinity values in various attribute locations
    ds = xr.Dataset(
        data_vars={
            "temperature": (
                ["x", "y"],
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                {"fill_value": float("nan"), "valid_max": float("inf")},
            ),
            "pressure": (
                ["x", "y"],
                np.array([[10.0, 20.0], [30.0, 40.0]]),
                {"fill_value": float("nan"), "valid_min": float("-inf")},
            ),
        },
        coords={
            "x": (["x"], [0, 1], {"missing_value": float("nan")}),
            "y": (["y"], [0, 1], {"scale_factor": 1.0}),
        },
        attrs={
            "global_nan": float("nan"),
            "global_inf": float("inf"),
            "nested": {"inner_nan": float("nan")},
        },
    )

    # Sanitize the dataset
    result = sanitize_dataset_attributes(ds)

    # Check dataset-level attributes
    assert result.attrs["global_nan"] == "NaN"
    assert result.attrs["global_inf"] == "Infinity"
    assert result.attrs["nested"]["inner_nan"] == "NaN"

    # Check data variable attributes
    assert result["temperature"].attrs["fill_value"] == "NaN"
    assert result["temperature"].attrs["valid_max"] == "Infinity"
    assert result["pressure"].attrs["fill_value"] == "NaN"
    assert result["pressure"].attrs["valid_min"] == "-Infinity"

    # Check coordinate attributes
    assert result.coords["x"].attrs["missing_value"] == "NaN"
    assert result.coords["y"].attrs["scale_factor"] == 1.0  # Normal float unchanged

    # Verify the original dataset was not modified
    assert ds.attrs["global_nan"] != ds.attrs["global_nan"]  # NaN != NaN
    assert ds["temperature"].attrs["fill_value"] != ds["temperature"].attrs["fill_value"]
