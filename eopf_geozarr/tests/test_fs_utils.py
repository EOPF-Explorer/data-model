"""Tests for filesystem utilities."""

from unittest.mock import Mock, patch

import pytest

from eopf_geozarr.conversion.fs_utils import (
    create_s3_store,
    get_s3_credentials_info,
    get_s3_storage_options,
    get_storage_options,
    is_s3_path,
    normalize_path,
    parse_s3_path,
    path_exists,
    read_json_metadata,
    validate_s3_access,
    write_json_metadata,
)


def test_is_s3_path():
    """Test S3 path detection."""
    assert is_s3_path("s3://bucket/path")
    assert is_s3_path("s3://my-bucket/data/file.zarr")
    assert not is_s3_path("/local/path")
    assert not is_s3_path("https://example.com")
    assert not is_s3_path("gs://bucket/path")


def test_parse_s3_path():
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

    with pytest.raises(ValueError):
        parse_s3_path("https://example.com")


def test_get_s3_credentials_info():
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
        assert creds["aws_access_key_id"] == "test-key"
        assert creds["aws_secret_access_key"] == "***"
        assert creds["aws_default_region"] == "us-west-2"


@patch("eopf_geozarr.conversion.fs_utils.s3fs.S3FileSystem")
def test_validate_s3_access_success(mock_s3fs):
    """Test successful S3 access validation."""
    mock_fs = Mock()
    mock_fs.ls.return_value = ["file1", "file2"]
    mock_s3fs.return_value = mock_fs

    success, error = validate_s3_access("s3://test-bucket/path")
    assert success is True
    assert error is None
    mock_fs.ls.assert_called_once_with("s3://test-bucket", detail=False)


@patch("eopf_geozarr.conversion.fs_utils.s3fs.S3FileSystem")
def test_validate_s3_access_failure(mock_s3fs):
    """Test failed S3 access validation."""
    mock_fs = Mock()
    mock_fs.ls.side_effect = Exception("Access denied")
    mock_s3fs.return_value = mock_fs

    success, error = validate_s3_access("s3://test-bucket/path")
    assert success is False
    assert "Access denied" in error


def test_create_s3_store_path_handling():
    """Test that create_s3_store returns the S3 path correctly."""
    # Test with S3 path
    result = create_s3_store("s3://test-bucket/path/to/data")
    assert result == "s3://test-bucket/path/to/data"

    # Test with bucket only
    result = create_s3_store("s3://test-bucket")
    assert result == "s3://test-bucket"


def test_get_s3_storage_options():
    """Test that get_s3_storage_options returns correct configuration."""
    with patch.dict(
        "os.environ",
        {"AWS_DEFAULT_REGION": "us-west-2", "AWS_S3_ENDPOINT": "https://s3.example.com"},
    ):
        options = get_s3_storage_options("s3://test-bucket/path")

        assert options["anon"] is False
        assert options["use_ssl"] is True
        assert options["client_kwargs"]["region_name"] == "us-west-2"
        assert options["endpoint_url"] == "https://s3.example.com"
        assert options["client_kwargs"]["endpoint_url"] == "https://s3.example.com"


def test_get_storage_options():
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


def test_normalize_path():
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
def test_path_exists(mock_get_filesystem):
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
def test_write_json_metadata(mock_get_filesystem):
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
def test_read_json_metadata(mock_get_filesystem):
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
