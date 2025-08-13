"""S3 utilities for GeoZarr conversion."""

import json
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import s3fs
import zarr


def normalize_s3_path(s3_path: str) -> str:
    """
    Normalize an S3 path by removing double slashes and ensuring proper format.

    This is important for OVH S3 which is sensitive to double slashes.

    Parameters
    ----------
    s3_path : str
        S3 path to normalize

    Returns
    -------
    str
        Normalized S3 path
    """
    if not s3_path.startswith("s3://"):
        return s3_path

    # Split into scheme and path parts
    scheme = "s3://"
    path_part = s3_path[5:]  # Remove "s3://"

    # Remove double slashes from the path part
    # But preserve the bucket/key structure
    parts = path_part.split("/")
    # Filter out empty parts (which come from double slashes)
    normalized_parts = [part for part in parts if part]

    # Reconstruct the path
    if normalized_parts:
        normalized_path = scheme + "/".join(normalized_parts)
    else:
        normalized_path = scheme

    return normalized_path


def is_s3_path(path: str) -> bool:
    """
    Check if a path is an S3 URL.

    Parameters
    ----------
    path : str
        Path to check

    Returns
    -------
    bool
        True if the path is an S3 URL
    """
    return path.startswith("s3://")


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """
    Parse an S3 path into bucket and key components.

    Parameters
    ----------
    s3_path : str
        S3 path in format s3://bucket/key

    Returns
    -------
    tuple[str, str]
        Tuple of (bucket, key)
    """
    parsed = urlparse(s3_path)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 path: {s3_path}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    return bucket, key


def get_s3_storage_options(s3_path: str, **s3_kwargs: Any) -> Dict[str, Any]:
    """
    Get storage options for S3 access with xarray.

    Parameters
    ----------
    s3_path : str
        S3 path in format s3://bucket/key
    **s3_kwargs
        Additional keyword arguments for s3fs.S3FileSystem

    Returns
    -------
    Dict[str, Any]
        Storage options dictionary for xarray
    """
    # Set up default S3 configuration
    default_s3_kwargs = {
        "anon": False,  # Use credentials
        "use_ssl": True,
        "client_kwargs": {
            "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        },
    }

    # Add custom endpoint support (e.g., for OVH Cloud)
    if "AWS_ENDPOINT_URL" in os.environ:
        default_s3_kwargs["endpoint_url"] = os.environ["AWS_ENDPOINT_URL"]
        client_kwargs = default_s3_kwargs.get("client_kwargs")
        if isinstance(client_kwargs, dict):
            client_kwargs["endpoint_url"] = os.environ["AWS_ENDPOINT_URL"]

    # Merge with user-provided kwargs
    s3_config = {**default_s3_kwargs, **s3_kwargs}

    return s3_config


def get_storage_options(path: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
    """
    Get storage options for any URL type, leveraging fsspec as the abstraction layer.

    This function eliminates the need for if/else branching by returning appropriate
    storage options based on the URL protocol.

    Parameters
    ----------
    path : str
        Path or URL (local path, s3://, etc.)
    **kwargs
        Additional keyword arguments for the storage backend

    Returns
    -------
    Optional[Dict[str, Any]]
        Storage options dictionary for xarray/zarr, or None for local paths
    """
    if is_s3_path(path):
        return get_s3_storage_options(path, **kwargs)
    # For local paths, return None (no storage options needed)
    # Future protocols (gcs://, azure://, etc.) can be added here
    return None


def normalize_path(path: str) -> str:
    """
    Normalize any path type (local or remote URL).

    This function handles path normalization for all filesystem types,
    ensuring proper path formatting and removing issues like double slashes.

    Parameters
    ----------
    path : str
        Path to normalize

    Returns
    -------
    str
        Normalized path
    """
    if is_s3_path(path):
        return normalize_s3_path(path)
    else:
        # For local paths, normalize by removing double slashes and cleaning up
        import os.path

        return os.path.normpath(path)


def create_s3_store(s3_path: str, **s3_kwargs: Any) -> str:
    """
    Create an S3 path with storage options for Zarr operations.

    This function now returns the S3 path directly, to be used with
    xarray's storage_options parameter instead of creating a store.

    Parameters
    ----------
    s3_path : str
        S3 path in format s3://bucket/key
    **s3_kwargs
        Additional keyword arguments for s3fs.S3FileSystem

    Returns
    -------
    str
        S3 path to be used with storage_options
    """
    # Just return the S3 path - storage options will be handled separately
    return s3_path


def write_s3_json_metadata(
    s3_path: str, metadata: Dict[str, Any], **s3_kwargs: Any
) -> None:
    """
    Write JSON metadata directly to S3.

    This is used for writing zarr.json files and other metadata that need
    to be written directly to S3 without going through the Zarr store.

    Parameters
    ----------
    s3_path : str
        S3 path for the JSON file
    metadata : dict
        Metadata dictionary to write as JSON
    **s3_kwargs
        Additional keyword arguments for s3fs.S3FileSystem
    """
    # Set up default S3 configuration
    default_s3_kwargs = {
        "anon": False,
        "use_ssl": True,
        "asynchronous": False,  # Force synchronous mode
        "client_kwargs": {
            "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        },
    }

    # Add custom endpoint support (e.g., for OVH Cloud)
    if "AWS_ENDPOINT_URL" in os.environ:
        default_s3_kwargs["endpoint_url"] = os.environ["AWS_ENDPOINT_URL"]
        client_kwargs = default_s3_kwargs.get("client_kwargs")
        if isinstance(client_kwargs, dict):
            client_kwargs["endpoint_url"] = os.environ["AWS_ENDPOINT_URL"]

    s3_config = {**default_s3_kwargs, **s3_kwargs}
    fs = s3fs.S3FileSystem(**s3_config)

    # Write JSON content
    json_content = json.dumps(metadata, indent=2)
    with fs.open(s3_path, "w") as f:
        f.write(json_content)


def read_s3_json_metadata(s3_path: str, **s3_kwargs: Any) -> Dict[str, Any]:
    """
    Read JSON metadata from S3.

    Parameters
    ----------
    s3_path : str
        S3 path for the JSON file
    **s3_kwargs
        Additional keyword arguments for s3fs.S3FileSystem

    Returns
    -------
    dict
        Parsed JSON metadata
    """
    # Set up default S3 configuration
    default_s3_kwargs = {
        "anon": False,
        "use_ssl": True,
        "asynchronous": False,  # Force synchronous mode
        "client_kwargs": {
            "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        },
    }

    # Add custom endpoint support (e.g., for OVH Cloud)
    if "AWS_ENDPOINT_URL" in os.environ:
        default_s3_kwargs["endpoint_url"] = os.environ["AWS_ENDPOINT_URL"]
        client_kwargs = default_s3_kwargs.get("client_kwargs")
        if isinstance(client_kwargs, dict):
            client_kwargs["endpoint_url"] = os.environ["AWS_ENDPOINT_URL"]

    s3_config = {**default_s3_kwargs, **s3_kwargs}
    fs = s3fs.S3FileSystem(**s3_config)

    with fs.open(s3_path, "r") as f:
        content = f.read()

    result: Dict[str, Any] = json.loads(content)
    return result


def s3_path_exists(s3_path: str, **s3_kwargs: Any) -> bool:
    """
    Check if an S3 path exists.

    Parameters
    ----------
    s3_path : str
        S3 path to check
    **s3_kwargs
        Additional keyword arguments for s3fs.S3FileSystem

    Returns
    -------
    bool
        True if the path exists
    """
    default_s3_kwargs = {
        "anon": False,
        "use_ssl": True,
        "asynchronous": False,  # Force synchronous mode
        "client_kwargs": {
            "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        },
    }

    # Add custom endpoint support (e.g., for OVH Cloud)
    if "AWS_ENDPOINT_URL" in os.environ:
        default_s3_kwargs["endpoint_url"] = os.environ["AWS_ENDPOINT_URL"]
        client_kwargs = default_s3_kwargs.get("client_kwargs")
        if isinstance(client_kwargs, dict):
            client_kwargs["endpoint_url"] = os.environ["AWS_ENDPOINT_URL"]

    s3_config = {**default_s3_kwargs, **s3_kwargs}
    fs = s3fs.S3FileSystem(**s3_config)

    result: bool = fs.exists(s3_path)
    return result


def open_s3_zarr_group(s3_path: str, mode: str = "r", **s3_kwargs: Any) -> zarr.Group:
    """
    Open a Zarr group from S3 using storage_options.

    Parameters
    ----------
    s3_path : str
        S3 path to the Zarr group
    mode : str, default "r"
        Access mode ("r", "r+", "w", "a")
    **s3_kwargs
        Additional keyword arguments for s3fs.S3FileSystem

    Returns
    -------
    zarr.Group
        Zarr group
    """
    storage_options = get_s3_storage_options(s3_path, **s3_kwargs)
    return zarr.open_group(
        s3_path, mode=mode, zarr_format=3, storage_options=storage_options
    )


def get_s3_credentials_info() -> Dict[str, Optional[str]]:
    """
    Get information about available S3 credentials.

    Returns
    -------
    dict
        Dictionary with credential information
    """
    return {
        "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": "***"
        if os.environ.get("AWS_SECRET_ACCESS_KEY")
        else None,
        "aws_session_token": "***" if os.environ.get("AWS_SESSION_TOKEN") else None,
        "aws_default_region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "aws_profile": os.environ.get("AWS_PROFILE"),
        "AWS_ENDPOINT_URL": os.environ.get("AWS_ENDPOINT_URL"),
    }


def validate_s3_access(s3_path: str, **s3_kwargs: Any) -> tuple[bool, Optional[str]]:
    """
    Validate that we can access the S3 path.

    Parameters
    ----------
    s3_path : str
        S3 path to validate
    **s3_kwargs
        Additional keyword arguments for s3fs.S3FileSystem

    Returns
    -------
    tuple[bool, Optional[str]]
        Tuple of (success, error_message)
    """
    try:
        bucket, key = parse_s3_path(s3_path)

        default_s3_kwargs = {
            "anon": False,
            "use_ssl": True,
            "asynchronous": False,  # Force synchronous mode
            "client_kwargs": {
                "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            },
        }

        # Add custom endpoint support (e.g., for OVH Cloud)
        if "AWS_ENDPOINT_URL" in os.environ:
            default_s3_kwargs["endpoint_url"] = os.environ["AWS_ENDPOINT_URL"]
            client_kwargs = default_s3_kwargs.get("client_kwargs")
            if isinstance(client_kwargs, dict):
                client_kwargs["endpoint_url"] = os.environ["AWS_ENDPOINT_URL"]

        s3_config = {**default_s3_kwargs, **s3_kwargs}
        fs = s3fs.S3FileSystem(**s3_config)

        # Try to list the bucket to check access
        fs.ls(f"s3://{bucket}", detail=False)

        return True, None

    except Exception as e:
        return False, str(e)


def get_filesystem(path: str, **kwargs: Any) -> Any:
    """
    Get the appropriate fsspec filesystem for any path type.

    Parameters
    ----------
    path : str
        Path or URL (local path, s3://, etc.)
    **kwargs
        Additional keyword arguments for the filesystem

    Returns
    -------
    fsspec.AbstractFileSystem
        Filesystem instance
    """
    import fsspec

    if is_s3_path(path):
        # Get S3 storage options and use them for fsspec
        storage_options = get_s3_storage_options(path, **kwargs)
        return fsspec.filesystem("s3", **storage_options)
    else:
        # For local paths, use the local filesystem
        return fsspec.filesystem("file")


def write_json_metadata(path: str, metadata: Dict[str, Any], **kwargs: Any) -> None:
    """
    Write JSON metadata to any path type using fsspec.

    Parameters
    ----------
    path : str
        Path where to write the JSON file (local path or URL)
    metadata : dict
        Metadata dictionary to write as JSON
    **kwargs
        Additional keyword arguments for the filesystem
    """
    fs = get_filesystem(path, **kwargs)

    # Ensure parent directory exists for local paths
    if not is_s3_path(path):
        parent_dir = os.path.dirname(path)
        if parent_dir:
            fs.makedirs(parent_dir, exist_ok=True)

    # Write JSON content using fsspec
    json_content = json.dumps(metadata, indent=2)
    with fs.open(path, "w") as f:
        f.write(json_content)


def read_json_metadata(path: str, **kwargs: Any) -> Dict[str, Any]:
    """
    Read JSON metadata from any path type using fsspec.

    Parameters
    ----------
    path : str
        Path to the JSON file (local path or URL)
    **kwargs
        Additional keyword arguments for the filesystem

    Returns
    -------
    dict
        Parsed JSON metadata
    """
    fs = get_filesystem(path, **kwargs)

    with fs.open(path, "r") as f:
        content = f.read()

    result: Dict[str, Any] = json.loads(content)
    return result


def path_exists(path: str, **kwargs: Any) -> bool:
    """
    Check if a path exists using fsspec.

    Parameters
    ----------
    path : str
        Path to check (local path or URL)
    **kwargs
        Additional keyword arguments for the filesystem

    Returns
    -------
    bool
        True if the path exists
    """
    fs = get_filesystem(path, **kwargs)
    result: bool = fs.exists(path)
    return result


def open_zarr_group(path: str, mode: str = "r", **kwargs: Any) -> zarr.Group:
    """
    Open a Zarr group from any path type using unified storage options.

    Parameters
    ----------
    path : str
        Path to the Zarr group (local path or URL)
    mode : str, default "r"
        Access mode ("r", "r+", "w", "a")
    **kwargs
        Additional keyword arguments for the storage backend

    Returns
    -------
    zarr.Group
        Zarr group
    """
    storage_options = get_storage_options(path, **kwargs)
    return zarr.open_group(
        path, mode=mode, zarr_format=3, storage_options=storage_options
    )
