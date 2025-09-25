from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import fsspec
import typer
import xarray as xr

from eopf_geozarr import create_geozarr_dataset
from eopf_geozarr.conversion.fs_utils import get_storage_options, is_s3_path

from .models import GeoZarrPayload

app = typer.Typer(help="Run GeoZarr conversion workflows driven by AMQP payloads")


def _configure_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("eopf_geozarr.pipeline")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger


def _resolve_zarr_root(src_item: str, logger: logging.Logger) -> str:
    candidate_paths = [
        Path("/app/scripts/override/resolve_zarr_from_item.py"),
        Path("/app/scripts/resolve_zarr_from_item.py"),
    ]
    for path in candidate_paths:
        if path.is_file():
            logger.debug("Using resolver script %s", path)
            result = subprocess.run(
                [sys.executable, str(path), "--src-item", src_item],
                check=True,
                capture_output=True,
                text=True,
            )
            output = result.stdout.strip()
            if not output:
                raise RuntimeError("Resolver script returned empty output.")
            return output
    raise FileNotFoundError("resolve_zarr_from_item.py not found in expected locations")


def _maybe_start_dask(enable: bool, logger: logging.Logger) -> Optional[Any]:
    if not enable:
        return None
    try:
        from dask.distributed import Client

        client = Client()
        logger.info("Dask cluster started: dashboard=%s", client.dashboard_link)
        return client
    except Exception as exc:  # pragma: no cover - dask optional at runtime
        logger.warning("Failed to start Dask cluster (%s). Proceeding without it.", exc)
        return None


def _configure_aws_environment(
    payload: GeoZarrPayload, zarr_root: str, logger: logging.Logger
) -> None:
    if payload.s3_endpoint:
        logger.debug("Using explicit S3 endpoint %s", payload.s3_endpoint)
        for key in ("AWS_S3_ENDPOINT", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3"):
            os.environ[key] = payload.s3_endpoint
    else:
        host = urlparse(zarr_root).netloc
        if host.startswith("s3.") and host.endswith(".io.cloud.ovh.net"):
            derived = f"https://{host}"
            logger.debug("Derived OVH endpoint %s from %s", derived, zarr_root)
            for key in ("AWS_S3_ENDPOINT", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3"):
                os.environ[key] = derived
            parts = host.split(".")
            if len(parts) >= 2:
                derived_region = parts[1]
                os.environ["AWS_DEFAULT_REGION"] = derived_region
                os.environ["AWS_REGION"] = derived_region
    os.environ.setdefault("AWS_DEFAULT_REGION", payload.s3_region)
    os.environ.setdefault("AWS_REGION", payload.s3_region)
    os.environ["AWS_S3_ADDRESSING_STYLE"] = payload.aws_addressing_style
    if payload.aws_session_token:
        os.environ["AWS_SESSION_TOKEN"] = payload.aws_session_token


def _drop_existing_store(output_path: str, logger: logging.Logger) -> None:
    if is_s3_path(output_path):
        fs = fsspec.filesystem("s3")
        if fs.exists(output_path):
            logger.info("Removing existing S3 store %s", output_path)
            fs.rm(output_path, recursive=True)
    else:
        path = Path(output_path)
        if path.exists():
            logger.info("Removing existing local store %s", path)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _run_conversion(
    payload: GeoZarrPayload, zarr_root: str, logger: logging.Logger
) -> None:
    storage_options = get_storage_options(zarr_root)
    logger.debug("Opening source DataTree %s", zarr_root)
    dt = xr.open_datatree(
        zarr_root, engine="zarr", chunks="auto", storage_options=storage_options
    )
    try:
        logger.info("Starting GeoZarr conversion to %s", payload.output_zarr)
        logger.debug("Payload context: %s", json.dumps(payload.as_log_dict()))
        create_geozarr_dataset(
            dt_input=dt,
            groups=list(payload.groups),
            output_path=payload.output_zarr,
            spatial_chunk=payload.spatial_chunk,
            min_dimension=payload.min_dimension,
            tile_width=payload.tile_width,
            max_retries=payload.max_retries,
            crs_groups=list(payload.crs_groups),
        )
        logger.info("Conversion completed: %s", payload.output_zarr)
    finally:
        try:
            dt.close()
        except AttributeError:  # DataTree close not available on older xarray
            pass


def run_pipeline(payload: GeoZarrPayload) -> None:
    payload.ensure_required()
    logger = _configure_logger(payload.verbose)
    logger.debug("Payload: %s", payload.as_log_dict())

    zarr_root = _resolve_zarr_root(payload.src_item, logger)
    logger.info("Resolved Zarr input: %s", zarr_root)

    _configure_aws_environment(payload, zarr_root, logger)

    if payload.overwrite == "replace":
        _drop_existing_store(payload.output_zarr, logger)

    dask_client = _maybe_start_dask(payload.dask_cluster, logger)
    try:
        _run_conversion(payload, zarr_root, logger)
    finally:
        if dask_client:
            logger.debug("Closing Dask client")
            try:
                dask_client.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning("Error while closing Dask client: %s", exc)


@app.command()  # type: ignore[misc]
def run(  # noqa: PLR0913 - CLI mirrors WorkflowTemplate parameters
    src_item: str = typer.Option(..., help="STAC item URL"),
    output_zarr: str = typer.Option(
        ..., help="Target GeoZarr store (s3:// or local path)"
    ),
    groups: Optional[str] = typer.Option(None, help="Comma-separated group paths"),
    crs_groups: Optional[str] = typer.Option(
        None, help="Comma-separated CRS group paths"
    ),
    register_url: Optional[str] = typer.Option(None),
    register_collection: Optional[str] = typer.Option(None),
    register_bearer_token: Optional[str] = typer.Option(None),
    register_href: Optional[str] = typer.Option(None),
    register_mode: Optional[str] = typer.Option(None),
    id_policy: Optional[str] = typer.Option(None),
    s3_endpoint: Optional[str] = typer.Option(None),
    s3_region: Optional[str] = typer.Option(None),
    aws_addressing_style: Optional[str] = typer.Option(None),
    max_retries: Optional[str] = typer.Option(None),
    overwrite: Optional[str] = typer.Option(None),
    metrics_out: Optional[str] = typer.Option(None),
    spatial_chunk: Optional[str] = typer.Option(None),
    min_dimension: Optional[str] = typer.Option(None),
    tile_width: Optional[str] = typer.Option(None),
    dask_cluster: Optional[str] = typer.Option(None),
    verbose: Optional[str] = typer.Option(None),
    aws_session_token: Optional[str] = typer.Option(None),
    collection_thumbnail: Optional[str] = typer.Option(None),
    owner: Optional[str] = typer.Option(None),
    service_account: Optional[str] = typer.Option(None),
) -> None:
    payload = GeoZarrPayload.from_cli(
        src_item=src_item,
        output_zarr=output_zarr,
        groups=groups,
        crs_groups=crs_groups,
        register_url=register_url,
        register_collection=register_collection,
        register_bearer_token=register_bearer_token,
        register_href=register_href,
        register_mode=register_mode,
        id_policy=id_policy,
        s3_endpoint=s3_endpoint,
        s3_region=s3_region,
        aws_addressing_style=aws_addressing_style,
        max_retries=max_retries,
        overwrite=overwrite,
        metrics_out=metrics_out,
        spatial_chunk=spatial_chunk,
        min_dimension=min_dimension,
        tile_width=tile_width,
        dask_cluster=dask_cluster,
        verbose=verbose,
        aws_session_token=aws_session_token,
        collection_thumbnail=collection_thumbnail,
        owner=owner,
        service_account=service_account,
    )
    run_pipeline(payload)


if __name__ == "__main__":  # pragma: no cover
    app()
