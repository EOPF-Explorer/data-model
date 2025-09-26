from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

DEFAULT_GROUPS: tuple[str, str, str] = (
    "/measurements/reflectance/r10m",
    "/measurements/reflectance/r20m",
    "/measurements/reflectance/r60m",
)


def _split_csv(values: str | Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw = values
    else:
        raw = ",".join(values)
    return tuple(
        item.strip()
        for item in raw.replace("\n", ",").replace("\t", ",").split(",")
        if item.strip()
    )


def _normalize_group_path(group: str) -> str:
    normalized = group.strip()
    if not normalized:
        return normalized
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    if normalized.startswith("/measurements/") and "/reflectance/" not in normalized:
        for suffix in ("/r10m", "/r20m", "/r60m"):
            if normalized.endswith(suffix):
                normalized = normalized.replace(
                    "/measurements/", "/measurements/reflectance/", 1
                )
                break
    return normalized


def _to_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int(value: str | int | None, default: int) -> int:
    if isinstance(value, int):
        return value
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return int(value)


def _ensure_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(v for v in values if v)


@dataclass(slots=True)
class GeoZarrPayload:
    src_item: str
    output_zarr: str
    groups: tuple[str, ...] = field(default_factory=lambda: DEFAULT_GROUPS)
    crs_groups: tuple[str, ...] = field(default_factory=tuple)
    register_url: str | None = None
    register_collection: str | None = None
    register_bearer_token: str | None = None
    register_href: str | None = None
    register_mode: str = "create-or-skip"
    id_policy: str = "src"
    s3_endpoint: str | None = None
    s3_region: str = "us-east-1"
    aws_addressing_style: str = "path"
    max_retries: int = 3
    overwrite: str = "replace"
    metrics_out: str | None = None
    spatial_chunk: int = 4096
    min_dimension: int = 256
    tile_width: int = 256
    dask_cluster: bool = False
    verbose: bool = True
    aws_session_token: str | None = None
    collection_thumbnail: str | None = None
    owner: str | None = None
    service_account: str | None = None
    scaling_strategy: str = "v0-hpa"
    worker_replicas_min: int = 2
    worker_replicas_max: int = 10
    queue_depth_target: int = 5

    @classmethod
    def from_cli(
        cls,
        *,
        src_item: str,
        output_zarr: str,
        groups: str | Iterable[str] | None = None,
        crs_groups: str | Iterable[str] | None = None,
        register_url: str | None = None,
        register_collection: str | None = None,
        register_bearer_token: str | None = None,
        register_href: str | None = None,
        register_mode: str | None = None,
        id_policy: str | None = None,
        s3_endpoint: str | None = None,
        s3_region: str | None = None,
        aws_addressing_style: str | None = None,
        max_retries: str | int | None = None,
        overwrite: str | None = None,
        metrics_out: str | None = None,
        spatial_chunk: str | int | None = None,
        min_dimension: str | int | None = None,
        tile_width: str | int | None = None,
        dask_cluster: str | bool | None = None,
        verbose: str | bool | None = None,
        aws_session_token: str | None = None,
        collection_thumbnail: str | None = None,
        owner: str | None = None,
        service_account: str | None = None,
        scaling_strategy: str | None = None,
        worker_replicas_min: str | int | None = None,
        worker_replicas_max: str | int | None = None,
        queue_depth_target: str | int | None = None,
    ) -> GeoZarrPayload:
        parsed_groups = _ensure_tuple(
            _normalize_group_path(g) for g in (_split_csv(groups) or DEFAULT_GROUPS)
        )
        parsed_crs_groups = _ensure_tuple(
            _normalize_group_path(g) for g in _split_csv(crs_groups)
        )

        return cls(
            src_item=src_item,
            output_zarr=output_zarr,
            groups=parsed_groups if parsed_groups else DEFAULT_GROUPS,
            crs_groups=parsed_crs_groups,
            register_url=register_url or None,
            register_collection=register_collection or None,
            register_bearer_token=register_bearer_token or None,
            register_href=register_href or None,
            register_mode=(register_mode or "create-or-skip"),
            id_policy=(id_policy or "src"),
            s3_endpoint=s3_endpoint or None,
            s3_region=(s3_region or "us-east-1"),
            aws_addressing_style=(aws_addressing_style or "path"),
            max_retries=_to_int(max_retries, default=3),
            overwrite=(overwrite or "replace"),
            metrics_out=metrics_out or None,
            spatial_chunk=_to_int(spatial_chunk, default=4096),
            min_dimension=_to_int(min_dimension, default=256),
            tile_width=_to_int(tile_width, default=256),
            dask_cluster=_to_bool(dask_cluster),
            verbose=_to_bool(verbose) or False,
            aws_session_token=aws_session_token or None,
            collection_thumbnail=collection_thumbnail or None,
            owner=owner or None,
            service_account=service_account or None,
            scaling_strategy=(scaling_strategy or "v0-hpa"),
            worker_replicas_min=_to_int(worker_replicas_min, default=2),
            worker_replicas_max=_to_int(worker_replicas_max, default=10),
            queue_depth_target=_to_int(queue_depth_target, default=5),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> GeoZarrPayload:
        return cls.from_cli(
            src_item=str(payload.get("src_item", "")),
            output_zarr=str(payload.get("output_zarr", "")),
            groups=payload.get("groups"),
            crs_groups=payload.get("crs_groups"),
            register_url=payload.get("register_url"),
            register_collection=payload.get("register_collection")
            or payload.get("collection"),
            register_bearer_token=(
                payload.get("register_bearer_token") or payload.get("bearer_token")
            ),
            register_href=payload.get("register_href"),
            register_mode=payload.get("register_mode"),
            id_policy=payload.get("id_policy"),
            s3_endpoint=payload.get("s3_endpoint"),
            s3_region=payload.get("s3_region"),
            aws_addressing_style=payload.get("aws_addressing_style"),
            max_retries=payload.get("max_retries"),
            overwrite=payload.get("overwrite"),
            metrics_out=payload.get("metrics_out"),
            spatial_chunk=payload.get("spatial_chunk"),
            min_dimension=payload.get("min_dimension"),
            tile_width=payload.get("tile_width"),
            dask_cluster=payload.get("dask_cluster"),
            verbose=payload.get("verbose"),
            aws_session_token=payload.get("aws_session_token"),
            collection_thumbnail=payload.get("collection_thumbnail"),
            owner=payload.get("owner"),
            service_account=payload.get("service_account"),
            scaling_strategy=payload.get("scaling_strategy"),
            worker_replicas_min=payload.get("worker_replicas_min"),
            worker_replicas_max=payload.get("worker_replicas_max"),
            queue_depth_target=payload.get("queue_depth_target"),
        )

    def ensure_required(self) -> None:
        if not self.src_item:
            raise ValueError("src_item is required")
        if not self.output_zarr:
            raise ValueError("output_zarr is required")

    def as_log_dict(self) -> dict[str, str | int | bool | None]:
        return {
            "src_item": self.src_item,
            "output_zarr": self.output_zarr,
            "groups": ",".join(self.groups),
            "crs_groups": ",".join(self.crs_groups),
            "s3_endpoint": self.s3_endpoint,
            "s3_region": self.s3_region,
            "overwrite": self.overwrite,
            "max_retries": self.max_retries,
            "dask_cluster": self.dask_cluster,
            "verbose": self.verbose,
        }

    def to_payload(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "src_item": self.src_item,
            "output_zarr": self.output_zarr,
            "groups": ",".join(self.groups),
        }
        if self.crs_groups:
            data["crs_groups"] = ",".join(self.crs_groups)
        optional_fields: tuple[tuple[str, str], ...] = (
            ("register_url", "register_url"),
            ("register_collection", "register_collection"),
            ("register_bearer_token", "register_bearer_token"),
            ("register_href", "register_href"),
            ("register_mode", "register_mode"),
            ("id_policy", "id_policy"),
            ("s3_endpoint", "s3_endpoint"),
            ("s3_region", "s3_region"),
            ("aws_addressing_style", "aws_addressing_style"),
            ("max_retries", "max_retries"),
            ("overwrite", "overwrite"),
            ("metrics_out", "metrics_out"),
            ("spatial_chunk", "spatial_chunk"),
            ("min_dimension", "min_dimension"),
            ("tile_width", "tile_width"),
            ("dask_cluster", "dask_cluster"),
            ("verbose", "verbose"),
            ("aws_session_token", "aws_session_token"),
            ("collection_thumbnail", "collection_thumbnail"),
            ("owner", "owner"),
            ("service_account", "service_account"),
            ("scaling_strategy", "scaling_strategy"),
            ("worker_replicas_min", "worker_replicas_min"),
            ("worker_replicas_max", "worker_replicas_max"),
            ("queue_depth_target", "queue_depth_target"),
        )
        for attr, key in optional_fields:
            value = getattr(self, attr)
            if value not in (None, "", (), False):
                data[key] = value
        if "register_collection" in data:
            data.setdefault("collection", data["register_collection"])
        return data
