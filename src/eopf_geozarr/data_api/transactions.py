"""Helpers for interacting with STAC Transactions endpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import httpx
import pystac


def _merge_json_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    base: dict[str, str] = {"Content-Type": "application/json"}
    if headers:
        base.update(headers)
    return base


@dataclass(slots=True)
class TransactionsClient:
    """Async convenience wrapper for a subset of the STAC Transactions API."""

    base_url: str
    headers: Mapping[str, str] | None = None
    timeout: float = 30.0
    transport: httpx.AsyncBaseTransport | None = None

    def _collections_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/collections"

    def _items_url(self, collection_id: str) -> str:
        return f"{self._collections_url()}/{collection_id}/items"

    def _client(self, *, headers: Mapping[str, str] | None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self.timeout,
            headers=headers,
            transport=self.transport,
        )

    async def get_collection(self, collection_id: str) -> pystac.Collection | None:
        url = f"{self._collections_url()}/{collection_id}"
        async with self._client(headers=self.headers) as client:
            response = await client.get(url)
        if response.status_code == 200:
            try:
                return pystac.Collection.from_dict(response.json())
            except Exception:  # pragma: no cover - fallback guard
                return None
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return None

    async def create_collection(
        self, collection: pystac.Collection
    ) -> pystac.Collection:
        async with self._client(headers=_merge_json_headers(self.headers)) as client:
            response = await client.post(
                self._collections_url(), json=collection.to_dict()
            )
        if response.status_code in (200, 201):
            try:
                return pystac.Collection.from_dict(response.json())
            except Exception:  # pragma: no cover - fallback guard
                return collection
        if response.status_code == 409:
            return collection
        response.raise_for_status()
        return collection

    async def create_item(self, item: pystac.Item, collection_id: str) -> pystac.Item:
        url = self._items_url(collection_id)
        async with self._client(headers=_merge_json_headers(self.headers)) as client:
            response = await client.post(url, json=item.to_dict())
        if response.status_code in (200, 201):
            try:
                return pystac.Item.from_dict(response.json())
            except Exception:  # pragma: no cover - fallback guard
                return item
        if response.status_code == 409:
            return item
        response.raise_for_status()
        return item

    async def get_item(self, collection_id: str, item_id: str) -> pystac.Item | None:
        url = f"{self._items_url(collection_id)}/{item_id}"
        async with self._client(headers=self.headers) as client:
            response = await client.get(url)
        if response.status_code == 200:
            try:
                return pystac.Item.from_dict(response.json())
            except Exception:  # pragma: no cover - fallback guard
                return None
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return None

    async def update_item(self, item: pystac.Item, collection_id: str) -> pystac.Item:
        url = f"{self._items_url(collection_id)}/{item.id}"
        async with self._client(headers=_merge_json_headers(self.headers)) as client:
            response = await client.put(url, json=item.to_dict())
        if response.status_code in (200, 201):
            try:
                return pystac.Item.from_dict(response.json())
            except Exception:  # pragma: no cover - fallback guard
                return item
        response.raise_for_status()
        return item

    async def delete_item(self, collection_id: str, item_id: str) -> bool:
        url = f"{self._items_url(collection_id)}/{item_id}"
        async with self._client(headers=self.headers) as client:
            response = await client.delete(url)
        if response.status_code in (200, 204):
            return True
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return False


async def ensure_collection(
    client: TransactionsClient,
    collection_id: str,
    allow_create: bool,
    title: str | None,
    description: str | None,
) -> pystac.Collection:
    existing = await client.get_collection(collection_id)
    if existing:
        return existing
    if not allow_create:
        msg = f"Collection '{collection_id}' not found and auto-create disabled"
        raise RuntimeError(msg)
    collection = pystac.Collection(
        id=collection_id,
        description=description or f"Auto-created collection {collection_id}",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
            temporal=pystac.TemporalExtent([[None, None]]),
        ),
        title=title or collection_id,
        license="provisional",
    )
    return await client.create_collection(collection)
