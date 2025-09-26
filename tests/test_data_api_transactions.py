from __future__ import annotations

import importlib
from datetime import UTC, datetime

import httpx
import pystac
import pytest

transactions = importlib.import_module("eopf_geozarr.data_api.transactions")
TransactionsClient = transactions.TransactionsClient
ensure_collection = transactions.ensure_collection


def _collection(collection_id: str) -> pystac.Collection:
    return pystac.Collection(
        id=collection_id,
        description=f"Collection {collection_id}",
        extent=pystac.Extent(
            spatial=pystac.SpatialExtent([[-180.0, -90.0, 180.0, 90.0]]),
            temporal=pystac.TemporalExtent([[None, None]]),
        ),
        title=f"Collection {collection_id}",
        license="provisional",
    )


def _item(collection_id: str, item_id: str) -> pystac.Item:
    return pystac.Item(
        id=item_id,
        geometry=None,
        bbox=None,
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
        properties={},
        stac_extensions=[],
        collection=collection_id,
    )


def _transport(responses: list[tuple[str, str, httpx.Response]]):
    remaining = responses.copy()

    def handler(request: httpx.Request) -> httpx.Response:
        if not remaining:
            raise AssertionError(f"Unexpected request: {request.method} {request.url}")
        method, url, response = remaining.pop(0)
        assert request.method == method
        assert str(request.url) == url
        return response

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_ensure_collection_returns_existing() -> None:
    base_url = "https://example.com/stac"
    responses = [
        (
            "GET",
            f"{base_url}/collections/demo",
            httpx.Response(200, json=_collection("demo").to_dict()),
        )
    ]
    client = TransactionsClient(base_url, transport=_transport(responses))
    collection = await ensure_collection(client, "demo", True, None, None)
    assert collection.id == "demo"


@pytest.mark.asyncio
async def test_ensure_collection_creates_when_missing() -> None:
    base_url = "https://example.com/stac"
    responses = [
        ("GET", f"{base_url}/collections/new", httpx.Response(404)),
        (
            "POST",
            f"{base_url}/collections",
            httpx.Response(201, json=_collection("new").to_dict()),
        ),
    ]
    client = TransactionsClient(base_url, transport=_transport(responses))
    collection = await ensure_collection(client, "new", True, "New", "Created")
    assert collection.id == "new"


@pytest.mark.asyncio
async def test_create_and_get_item_roundtrip() -> None:
    base_url = "https://example.com/stac"
    responses = [
        (
            "POST",
            f"{base_url}/collections/demo/items",
            httpx.Response(201, json=_item("demo", "item-1").to_dict()),
        ),
        (
            "GET",
            f"{base_url}/collections/demo/items/item-1",
            httpx.Response(200, json=_item("demo", "item-1").to_dict()),
        ),
    ]
    client = TransactionsClient(base_url, transport=_transport(responses))
    item = await client.create_item(_item("demo", "item-1"), "demo")
    assert item.id == "item-1"
    fetched = await client.get_item("demo", "item-1")
    assert fetched and fetched.id == "item-1"


@pytest.mark.asyncio
async def test_delete_item_handles_missing() -> None:
    base_url = "https://example.com/stac"
    responses = [
        (
            "DELETE",
            f"{base_url}/collections/demo/items/item-404",
            httpx.Response(404),
        )
    ]
    client = TransactionsClient(base_url, transport=_transport(responses))
    deleted = await client.delete_item("demo", "item-404")
    assert deleted is False
