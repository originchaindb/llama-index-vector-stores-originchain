"""Real SDK + HTTP transport fixture; this is not live database validation."""

import json
import sqlite3

import httpx
import pytest
from originchain import OriginChain


class ApiFixture:
    def __init__(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            "CREATE TABLE doc_rows (id TEXT PRIMARY KEY, text TEXT, metadata TEXT, ref_doc_id TEXT)"
        )
        self.vectors = {}
        self.requests = []
        self.failure = None
        self.translate_only = False
        self.stale_hits = []

    def handle(self, request):
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        if self.failure:
            response = self.failure(request, body)
            if response is not None:
                return response
        if request.url.path.endswith("/rows/doc_rows"):
            self.connection.execute(
                "INSERT OR REPLACE INTO doc_rows VALUES (?, ?, ?, ?)",
                [
                    body["id"],
                    body["text"],
                    json.dumps(body["metadata"]),
                    body.get("ref_doc_id", ""),
                ],
            )
            return httpx.Response(200, json={"ok": True})
        if request.url.path.endswith("/put"):
            self.vectors[body["id"]] = body
            return httpx.Response(200, json={"ok": True})
        if request.url.path.endswith("/topk"):
            hits = [
                {"id": key, "score": 0.9 - i / 100}
                for i, (key, value) in enumerate(self.vectors.items())
                if all(
                    value.get("metadata", {}).get(k) == v for k, v in body.get("filter", {}).items()
                )
            ]
            return httpx.Response(200, json=(self.stale_hits + hits)[: body["k"]])
        if request.url.path.endswith("/delete-bulk"):
            deleted = sum(self.vectors.pop(key, None) is not None for key in set(body["ids"]))
            return httpx.Response(
                200,
                json={"deleted_count": deleted, "missing_count": len(set(body["ids"])) - deleted},
            )
        if request.url.path.endswith("/sql"):
            sql = body["sql"]
            if sql.startswith("DELETE") and self.translate_only:
                return httpx.Response(
                    200, json={"kind": "delete", "schema": "doc_rows", "pk": "translated"}
                )
            params = {str(i): value for i, value in enumerate(body.get("params", []), 1)}
            cursor = self.connection.execute(sql, params)
            if sql.startswith("SELECT"):
                rows = [dict(row) for row in cursor.fetchall()]
                for row in rows:
                    if "metadata" in row:
                        row["metadata"] = json.loads(row["metadata"])
                return httpx.Response(200, json={"kind": "select", "rows": rows})
            return httpx.Response(
                200,
                json={
                    "kind": "delete",
                    "schema": "doc_rows",
                    "pk": "deleted",
                    "rows_affected": cursor.rowcount,
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")


@pytest.fixture
def api():
    state = ApiFixture()
    client = OriginChain(
        base_url="https://test.invalid", tenant="test", bearer="test-only", max_retries=0
    )
    client._client.close()
    client._client = httpx.Client(
        base_url=client.base_url, transport=httpx.MockTransport(state.handle)
    )
    yield client, state
    client.close()
    state.connection.close()
