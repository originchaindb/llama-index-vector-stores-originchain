"""Shared adapter rules, kept private to this distribution."""

from __future__ import annotations

import math
import re
from typing import Any


def identifier(value: str) -> str:
    # The SQL parser accepts unquoted names, optionally namespace-qualified.
    # Reject syntax rather than interpolating arbitrary caller text as SQL.
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?", value
    ):
        raise ValueError("table/schema must be an ASCII SQL identifier, optionally namespace.table")
    return value


def validate_id(value: str) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("ids must be non-empty strings without NUL characters")


def validate_config(
    table: str, schema: str | None, dim: int, metric: str, mode: str | None
) -> None:
    identifier(table)
    if schema is not None:
        identifier(schema)
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise ValueError("dim must be a positive integer")
    if metric not in ("cosine", "dot", "l2"):
        raise ValueError("metric must be cosine, dot, or l2")
    if mode not in (None, "fast", "high_recall"):
        raise ValueError("mode must be fast or high_recall")


def validate_vector(vector: list[float], dim: int) -> None:
    if len(vector) != dim or any(
        isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
        for x in vector
    ):
        raise ValueError("embedding must contain dim finite numbers")


def validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def equality_filter(value: dict | None) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("filter must be a flat dictionary of scalar equality values")
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or key.startswith("$")
            or item is None
            or not isinstance(item, (str, int, float, bool))
            or (isinstance(item, float) and not math.isfinite(item))
        ):
            raise ValueError("only flat scalar equality filters are supported")
    return dict(value) or None


def read_row(client: Any, schema: str, row_id: str) -> dict | None:
    # SQL avoids raw row IDs being interpreted as URL paths/query strings.
    validate_id(row_id)
    result = client.sql.query(
        f"SELECT * FROM {identifier(schema)} WHERE id = $1 LIMIT 1", params=[row_id]
    )
    if not result.rows:
        return None
    row = result.rows[0]
    if (
        not isinstance(row, dict)
        or not isinstance(row.get("text"), str)
        or not isinstance(row.get("metadata", {}), dict)
    ):
        raise ValueError("stored document must contain text and object metadata")
    return row


def delete_ids(client: Any, table: str, schema: str | None, ids: list[str]) -> None:
    # Validate the complete input before any mutation. Bulk API bodies preserve
    # IDs containing quotes, slash, # or ? without path interpolation.
    if not isinstance(ids, (list, tuple)):
        raise ValueError("ids must be an explicit list of strings")
    for row_id in ids:
        validate_id(row_id)
    for row_id in dict.fromkeys(ids):
        result = client.vector.delete_bulk(table, [row_id])
        if result.deleted_count + result.missing_count != 1:
            raise RuntimeError("server did not confirm vector deletion")
        if schema is not None:
            result = client.sql(f"DELETE FROM {identifier(schema)} WHERE id = $1", params=[row_id])
            if getattr(result, "kind", None) != "delete":
                raise RuntimeError("server did not acknowledge row deletion")
            # Older engines returned a translation without executing DELETE.
            # Verify absence instead of trusting a 200 or a translated plan.
            remaining = client.sql.query(
                f"SELECT id FROM {identifier(schema)} WHERE id = $1 LIMIT 1", params=[row_id]
            )
            if remaining.rows:
                raise RuntimeError("row still exists after DELETE; server may only translate SQL")
