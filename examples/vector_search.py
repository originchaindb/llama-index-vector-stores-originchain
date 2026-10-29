"""Read an existing vector index with a precomputed embedding."""

import argparse
import json
import math
import os

from llama_index.core.vector_stores.types import VectorStoreQuery
from originchain import OriginChain

from llama_index.vector_stores.originchain import OriginChainVectorStore


def search(client, table, vector):
    store = OriginChainVectorStore(client=client, table=table, dim=len(vector))
    result = store.query(VectorStoreQuery(query_embedding=vector, similarity_top_k=5))
    return {"ids": result.ids, "similarities": result.similarities}


def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for name in ["OC_BASE_URL", "OC_TENANT", "OC_BEARER"]:
        required(name)
    table = required("OC_TABLE")
    vector = json.loads(required("OC_QUERY_VECTOR"))
    if (
        not isinstance(vector, list)
        or not vector
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        )
    ):
        raise ValueError("OC_QUERY_VECTOR must be a nonempty array of finite numbers")
    db_user = os.environ.get("OC_DB_USER")
    db_key = os.environ.get("OC_DB_KEY")
    if bool(db_user) != bool(db_key):
        raise ValueError("Set both OC_DB_USER and OC_DB_KEY, or neither")
    with OriginChain.from_env(db_user=db_user, db_key=db_key, timeout=30, max_retries=0) as client:
        print(json.dumps(search(client, table, vector), indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit(
            "Query failed. Check connection, permissions and example inputs."
        ) from None
