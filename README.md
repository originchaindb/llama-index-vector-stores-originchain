# LlamaIndex for OriginChainDB

Store embedded LlamaIndex nodes in OriginChainDB, retrieve their text and
metadata, and delete all nodes belonging to a source document.

**Status:** `0.1.0rc1` preview. **License:** Proprietary; the existing license
declaration is unchanged. No open-source license is granted by this release.

## Install

Requires Python 3.10+, LlamaIndex Core 0.11 or newer (below 1.0), and the
OriginChain Python SDK 0.10.0rc1 or newer (below 1.0). These commands install the
tested core SDK and adapter from GitHub; the adapter is not published to PyPI.

```bash
python -m pip install "https://github.com/originchaindb/originchain-python/releases/download/v0.10.0-rc.3/originchain-0.10.0rc3-py3-none-any.whl#sha256=1d6456e7bc51e0dfb4b0d03ef4e0713f2cd3c902e4579a7fb956e5fb6e59c647"
python -m pip install https://github.com/originchaindb/llama-index-vector-stores-originchain/releases/download/v0.1.0-rc.1/llama_index_vector_stores_originchain-0.1.0rc1-py3-none-any.whl
```

## Store, query and delete a node

Set `OC_BASE_URL`, `OC_TENANT` and `OC_BEARER` to your database connection values
through your environment or secrets manager.

The example creates a content schema, stores one node, queries it with a metadata
filter, and deletes it by source-document ID. The two-dimensional vectors make
the storage example reproducible without an embedding service. In an application,
supply vectors from your embedding model and set `dim` to that model's output size.

```python
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.originchain import OriginChainVectorStore
from originchain import OriginChain

with OriginChain.from_env() as db:
    db.schemas.register("""
namespace = "sdk_examples"
table = "llamaindex_nodes"
primary_key = ["id"]
[[columns]]
name = "id"
ty = "str"
required = true
[[columns]]
name = "text"
ty = "str"
[[columns]]
name = "metadata"
ty = "json"
[[columns]]
name = "ref_doc_id"
ty = "str"
""")

    store = OriginChainVectorStore(
        client=db,
        table="llamaindex_vectors",
        text_schema="sdk_examples.llamaindex_nodes",
        dim=2,
    )
    store.add([
        TextNode(
            id_="example-chunk-1",
            text="Returns are accepted within 30 days.",
            embedding=[1.0, 0.0],
            metadata={"topic": "returns"},
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="example-policy")
            },
        )
    ])
    result = store.query(VectorStoreQuery(
        query_embedding=[1.0, 0.0],
        similarity_top_k=2,
        filters=MetadataFilters(
            condition=FilterCondition.AND,
            filters=[MetadataFilter(
                key="topic", value="returns", operator=FilterOperator.EQ
            )],
        ),
    ))
    for node, score in zip(result.nodes or [], result.similarities or []):
        print(node.node_id, node.get_content(), score, node.ref_doc_id)

    store.delete(ref_doc_id="example-policy")
```

`table` names the vector table; `text_schema` names the separate content schema.
Schema registration belongs in application setup. An existing content schema
must have the columns and primary key shown above. `ref_doc_id` persists the
source relationship used by `delete`. Names must be ASCII identifiers, optionally
qualified as `namespace.table`.

## Framework behavior

| Operation | Behavior |
| --- | --- |
| `add(nodes)` | Store precomputed embeddings and return node IDs. Every node needs an embedding containing exactly `dim` finite numbers. |
| `query(VectorStoreQuery(...))` | Return aligned IDs, scores and hydrated nodes for a default dense-vector query. |
| `delete(ref_doc_id)` | Find all stored node IDs for the source document, then delete their vectors and rows. Requires `text_schema`. |
| `delete_nodes(node_ids=[...])` | Delete explicit node IDs. An empty list is a no-op; omitting IDs raises `ValueError`. |

With `text_schema`, `stores_text` is true and source-document relationships are
restored when nodes are read. Without it, `stores_text` is false and query results
have `nodes=None`; LlamaIndex must hydrate nodes from its own docstore. Use
`StorageContext.from_defaults(vector_store=store)` when constructing a
`VectorStoreIndex`. The framework owns embedding generation; the adapter never
chooses or calls an embedding model.

Metadata filters support flat `AND` conditions with `EQ` comparisons on strings,
finite numbers or booleans. Nested groups, OR/NOT, other operators, lists, and
conflicting values for the same key raise `ValueError` before any request.

Only `VectorStoreQueryMode.DEFAULT` is supported. Hybrid, sparse and MMR modes,
nonempty `node_ids`/`doc_ids` query restrictions, and filter-based deletion are
rejected. `metric` accepts `cosine` (default), `dot` or `l2`; scores are returned
as supplied by the server. `mode` optionally selects `fast` or `high_recall`.

## Storage, failures and client ownership

The caller owns the synchronous `OriginChain` client. Keep it open while the
store or index is in use; the context manager above closes it afterward.

Writes store a content row before its vector. Deletes remove each vector before
its row, then check that the row is absent. These requests are **not one
transaction**: an interrupted operation can leave partial state. Exceptions
propagate. Supply stable node IDs, retry those IDs after failures, and serialize
concurrent operations on the same ID. Repeated deletion of absent IDs is safe.

A vector hit with no content row is skipped, so fewer than the requested number
of nodes may be returned. Authentication, schema and transport failures raise
instead of being treated as missing content.

The server must execute SQL DELETE and support positional SQL parameters.
Invalid inputs raise `ValueError`; network and service failures use the core SDK's
exceptions. If a server acknowledges DELETE but retains the row, deletion raises
`RuntimeError`.

## Development

After installing the core SDK above:

```bash
python -m pip install -e ".[dev]" build
python -m pytest -q tests
python -m ruff check src tests
python -m build
```

Tests use real LlamaIndex types and mocked HTTP. Separate live database results
and package checksums are attached to the [GitHub release](https://github.com/originchaindb/llama-index-vector-stores-originchain/releases/tag/v0.1.0-rc.1).
