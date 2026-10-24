import httpx
import pytest
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from originchain import OCAuthError, OCServerError

from llama_index.vector_stores.originchain import OriginChainVectorStore


def store(client, **kwargs):
    return OriginChainVectorStore(client=client, table="docs", dim=2, **kwargs)


def node(node_id="a", ref="document", text="hello"):
    return TextNode(
        id_=node_id,
        text=text,
        embedding=[1.0, 0.0],
        metadata={"tag": "x"},
        relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=ref)},
    )


def query(**kwargs):
    kwargs.setdefault("similarity_top_k", 4)
    return VectorStoreQuery(query_embedding=[1.0, 0.0], **kwargs)


def test_real_nodes_round_trip_and_reference_deletion(api):
    client, state = api
    vector_store = store(client, text_schema="doc_rows")
    assert vector_store.stores_text is True
    vector_store.add([node("a"), node("b"), node("c", ref="other")])
    results = vector_store.query(query())
    assert results.nodes[0].get_content() == "hello"
    assert results.nodes[0].ref_doc_id == "document"
    vector_store.delete("document")
    assert vector_store.query(query()).ids == ["c"]
    assert state.connection.execute("SELECT COUNT(*) FROM doc_rows").fetchone()[0] == 1
    vector_store.delete("document")  # idempotent


def test_framework_index_retrieval_and_delete_ref_doc(api):
    client, state = api
    vector_store = store(client, text_schema="doc_rows")
    index = VectorStoreIndex.from_documents(
        [Document(id_="source-document", text="A complete framework round trip")],
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=MockEmbedding(embed_dim=2),
    )
    hits = index.as_retriever(similarity_top_k=2).retrieve("round trip")
    assert hits[0].node.get_content() == "A complete framework round trip"
    assert hits[0].node.ref_doc_id == "source-document"
    index.delete_ref_doc("source-document", delete_from_docstore=True)
    assert index.as_retriever(similarity_top_k=2).retrieve("round trip") == []
    assert state.connection.execute("SELECT COUNT(*) FROM doc_rows").fetchone()[0] == 0


@pytest.mark.parametrize(
    "ref", ["o'brien", "x' OR '1'='1", "x'; DELETE FROM doc_rows; --", "a/b?c#d"]
)
def test_reference_and_node_ids_are_safely_quoted(api, ref):
    client, state = api
    vector_store = store(client, text_schema="doc_rows")
    vector_store.add([node(ref, ref), node("safe", "safe")])
    vector_store.delete(ref)
    assert vector_store.query(query()).ids == ["safe"]
    assert list(state.vectors) == ["safe"]


def test_no_text_schema_allows_framework_docstore_hydration(api):
    client, state = api
    vector_store = store(client)
    assert vector_store.stores_text is False
    vector_store.add([node()])
    results = vector_store.query(query())
    assert results.nodes is None
    assert results.ids == ["a"]
    with pytest.raises(ValueError, match="requires text_schema"):
        vector_store.delete("document")
    vector_store.delete_nodes(["a"])
    assert state.vectors == {}


def test_stale_hits_preserve_result_alignment(api):
    client, state = api
    vector_store = store(client, text_schema="doc_rows")
    vector_store.add([node()])
    state.stale_hits = [{"id": "missing", "score": 1.0}]
    result = vector_store.query(query())
    assert result.ids == ["a"]
    assert result.similarities == [0.9]
    assert [n.node_id for n in result.nodes] == ["a"]


@pytest.mark.parametrize("status, error", [(401, OCAuthError), (500, OCServerError)])
def test_hydration_errors_propagate(api, status, error):
    client, state = api
    vector_store = store(client, text_schema="doc_rows")
    vector_store.add([node()])
    state.failure = lambda req, body: (
        httpx.Response(status, json={"error": "outage"}) if req.url.path.endswith("/sql") else None
    )
    with pytest.raises(error):
        vector_store.query(query())


def test_delete_failure_and_translation_are_not_success(api):
    client, state = api
    vector_store = store(client, text_schema="doc_rows")
    vector_store.add([node()])
    state.translate_only = True
    with pytest.raises(RuntimeError, match="row still exists"):
        vector_store.delete("document")
    state.translate_only = False
    vector_store.delete("document")
    assert vector_store.query(query()).ids == []


def test_equality_filters_preserve_constraint(api):
    client, state = api
    vector_store = store(client)
    vector_store.add([node()])
    filters = MetadataFilters(filters=[MetadataFilter(key="tag", value="secret")])
    assert vector_store.query(query(filters=filters)).ids == []
    assert state.requests[-1][2]["filter"] == {"tag": "secret"}


@pytest.mark.parametrize("operator", [op for op in FilterOperator if op != FilterOperator.EQ])
def test_every_unsupported_operator_fails_before_network(api, operator):
    client, state = api
    filters = MetadataFilters(filters=[MetadataFilter(key="tag", value="x", operator=operator)])
    with pytest.raises(ValueError, match="EQ"):
        store(client).query(query(filters=filters))
    assert state.requests == []


def test_or_nested_conflicting_and_list_filters_fail_closed(api):
    client, state = api
    equality = MetadataFilter(key="tag", value="x")
    cases = [
        MetadataFilters(filters=[equality], condition=FilterCondition.OR),
        MetadataFilters(filters=[MetadataFilters(filters=[equality])]),
        MetadataFilters(filters=[equality, MetadataFilter(key="tag", value="y")]),
        MetadataFilters(filters=[MetadataFilter(key="tag", value=["x", "y"])]),
    ]
    for filters in cases:
        with pytest.raises(ValueError):
            store(client).query(query(filters=filters))
    assert state.requests == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": VectorStoreQueryMode.HYBRID},
        {"node_ids": ["a"]},
        {"doc_ids": ["doc"]},
        {"similarity_top_k": 0},
    ],
)
def test_query_modes_and_constraints_are_not_ignored(api, kwargs):
    client, state = api
    with pytest.raises(ValueError):
        store(client).query(query(**kwargs))
    assert state.requests == []


def test_delete_filters_are_not_ignored(api):
    client, state = api
    with pytest.raises(ValueError, match="filters"):
        store(client).delete_nodes(["a"], filters=MetadataFilters(filters=[]))
    assert state.requests == []


def test_batch_validation_happens_before_first_write(api):
    client, state = api
    with pytest.raises(ValueError, match="no embedding"):
        store(client).add([node(), TextNode(id_="b", text="missing embedding")])
    assert state.requests == []


@pytest.mark.parametrize("schema", ["doc_rows; --", "a/b", "", "x.y.z"])
def test_schema_validation(api, schema):
    client, state = api
    with pytest.raises(ValueError, match="identifier"):
        store(client, text_schema=schema)
    assert state.requests == []
