"""LlamaIndex vector store using OriginChainDB's official Python client."""

from __future__ import annotations

from typing import Any

from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from pydantic import PrivateAttr

from ._support import (
    delete_ids,
    equality_filter,
    identifier,
    read_row,
    validate_config,
    validate_id,
    validate_k,
    validate_vector,
)


class OriginChainVectorStore(BasePydanticVectorStore):
    """Store pre-embedded nodes; caller owns the OriginChain client lifecycle."""

    stores_text: bool = False
    is_embedding_query: bool = True
    flat_metadata: bool = False
    _client: Any = PrivateAttr()
    _table: str = PrivateAttr()
    _dim: int = PrivateAttr()
    _metric: str = PrivateAttr()
    _text_schema: str | None = PrivateAttr()
    _mode: str | None = PrivateAttr()

    def __init__(
        self,
        *,
        client: Any,
        table: str,
        dim: int,
        metric: str = "cosine",
        text_schema: str | None = None,
        mode: str | None = None,
    ) -> None:
        validate_config(table, text_schema, dim, metric, mode)
        super().__init__(stores_text=text_schema is not None)
        self._client = client
        self._table = table
        self._dim = dim
        self._metric = metric
        self._text_schema = text_schema
        self._mode = mode

    @property
    def client(self) -> Any:
        return self._client

    def add(self, nodes: list[BaseNode], **kwargs: Any) -> list[str]:
        for node in nodes:
            validate_id(node.node_id)
            if node.embedding is None:
                raise ValueError(f"node {node.node_id} has no embedding")
            validate_vector(node.embedding, self._dim)
        for node in nodes:
            meta = dict(node.metadata or {})
            if self._text_schema is not None:
                self._client.rows.put(
                    self._text_schema,
                    {
                        "id": node.node_id,
                        "text": node.get_content(metadata_mode=MetadataMode.NONE),
                        "metadata": meta,
                        "ref_doc_id": node.ref_doc_id or "",
                    },
                )
            self._client.vector_put(
                self._table,
                id=node.node_id,
                embedding=list(node.get_embedding()),
                dim=self._dim,
                metric=self._metric,
                metadata=meta or None,
            )
        return [node.node_id for node in nodes]

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Resolve persisted source-document links, then delete every matching ID.

        Without text_schema there is no persistent ref_doc_id lookup; use
        delete_nodes(node_ids=...) instead. Failures are always surfaced.
        """
        if self._text_schema is None:
            raise ValueError("delete(ref_doc_id) requires text_schema; use delete_nodes for IDs")
        validate_id(ref_doc_id)
        result = self._client.sql.query(
            f"SELECT id FROM {identifier(self._text_schema)} WHERE ref_doc_id = $1",
            params=[ref_doc_id],
        )
        ids = [row["id"] for row in result.rows]
        delete_ids(self._client, self._table, self._text_schema, ids)

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **kwargs: Any,
    ) -> None:
        if filters is not None:
            raise ValueError(
                "delete_nodes does not support metadata filters; pass explicit node_ids"
            )
        if node_ids is None:
            raise ValueError("delete_nodes requires explicit node_ids")
        delete_ids(self._client, self._table, self._text_schema, node_ids)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise ValueError("only the default dense vector query mode is supported")
        # VectorStoreIndex passes node_ids=[] when this store owns the text.
        # Empty lists mean no restriction in that framework-generated query.
        if query.node_ids or query.doc_ids:
            raise ValueError("node_ids/doc_ids query constraints are not supported")
        if query.query_embedding is None:
            raise ValueError("query.query_embedding is required")
        validate_k(query.similarity_top_k)
        validate_vector(query.query_embedding, self._dim)
        ann_filter = _translate_filters(query.filters)
        hits = self._client.vector_topk(
            self._table,
            query=list(query.query_embedding),
            k=query.similarity_top_k,
            dim=self._dim,
            metric=self._metric,
            filter=ann_filter,
            mode=self._mode,
        )
        ids: list[str] = []
        scores: list[float] = []
        nodes: list[TextNode] = []
        for hit in hits:
            if self._text_schema is not None:
                row = read_row(self._client, self._text_schema, hit.id)
                if row is None:
                    continue
                relationships = {}
                if row.get("ref_doc_id"):
                    relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                        node_id=row["ref_doc_id"]
                    )
                nodes.append(
                    TextNode(
                        id_=hit.id,
                        text=row["text"],
                        metadata=dict(row.get("metadata", {})),
                        relationships=relationships,
                    )
                )
            ids.append(hit.id)
            scores.append(float(hit.score))
        return VectorStoreQueryResult(
            nodes=nodes if self.stores_text else None, similarities=scores, ids=ids
        )


def _translate_filters(filters: MetadataFilters | None) -> dict | None:
    """Support flat AND/EQ only; reject everything we cannot enforce exactly."""
    if filters is None:
        return None
    if filters.condition != FilterCondition.AND:
        raise ValueError("only AND metadata filter conditions are supported")
    out: dict = {}
    for item in filters.filters:
        if isinstance(item, MetadataFilters):
            raise ValueError("nested metadata filters are not supported")
        if item.operator != FilterOperator.EQ:
            raise ValueError("only EQ metadata filter operators are supported")
        if item.key in out and out[item.key] != item.value:
            raise ValueError("conflicting equality filters for the same key")
        out[item.key] = item.value
    return equality_filter(out)
