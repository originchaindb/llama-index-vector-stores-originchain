"""LlamaIndex VectorStore integration for OriginChain.

Usage::

    from llama_index.vector_stores.originchain import OriginChainVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.embeddings.openai import OpenAIEmbedding
    from originchain import OriginChain

    db = OriginChain.from_env()
    vs = OriginChainVectorStore(
        client=db, table="docs", dim=1536, text_schema="doc_rows"
    )
    storage = StorageContext.from_defaults(vector_store=vs)
    index = VectorStoreIndex.from_documents(
        docs, storage_context=storage, embed_model=OpenAIEmbedding()
    )
"""

from .base import OriginChainVectorStore

__all__ = ["OriginChainVectorStore"]
__version__ = "0.1.0rc1"
