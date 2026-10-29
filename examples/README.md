# Query a prepared vector index

This example runs the real LlamaIndex vector store with a supplied query vector. It calls no embedding provider and creates no rows or indexes. Use a pre-existing cosine vector index whose dimension and embedding model match the vector. `[1,0]` is suitable only for a deliberately prepared two-dimensional fixture; it is not a text embedding.

Install the core SDK and this adapter using [Contributing](../CONTRIBUTING.md), then check the command without credentials:

```sh
python examples/vector_search.py --help
```

Set `OC_BASE_URL`, `OC_TENANT`, `OC_BEARER`, `OC_TABLE` and `OC_QUERY_VECTOR` in your process environment. `OC_QUERY_VECTOR` is a JSON array, for example `[1,0]`. If database identity is enabled, also set both `OC_DB_USER` and `OC_DB_KEY`. Use your own endpoint and scoped credentials; keep them out of files and shell history.

```sh
python examples/vector_search.py
```

The command returns up to five IDs and similarity scores. It does not configure a text schema, so LlamaIndex returns IDs and similarities without hydrated nodes. This keeps the example focused on an existing vector index; the main README covers adding document storage.

`tests/test_examples.py` runs this exact command through a loopback HTTP fixture using the installed adapter and Python SDK. The test checks the route, vector payload, authentication, result and rejection of malformed input. It proves client integration, not live engine index behaviour.
