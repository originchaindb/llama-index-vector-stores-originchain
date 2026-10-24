# Changelog

## 0.1.0rc1

- Validate identifiers and bind SQL values separately.
- Delete vector entries and row content, confirming that rows are gone.
- Propagate write, deletion, hydration and authentication failures.
- Skip stale hits and preserve IDs, scores and hydrated node alignment.
- Reject unsupported filters and constraints before network requests.
- Validate all embeddings and IDs before starting an ingestion batch.
- Test with installed framework types and the real SDK HTTP transport.
