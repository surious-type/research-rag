# Metric Definitions

This project uses a GraphRAG-Bench-inspired evaluation, not a full reproduction of GraphRAG-Bench.

## Shared graph schema

- `nodes_total`: all nodes stored by the framework backend.
- `edges_total`: all edges stored by the framework backend.
- `entities_count`: semantic entities only.
- `relationships_count`: semantic `Entity -> Entity` relations only.
- `documents_count`: document-level objects created by the framework.
- `chunks_count`: chunk or text-unit objects created by the framework.
- `communities_count`: community-level objects when the framework exposes them.
- `isolated_entities_count`: semantic entities with zero semantic degree.
- `connected_entities_count`: semantic entities with at least one semantic relation.
- `connected_entities_ratio`: `connected_entities_count / entities_count`.

Technical nodes excluded from `entities_count`:

- `Chunk`
- `Document`
- `TextUnit`
- `Community`

Technical relations excluded from `relationships_count`:

- framework-internal links between documents, chunks, communities, and support tables
- attachment edges such as `HAS_CHUNK`, `MENTIONS`, `PART_OF`, and similar non-semantic links

## Framework-specific counting

### Microsoft GraphRAG

- Reads `documents.parquet`, `text_units.parquet`, `entities.parquet`, `relationships.parquet`, and `communities.parquet`.
- Uses `documents.parquet` for `documents_count`.
- Uses `text_units.parquet` for `chunks_count`.
- Uses `communities.parquet` for `communities_count`.
- Uses entity and relationship tables to exclude technical node and edge types from semantic counts.

### LightRAG

- Reads `graph_chunk_entity_relation.graphml` and JSON stores in `rag_storage/`.
- Uses `kv_store_full_docs.json` for `documents_count`.
- Uses `kv_store_text_chunks.json` for `chunks_count`.
- Derives `nodes_total` and `edges_total` from GraphML.
- Filters technical chunk/document/community rows out of semantic counts.

### OpenSPG KAG

- Reads Neo4j counts and labels directly from the running graph backend.
- Uses Neo4j label counts for `Document`, `Chunk`, and `Entity`.
- Uses Neo4j relationship types to separate semantic `Entity -> Entity` relations from technical links.

## Context capture

- Microsoft GraphRAG: contexts come from the actual local-search callback payload returned during answer generation.
- LightRAG: contexts come from the same hybrid query result object that also returns the generated answer.
- KAG: contexts come from the solver trace or evidence payload returned by the runtime.
