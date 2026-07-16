# Metric Definitions

This project uses a GraphRAG-Bench-inspired evaluation, not a full reproduction of GraphRAG-Bench.

## Shared graph schema

- `nodes_total`: all nodes stored by the framework backend.
- `edges_total`: all edges stored by the framework backend.
- `entities_count`: semantic entities only.
- `relationships_count`: semantic `Entity -> Entity` relations only.
- `documents_count`: input documents passed into the benchmark workflow.
- `input_documents_count`: explicit copy of the benchmark input document count.
- `backend_document_nodes_count`: document-like backend nodes materialized by the framework storage layer.
- `chunks_count`: chunk or text-unit objects created by the framework.
- `communities_count`: community-level objects when the framework exposes them.
- `isolated_entities_count`: semantic entities with zero semantic degree.
- `connected_entities_count`: semantic entities with at least one semantic relation.
- `connected_entities_ratio`: `connected_entities_count / entities_count`.

Technical nodes excluded from `entities_count`:

- `Chunk`
- `Document`
- `Doc`
- `TextUnit`
- `Community`
- `AtomicQuery`
- `Summary`
- `Outline`

`KnowledgeUnit` is treated as a semantic KAG node for metrics in this repository. When present in the namespace-scoped Neo4j snapshot, it contributes to `entities_count` and is also recorded as `knowledge_units_count` in the raw KAG namespace summary.

Technical relations excluded from `relationships_count`:

- framework-internal links between documents, chunks, communities, and support tables
- attachment edges such as `HAS_CHUNK`, `MENTIONS`, `PART_OF`, and similar non-semantic links
- opaque numeric OpenSPG relation labels such as `128` until they are mapped with documented semantics

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
- Uses `documents_count=1` for the current single-file benchmark input even if KAG does not materialize `Doc` or `Document` nodes.
- Stores Neo4j `Doc` or `Document` materialization separately as `backend_document_nodes_count`.
- Filters Neo4j nodes and relationships strictly to the namespace-specific database and labels of the current run.
- Audits `SHOW FULLTEXT INDEXES` and `SHOW VECTOR INDEXES` after build, and provisions `_default_text_index` only for the current namespace database if schema-free build omitted it.
- Uses both raw vector index metadata and real OpenSPG `SearchClient.search_vector()` probes when deciding whether vector retrieval is healthy.
- Uses only semantic `Entity -> Entity` relations for `connected_entities_count`, `connected_entities_ratio`, and `relationships_count`.

### KAG vector retrieval verification

The repository distinguishes between raw index visibility and real retrieval behavior.

- `fulltext_index`: diagnostic snapshot of `_default_text_index` when present.
- `vector_indexes`: raw rows returned by `SHOW VECTOR INDEXES`.
- `vector_search_probes`: benchmark probes executed against OpenSPG search.
- `vector_search_probe_status`: summarized probe verdict used by verification.

Probe statuses mean:

- `PASS`: at least one expected vector probe returned a valid result.
- `WARN`: vector query executed successfully but the smoke corpus returned no match.
- `FAIL`: vector query failed with a technical/server error.

Verification does not rely on `SHOW VECTOR INDEXES` alone. A run can still be usable when raw index rows are incomplete but real OpenSPG vector search works.

### KAG retriever diagnostics

Per-question KAG diagnostics can include:

- `retriever_results`: separate status for `kg_cs_open_spg`, `kg_fr_open_spg`, and `rc_open_spg`
- `ppr_requested_doc_ids`
- `ppr_loaded_doc_ids`
- `ppr_missing_doc_ids`
- `ppr_chunk_properties`
- `ppr_errors`

These fields exist to separate three different failure classes:

- PPR document loading problems
- OpenSPG vector retrieval problems
- absence of relevant data in a valid retrieval path

## Context capture

- Microsoft GraphRAG: contexts come from the actual local-search callback payload returned during answer generation.
- LightRAG: contexts come from the same hybrid query result object that also returns the generated answer.
- KAG: contexts come from the solver trace or evidence payload returned by the runtime.

For KAG, a successful answer is expected to have non-empty real contexts extracted from trace/evidence payloads rather than synthetic placeholder contexts.

## Progress artifacts

Every benchmark run may emit progress artifacts alongside stage outputs:

- `progress.log`: human-readable timeline
- `progress.jsonl`: machine-readable stage/process events

These are operational diagnostics rather than evaluation metrics, but they are part of the benchmark observability layer and are useful when long-running local experiments need to be inspected without interruption.
