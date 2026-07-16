# Graph Report - research-rag  (2026-07-10)

## Corpus Check
- 145 files · ~1,194,524 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 520 nodes · 770 edges · 54 communities (35 shown, 19 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.81)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- MSGraphRAG Prompting
- Question Schema Links
- Embedding Service Code
- Question Schema Core
- Graphify Skill Docs
- Question Schema Evidence
- CAD Corpus Variants
- Embedding Stack
- Add and Watch Docs
- run_runtime_repair.py
- container_runtime_repair.py
- container_runtime_repair.py
- container_runtime_repair.py
- container_runtime_repair.py
- container_runtime_repair.py
- container_runtime_repair.py
- container_runtime_repair.py
- graphify reference: extra exports and benchmark
- graphify reference: query, path, explain
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- AGENTS.md
- extraction-spec.md
- Exports Reference
- Extraction Subagent Prompt
- GitHub Clone and Merge Reference
- Hooks and CLAUDE Integration Reference
- Query Path Explain Reference
- Vocabulary-Constrained Query Expansion
- Transcribe Video and Audio Reference
- Incremental Update Reference
- Fast Path Existing Graph
- graphify Skill
- Project Graphify Policy
- multilingual-e5-large README
- embedding service README
- container_validation.py
- container_validation.py
- container_validation.py
- container_validation.py
- container_validation.py
- container_validation.py
- run_validation.py
- KAG Automatic Builder Validation
- container_validation.py

## God Nodes (most connected - your core abstractions)
1. `KAG Automatic Builder Validation` - 22 edges
2. `Pre-Benchmark Validation Report` - 18 edges
3. `msgraphrag settings` - 14 edges
4. `main()` - 13 edges
5. `main()` - 13 edges
6. `main()` - 13 edges
7. `main()` - 13 edges
8. `main()` - 13 edges
9. `main()` - 13 edges
10. `main()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Default embedding model` --conceptually_related_to--> `Embedding service`  [INFERRED]
  frameworks/msgraphrag/settings.yaml → services/embeddings/README.md
- `MSGraphRAG CAD Input` --shares_data_with--> `CAD Manual Baseline`  [INFERRED]
  frameworks/msgraphrag/input/D0.txt → data/corpus/D0/source.txt
- `msgraphrag settings` --references--> `drift_reduce_prompt prompt`  [AMBIGUOUS]
  frameworks/msgraphrag/settings.yaml → frameworks/msgraphrag/prompts/drift_reduce_prompt.txt
- `CAD entity and relationship extraction` --conceptually_related_to--> `RAG orchestrator graph fragment`  [INFERRED]
  frameworks/msgraphrag/prompts/extract_graph.txt → output/debug/preflight/extraction_response.txt
- `Default embedding model` --references--> `Multilingual-E5-large`  [EXTRACTED]
  frameworks/msgraphrag/settings.yaml → models/embeddings/multilingual-e5-large/README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **CAD Manual Variant Corpus** — data_corpus_d0_source_cad_manual, data_corpus_d1_add_source_cad_manual_update, data_corpus_d2_edit_source_cad_manual_edit, data_corpus_d3_delete_source_cad_manual_delete, frameworks_msgraphrag_input_d0_cad_manual [INFERRED 0.95]
- **Local GraphRAG Framework Assets** — frameworks_kag_docker_compose_west_kag_stack, frameworks_lightrag_docker_compose_lightrag_stack, frameworks_msgraphrag_prompts_basic_search_system_prompt_basic_search_prompt [INFERRED 0.75]
- **Global Search Prompt Suite** — frameworks_msgraphrag_prompts_global_search_map_system_prompt_document, frameworks_msgraphrag_prompts_global_search_reduce_system_prompt_document, frameworks_msgraphrag_prompts_global_search_knowledge_system_prompt_document [EXTRACTED 1.00]
- **CAD Extraction and Reporting Chain** — frameworks_msgraphrag_prompts_extract_graph_cad_entity_relationship_extraction, frameworks_msgraphrag_prompts_summarize_descriptions_document, frameworks_msgraphrag_prompts_community_report_graph_cad_domain_prompting [INFERRED 0.85]

## Communities (54 total, 19 thin omitted)

### Community 0 - "MSGraphRAG Prompting"
Cohesion: 0.10
Nodes (25): CAD-domain community report prompting, community_report_graph prompt, Community report generation, community_report_text prompt, drift_reduce_prompt prompt, Grounded report summarization, drift_search_system_prompt prompt, Report-grounded question answering (+17 more)

### Community 1 - "Question Schema Links"
Cohesion: 0.05
Nodes (40): enum, type, items, type, items, type, items, type (+32 more)

### Community 2 - "Embedding Service Code"
Cohesion: 0.21
Nodes (13): Any, AutoModel, AutoTokenizer, BaseModel, EmbeddingRequest, embeddings(), health(), load_model() (+5 more)

### Community 3 - "Question Schema Core"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 5 - "Question Schema Evidence"
Cohesion: 0.06
Nodes (30): Changed Files, Created Files, Exact Re-Run Commands, Final Repaired Runtime Path, Graph Validation Summary, KAG Runtime Repair Report, Limitations, Root Cause Of The Original Failure (+22 more)

### Community 6 - "CAD Corpus Variants"
Cohesion: 0.38
Nodes (7): CAD Manual Baseline, CAD Manual Added Revision, CAD Manual Edited Revision, CAD Manual Deletion Revision, KAG Docker Compose Stack, LightRAG Docker Compose Stack, MSGraphRAG CAD Input

### Community 7 - "Embedding Stack"
Cohesion: 0.12
Nodes (14): Default embedding model, Benchmark Results on [Mr. TyDi](https://arxiv.org/abs/2108.08787), Citation, FAQ, Limitations, MTEB Benchmark Evaluation, Multilingual-E5-large, Support for Sentence Transformers (+6 more)

### Community 9 - "run_runtime_repair.py"
Cohesion: 0.33
Nodes (12): build_config(), build_schema(), choose_namespace(), docker_python(), ensure_ok(), main(), CompletedProcess, Path (+4 more)

### Community 10 - "container_runtime_repair.py"
Cohesion: 0.35
Nodes (10): config_validation(), graph_validation(), load_config(), main(), Path, run_async_component_build(), save_json(), solver_queries() (+2 more)

### Community 11 - "container_runtime_repair.py"
Cohesion: 0.33
Nodes (10): config_validation(), graph_validation(), load_config(), main(), Path, run_async_component_build(), run_builder_main(), save_json() (+2 more)

### Community 12 - "container_runtime_repair.py"
Cohesion: 0.35
Nodes (10): config_validation(), graph_validation(), load_config(), main(), Path, run_async_component_build(), save_json(), solver_queries() (+2 more)

### Community 13 - "container_runtime_repair.py"
Cohesion: 0.35
Nodes (10): config_validation(), graph_validation(), load_config(), main(), Path, run_async_component_build(), save_json(), solver_queries() (+2 more)

### Community 14 - "container_runtime_repair.py"
Cohesion: 0.36
Nodes (9): config_validation(), graph_validation(), load_config(), main(), Path, run_async_component_build(), run_builder_main(), save_json() (+1 more)

### Community 15 - "container_runtime_repair.py"
Cohesion: 0.36
Nodes (9): config_validation(), graph_validation(), load_config(), main(), Path, run_async_component_build(), run_builder_main(), save_json() (+1 more)

### Community 16 - "container_runtime_repair.py"
Cohesion: 0.36
Nodes (9): config_validation(), graph_validation(), load_config(), main(), Path, run_async_component_build(), save_json(), solver_queries() (+1 more)

### Community 17 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 18 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 19 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 20 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 21 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 39 - "container_validation.py"
Cohesion: 0.17
Nodes (23): append_jsonl(), chunk_hit_payload(), chunk_label_schema(), component_validation(), entity_validation(), expected_fact_present(), graph_statistics(), load_config() (+15 more)

### Community 40 - "container_validation.py"
Cohesion: 0.17
Nodes (23): append_jsonl(), chunk_hit_payload(), chunk_label_schema(), component_validation(), entity_validation(), expected_fact_present(), graph_statistics(), load_config() (+15 more)

### Community 41 - "container_validation.py"
Cohesion: 0.17
Nodes (23): append_jsonl(), chunk_hit_payload(), chunk_label_schema(), component_validation(), entity_validation(), expected_fact_present(), graph_statistics(), load_config() (+15 more)

### Community 42 - "container_validation.py"
Cohesion: 0.17
Nodes (23): append_jsonl(), chunk_hit_payload(), chunk_label_schema(), component_validation(), entity_validation(), expected_fact_present(), graph_statistics(), load_config() (+15 more)

### Community 43 - "container_validation.py"
Cohesion: 0.17
Nodes (23): append_jsonl(), chunk_hit_payload(), chunk_label_schema(), component_validation(), entity_validation(), expected_fact_present(), graph_statistics(), load_config() (+15 more)

### Community 44 - "container_validation.py"
Cohesion: 0.17
Nodes (23): append_jsonl(), chunk_hit_payload(), chunk_label_schema(), component_validation(), entity_validation(), expected_fact_present(), graph_statistics(), load_config() (+15 more)

### Community 45 - "run_validation.py"
Cohesion: 0.28
Nodes (15): build_config(), build_schema(), choose_namespace(), collect_corpus_stats(), docker_python(), ensure_ok(), main(), parse_peak_memory_kb() (+7 more)

### Community 46 - "KAG Automatic Builder Validation"
Cohesion: 0.09
Nodes (22): 10. Automatic Build, 11. Graph Statistics, 12. Entity Extraction Validation, 13. Relation Extraction Validation, 14. Stock Retriever, 15. Retriever Diagnostics, 16. Query Results, 17. Graph Usage Analysis (+14 more)

### Community 52 - "container_validation.py"
Cohesion: 0.17
Nodes (23): append_jsonl(), chunk_hit_payload(), chunk_label_schema(), component_validation(), entity_validation(), expected_fact_present(), graph_statistics(), load_config() (+15 more)

## Ambiguous Edges - Review These
- `drift_reduce_prompt prompt` → `msgraphrag settings`  [AMBIGUOUS]
  frameworks/msgraphrag/settings.yaml · relation: references

## Knowledge Gaps
- **142 isolated node(s):** `$schema`, `type`, `required`, `type`, `type` (+137 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `drift_reduce_prompt prompt` and `msgraphrag settings`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `KAG Automatic Builder Validation` connect `KAG Automatic Builder Validation` to `Question Schema Evidence`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **Why does `msgraphrag settings` connect `MSGraphRAG Prompting` to `Embedding Stack`?**
  _High betweenness centrality (0.005) - this node is a cross-community bridge._
- **What connects `$schema`, `type`, `required` to the rest of the system?**
  _144 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `MSGraphRAG Prompting` be split into smaller, more focused modules?**
  _Cohesion score 0.09666666666666666 - nodes in this community are weakly interconnected._
- **Should `Question Schema Links` be split into smaller, more focused modules?**
  _Cohesion score 0.05365853658536585 - nodes in this community are weakly interconnected._
- **Should `Question Schema Core` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._