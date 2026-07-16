# Adapters

The benchmark treats each framework as an adapter behind one shared workflow contract.

## Adapter contract

Each adapter must support two operations:

- `build(source_path, run_dir)`
- `query(run_id, run_dir, questions)`

The workflow should not need to understand framework internals beyond those calls.

The registry for adapter selection lives in:

- [research_bench/adapters/registry.py](/home/surious-type/projects/research-rag/research_bench/adapters/registry.py)

## Current adapters

## Microsoft GraphRAG

Defined in:

- [research_bench/adapters/msgraphrag.py](/home/surious-type/projects/research-rag/research_bench/adapters/msgraphrag.py)

Role:
- copy project files
- run local GraphRAG indexing
- parse produced parquet artifacts
- log external `graphrag index` process start/finish into the run progress timeline

## LightRAG

Defined in:

- [research_bench/adapters/lightrag.py](/home/surious-type/projects/research-rag/research_bench/adapters/lightrag.py)

Role:
- initialize LightRAG storage
- insert the source text
- run hybrid queries
- normalize contexts from LightRAG output

## KAG

Defined in:

- [research_bench/adapters/kag.py](/home/surious-type/projects/research-rag/research_bench/adapters/kag.py)

Role:
- create/update the KAG project
- prepare supported schema path
- launch KAG build
- inspect Neo4j/OpenSPG runtime state
- normalize query traces and contexts
- capture namespace-scoped index, probe, and retriever diagnostics

Key KAG-specific responsibilities now include:

- namespace-scoped Neo4j graph metrics
- vector index audit plus real OpenSPG vector probes
- query-time trace normalization
- verification-oriented diagnostics such as `vector_search_probe_status`

Related local compatibility hooks:

- [research_bench/_kag_registry/__init__.py](/home/surious-type/projects/research-rag/research_bench/_kag_registry/__init__.py)

The local KAG registry adds benchmark-only compatibility shims without modifying upstream KAG runtime files. In particular it provides:

- `benchmark_ppr_chunk_retriever`
- KAG retriever diagnostics
- per-question compatibility notes for `kg_fr_open_spg`, `rc_open_spg`, and PPR loading

## Design rule

Framework-specific behavior belongs inside adapters or framework-local helper modules.

Workflow code should only coordinate stages and operate on normalized results.

Compatibility facades for older imports still exist in:

- [research_bench/frameworks.py](/home/surious-type/projects/research-rag/research_bench/frameworks.py)

That file should be treated as a public compatibility surface, not as the primary place to extend adapter behavior.

## Diagnostics boundary

Adapter code may write framework-specific artifacts under the run directory, but it should still emit normalized outputs for the rest of the benchmark:

- build metrics
- graph metrics
- normalized answer rows
- optional diagnostics and traces

Long-running or external adapter operations should also feed the run progress timeline through:

- `progress.log`
- `progress.jsonl`

## Best later refactor target

The KAG adapter is currently the strongest candidate for further decomposition into:

- project/schema lifecycle helpers
- index and diagnostics helpers
- query/trace normalization helpers
- adapter shell class
