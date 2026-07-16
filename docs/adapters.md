# Adapters

The benchmark treats each framework as an adapter behind one shared workflow contract.

## Adapter contract

Each adapter must support two operations:

- `build(source_path, run_dir)`
- `query(run_id, run_dir, questions)`

The workflow should not need to understand framework internals beyond those calls.

## Current adapters

## Microsoft GraphRAG

Defined in:

- [MsGraphRAGAdapter](/home/surious-type/projects/research-rag/research_bench/frameworks.py)

Role:
- copy project files
- run local GraphRAG indexing
- parse produced parquet artifacts

## LightRAG

Defined in:

- [LightRAGAdapter](/home/surious-type/projects/research-rag/research_bench/frameworks.py)

Role:
- initialize LightRAG storage
- insert the source text
- run hybrid queries
- normalize contexts from LightRAG output

## KAG

Defined in:

- [KAGAdapter](/home/surious-type/projects/research-rag/research_bench/frameworks.py)

Role:
- create/update the KAG project
- prepare supported schema path
- launch KAG build
- inspect Neo4j/OpenSPG runtime state
- normalize query traces and contexts

Related local compatibility hooks:

- [research_bench/_kag_registry/__init__.py](/home/surious-type/projects/research-rag/research_bench/_kag_registry/__init__.py)

## Design rule

Framework-specific behavior belongs inside adapters or framework-local helper modules.

Workflow code should only coordinate stages and operate on normalized results.

## Best phase 2 refactor target

The KAG adapter is currently the strongest candidate for further decomposition into:

- project/schema lifecycle helpers
- index and diagnostics helpers
- query/trace normalization helpers
- adapter shell class
