# Architecture

The benchmark uses one shared contract across three frameworks.

## High-level dependency flow

CLI
↓
Workflow
↓
Framework adapter
↓
Framework runtime / external services
↓
Artifacts
↓
RAGAS / verification / reports

## Layer responsibilities

## CLI

- Parses commands
- Selects run mode
- Prints final artifacts or summaries

Files:
- [experiment.py](/home/surious-type/projects/research-rag/experiment.py)
- [research_bench/cli.py](/home/surious-type/projects/research-rag/research_bench/cli.py)

## Workflow orchestration

- Validates environment
- Creates run directories
- Writes manifests
- Invokes adapter build/query stages
- Runs RAGAS
- Verifies final artifacts

Files:
- [research_bench/workflows/run_stages.py](/home/surious-type/projects/research-rag/research_bench/workflows/run_stages.py)
- [research_bench/workflows/checks.py](/home/surious-type/projects/research-rag/research_bench/workflows/checks.py)
- [research_bench/workflows/ragas_verification.py](/home/surious-type/projects/research-rag/research_bench/workflows/ragas_verification.py)
- [research_bench/workflow.py](/home/surious-type/projects/research-rag/research_bench/workflow.py)

## Framework adapters

Each adapter implements the same conceptual interface:

- `build(source_path, run_dir)`
- `query(run_id, run_dir, questions)`

Files:
- [research_bench/adapters/base.py](/home/surious-type/projects/research-rag/research_bench/adapters/base.py)
- [research_bench/adapters/registry.py](/home/surious-type/projects/research-rag/research_bench/adapters/registry.py)
- [research_bench/adapters/msgraphrag.py](/home/surious-type/projects/research-rag/research_bench/adapters/msgraphrag.py)
- [research_bench/adapters/lightrag.py](/home/surious-type/projects/research-rag/research_bench/adapters/lightrag.py)
- [research_bench/adapters/kag.py](/home/surious-type/projects/research-rag/research_bench/adapters/kag.py)
- [research_bench/frameworks.py](/home/surious-type/projects/research-rag/research_bench/frameworks.py)

## Evaluation helpers

- graph metrics normalization
- per-framework output parsing
- RAGAS row preparation and persistence
- aggregated reporting

Files:
- [research_bench/metrics.py](/home/surious-type/projects/research-rag/research_bench/metrics.py)
- [research_bench/parsers.py](/home/surious-type/projects/research-rag/research_bench/parsers.py)
- [research_bench/ragas_eval.py](/home/surious-type/projects/research-rag/research_bench/ragas_eval.py)
- [research_bench/reporting.py](/home/surious-type/projects/research-rag/research_bench/reporting.py)

## Shared infrastructure

- atomic artifact writes
- subprocess execution
- progress logging
- trace path helpers

Files:
- [research_bench/shared/io.py](/home/surious-type/projects/research-rag/research_bench/shared/io.py)
- [research_bench/shared/subprocess.py](/home/surious-type/projects/research-rag/research_bench/shared/subprocess.py)
- [research_bench/diagnostics/logging.py](/home/surious-type/projects/research-rag/research_bench/diagnostics/logging.py)
- [research_bench/diagnostics/traces.py](/home/surious-type/projects/research-rag/research_bench/diagnostics/traces.py)

## External boundary

The repository intentionally treats these as external systems:

- Microsoft GraphRAG runtime
- LightRAG runtime
- OpenSPG KAG runtime
- local LLM endpoint
- local embeddings endpoint
- Neo4j / OpenSPG services

The benchmark code should orchestrate them, not modify their upstream behavior.

## Current hotspots

- The heaviest framework-specific module is [research_bench/adapters/kag.py](/home/surious-type/projects/research-rag/research_bench/adapters/kag.py).
- The public compatibility surface is still centered on [research_bench/workflow.py](/home/surious-type/projects/research-rag/research_bench/workflow.py) and [research_bench/frameworks.py](/home/surious-type/projects/research-rag/research_bench/frameworks.py).

`adapters/kag.py` currently combines:

- adapter definitions
- KAG project creation
- schema diagnostics
- Neo4j diagnostics
- context normalization
- query-time KAG helpers

That is the strongest candidate for a later phase if we continue decomposing by responsibility.
