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

File:
- [research_bench/workflow.py](/home/surious-type/projects/research-rag/research_bench/workflow.py)

## Framework adapters

Each adapter implements the same conceptual interface:

- `build(source_path, run_dir)`
- `query(run_id, run_dir, questions)`

File:
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

## External boundary

The repository intentionally treats these as external systems:

- Microsoft GraphRAG runtime
- LightRAG runtime
- OpenSPG KAG runtime
- local LLM endpoint
- local embeddings endpoint
- Neo4j / OpenSPG services

The benchmark code should orchestrate them, not modify their upstream behavior.

## Current hotspot

The main architectural hotspot is [research_bench/frameworks.py](/home/surious-type/projects/research-rag/research_bench/frameworks.py).

It currently combines:

- adapter definitions
- KAG project creation
- schema diagnostics
- Neo4j diagnostics
- context normalization
- adapter registry

That is the best candidate for phase 2 decomposition after the workflow layer has been made easier to read.
