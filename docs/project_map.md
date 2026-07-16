# Project Map

This repository is a local benchmark harness for three graph-RAG stacks:

- Microsoft GraphRAG
- LightRAG
- OpenSPG KAG

The project is easiest to understand as four layers:

1. CLI entrypoints
2. Workflow orchestration
3. Framework adapters
4. Evaluation, diagnostics, and reporting helpers

## Entry points

- [experiment.py](/home/surious-type/projects/research-rag/experiment.py): thin executable wrapper
- [research_bench/cli.py](/home/surious-type/projects/research-rag/research_bench/cli.py): command parsing and top-level command dispatch

## Workflow layer

- [research_bench/workflow.py](/home/surious-type/projects/research-rag/research_bench/workflow.py): compatibility facade for the public workflow API
- [research_bench/workflows/checks.py](/home/surious-type/projects/research-rag/research_bench/workflows/checks.py): environment checks
- [research_bench/workflows/run_stages.py](/home/surious-type/projects/research-rag/research_bench/workflows/run_stages.py): run lifecycle and stage orchestration
- [research_bench/workflows/ragas_verification.py](/home/surious-type/projects/research-rag/research_bench/workflows/ragas_verification.py): RAGAS execution and verification logic
- [research_bench/workflows/check_workflow.py](/home/surious-type/projects/research-rag/research_bench/workflows/check_workflow.py): public check entrypoints
- [research_bench/workflows/run_workflow.py](/home/surious-type/projects/research-rag/research_bench/workflows/run_workflow.py): public run/query/ragas/verify entrypoints

Read `run_stages.py` after the CLI if you want to understand how a benchmark run moves from build to query to RAGAS to verification.

## Framework adapters

- [research_bench/adapters/base.py](/home/surious-type/projects/research-rag/research_bench/adapters/base.py): adapter contract
- [research_bench/adapters/registry.py](/home/surious-type/projects/research-rag/research_bench/adapters/registry.py): adapter lookup
- [research_bench/adapters/msgraphrag.py](/home/surious-type/projects/research-rag/research_bench/adapters/msgraphrag.py): Microsoft GraphRAG adapter
- [research_bench/adapters/lightrag.py](/home/surious-type/projects/research-rag/research_bench/adapters/lightrag.py): LightRAG adapter
- [research_bench/adapters/kag.py](/home/surious-type/projects/research-rag/research_bench/adapters/kag.py): KAG adapter, Neo4j diagnostics, and KAG-specific normalization
- [research_bench/frameworks.py](/home/surious-type/projects/research-rag/research_bench/frameworks.py): compatibility facade for older imports and monkeypatch-based tests
- [research_bench/_kag_registry/__init__.py](/home/surious-type/projects/research-rag/research_bench/_kag_registry/__init__.py): local KAG compatibility overrides and diagnostics hooks

Read adapters after you understand the workflow contract that each adapter must satisfy.

## Data and models

- [research_bench/data.py](/home/surious-type/projects/research-rag/research_bench/data.py): canonical paths and source/question loading
- [research_bench/models.py](/home/surious-type/projects/research-rag/research_bench/models.py): typed result models and framework lists

## Evaluation helpers

- [research_bench/ragas_eval.py](/home/surious-type/projects/research-rag/research_bench/ragas_eval.py): RAGAS row preparation and persistence
- [research_bench/metrics.py](/home/surious-type/projects/research-rag/research_bench/metrics.py): normalized graph metrics
- [research_bench/parsers.py](/home/surious-type/projects/research-rag/research_bench/parsers.py): output parsing helpers
- [research_bench/reporting.py](/home/surious-type/projects/research-rag/research_bench/reporting.py): aggregate report generation

## Shared helpers

- [research_bench/shared/io.py](/home/surious-type/projects/research-rag/research_bench/shared/io.py): atomic writes, file copies, JSON helpers
- [research_bench/shared/paths.py](/home/surious-type/projects/research-rag/research_bench/shared/paths.py): run id and hashing helpers
- [research_bench/shared/subprocess.py](/home/surious-type/projects/research-rag/research_bench/shared/subprocess.py): subprocess wrapper with optional progress logging
- [research_bench/shared/text.py](/home/surious-type/projects/research-rag/research_bench/shared/text.py): numeric/text normalization helpers
- [research_bench/utils.py](/home/surious-type/projects/research-rag/research_bench/utils.py): compatibility facade that re-exports shared helpers

## Diagnostics

- [research_bench/diagnostics/artifacts.py](/home/surious-type/projects/research-rag/research_bench/diagnostics/artifacts.py): run artifact directory helpers
- [research_bench/diagnostics/logging.py](/home/surious-type/projects/research-rag/research_bench/diagnostics/logging.py): `progress.log` and `progress.jsonl`
- [research_bench/diagnostics/traces.py](/home/surious-type/projects/research-rag/research_bench/diagnostics/traces.py): per-question trace paths

## Config and runtime glue

- [configs/kag/graph_config.template.yaml](/home/surious-type/projects/research-rag/configs/kag/graph_config.template.yaml): KAG runtime template
- [scripts/kag/build.py](/home/surious-type/projects/research-rag/scripts/kag/build.py): KAG build launcher
- [frameworks/](/home/surious-type/projects/research-rag/frameworks): vendor/runtime assets; treat these as external dependencies

## Tests

- [tests/test_foundations.py](/home/surious-type/projects/research-rag/tests/test_foundations.py): foundational helpers, parsers, data loading, and KAG utility behavior
- [tests/workflows/test_runtime.py](/home/surious-type/projects/research-rag/tests/workflows/test_runtime.py): workflow, verify, reruns, reporting, and CLI checks
- [tests/adapters/test_kag_adapter.py](/home/surious-type/projects/research-rag/tests/adapters/test_kag_adapter.py): KAG adapter build/schema/vector behavior
- [tests/adapters/test_kag_registry_runtime.py](/home/surious-type/projects/research-rag/tests/adapters/test_kag_registry_runtime.py): KAG compatibility registry runtime behavior
- [tests/shared/](/home/surious-type/projects/research-rag/tests/shared): shared helper tests
- [tests/diagnostics/](/home/surious-type/projects/research-rag/tests/diagnostics): diagnostics and progress logging tests
- [tests/workflows/](/home/surious-type/projects/research-rag/tests/workflows): public workflow module tests

## Read this, then that

If you are orienting yourself for the first time:

1. [README.md](/home/surious-type/projects/research-rag/README.md)
2. [docs/reading_order.md](/home/surious-type/projects/research-rag/docs/reading_order.md)
3. [research_bench/cli.py](/home/surious-type/projects/research-rag/research_bench/cli.py)
4. [research_bench/workflows/run_stages.py](/home/surious-type/projects/research-rag/research_bench/workflows/run_stages.py)
5. [research_bench/workflow.py](/home/surious-type/projects/research-rag/research_bench/workflow.py)
6. [research_bench/adapters/registry.py](/home/surious-type/projects/research-rag/research_bench/adapters/registry.py)
7. [research_bench/adapters/](/home/surious-type/projects/research-rag/research_bench/adapters)
