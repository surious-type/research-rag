# Project Map

This repository is a local benchmark harness for three graph-RAG stacks:

- Microsoft GraphRAG
- LightRAG
- OpenSPG KAG

The project is easiest to understand as four layers:

1. CLI entrypoints
2. Workflow orchestration
3. Framework adapters
4. Evaluation and reporting helpers

## Entry points

- [experiment.py](/home/surious-type/projects/research-rag/experiment.py): thin executable wrapper
- [research_bench/cli.py](/home/surious-type/projects/research-rag/research_bench/cli.py): command parsing and top-level command dispatch

## Workflow layer

- [research_bench/workflow.py](/home/surious-type/projects/research-rag/research_bench/workflow.py): run lifecycle, reruns, verification, environment checks

This is the file to read after the CLI if you want to understand how a benchmark run moves from build to query to RAGAS to verification.

## Framework adapters

- [research_bench/frameworks.py](/home/surious-type/projects/research-rag/research_bench/frameworks.py): adapter implementations for `msgraphrag`, `lightrag`, and `kag`
- [research_bench/_kag_registry/__init__.py](/home/surious-type/projects/research-rag/research_bench/_kag_registry/__init__.py): local KAG compatibility overrides and diagnostics hooks

This is the heaviest area of the codebase. Read it after you understand the workflow contract that each adapter must satisfy.

## Data and models

- [research_bench/data.py](/home/surious-type/projects/research-rag/research_bench/data.py): canonical paths and source/question loading
- [research_bench/models.py](/home/surious-type/projects/research-rag/research_bench/models.py): typed result models and framework lists

## Evaluation helpers

- [research_bench/ragas_eval.py](/home/surious-type/projects/research-rag/research_bench/ragas_eval.py): RAGAS row preparation and persistence
- [research_bench/metrics.py](/home/surious-type/projects/research-rag/research_bench/metrics.py): normalized graph metrics
- [research_bench/parsers.py](/home/surious-type/projects/research-rag/research_bench/parsers.py): output parsing helpers
- [research_bench/reporting.py](/home/surious-type/projects/research-rag/research_bench/reporting.py): aggregate report generation

## Utilities

- [research_bench/utils.py](/home/surious-type/projects/research-rag/research_bench/utils.py): atomic writes, shell execution, hashing, latency summaries

## Config and runtime glue

- [configs/kag/graph_config.template.yaml](/home/surious-type/projects/research-rag/configs/kag/graph_config.template.yaml): KAG runtime template
- [scripts/kag/build.py](/home/surious-type/projects/research-rag/scripts/kag/build.py): KAG build launcher
- [frameworks/](/home/surious-type/projects/research-rag/frameworks): vendor/runtime assets; treat these as external dependencies

## Tests

- [tests/test_core.py](/home/surious-type/projects/research-rag/tests/test_core.py): coverage for the local benchmark workflow code

## Read this, then that

If you are orienting yourself for the first time:

1. [README.md](/home/surious-type/projects/research-rag/README.md)
2. [docs/reading_order.md](/home/surious-type/projects/research-rag/docs/reading_order.md)
3. [research_bench/cli.py](/home/surious-type/projects/research-rag/research_bench/cli.py)
4. [research_bench/workflow.py](/home/surious-type/projects/research-rag/research_bench/workflow.py)
5. [research_bench/frameworks.py](/home/surious-type/projects/research-rag/research_bench/frameworks.py)
