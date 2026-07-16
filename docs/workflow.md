# Workflow

This document describes the lifecycle of one benchmark run.

## Command entry

A command starts in:

- [experiment.py](/home/surious-type/projects/research-rag/experiment.py)
- [research_bench/cli.py](/home/surious-type/projects/research-rag/research_bench/cli.py)

The CLI dispatches to workflow functions instead of implementing run logic itself.

## Main run stages

The public workflow entrypoints live in:

- [research_bench/workflows/check_workflow.py](/home/surious-type/projects/research-rag/research_bench/workflows/check_workflow.py)
- [research_bench/workflows/run_workflow.py](/home/surious-type/projects/research-rag/research_bench/workflows/run_workflow.py)

The main stage orchestration lives in:

- [research_bench/workflows/run_stages.py](/home/surious-type/projects/research-rag/research_bench/workflows/run_stages.py)

The compatibility facade remains in:

- [research_bench/workflow.py](/home/surious-type/projects/research-rag/research_bench/workflow.py)

The logical stages are:

1. `check_environment`
2. `create_run_dir`
3. `resolve_run_inputs`
4. `write manifest`
5. `adapter.build`
6. `adapter.query`
7. `run ragas`
8. `verify_run`

Each stage writes normalized artifacts under the run directory and appends progress events to:

- `progress.log`
- `progress.jsonl`

## Build stage

Each framework adapter receives:

- the source file path
- the run directory

It is responsible for creating framework-specific artifacts under `build/` and returning normalized metrics for the local benchmark code.

## Query stage

The workflow sends normalized question rows into the adapter and expects normalized answer rows back.

The workflow then writes:

- `query/answers.jsonl`
- `query/metrics.json`

## RAGAS stage

Only successful answers with non-empty contexts are eligible for scoring.

For smoke runs, the workflow intentionally limits evaluation to the first eligible answer to keep runtime bounded while still checking that RAGAS works.

## Verification stage

`verify_run()` checks:

- build success
- expected answer count
- non-empty answers for successful rows
- presence of contexts
- KAG diagnostics when present
- RAGAS summary consistency

## Replay stages

Two commands rerun later stages without rebuilding:

- `query <run_id>`
- `ragas <run_id>`

These use:

- `rerun_query_stage()`
- `rerun_ragas_stage()`

They also append progress events so long-running reruns can be inspected without stopping the process.

## Where to extend

If you add a new framework:

1. add its adapter contract implementation
2. register it in [research_bench/adapters/registry.py](/home/surious-type/projects/research-rag/research_bench/adapters/registry.py)
3. make sure it produces normalized `BuildMetrics` and `QueryAnswer` rows
4. verify that report and RAGAS consumers work without framework-specific branches
