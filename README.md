# Research RAG Benchmark

This repository contains a local GraphRAG-Bench-inspired evaluation workflow for comparing:

- Microsoft GraphRAG
- LightRAG
- OpenSPG KAG

The workflow uses one shared corpus in [data/corpus/source.txt](/home/surious-type/projects/research-rag/data/corpus/source.txt) and one shared question set in [data/questions/questions.jsonl](/home/surious-type/projects/research-rag/data/questions/questions.jsonl). It does not implement D0-D3 version comparisons, incremental updates, or DOCX parsing.

## Project structure

- [experiment.py](/home/surious-type/projects/research-rag/experiment.py): main CLI entrypoint
- [research_bench](/home/surious-type/projects/research-rag/research_bench): benchmark implementation
- [tests](/home/surious-type/projects/research-rag/tests): unit tests for local workflow code
- [frameworks/kag](/home/surious-type/projects/research-rag/frameworks/kag): vendored external KAG source tree kept as a separate nested git checkout
- [docs/project_map.md](/home/surious-type/projects/research-rag/docs/project_map.md): high-level codebase map
- [docs/architecture.md](/home/surious-type/projects/research-rag/docs/architecture.md): architecture summary
- [docs/workflow.md](/home/surious-type/projects/research-rag/docs/workflow.md): benchmark run lifecycle
- [docs/adapters.md](/home/surious-type/projects/research-rag/docs/adapters.md): adapter responsibilities
- [docs/reading_order.md](/home/surious-type/projects/research-rag/docs/reading_order.md): recommended onboarding order
- [frameworks/msgraphrag/settings.yaml](/home/surious-type/projects/research-rag/frameworks/msgraphrag/settings.yaml): GraphRAG base config
- [frameworks/lightrag/docker-compose.yml](/home/surious-type/projects/research-rag/frameworks/lightrag/docker-compose.yml): LightRAG runtime reference
- [configs/kag/graph_config.template.yaml](/home/surious-type/projects/research-rag/configs/kag/graph_config.template.yaml): KAG config template
- [scripts/kag/build.py](/home/surious-type/projects/research-rag/scripts/kag/build.py): KAG build entrypoint
- [docs/metric_definitions.md](/home/surious-type/projects/research-rag/docs/metric_definitions.md): graph metric rules

## Recommended reading order

If you are new to the repository, read in this order:

1. [README.md](/home/surious-type/projects/research-rag/README.md)
2. [docs/project_map.md](/home/surious-type/projects/research-rag/docs/project_map.md)
3. [docs/architecture.md](/home/surious-type/projects/research-rag/docs/architecture.md)
4. [docs/workflow.md](/home/surious-type/projects/research-rag/docs/workflow.md)
5. [experiment.py](/home/surious-type/projects/research-rag/experiment.py)
6. [research_bench/cli.py](/home/surious-type/projects/research-rag/research_bench/cli.py)
7. [research_bench/workflows/run_stages.py](/home/surious-type/projects/research-rag/research_bench/workflows/run_stages.py)
8. [research_bench/workflow.py](/home/surious-type/projects/research-rag/research_bench/workflow.py)
9. [docs/adapters.md](/home/surious-type/projects/research-rag/docs/adapters.md)
10. [research_bench/adapters/kag.py](/home/surious-type/projects/research-rag/research_bench/adapters/kag.py)
11. [tests/workflows/test_runtime.py](/home/surious-type/projects/research-rag/tests/workflows/test_runtime.py)

## Environment check

Runtime model selection is now driven by `.env` in the project root. If `.env` is absent, the benchmark falls back to the current local defaults.
Use [.env.example](/home/surious-type/projects/research-rag/.env.example) as the starting template for a new local setup.

Supported variables:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
OPENAI_EMBEDDING_BASE_URL=http://127.0.0.1:8010/v1
OPENAI_API_KEY=local
OPENAI_MODEL=/models/Qwen3.5-35B-A3B-Q4_K_M.gguf
OPENAI_EMBEDDING_MODEL=multilingual-e5-large
```

Examples:

Local default-compatible setup:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
OPENAI_EMBEDDING_BASE_URL=http://127.0.0.1:8010/v1
OPENAI_API_KEY=local
OPENAI_MODEL=/models/Qwen3.5-35B-A3B-Q4_K_M.gguf
OPENAI_EMBEDDING_MODEL=multilingual-e5-large
```

OpenAI API setup:

```bash
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-5-nano
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

In this shell, use `.venv/bin/python` instead of `python`, because the plain `python` alias is not installed.

Run:

```bash
.venv/bin/python experiment.py check
```

The check command validates:

- `source.txt`
- `questions.jsonl`
- configured LLM endpoint and model from `.env`
- configured embedding endpoint and model from `.env`
- installed framework assets
- Docker and running containers
- Neo4j visibility for KAG
- free disk space
- writability of `results/`

Output format:

```text
CHECK | STATUS | DETAILS
```

## Tests

Run:

```bash
.venv/bin/python experiment.py test
```

This runs `pytest` with `pytest-cov` against the local benchmark code only.

## Smoke runs

Smoke runs use the small corpus in [output/smoke_tests/data/smoke_document.txt](/home/surious-type/projects/research-rag/output/smoke_tests/data/smoke_document.txt) and the two smoke questions in [output/smoke_tests/data/smoke_questions.json](/home/surious-type/projects/research-rag/output/smoke_tests/data/smoke_questions.json).

Run:

```bash
.venv/bin/python experiment.py smoke msgraphrag
.venv/bin/python experiment.py smoke lightrag
.venv/bin/python experiment.py smoke kag
.venv/bin/python experiment.py smoke all
```

Smoke results are stored under `results/_smoke/` and are excluded from the final comparison report.
It is safe to periodically clean older smoke runs and keep only the latest one per framework.

## Full runs

Run one framework at a time:

```bash
.venv/bin/python experiment.py run msgraphrag
.venv/bin/python experiment.py run lightrag
.venv/bin/python experiment.py run kag
```

Run IDs are generated automatically in the format:

```text
<framework>_YYYYMMDD_HHMMSS
```

If a collision happens, a numeric suffix is appended automatically.

`run-all` exists for manual use but is intentionally not triggered automatically:

```bash
.venv/bin/python experiment.py run-all
```

## RAGAS

Pinned support packages live in [requirements-ragas.txt](/home/surious-type/projects/research-rag/requirements-ragas.txt).

RAGAS uses the same runtime configuration as the benchmark adapters and `check`. Changing `.env` is enough to switch the LLM provider and embedding model for RAGAS as well.

Per-run command:

```bash
.venv/bin/python experiment.py ragas <run_id>
```

## Verify

Run:

```bash
.venv/bin/python experiment.py verify <run_id>
```

This validates build success, question counts, answer/context presence, latency fields, graph metrics, and RAGAS summary consistency.

## Report

Run:

```bash
.venv/bin/python experiment.py report
```

The report command selects the latest successful non-smoke run for each framework and writes:

- `reports/build_comparison.csv`
- `reports/graph_comparison.csv`
- `reports/query_comparison.csv`
- `reports/ragas_comparison.csv`
- `reports/ragas_by_question_type.csv`
- `reports/per_question_comparison.csv`
- `reports/summary.md`

## Vendored frameworks

`frameworks/kag` is stored in this repository as a vendored external framework source tree and currently exists as its own nested git checkout. The benchmark uses it locally, but `research_bench` code should treat it as upstream framework code rather than internal application code.

## Result structure

Each full run is stored as:

```text
results/
  <run_id>/
    manifest.json
    build/
      metrics.json
      graph_metrics.json
      stdout.log
      stderr.log
    query/
      answers.jsonl
      metrics.json
    ragas/
      scores.csv
      scores.jsonl
      summary.json
      errors.jsonl
    verification.json
```
