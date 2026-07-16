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
- [frameworks/msgraphrag/settings.yaml](/home/surious-type/projects/research-rag/frameworks/msgraphrag/settings.yaml): GraphRAG base config
- [frameworks/lightrag/docker-compose.yml](/home/surious-type/projects/research-rag/frameworks/lightrag/docker-compose.yml): LightRAG runtime reference
- [configs/kag/graph_config.template.yaml](/home/surious-type/projects/research-rag/configs/kag/graph_config.template.yaml): KAG config template
- [scripts/kag/build.py](/home/surious-type/projects/research-rag/scripts/kag/build.py): KAG build entrypoint
- [docs/metric_definitions.md](/home/surious-type/projects/research-rag/docs/metric_definitions.md): graph metric rules

## Environment check

In this shell, use `.venv/bin/python` instead of `python`, because the plain `python` alias is not installed.

Run:

```bash
.venv/bin/python experiment.py check
```

The check command validates:

- `source.txt`
- `questions.jsonl`
- LLM endpoint `http://127.0.0.1:8080/v1`
- embedding endpoint `http://127.0.0.1:8010/v1`
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

The workflow uses local endpoints only:

- LLM: `http://127.0.0.1:8080/v1`
- embeddings: `http://127.0.0.1:8010/v1`

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
