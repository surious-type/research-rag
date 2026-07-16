# Reading Order

If you want to understand the project quickly, read in this order.

## 1. Start with the overview

- [README.md](/home/surious-type/projects/research-rag/README.md)
- [docs/project_map.md](/home/surious-type/projects/research-rag/docs/project_map.md)
- [docs/architecture.md](/home/surious-type/projects/research-rag/docs/architecture.md)

## 2. Understand the command surface

- [experiment.py](/home/surious-type/projects/research-rag/experiment.py)
- [research_bench/cli.py](/home/surious-type/projects/research-rag/research_bench/cli.py)

These show what commands exist and which workflow entrypoints they call.

## 3. Understand the run lifecycle

- [docs/workflow.md](/home/surious-type/projects/research-rag/docs/workflow.md)
- [research_bench/workflow.py](/home/surious-type/projects/research-rag/research_bench/workflow.py)

This is the best place to understand the benchmark as a sequence of stages.

## 4. Understand the shared data model

- [research_bench/models.py](/home/surious-type/projects/research-rag/research_bench/models.py)
- [research_bench/data.py](/home/surious-type/projects/research-rag/research_bench/data.py)

## 5. Understand framework adapters

- [docs/adapters.md](/home/surious-type/projects/research-rag/docs/adapters.md)
- [research_bench/frameworks.py](/home/surious-type/projects/research-rag/research_bench/frameworks.py)

Read this only after the workflow is clear; otherwise the file feels larger than it really is.

## 6. Understand evaluation and reports

- [research_bench/ragas_eval.py](/home/surious-type/projects/research-rag/research_bench/ragas_eval.py)
- [research_bench/metrics.py](/home/surious-type/projects/research-rag/research_bench/metrics.py)
- [research_bench/parsers.py](/home/surious-type/projects/research-rag/research_bench/parsers.py)
- [research_bench/reporting.py](/home/surious-type/projects/research-rag/research_bench/reporting.py)

## 7. Use tests as executable documentation

- [tests/test_core.py](/home/surious-type/projects/research-rag/tests/test_core.py)

Focus first on tests around:

- environment checks
- run execution
- verification
- KAG smoke behavior
