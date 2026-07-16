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
- [research_bench/workflows/run_stages.py](/home/surious-type/projects/research-rag/research_bench/workflows/run_stages.py)
- [research_bench/workflow.py](/home/surious-type/projects/research-rag/research_bench/workflow.py)

Read `run_stages.py` first for the real sequence of stages, then `workflow.py` to see the public compatibility facade.

## 4. Understand the shared data model

- [research_bench/models.py](/home/surious-type/projects/research-rag/research_bench/models.py)
- [research_bench/data.py](/home/surious-type/projects/research-rag/research_bench/data.py)

## 5. Understand framework adapters

- [docs/adapters.md](/home/surious-type/projects/research-rag/docs/adapters.md)
- [research_bench/adapters/registry.py](/home/surious-type/projects/research-rag/research_bench/adapters/registry.py)
- [research_bench/adapters/msgraphrag.py](/home/surious-type/projects/research-rag/research_bench/adapters/msgraphrag.py)
- [research_bench/adapters/lightrag.py](/home/surious-type/projects/research-rag/research_bench/adapters/lightrag.py)
- [research_bench/adapters/kag.py](/home/surious-type/projects/research-rag/research_bench/adapters/kag.py)
- [research_bench/frameworks.py](/home/surious-type/projects/research-rag/research_bench/frameworks.py)

Read `frameworks.py` last in this group; it is now mostly a compatibility facade.

## 6. Understand evaluation and reports

- [research_bench/ragas_eval.py](/home/surious-type/projects/research-rag/research_bench/ragas_eval.py)
- [research_bench/metrics.py](/home/surious-type/projects/research-rag/research_bench/metrics.py)
- [research_bench/parsers.py](/home/surious-type/projects/research-rag/research_bench/parsers.py)
- [research_bench/reporting.py](/home/surious-type/projects/research-rag/research_bench/reporting.py)

## 7. Understand shared helpers and diagnostics

- [research_bench/shared/io.py](/home/surious-type/projects/research-rag/research_bench/shared/io.py)
- [research_bench/shared/subprocess.py](/home/surious-type/projects/research-rag/research_bench/shared/subprocess.py)
- [research_bench/diagnostics/logging.py](/home/surious-type/projects/research-rag/research_bench/diagnostics/logging.py)
- [research_bench/diagnostics/traces.py](/home/surious-type/projects/research-rag/research_bench/diagnostics/traces.py)

## 8. Use tests as executable documentation

- [tests/test_foundations.py](/home/surious-type/projects/research-rag/tests/test_foundations.py)
- [tests/workflows/test_runtime.py](/home/surious-type/projects/research-rag/tests/workflows/test_runtime.py)
- [tests/adapters/test_kag_adapter.py](/home/surious-type/projects/research-rag/tests/adapters/test_kag_adapter.py)
- [tests/adapters/test_kag_registry_runtime.py](/home/surious-type/projects/research-rag/tests/adapters/test_kag_registry_runtime.py)

Focus first on tests around:

- environment checks and run execution
- verification and RAGAS gating
- KAG adapter build and vector diagnostics
- KAG compatibility registry behavior
