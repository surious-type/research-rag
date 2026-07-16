from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import research_bench.cli as cli_mod
import research_bench.reporting as reporting_mod
import research_bench.workflow as workflow_mod
from research_bench.utils import atomic_write_json, atomic_write_jsonl


def test_verify_run_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(workflow_mod, "RESULTS_DIR", tmp_path / "results")
    run_dir = workflow_mod.RESULTS_DIR / "msgraphrag_20260715_183542"
    for part in ("build", "query", "ragas"):
        (run_dir / part).mkdir(parents=True)
    atomic_write_json(run_dir / "manifest.json", {"run_id": run_dir.name})
    atomic_write_json(run_dir / "build" / "metrics.json", {"build_status": "success", "documents_count": 1, "chunks_count": 1})
    atomic_write_json(run_dir / "build" / "graph_metrics.json", {"nodes_total": 1})
    answers = [{"question_id": f"q{i:03d}", "status": "success", "answer": "ok", "contexts": [{"text": "ctx"}]} for i in range(1, 101)]
    atomic_write_jsonl(run_dir / "query" / "answers.jsonl", answers)
    atomic_write_json(
        run_dir / "ragas" / "summary.json",
        {metric: {"mean": 0.5, "median": 0.5, "valid_count": 100, "failed_count": 0} for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")},
    )
    result = workflow_mod.verify_run(run_dir.name)
    assert result["status"] == "PASS"


def test_verify_run_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(workflow_mod, "RESULTS_DIR", tmp_path / "results")
    run_dir = workflow_mod.RESULTS_DIR / "lightrag_20260715_183542"
    for part in ("build", "query", "ragas"):
        (run_dir / part).mkdir(parents=True)
    atomic_write_json(run_dir / "manifest.json", {"run_id": run_dir.name})
    atomic_write_json(run_dir / "build" / "metrics.json", {"build_status": "failed", "documents_count": 0, "chunks_count": 0})
    atomic_write_json(run_dir / "build" / "graph_metrics.json", {})
    atomic_write_jsonl(run_dir / "query" / "answers.jsonl", [])
    atomic_write_json(
        run_dir / "ragas" / "summary.json",
        {metric: {"mean": 2.0, "median": 2.0, "valid_count": 1, "failed_count": 1} for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")},
    )
    result = workflow_mod.verify_run(run_dir.name)
    assert result["status"] == "FAIL"


def test_verify_run_allows_kag_without_backend_doc_nodes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(workflow_mod, "RESULTS_DIR", tmp_path / "results")
    run_dir = workflow_mod.RESULTS_DIR / "_smoke" / "kag_20260716_183542"
    for part in ("build", "query", "ragas"):
        (run_dir / part).mkdir(parents=True)
    atomic_write_json(run_dir / "manifest.json", {"run_id": run_dir.name})
    atomic_write_json(
        run_dir / "build" / "metrics.json",
        {"build_status": "success", "documents_count": 1, "input_documents_count": 1, "backend_document_nodes_count": 0, "chunks_count": 2},
    )
    atomic_write_json(run_dir / "build" / "graph_metrics.json", {"nodes_total": 3, "documents_count": 1, "backend_document_nodes_count": 0})
    atomic_write_jsonl(
        run_dir / "query" / "answers.jsonl",
        [{"question_id": "smoke_q1", "status": "success", "answer": "Atlas", "contexts": [{"text": "ctx"}]}],
    )
    atomic_write_json(
        run_dir / "query" / "diagnostics.json",
        {
            "selected_pipeline": "kag_solver_pipeline",
            "resolved_retriever_types": ["kg_cs_open_spg", "kg_fr_open_spg", "rc_open_spg"],
            "reporter_type": "trace_log_reporter",
            "fulltext_index": {"name": "_default_text_index", "state": "ONLINE"},
            "vector_indexes_online": False,
            "vector_search_probe_status": "PASS",
        },
    )
    atomic_write_json(
        run_dir / "ragas" / "summary.json",
        {
            "faithfulness": {"mean": 0.5, "median": 0.5, "valid_count": 1, "failed_count": 0},
            "answer_relevancy": {"mean": 0.5, "median": 0.5, "valid_count": 1, "failed_count": 0},
            "context_precision": {"mean": 0.5, "median": 0.5, "valid_count": 1, "failed_count": 0},
            "context_recall": {"mean": 0.5, "median": 0.5, "valid_count": 1, "failed_count": 0},
        },
    )
    monkeypatch.setattr(workflow_mod, "load_smoke_questions", lambda: [{"id": "smoke_q1"}])
    result = workflow_mod.verify_run(run_dir.name)
    assert result["status"] == "PASS"


def test_verify_run_fails_on_vector_probe_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(workflow_mod, "RESULTS_DIR", tmp_path / "results")
    run_dir = workflow_mod.RESULTS_DIR / "_smoke" / "kag_20260716_183600"
    for part in ("build", "query", "ragas"):
        (run_dir / part).mkdir(parents=True)
    atomic_write_json(run_dir / "manifest.json", {"run_id": run_dir.name})
    atomic_write_json(run_dir / "build" / "metrics.json", {"build_status": "success", "documents_count": 1, "chunks_count": 1})
    atomic_write_json(run_dir / "build" / "graph_metrics.json", {"nodes_total": 1})
    atomic_write_jsonl(run_dir / "query" / "answers.jsonl", [{"question_id": "smoke_q1", "status": "success", "answer": "ok", "contexts": [{"text": "ctx"}]}])
    atomic_write_json(
        run_dir / "query" / "diagnostics.json",
        {
            "selected_pipeline": "kag_solver_pipeline",
            "resolved_retriever_types": ["kg_cs_open_spg"],
            "reporter_type": "trace_log_reporter",
            "fulltext_index": {"name": "_default_text_index", "state": "ONLINE"},
            "vector_search_probe_status": "FAIL",
        },
    )
    atomic_write_json(
        run_dir / "ragas" / "summary.json",
        {metric: {"mean": 0.5, "median": 0.5, "valid_count": 1, "failed_count": 0} for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")},
    )
    monkeypatch.setattr(workflow_mod, "load_smoke_questions", lambda: [{"id": "smoke_q1"}])
    result = workflow_mod.verify_run(run_dir.name)
    assert result["status"] == "FAIL"
    assert "vector search probes failed" in result["issues"]


def test_build_report(tmp_path: Path, monkeypatch) -> None:
    results_dir = tmp_path / "results"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporting_mod, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(reporting_mod, "REPORTS_DIR", reports_dir)
    for framework in ("msgraphrag", "lightrag", "kag"):
        run_dir = results_dir / f"{framework}_20260715_183542"
        for part in ("build", "query", "ragas"):
            (run_dir / part).mkdir(parents=True)
        atomic_write_json(run_dir / "manifest.json", {"source": {"sha256": "s"}, "questions_sha256": "q"})
        atomic_write_json(run_dir / "verification.json", {"status": "PASS"})
        atomic_write_json(run_dir / "build" / "metrics.json", {"build_time_seconds": 1, "documents_count": 1, "chunks_count": 1})
        atomic_write_json(run_dir / "build" / "graph_metrics.json", {"nodes_total": 1})
        atomic_write_json(run_dir / "query" / "metrics.json", {"successful_questions": 1})
        atomic_write_jsonl(run_dir / "query" / "answers.jsonl", [{"question_id": "q001", "question_type": "fact", "status": "success", "latency_seconds": 1.0}])
        atomic_write_json(run_dir / "ragas" / "summary.json", {metric: {"mean": 0.5} for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")})
        atomic_write_jsonl(run_dir / "ragas" / "scores.jsonl", [{"question_id": "q001", "faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.5}])
    summary_path = reporting_mod.build_report()
    assert summary_path.exists()
    assert (reports_dir / "build_comparison.csv").exists()


def test_ensure_source_txt_and_create_run_dir(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_text("content", encoding="utf-8")
    monkeypatch.setattr(workflow_mod, "SOURCE_PATH", source_path)
    monkeypatch.setattr(workflow_mod, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(workflow_mod, "canonical_source_path", lambda: source_path)
    ensured = workflow_mod.ensure_source_txt()
    run_id, run_dir = workflow_mod.create_run_dir("msgraphrag")
    assert ensured == source_path
    assert source_path.read_text(encoding="utf-8") == "content"
    assert run_id.startswith("msgraphrag_")
    assert run_dir.exists()


def test_execute_run_success(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_text("content", encoding="utf-8")
    questions_path = tmp_path / "questions.jsonl"
    rows = [{"id": f"q{i:03d}", "question": f"Question {i}", "reference_answer": f"Answer {i}", "question_type": "fact"} for i in range(1, 101)]
    questions_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(workflow_mod, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(workflow_mod, "QUESTIONS_PATH", questions_path)
    monkeypatch.setattr(workflow_mod, "SOURCE_PATH", source_path)
    monkeypatch.setattr(workflow_mod, "ensure_source_txt", lambda: source_path)
    monkeypatch.setattr(workflow_mod, "load_questions", lambda: [type("Q", (), {"payload": row})() for row in rows])
    monkeypatch.setattr(workflow_mod, "check_environment", lambda: [])
    monkeypatch.setattr(workflow_mod, "_run_ragas", lambda run_dir, answers, limit=None: atomic_write_json(run_dir / "ragas" / "summary.json", {metric: {"mean": 0.5, "median": 0.5, "valid_count": 100, "failed_count": 0} for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}))
    monkeypatch.setattr(workflow_mod, "verify_run", lambda run_id, smoke=False: {"status": "PASS", "issues": []})

    class DummyAdapter:
        def build(self, source_path, run_dir):
            return type("BM", (), {"build_status": "success", "to_dict": lambda self: {"build_status": "success", "documents_count": 1, "chunks_count": 1}})(), {"nodes_total": 1}

        def query(self, run_id, run_dir, question_rows):
            return [
                type(
                    "Answer",
                    (),
                    {
                        "to_dict": lambda self, row=row: {
                            "run_id": run_id,
                            "framework": "msgraphrag",
                            "question_id": row["id"],
                            "question_type": row["question_type"],
                            "question": row["question"],
                            "reference_answer": row["reference_answer"],
                            "answer": "ok",
                            "contexts": [{"text": "ctx"}],
                            "latency_seconds": 1.0,
                            "retrieval_time_seconds": None,
                            "generation_time_seconds": None,
                            "status": "success",
                            "error": None,
                        },
                    },
                )()
                for row in question_rows
            ]

    monkeypatch.setattr(workflow_mod, "get_adapter", lambda framework: DummyAdapter())
    run_id, run_dir = workflow_mod.execute_run("msgraphrag")
    assert run_id.startswith("msgraphrag_")
    assert (run_dir / "manifest.json").exists()


def test_smoke_run_requires_one_valid_ragas_score(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(workflow_mod, "RESULTS_DIR", tmp_path / "results")
    run_dir = workflow_mod.RESULTS_DIR / "_smoke" / "msgraphrag_20260715_183542"
    for part in ("build", "query", "ragas"):
        (run_dir / part).mkdir(parents=True)
    atomic_write_json(run_dir / "manifest.json", {"run_id": run_dir.name})
    atomic_write_json(run_dir / "build" / "metrics.json", {"build_status": "success", "documents_count": 1, "chunks_count": 1})
    atomic_write_json(run_dir / "build" / "graph_metrics.json", {"nodes_total": 1})
    atomic_write_jsonl(
        run_dir / "query" / "answers.jsonl",
        [
            {"question_id": "smoke_q1", "status": "success", "answer": "ok", "contexts": [{"text": "ctx"}]},
            {"question_id": "smoke_q2", "status": "success", "answer": "ok", "contexts": [{"text": "ctx"}]},
        ],
    )
    atomic_write_json(
        run_dir / "ragas" / "summary.json",
        {metric: {"mean": None, "median": None, "valid_count": 0, "failed_count": 2} for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")},
    )
    monkeypatch.setattr(workflow_mod, "load_smoke_questions", lambda: [{"id": "smoke_q1"}, {"id": "smoke_q2"}])
    result = workflow_mod.verify_run(run_dir.name)
    assert result["status"] == "FAIL"
    assert "smoke run has no valid ragas scores" in result["issues"]


def test_find_run_dir_prefers_non_smoke_then_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(workflow_mod, "RESULTS_DIR", tmp_path / "results")
    non_smoke = workflow_mod.RESULTS_DIR / "run_1"
    smoke = workflow_mod.RESULTS_DIR / "_smoke" / "run_2"
    non_smoke.mkdir(parents=True)
    smoke.mkdir(parents=True)
    assert workflow_mod.find_run_dir("run_1") == (non_smoke, False)
    assert workflow_mod.find_run_dir("run_2") == (smoke, True)


def test_rerun_query_and_ragas_stage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(workflow_mod, "RESULTS_DIR", tmp_path / "results")
    run_dir = workflow_mod.RESULTS_DIR / "msgraphrag_20260715_183542"
    for part in ("build", "query", "ragas"):
        (run_dir / part).mkdir(parents=True)
    atomic_write_json(run_dir / "manifest.json", {"framework": "msgraphrag"})
    monkeypatch.setattr(workflow_mod, "load_questions", lambda: [type("Q", (), {"payload": {"id": "q001", "question": "Q", "reference_answer": "A"}})()])

    class DummyAdapter:
        def query(self, run_id, current_run_dir, question_rows):
            assert run_id == run_dir.name
            assert current_run_dir == run_dir
            return [
                type(
                    "Answer",
                    (),
                    {"to_dict": lambda self: {"question_id": "q001", "question": "Q", "reference_answer": "A", "answer": "ok", "contexts": [{"text": "ctx"}], "latency_seconds": 1.0, "status": "success"}},
                )()
            ]

    monkeypatch.setattr(workflow_mod, "get_adapter", lambda framework: DummyAdapter())
    monkeypatch.setattr(workflow_mod, "_run_ragas", lambda current_run_dir, answers, limit=None: atomic_write_json(current_run_dir / "ragas" / "summary.json", {"faithfulness": {"mean": 0.5, "median": 0.5, "valid_count": 1, "failed_count": 0}}))
    assert workflow_mod.rerun_query_stage(run_dir.name) == run_dir
    assert (run_dir / "query" / "answers.jsonl").exists()
    assert workflow_mod.rerun_ragas_stage(run_dir.name) == run_dir
    assert (run_dir / "ragas" / "summary.json").exists()


def test_check_environment_pass_with_monkeypatches(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_text("content", encoding="utf-8")
    questions_path = tmp_path / "questions.jsonl"
    rows = [{"id": f"q{i:03d}", "question": f"Question {i}", "reference_answer": f"Answer {i}"} for i in range(1, 101)]
    questions_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(workflow_mod, "SOURCE_PATH", source_path)
    monkeypatch.setattr(workflow_mod, "QUESTIONS_PATH", questions_path)
    monkeypatch.setattr(workflow_mod, "ensure_source_txt", lambda: source_path)
    monkeypatch.setattr(workflow_mod, "load_questions", lambda: [type("Q", (), {"payload": row})() for row in rows])
    monkeypatch.setattr(workflow_mod, "_check_llm", lambda: [workflow_mod.CheckResult("llm", "PASS", "ok")])
    monkeypatch.setattr(workflow_mod, "_check_embeddings", lambda: [workflow_mod.CheckResult("emb", "PASS", "ok")])
    monkeypatch.setattr(workflow_mod, "_check_frameworks", lambda: [workflow_mod.CheckResult("fw", "PASS", "ok")])
    monkeypatch.setattr(workflow_mod, "_check_kag_neo4j", lambda: [workflow_mod.CheckResult("kag neo4j", "PASS", "ok")])
    monkeypatch.setattr(workflow_mod, "_check_docker", lambda: [workflow_mod.CheckResult("docker", "PASS", "ok")])
    monkeypatch.setattr(workflow_mod, "_check_storage", lambda: [workflow_mod.CheckResult("storage", "PASS", "ok")])
    results = workflow_mod.check_environment()
    assert any(row.name == "source.txt" for row in results)


def test_check_helpers(monkeypatch) -> None:
    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    monkeypatch.setattr(workflow_mod.requests, "get", lambda *args, **kwargs: DummyResponse({"data": [{"id": "model-x"}]}))
    monkeypatch.setattr(
        workflow_mod.requests,
        "post",
        lambda url, **kwargs: DummyResponse({"choices": [{"message": {"content": "ГОТОВО"}}], "data": [{"embedding": [0.0] * 1024}]}),
    )
    monkeypatch.setattr(workflow_mod.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        workflow_mod.subprocess,
        "run",
        lambda *args, **kwargs: type("Proc", (), {"stdout": "release-openspg-neo4j\n", "stderr": "", "returncode": 0})(),
    )
    results = workflow_mod._check_llm() + workflow_mod._check_embeddings() + workflow_mod._check_docker()
    assert any(row.name == "llm model" and row.status == "PASS" for row in results)


def test_check_frameworks_uses_current_python_environment(monkeypatch) -> None:
    def fake_find_spec(name: str):
        if name in {"graphrag", "lightrag"}:
            return types.SimpleNamespace(name=name)
        return None

    monkeypatch.setattr(workflow_mod.checks_mod.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(workflow_mod.checks_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(workflow_mod.checks_mod, "FRAMEWORKS", ("msgraphrag", "lightrag", "kag"))
    monkeypatch.setattr(workflow_mod.checks_mod, "ROOT", Path("/tmp/repo"))
    rows = workflow_mod._check_frameworks()
    assert [row.status for row in rows] == ["PASS", "PASS", "FAIL"]


def test_check_kag_neo4j(monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, str]]] = []

    class DummySession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def run(self, query):
            assert query == "RETURN 1 AS value"
            return type("Result", (), {"single": lambda self: {"value": 1}})()

    class DummyDriver:
        def verify_connectivity(self):
            return None

        def session(self, database):
            assert database == "neo4j"
            return DummySession()

        def close(self):
            return None

    class DummyGraphDatabase:
        @staticmethod
        def driver(uri, auth):
            calls.append((uri, auth))
            return DummyDriver()

    monkeypatch.setenv("KAG_NEO4J_URI", "bolt://127.0.0.1:17687")
    monkeypatch.setitem(sys.modules, "neo4j", type("Neo4jModule", (), {"GraphDatabase": DummyGraphDatabase})())
    results = workflow_mod._check_kag_neo4j()
    assert results[0].status == "PASS"
    assert calls == [("bolt://127.0.0.1:17687", ("neo4j", "neo4j@openspg"))]


def test_run_ragas_placeholder_when_no_answers(tmp_path: Path) -> None:
    workflow_mod._run_ragas(tmp_path / "run" / "ragas_base", [])
    summary = json.loads((tmp_path / "run" / "ragas_base" / "ragas" / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed_questions"] == 0


def test_run_ragas_excludes_degraded_failed_and_contextless_answers(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_save(base_dir, rows, answer_rows=None):
        captured["base_dir"] = base_dir
        captured["rows"] = rows
        captured["answer_rows"] = answer_rows

    monkeypatch.setattr(workflow_mod, "save_ragas_outputs", fake_save)
    workflow_mod._run_ragas(
        tmp_path / "run",
        [
            {"question_id": "q1", "status": "degraded", "answer": "ok", "contexts": [{"text": "ctx"}]},
            {"question_id": "q2", "status": "failed", "answer": "", "contexts": [{"text": "ctx"}]},
            {"question_id": "q3", "status": "success", "answer": "ok", "contexts": []},
        ],
    )
    rows = captured["rows"]
    assert isinstance(rows, list)
    assert len(rows) == 3
    assert all(row["faithfulness"] is None for row in rows)
    assert {row["question_id"]: row["error"] for row in rows} == {
        "q1": "degraded answer is excluded from ragas",
        "q2": "failed answer is excluded from ragas",
        "q3": "contextless answer is excluded from ragas",
    }


def test_execute_run_stops_before_query_when_build_failed(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_text("content", encoding="utf-8")
    monkeypatch.setattr(workflow_mod, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(workflow_mod, "ensure_source_txt", lambda: source_path)
    monkeypatch.setattr(workflow_mod, "check_environment", lambda: [])
    monkeypatch.setattr(workflow_mod, "load_questions", lambda: [type("Q", (), {"payload": {"id": "q001", "question": "Q", "reference_answer": "A"}})()])

    class DummyAdapter:
        def build(self, source_path, run_dir):
            return (
                type(
                    "BM",
                    (),
                    {
                        "build_status": "failed",
                        "build_error": "empty namespace",
                        "to_dict": lambda self: {"build_status": "failed", "build_error": "empty namespace", "documents_count": 0, "chunks_count": 0},
                    },
                )(),
                {"nodes_total": 0, "chunks_count": 0},
            )

        def query(self, run_id, run_dir, question_rows):
            raise AssertionError("query must not run after failed build")

    monkeypatch.setattr(workflow_mod, "get_adapter", lambda framework: DummyAdapter())
    try:
        workflow_mod.execute_run("kag")
    except RuntimeError as exc:
        assert "empty namespace" in str(exc)
    else:
        raise AssertionError("Expected failed KAG build to stop workflow")


def test_cli_check_returns_nonzero_on_fail(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli_mod, "check_environment", lambda: [workflow_mod.CheckResult("source.txt", "FAIL", "missing")])
    exit_code = cli_mod.main(["check"])
    assert exit_code == 1
    assert "FAIL" in capsys.readouterr().out
