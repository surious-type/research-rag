from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import research_bench.data as data_mod
import research_bench.reporting as reporting_mod
import research_bench.workflow as workflow_mod
from research_bench.data import load_questions, load_source_info
from research_bench.metrics import normalize_graph_metrics
from research_bench.parsers import parse_lightrag_outputs, parse_msgraphrag_outputs, parse_neo4j_summary
from research_bench.ragas_eval import prepare_ragas_rows, save_ragas_placeholder, summarize_ragas_scores
from research_bench.utils import aggregate_numeric, atomic_write_json, atomic_write_jsonl, latency_summary, sha256_file, utc_run_id


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello\n", encoding="utf-8")
    assert sha256_file(path) == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"


def test_run_id_generation_handles_collisions() -> None:
    now = datetime(2026, 7, 15, 18, 35, 42)
    assert utc_run_id("msgraphrag", now=now, existing=set()) == "msgraphrag_20260715_183542"
    assert utc_run_id("msgraphrag", now=now, existing={"msgraphrag_20260715_183542"}) == "msgraphrag_20260715_183542_001"


def test_atomic_json_and_jsonl_write(tmp_path: Path) -> None:
    json_path = tmp_path / "data.json"
    jsonl_path = tmp_path / "data.jsonl"
    atomic_write_json(json_path, {"ok": True})
    atomic_write_jsonl(jsonl_path, [{"a": 1}, {"a": 2}])
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True}
    assert len(jsonl_path.read_text(encoding="utf-8").splitlines()) == 2


def test_latency_summary_and_percentiles() -> None:
    result = latency_summary([1.0, 2.0, 3.0, None])
    assert result["questions_count"] == 4
    assert result["successful_questions"] == 3
    assert result["mean_latency_seconds"] == 2.0
    assert result["median_latency_seconds"] == 2.0
    assert result["p95_latency_seconds"] is not None


def test_aggregate_numeric() -> None:
    result = aggregate_numeric([0.1, 0.2, None])
    assert result["valid_count"] == 2
    assert result["failed_count"] == 1


def test_graph_metric_normalization_excludes_technical_rows() -> None:
    graph = normalize_graph_metrics(
        nodes_total=10,
        edges_total=12,
        documents_count=1,
        chunks_count=2,
        communities_count=1,
        entity_rows=[
            {"type": "Entity", "degree": 1},
            {"type": "Chunk", "degree": 5},
            {"type": "Person", "degree": 0},
        ],
        relationship_rows=[
            {"type": "RELATED_TO", "source_type": "Entity", "target_type": "Person"},
            {"type": "HAS_CHUNK", "source_type": "Entity", "target_type": "Chunk"},
        ],
    )
    assert graph.entities_count == 2
    assert graph.relationships_count == 1
    assert graph.connected_entities_count == 1
    assert graph.isolated_entities_count == 1


def test_parse_neo4j_summary() -> None:
    graph = parse_neo4j_summary(
        {
            "nodes_total": 4,
            "edges_total": 3,
            "documents_count": 1,
            "chunks_count": 1,
            "communities_count": None,
            "entity_rows": [{"type": "Entity", "degree": 1}],
            "relationship_rows": [{"type": "REL", "source_type": "Entity", "target_type": "Entity"}],
        }
    )
    assert graph.nodes_total == 4
    assert graph.relationships_count == 1


def test_prepare_ragas_rows() -> None:
    rows = prepare_ragas_rows(
        [
            {
                "question_id": "q1",
                "question": "What?",
                "answer": "Answer",
                "reference_answer": "Reference",
                "contexts": [{"text": "Context"}],
            }
        ]
    )
    assert rows[0]["retrieved_contexts"] == ["Context"]


def test_summarize_ragas_scores() -> None:
    summary = summarize_ragas_scores(
        [
            {"faithfulness": 0.8, "answer_relevancy": 0.7, "context_precision": None, "context_recall": 0.9},
            {"faithfulness": None, "answer_relevancy": 0.5, "context_precision": 0.6, "context_recall": None},
        ]
    )
    assert summary["faithfulness"]["valid_count"] == 1
    assert summary["answer_relevancy"]["failed_count"] == 0


def test_load_source_info_and_questions(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_text("one two\nthree", encoding="utf-8")
    questions_path = tmp_path / "questions.jsonl"
    rows = [
        {"id": f"q{i:03d}", "question": f"Question {i}", "reference_answer": f"Answer {i}", "question_type": "fact"}
        for i in range(1, 101)
    ]
    questions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(data_mod, "SOURCE_PATH", source_path)
    monkeypatch.setattr(data_mod, "MERGED_PATH", source_path)
    monkeypatch.setattr(data_mod, "QUESTIONS_PATH", questions_path)
    source = load_source_info(source_path)
    questions = load_questions(questions_path)
    assert source.words_count == 3
    assert len(questions) == 100


def test_load_questions_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    rows = [{"id": "q001", "question": "Q", "reference_answer": "A"}] * 100
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    try:
        load_questions(path)
    except ValueError as exc:
        assert "Duplicate question id" in str(exc)
    else:
        raise AssertionError("Expected duplicate id validation to fail")


def test_parse_msgraphrag_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pd.DataFrame([{"id": 1}]).to_parquet(output_dir / "documents.parquet")
    pd.DataFrame([{"id": 1}, {"id": 2}]).to_parquet(output_dir / "text_units.parquet")
    pd.DataFrame([{"type": "Person", "degree": 1}, {"type": "Chunk", "degree": 2}]).to_parquet(output_dir / "entities.parquet")
    pd.DataFrame([{"type": "RELATED_TO", "source_type": "Person", "target_type": "Person"}]).to_parquet(output_dir / "relationships.parquet")
    pd.DataFrame([{"community": 1}]).to_parquet(output_dir / "communities.parquet")
    graph = parse_msgraphrag_outputs(output_dir)
    assert graph.documents_count == 1
    assert graph.chunks_count == 2
    assert graph.entities_count == 1


def test_parse_lightrag_outputs(tmp_path: Path) -> None:
    work_dir = tmp_path / "rag_storage"
    work_dir.mkdir()
    graphml = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <graph edgedefault="directed">
    <node id="n0"><data key="type">Entity</data></node>
    <node id="n1"><data key="type">Chunk</data></node>
    <edge id="e0" source="n0" target="n0" label="RELATED_TO" />
  </graph>
</graphml>
"""
    (work_dir / "graph_chunk_entity_relation.graphml").write_text(graphml, encoding="utf-8")
    (work_dir / "kv_store_full_docs.json").write_text(json.dumps({"doc1": {}}), encoding="utf-8")
    (work_dir / "kv_store_text_chunks.json").write_text(json.dumps({"chunk1": {}}), encoding="utf-8")
    graph = parse_lightrag_outputs(work_dir)
    assert graph.nodes_total == 2
    assert graph.edges_total == 1


def test_save_ragas_placeholder(tmp_path: Path) -> None:
    base = tmp_path / "ragas"
    save_ragas_placeholder(base, [{"question_id": "q1"}, {"question_id": "q2"}], "missing ragas")
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    assert summary["faithfulness"]["failed_count"] == 2


def test_utils_misc_helpers(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    dst = tmp_path / "nested" / "dst.txt"
    src.write_text("123", encoding="utf-8")
    from research_bench.utils import copy_file, file_size, load_json, percentile, safe_float

    copy_file(src, dst)
    atomic_write_json(tmp_path / "payload.json", {"x": 1})
    assert dst.read_text(encoding="utf-8") == "123"
    assert file_size(tmp_path) >= 3
    assert load_json(tmp_path / "payload.json") == {"x": 1}
    assert percentile([1.0, 2.0, 3.0], 0.95) is not None
    assert safe_float("1.5") == 1.5
    assert safe_float("bad") is None


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
    source_path = tmp_path / "merged.txt"
    source_path.write_text("content", encoding="utf-8")
    target_path = tmp_path / "source.txt"
    monkeypatch.setattr(workflow_mod, "SOURCE_PATH", target_path)
    monkeypatch.setattr(workflow_mod, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(workflow_mod, "canonical_source_path", lambda: source_path)
    ensured = workflow_mod.ensure_source_txt()
    run_id, run_dir = workflow_mod.create_run_dir("msgraphrag")
    assert ensured == target_path
    assert target_path.read_text(encoding="utf-8") == "content"
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
    monkeypatch.setattr(workflow_mod, "_run_ragas", lambda run_dir, answers: atomic_write_json(run_dir / "ragas" / "summary.json", {metric: {"mean": 0.5, "median": 0.5, "valid_count": 100, "failed_count": 0} for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}))
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
