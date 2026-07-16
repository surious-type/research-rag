from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import datetime
from pathlib import Path

import pandas as pd

import research_bench.cli as cli_mod
import research_bench.data as data_mod
import research_bench.frameworks as frameworks_mod
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


def test_extract_kag_label_type() -> None:
    assert frameworks_mod._extract_kag_label_type(["Kag123.Person"], "Kag123") == "Person"
    assert frameworks_mod._extract_kag_label_type(["Kag123.Chunk"], "Kag123") == "Chunk"
    assert frameworks_mod._extract_kag_label_type(["Other.Person"], "Kag123") is None


def test_summarize_kag_namespace_graph_filters_namespace_and_technical_labels() -> None:
    summary = frameworks_mod.summarize_kag_namespace_graph(
        node_rows=[
            {"node_id": "1", "labels": ["Kag123.Person"]},
            {"node_id": "2", "labels": ["Kag123.Organization"]},
            {"node_id": "3", "labels": ["Kag123.Chunk"]},
            {"node_id": "4", "labels": ["Kag123.Doc"]},
            {"node_id": "5", "labels": ["Other.Person"]},
            {"node_id": "6", "labels": ["Kag123.AtomicQuery"]},
        ],
        edge_rows=[
            {"source_id": "1", "target_id": "2", "source_labels": ["Kag123.Person"], "target_labels": ["Kag123.Organization"], "type": "WORKS_WITH"},
            {"source_id": "1", "target_id": "3", "source_labels": ["Kag123.Person"], "target_labels": ["Kag123.Chunk"], "type": "MENTIONS"},
            {"source_id": "1", "target_id": "5", "source_labels": ["Kag123.Person"], "target_labels": ["Other.Person"], "type": "CROSS_NS"},
        ],
        namespace="Kag123",
    )
    graph = parse_neo4j_summary(summary)
    assert summary["documents_count"] == 1
    assert summary["chunks_count"] == 1
    assert graph.entities_count == 2
    assert graph.relationships_count == 1
    assert graph.connected_entities_count == 2
    assert graph.connected_entities_ratio == 1.0
    assert graph.entity_types == {"Person": 1, "Organization": 1}


def test_get_kag_neo4j_config_defaults(monkeypatch) -> None:
    monkeypatch.delenv("KAG_NEO4J_URI", raising=False)
    monkeypatch.delenv("KAG_NEO4J_USER", raising=False)
    monkeypatch.delenv("KAG_NEO4J_PASSWORD", raising=False)
    monkeypatch.delenv("KAG_NEO4J_DATABASE", raising=False)
    config = frameworks_mod.get_kag_neo4j_config()
    assert config["uri"] == "bolt://127.0.0.1:17687"
    assert config["database"] == "neo4j"


def test_resolve_kag_neo4j_database_prefers_namespace() -> None:
    assert frameworks_mod._resolve_kag_neo4j_database("Kag260716095410", "neo4j") == "kag260716095410"
    assert frameworks_mod._resolve_kag_neo4j_database("", "neo4j") == "neo4j"


def test_build_kag_server_project_config_has_no_mock_vectorizer() -> None:
    template = """
llm:
  type: openai
  base_url: http://127.0.0.1:8080/v1
vectorize_model:
  type: openai
  base_url: http://127.0.0.1:8010/v1
  model: multilingual-e5-large
  vector_dimensions: 1024
vectorizer:
  type: openai
  base_url: http://127.0.0.1:8010/v1
  model: multilingual-e5-large
  vector_dimensions: 1024
"""
    config = frameworks_mod.build_kag_server_project_config(template)
    assert config["vectorize_model"]["type"] == "openai"
    assert config["vectorize_model"]["base_url"] == "http://multilingual-e5-server/v1"
    assert config["vectorizer"]["type"] == "openai"
    assert config["vectorizer"]["base_url"] == "http://multilingual-e5-server/v1"


def test_resolve_kag_query_diagnostics_uses_expected_pipeline_and_retrievers() -> None:
    config = {
        "kag_solver_pipeline": {
            "type": "kag_static_pipeline",
            "executors": [
                {
                    "type": "kag_hybrid_retrieval_executor",
                    "retrievers": [
                        {"type": "kg_cs_open_spg"},
                        {"type": "kg_fr_open_spg"},
                        {"type": "rc_open_spg"},
                    ],
                }
            ],
        }
    }
    _, diagnostics = frameworks_mod._resolve_kag_query_diagnostics(config, {"id": "24", "namespace": "KagNs"})
    assert diagnostics["selected_pipeline"] == "kag_solver_pipeline"
    assert diagnostics["resolved_retriever_types"] == ["kg_cs_open_spg", "kg_fr_open_spg", "rc_open_spg"]


def test_resolve_kag_query_diagnostics_fails_on_empty_retrievers() -> None:
    config = {"kag_solver_pipeline": {"type": "kag_static_pipeline", "executors": []}}
    try:
        frameworks_mod._resolve_kag_query_diagnostics(config, {"id": "24", "namespace": "KagNs"})
    except RuntimeError as exc:
        assert "resolved_retriever_types is empty" in str(exc)
    else:
        raise AssertionError("Expected empty KAG retriever resolution to fail")


def test_prepare_kag_query_environment_calls_init_env(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "project.json").write_text(json.dumps({"id": "26", "namespace": "KagNs"}), encoding="utf-8")
    (run_dir / "generated_kag_config.yaml").write_text(
        "kag_builder_pipeline:\n  register_path: /tmp/fake-kag\n",
        encoding="utf-8",
    )
    called: dict[str, str] = {}
    monkeypatch.setitem(
        sys.modules,
        "kag.common.conf",
        types.SimpleNamespace(init_env=lambda path: called.setdefault("config_path", path)),
    )
    monkeypatch.setitem(
        sys.modules,
        "kag.common.registry",
        types.SimpleNamespace(import_modules_from_path=lambda path: called.setdefault("register_path", path)),
    )
    project, config, config_path = frameworks_mod._prepare_kag_query_environment(run_dir)
    assert project["id"] == "26"
    assert config_path == run_dir / "generated_kag_config.yaml"
    assert called["config_path"] == str(config_path)
    assert called["register_path"] == "/tmp/fake-kag"
    assert config["kag_builder_pipeline"]["register_path"] == "/tmp/fake-kag"


def test_normalize_kag_context_from_chunk_trace() -> None:
    contexts = frameworks_mod._normalize_kag_context(
        {
            "decompose": [
                {
                    "chunk_datas": [
                        {
                            "content": "Chunk text",
                            "document_id": "doc-1",
                            "document_name": "source.txt",
                            "score": 0.8,
                        }
                    ]
                }
            ]
        }
    )
    assert len(contexts) == 1
    assert contexts[0].text == "Chunk text"
    assert contexts[0].metadata["document_id"] == "doc-1"


def test_normalize_kag_context_from_graph_trace() -> None:
    contexts = frameworks_mod._normalize_kag_context({"decompose": [{"graph_data": ["Person[A] -WORKS_WITH-> Org[B]"]}]})
    assert len(contexts) == 1
    assert contexts[0].source == "graph"
    assert "WORKS_WITH" in contexts[0].text


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
    monkeypatch.setattr(data_mod, "QUESTIONS_PATH", questions_path)
    source = load_source_info(source_path)
    questions = load_questions(questions_path)
    assert source.words_count == 3
    assert len(questions) == 100


def test_canonical_source_requires_source_txt(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source.txt"
    monkeypatch.setattr(data_mod, "SOURCE_PATH", source_path)
    try:
        data_mod.canonical_source_path()
    except FileNotFoundError as exc:
        assert "source.txt" in str(exc)
    else:
        raise AssertionError("Expected source.txt lookup to fail without fallback")


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


def test_kag_build_skips_neo4j_on_failed_subprocess(tmp_path: Path, monkeypatch) -> None:
    adapter = frameworks_mod.KAGAdapter()
    source_path = tmp_path / "source.txt"
    source_path.write_text("data", encoding="utf-8")
    run_dir = tmp_path / "run"
    (run_dir / "build").mkdir(parents=True)
    monkeypatch.setattr(frameworks_mod, "_normalize_kag_namespace", lambda _run_dir: "Kag123456789012")
    monkeypatch.setattr(adapter, "_create_project", lambda namespace, config_text: {"id": "1", "namespace": namespace})
    monkeypatch.setattr(adapter, "_sync_project_config", lambda project_id, namespace, config_text: None)
    monkeypatch.setattr(frameworks_mod, "run_command", lambda *args, **kwargs: type("Proc", (), {"returncode": 9})())
    monkeypatch.setattr(adapter, "_neo4j_summary", lambda namespace: (_ for _ in ()).throw(AssertionError("should not query neo4j on failed build")))
    metrics, graph = adapter.build(source_path, run_dir)
    assert metrics.build_status == "failed"
    assert metrics.build_error == "kag build exit code 9"
    assert graph == {}


def test_kag_build_runs_from_run_dir_and_writes_canonical_config(tmp_path: Path, monkeypatch) -> None:
    adapter = frameworks_mod.KAGAdapter()
    source_path = tmp_path / "source.txt"
    source_path.write_text("data", encoding="utf-8")
    run_dir = tmp_path / "run"
    (run_dir / "build").mkdir(parents=True)
    monkeypatch.setattr(frameworks_mod, "_normalize_kag_namespace", lambda _run_dir: "Kag123456789012")
    monkeypatch.setattr(adapter, "_create_project", lambda namespace, config_text: {"id": "31", "namespace": namespace})
    sync_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(adapter, "_sync_project_config", lambda project_id, namespace, config_text: sync_calls.append((project_id, namespace)))
    monkeypatch.setattr(adapter, "_neo4j_summary", lambda namespace, run_dir=None: {"nodes_total": 1, "documents_count": 1, "chunks_count": 1, "entity_rows": [], "relationship_rows": []})
    call: dict[str, Any] = {}

    def fake_run_command(*args, **kwargs):
        call["cwd"] = kwargs["cwd"]
        call["cmd"] = args[0]
        return type("Proc", (), {"returncode": 0})()

    monkeypatch.setattr(frameworks_mod, "run_command", fake_run_command)
    metrics, _ = adapter.build(source_path, run_dir)
    assert metrics.build_status == "success"
    assert call["cwd"] == run_dir
    assert (run_dir / "kag_config.yaml").exists()
    assert sync_calls == [("31", "Kag123456789012")]


def test_sync_project_config_updates_server_with_real_project_id(monkeypatch) -> None:
    adapter = frameworks_mod.KAGAdapter()
    calls: list[dict[str, Any]] = []

    class FakeProjectClient:
        def __init__(self, host_addr):
            calls.append({"host_addr": host_addr})

        def update(self, id, namespace, config, visibility=None, tag=None, userNo=None):
            calls.append({"id": id, "namespace": namespace, "config": config})

    monkeypatch.setattr(frameworks_mod, "_ensure_kag_imports", lambda: None)
    monkeypatch.setitem(sys.modules, "knext.project.client", types.SimpleNamespace(ProjectClient=FakeProjectClient))
    adapter._sync_project_config(
        "42",
        "Kag420000000000",
        """
project:
  id: "42"
  namespace: Kag420000000000
vectorize_model:
  type: openai
  base_url: http://127.0.0.1:8010/v1
""",
    )
    assert calls[0]["host_addr"] == frameworks_mod.KAG_LOCAL_HOST_ADDR
    assert calls[1]["id"] == "42"
    assert calls[1]["namespace"] == "Kag420000000000"
    assert calls[1]["config"]["project"]["id"] == "42"


def test_invoke_kag_query_with_trace_uses_trace_reporter(monkeypatch) -> None:
    events: list[str] = []

    class FakeTraceReporter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def start(self):
            events.append("start")

        async def stop(self):
            events.append("stop")

        def add_report_line(self, segment, tag_name, content, status, **kwargs):
            events.append(f"{segment}:{status}")

        def generate_report_data(self):
            return (
                type(
                    "TracePayload",
                    (),
                    {
                        "to_dict": lambda self: {
                            "decompose": [{"chunk_datas": [{"content": "ctx", "document_name": "doc.txt", "document_id": "doc-1"}]}],
                            "answer": "Atlas",
                        }
                    },
                )(),
                "FINISH",
            )

    monkeypatch.setitem(
        sys.modules,
        "kag.common.conf",
        types.SimpleNamespace(KAG_PROJECT_CONF=types.SimpleNamespace(host_addr="", language="en")),
    )
    monkeypatch.setitem(
        sys.modules,
        "kag.solver.main_solver",
        types.SimpleNamespace(
            do_qa_pipeline=lambda use_pipeline, query, qa_config, reporter, task_id, kb_project_ids: asyncio.sleep(0, result="Atlas"),
            is_chinese=lambda text: True,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "kag.solver.reporter.trace_log_reporter",
        types.SimpleNamespace(TraceLogReporter=FakeTraceReporter),
    )
    payload = asyncio.run(
        frameworks_mod._invoke_kag_query_with_trace(
            run_id="kag_20260716_090000",
            question="Как называется главный модуль проекта?",
            config={"kag_solver_pipeline": {}},
            project={"id": "26", "namespace": "KagNs"},
        )
    )
    assert payload["answer"] == "Atlas"
    assert payload["trace"]["decompose"][0]["chunk_datas"][0]["content"] == "ctx"
    assert events[0] == "start"
    assert "answer:FINISH" in events


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
