from __future__ import annotations

import json
import math
import sys
import types
from datetime import datetime
from pathlib import Path

import pandas as pd

import research_bench.data as data_mod
import research_bench.frameworks as frameworks_mod
import research_bench.runtime_config as runtime_config_mod
from research_bench.data import load_questions, load_source_info
from research_bench.metrics import normalize_graph_metrics
from research_bench.parsers import (
    parse_lightrag_outputs,
    parse_msgraphrag_outputs,
    parse_neo4j_summary,
)
from research_bench.ragas_eval import (
    prepare_ragas_rows,
    save_ragas_placeholder,
    summarize_ragas_scores,
)
from research_bench.utils import (
    aggregate_numeric,
    atomic_write_json,
    atomic_write_jsonl,
    latency_summary,
    sha256_file,
    utc_run_id,
)
from research_bench.shared.text import safe_float


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello\n", encoding="utf-8")
    assert (
        sha256_file(path)
        == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    )


def test_load_model_runtime_config_defaults(monkeypatch) -> None:
    for key in (
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(runtime_config_mod, "ROOT", Path("/tmp/does-not-exist"))
    config = runtime_config_mod.load_model_runtime_config()
    assert config.base_url == "http://127.0.0.1:8080/v1"
    assert config.embedding_base_url == "http://127.0.0.1:8010/v1"
    assert config.api_key == "local"
    assert config.model == "/models/Qwen3.5-35B-A3B-Q4_K_M.gguf"
    assert config.embedding_model == "multilingual-e5-large"


def test_load_model_runtime_config_reads_dotenv(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "OPENAI_BASE_URL=https://api.openai.com/v1",
                "OPENAI_API_KEY=test-key",
                "OPENAI_MODEL=gpt-5-nano",
                "OPENAI_EMBEDDING_MODEL=text-embedding-3-small",
            ]
        ),
        encoding="utf-8",
    )
    for key in (
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(runtime_config_mod, "ROOT", tmp_path)
    config = runtime_config_mod.load_model_runtime_config()
    assert config.base_url == "https://api.openai.com/v1"
    assert config.embedding_base_url == "https://api.openai.com/v1"
    assert config.api_key == "test-key"
    assert config.model == "gpt-5-nano"
    assert config.embedding_model == "text-embedding-3-small"


def test_load_model_runtime_config_reads_separate_embedding_base_url(
    tmp_path: Path, monkeypatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "OPENAI_BASE_URL=https://api.openai.com/v1",
                "OPENAI_EMBEDDING_BASE_URL=http://127.0.0.1:8010/v1",
                "OPENAI_API_KEY=test-key",
                "OPENAI_MODEL=gpt-5-nano",
                "OPENAI_EMBEDDING_MODEL=multilingual-e5-large",
            ]
        ),
        encoding="utf-8",
    )
    for key in (
        "OPENAI_BASE_URL",
        "OPENAI_EMBEDDING_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(runtime_config_mod, "ROOT", tmp_path)
    config = runtime_config_mod.load_model_runtime_config()
    assert config.base_url == "https://api.openai.com/v1"
    assert config.embedding_base_url == "http://127.0.0.1:8010/v1"
    assert config.api_key == "test-key"
    assert config.model == "gpt-5-nano"
    assert config.embedding_model == "multilingual-e5-large"


def test_load_model_runtime_config_keeps_local_embedding_endpoint_when_base_url_is_local(
    tmp_path: Path, monkeypatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_BASE_URL=http://127.0.0.1:8080/v1\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(runtime_config_mod, "ROOT", tmp_path)
    config = runtime_config_mod.load_model_runtime_config()
    assert config.base_url == "http://127.0.0.1:8080/v1"
    assert config.embedding_base_url == "http://127.0.0.1:8010/v1"


def test_run_id_generation_handles_collisions() -> None:
    now = datetime(2026, 7, 15, 18, 35, 42)
    assert (
        utc_run_id("msgraphrag", now=now, existing=set())
        == "msgraphrag_20260715_183542"
    )
    assert (
        utc_run_id("msgraphrag", now=now, existing={"msgraphrag_20260715_183542"})
        == "msgraphrag_20260715_183542_001"
    )


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


def test_aggregate_numeric_ignores_nan() -> None:
    result = aggregate_numeric([0.1, math.nan, None])
    assert result["mean"] == 0.1
    assert result["valid_count"] == 1
    assert result["failed_count"] == 2


def test_safe_float_rejects_nan() -> None:
    assert safe_float("nan") is None


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
            "relationship_rows": [
                {"type": "REL", "source_type": "Entity", "target_type": "Entity"}
            ],
        }
    )
    assert graph.nodes_total == 4
    assert graph.relationships_count == 1


def test_extract_kag_label_type() -> None:
    assert (
        frameworks_mod._extract_kag_label_type(["Kag123.Person"], "Kag123") == "Person"
    )
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
            {
                "source_id": "1",
                "target_id": "2",
                "source_labels": ["Kag123.Person"],
                "target_labels": ["Kag123.Organization"],
                "type": "WORKS_WITH",
            },
            {
                "source_id": "1",
                "target_id": "3",
                "source_labels": ["Kag123.Person"],
                "target_labels": ["Kag123.Chunk"],
                "type": "MENTIONS",
            },
            {
                "source_id": "1",
                "target_id": "5",
                "source_labels": ["Kag123.Person"],
                "target_labels": ["Other.Person"],
                "type": "CROSS_NS",
            },
        ],
        namespace="Kag123",
    )
    graph = parse_neo4j_summary(summary)
    assert summary["backend_document_nodes_count"] == 1
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
    assert (
        frameworks_mod._resolve_kag_neo4j_database("Kag260716095410", "neo4j")
        == "kag260716095410"
    )
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
    runtime = runtime_config_mod.ModelRuntimeConfig(
        base_url="https://api.openai.com/v1",
        embedding_base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5-nano",
        embedding_model="text-embedding-3-small",
        embedding_dimension=1536,
    )
    original_loader = frameworks_mod.kag_mod.load_model_runtime_config
    frameworks_mod.kag_mod.load_model_runtime_config = lambda: runtime
    config = frameworks_mod.build_kag_server_project_config(template)
    frameworks_mod.kag_mod.load_model_runtime_config = original_loader
    assert config["llm"]["base_url"] == "https://api.openai.com/v1"
    assert config["llm"]["api_key"] == "test-key"
    assert config["llm"]["model"] == "gpt-5-nano"
    assert config["vectorize_model"]["type"] == "openai"
    assert config["vectorize_model"]["base_url"] == "https://api.openai.com/v1"
    assert config["vectorize_model"]["model"] == "text-embedding-3-small"
    assert config["vectorize_model"]["vector_dimensions"] == 1536
    assert config["vectorizer"]["type"] == "openai"
    assert config["vectorizer"]["base_url"] == "https://api.openai.com/v1"


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
    _, diagnostics = frameworks_mod._resolve_kag_query_diagnostics(
        config, {"id": "24", "namespace": "KagNs"}
    )
    assert diagnostics["selected_pipeline"] == "kag_solver_pipeline"
    assert diagnostics["resolved_retriever_types"] == [
        "kg_cs_open_spg",
        "kg_fr_open_spg",
        "rc_open_spg",
    ]


def test_resolve_kag_query_diagnostics_fails_on_empty_retrievers() -> None:
    config = {"kag_solver_pipeline": {"type": "kag_static_pipeline", "executors": []}}
    try:
        frameworks_mod._resolve_kag_query_diagnostics(
            config, {"id": "24", "namespace": "KagNs"}
        )
    except RuntimeError as exc:
        assert "resolved_retriever_types is empty" in str(exc)
    else:
        raise AssertionError("Expected empty KAG retriever resolution to fail")


def test_prepare_kag_query_environment_calls_init_env(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "project.json").write_text(
        json.dumps({"id": "26", "namespace": "KagNs"}), encoding="utf-8"
    )
    (run_dir / "generated_kag_config.yaml").write_text(
        "kag_builder_pipeline:\n  register_path: /tmp/fake-kag\n",
        encoding="utf-8",
    )
    called: dict[str, str] = {}
    monkeypatch.setitem(
        sys.modules,
        "kag.common.conf",
        types.SimpleNamespace(
            init_env=lambda path: called.setdefault("config_path", path)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "kag.common.registry",
        types.SimpleNamespace(
            import_modules_from_path=lambda path: called.setdefault(
                "register_path", path
            )
        ),
    )
    project, config, config_path = frameworks_mod._prepare_kag_query_environment(
        run_dir
    )
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
    contexts = frameworks_mod._normalize_kag_context(
        {"decompose": [{"graph_data": ["Person[A] -WORKS_WITH-> Org[B]"]}]}
    )
    assert len(contexts) == 1
    assert contexts[0].source == "graph"
    assert "WORKS_WITH" in contexts[0].text


def test_normalize_kag_context_from_retriever_output_chunks_and_docs() -> None:
    contexts = frameworks_mod._normalize_kag_context(
        {
            "decompose": [
                {
                    "chunks": [
                        {
                            "content": "Chunk ctx",
                            "document_name": "source.txt",
                            "document_id": "doc-1",
                            "score": 0.7,
                        }
                    ],
                    "docs": [{"content": "Doc ctx", "title": "source.txt"}],
                }
            ]
        }
    )
    assert [item.text for item in contexts] == ["Chunk ctx", "Doc ctx"]


def test_extract_kag_retriever_errors_and_degraded_classification() -> None:
    trace = {
        "decompose": [
            {"retriever_method": "kg_fr_open_spg", "err_msg": "boom", "task": "Task<0>"}
        ]
    }
    errors = frameworks_mod._extract_kag_retriever_errors(trace)
    status, error = frameworks_mod._classify_kag_query_answer(
        "Atlas",
        [frameworks_mod.ContextItem(rank=1, text="ctx", source="doc")],
        errors,
        None,
    )
    assert errors == [
        {"retriever_method": "kg_fr_open_spg", "error": "boom", "task": "Task<0>"}
    ]
    assert status == "degraded"
    assert "boom" in str(error)


def test_prepare_ragas_rows() -> None:
    rows = prepare_ragas_rows(
        [
            {
                "question_id": "q1",
                "question": "What?",
                "answer": "Answer",
                "reference_answer": "Reference",
                "contexts": [{"text": "Context"}],
                "status": "success",
            }
        ]
    )
    assert rows[0]["retrieved_contexts"] == ["Context"]


def test_summarize_ragas_scores() -> None:
    summary = summarize_ragas_scores(
        [
            {
                "faithfulness": 0.8,
                "answer_relevancy": 0.7,
                "context_precision": None,
                "context_recall": 0.9,
            },
            {
                "faithfulness": None,
                "answer_relevancy": 0.5,
                "context_precision": 0.6,
                "context_recall": None,
            },
        ]
    )
    assert summary["faithfulness"]["valid_count"] == 1
    assert summary["answer_relevancy"]["failed_count"] == 0


def test_load_source_info_and_questions(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_text("one two\nthree", encoding="utf-8")
    questions_path = tmp_path / "questions.jsonl"
    rows = [
        {
            "id": f"q{i:03d}",
            "question": f"Question {i}",
            "reference_answer": f"Answer {i}",
            "question_type": "fact",
        }
        for i in range(1, 101)
    ]
    questions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
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
    pd.DataFrame(
        [{"type": "Person", "degree": 1}, {"type": "Chunk", "degree": 2}]
    ).to_parquet(output_dir / "entities.parquet")
    pd.DataFrame(
        [{"type": "RELATED_TO", "source_type": "Person", "target_type": "Person"}]
    ).to_parquet(output_dir / "relationships.parquet")
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
    (work_dir / "graph_chunk_entity_relation.graphml").write_text(
        graphml, encoding="utf-8"
    )
    (work_dir / "kv_store_full_docs.json").write_text(
        json.dumps({"doc1": {}}), encoding="utf-8"
    )
    (work_dir / "kv_store_text_chunks.json").write_text(
        json.dumps({"chunk1": {}}), encoding="utf-8"
    )
    graph = parse_lightrag_outputs(work_dir)
    assert graph.nodes_total == 2
    assert graph.edges_total == 1


def test_save_ragas_placeholder(tmp_path: Path) -> None:
    base = tmp_path / "ragas"
    save_ragas_placeholder(
        base, [{"question_id": "q1"}, {"question_id": "q2"}], "missing ragas"
    )
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    assert summary["faithfulness"]["failed_count"] == 2


def test_utils_misc_helpers(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    dst = tmp_path / "nested" / "dst.txt"
    src.write_text("123", encoding="utf-8")
    from research_bench.utils import (
        copy_file,
        file_size,
        load_json,
        percentile,
        safe_float,
    )

    copy_file(src, dst)
    atomic_write_json(tmp_path / "payload.json", {"x": 1})
    assert dst.read_text(encoding="utf-8") == "123"
    assert file_size(tmp_path) >= 3
    assert load_json(tmp_path / "payload.json") == {"x": 1}
    assert percentile([1.0, 2.0, 3.0], 0.95) is not None
    assert safe_float("1.5") == 1.5
    assert safe_float("bad") is None


def test_prepare_ragas_rows_skips_degraded_answers() -> None:
    rows = prepare_ragas_rows(
        [
            {
                "question_id": "q1",
                "question": "What?",
                "answer": "Answer",
                "reference_answer": "Reference",
                "contexts": [{"text": "Context"}],
                "status": "degraded",
            },
            {
                "question_id": "q2",
                "question": "What?",
                "answer": "Answer",
                "reference_answer": "Reference",
                "contexts": [{"text": "Context"}],
                "status": "success",
            },
        ]
    )
    assert [row["question_id"] for row in rows] == ["q2"]


def test_normalize_graph_metrics_excludes_numeric_relation_without_rules() -> None:
    graph = normalize_graph_metrics(
        nodes_total=2,
        edges_total=1,
        documents_count=1,
        chunks_count=0,
        communities_count=None,
        entity_rows=[{"type": "Works", "degree": 1}, {"type": "Others", "degree": 1}],
        relationship_rows=[
            {"type": "128", "source_type": "Works", "target_type": "Others"}
        ],
    )
    assert graph.relationships_count == 0
    assert "excluded_numeric_relationship_types" in graph.notes


def test_invoke_kag_query_with_trace_uses_trace_reporter(monkeypatch) -> None:
    import asyncio

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
                            "decompose": [
                                {
                                    "chunk_datas": [
                                        {
                                            "content": "ctx",
                                            "document_name": "doc.txt",
                                            "document_id": "doc-1",
                                        }
                                    ]
                                }
                            ],
                            "answer": "Atlas",
                        }
                    },
                )(),
                "FINISH",
            )

    monkeypatch.setitem(
        sys.modules,
        "kag.common.conf",
        types.SimpleNamespace(
            KAG_PROJECT_CONF=types.SimpleNamespace(host_addr="", language="en")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "kag.solver.main_solver",
        types.SimpleNamespace(
            do_qa_pipeline=lambda use_pipeline, query, qa_config, reporter, task_id, kb_project_ids: (
                asyncio.sleep(0, result="Atlas")
            ),
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
            question='Как называется главный модуль проекта "Орион-128"?',
            config={"kag_solver_pipeline": {}},
            project={"id": "26", "namespace": "KagNs"},
        )
    )
    assert payload["answer"] == "Atlas"
    assert payload["trace"]["decompose"][0]["chunk_datas"][0]["content"] == "ctx"
    assert events[0] == "start"
    assert "answer:FINISH" in events
