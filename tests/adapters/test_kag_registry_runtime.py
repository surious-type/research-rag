from __future__ import annotations

import json
import types
from typing import Any


def test_registry_normalizes_json_string_task_arguments() -> None:
    import research_bench._kag_registry as kag_registry

    args, normalized, error = kag_registry._normalize_task_arguments('{"query":"Q","logic_form_node":{"type":"x"}}')
    assert normalized is True
    assert error is None
    assert args["query"] == "Q"


def test_registry_rejects_invalid_string_task_arguments() -> None:
    import research_bench._kag_registry as kag_registry

    args, normalized, error = kag_registry._normalize_task_arguments("not-json")
    assert normalized is False
    assert args == "not-json"
    assert "not a JSON object" in str(error)


def test_registry_normalize_ner_item_and_diag_helpers(tmp_path, monkeypatch) -> None:
    import research_bench._kag_registry as kag_registry

    monkeypatch.setenv("KAG_BENCH_QUERY_DIR", str(tmp_path / "query"))
    monkeypatch.setenv("KAG_BENCH_QUESTION_ID", "q001")
    item, note = kag_registry._normalize_ner_item("Atlas")
    assert item == {"name": "Atlas", "category": "Others", "official_name": "Atlas"}
    assert "normalized plain-string" in str(note)
    kag_registry._merge_diag({"kg_fr_arguments_type": "str"})
    kag_registry._append_error_log("boom")
    diag = json.loads((tmp_path / "query" / "_bench_diag" / "q001.json").read_text(encoding="utf-8"))
    assert diag["kg_fr_arguments_type"] == "str"
    assert "boom" in (tmp_path / "query" / "kg_fr_error.log").read_text(encoding="utf-8")


def test_registry_chunk_title_from_node() -> None:
    import research_bench._kag_registry as kag_registry

    assert kag_registry._chunk_title_from_node({"title": "Atlas"}, "chunk-1") == "Atlas"
    assert kag_registry._chunk_title_from_node({"document_name": "doc.txt"}, "chunk-1") == "doc.txt"
    assert kag_registry._chunk_title_from_node({}, "chunk-1") == "chunk-1"


def test_registry_compatibility_ner_invoke_normalizes_strings(monkeypatch) -> None:
    import research_bench._kag_registry as kag_registry

    monkeypatch.setattr(kag_registry.Ner, "_parse_ner_list", lambda self, query: ["Atlas", {"name": "Orion-128", "category": "Works"}])
    ner = object.__new__(kag_registry.BenchmarkCompatibilityNer)
    result = kag_registry.BenchmarkCompatibilityNer.invoke(ner, "query")
    assert [item.entity_name for item in result] == ["Atlas", "Orion-128"]


def test_registry_compatibility_kg_fr_invoke_handles_invalid_arguments(monkeypatch, tmp_path) -> None:
    import research_bench._kag_registry as kag_registry

    monkeypatch.setenv("KAG_BENCH_QUERY_DIR", str(tmp_path / "query"))
    monkeypatch.setenv("KAG_BENCH_QUESTION_ID", "q001")
    retriever = object.__new__(kag_registry.BenchmarkCompatibilityKgFrRetriever)
    retriever._name = "kg_fr_open_spg"
    task = type("Task", (), {"arguments": "not-json", "__str__": lambda self: "Task<0>"})()
    output = kag_registry.BenchmarkCompatibilityKgFrRetriever.invoke(retriever, task)
    assert output.err_msg
    diag = json.loads((tmp_path / "query" / "_bench_diag" / "q001.json").read_text(encoding="utf-8"))
    assert diag["kg_fr_arguments_type"] == "str"
    assert diag["kg_fr_arguments_normalized"] is False


def test_registry_compatibility_kg_fr_invoke_normalizes_json_and_calls_super(monkeypatch, tmp_path) -> None:
    import research_bench._kag_registry as kag_registry

    monkeypatch.setenv("KAG_BENCH_QUERY_DIR", str(tmp_path / "query"))
    monkeypatch.setenv("KAG_BENCH_QUESTION_ID", "q002")
    captured: dict[str, Any] = {}

    def fake_super(self, task, **kwargs):
        captured["arguments"] = task.arguments
        return type("Output", (), {"err_msg": "", "retriever_method": "kg_fr_open_spg"})()

    monkeypatch.setattr(kag_registry.KgFreeRetrieverWithOpenSPGRetriever, "invoke", fake_super)
    retriever = object.__new__(kag_registry.BenchmarkCompatibilityKgFrRetriever)
    retriever._name = "kg_fr_open_spg"
    task = type("Task", (), {"arguments": '{"query":"Q","logic_form_node":{"type":"x"}}', "__str__": lambda self: "Task<1>"})()
    output = kag_registry.BenchmarkCompatibilityKgFrRetriever.invoke(retriever, task)
    assert output.err_msg == ""
    assert captured["arguments"]["query"] == "Q"
    diag = json.loads((tmp_path / "query" / "_bench_diag" / "q002.json").read_text(encoding="utf-8"))
    assert diag["kg_fr_arguments_normalized"] is True


def test_registry_compatibility_kg_fr_invoke_logs_traceback_on_exception(monkeypatch, tmp_path) -> None:
    import research_bench._kag_registry as kag_registry

    monkeypatch.setenv("KAG_BENCH_QUERY_DIR", str(tmp_path / "query"))
    monkeypatch.setenv("KAG_BENCH_QUESTION_ID", "q003")

    def fake_super(self, task, **kwargs):
        raise AttributeError("'str' object has no attribute 'get'")

    monkeypatch.setattr(kag_registry.KgFreeRetrieverWithOpenSPGRetriever, "invoke", fake_super)
    retriever = object.__new__(kag_registry.BenchmarkCompatibilityKgFrRetriever)
    retriever._name = "kg_fr_open_spg"
    task = type("Task", (), {"arguments": {"query": "Q", "logic_form_node": {"type": "x"}}, "__str__": lambda self: "Task<2>"})()
    try:
        kag_registry.BenchmarkCompatibilityKgFrRetriever.invoke(retriever, task)
    except AttributeError as exc:
        assert "has no attribute 'get'" in str(exc)
    else:
        raise AssertionError("Expected retriever exception to be re-raised")
    log_text = (tmp_path / "query" / "kg_fr_error.log").read_text(encoding="utf-8")
    assert "AttributeError" in log_text


def test_registry_benchmark_ppr_retriever_handles_missing_name_and_none_results(monkeypatch, tmp_path) -> None:
    import research_bench._kag_registry as kag_registry

    monkeypatch.setenv("KAG_BENCH_QUERY_DIR", str(tmp_path / "query"))
    monkeypatch.setenv("KAG_BENCH_QUESTION_ID", "q004")
    retriever = object.__new__(kag_registry.BenchmarkCompatibilityPprChunkRetriever)
    retriever.graph_api = types.SimpleNamespace(
        get_entity_prop_by_id=lambda label, biz_id: (
            types.SimpleNamespace(items=lambda: {"content": "Chunk body", "title": "Atlas"}.items())
            if biz_id == "chunk-1"
            else None
        )
    )
    retriever.schema_helper = types.SimpleNamespace(get_label_within_prefix=lambda label: "Kag123.Chunk")
    docs = kag_registry.BenchmarkCompatibilityPprChunkRetriever.get_all_docs_by_id(
        retriever,
        ["query"],
        [("chunk-1", 0.8), ("chunk-2", 0.2), None],
        3,
    )
    assert len(docs) == 1
    assert docs[0].title == "Atlas"
    assert docs[0].content == "Chunk body"
    diag = json.loads((tmp_path / "query" / "_bench_diag" / "q004.json").read_text(encoding="utf-8"))
    assert diag["ppr_requested_doc_ids"] == ["chunk-1", "chunk-2"]
    assert diag["ppr_loaded_doc_ids"] == ["chunk-1"]
    assert "chunk-2" in diag["ppr_missing_doc_ids"]


def test_registry_benchmark_ppr_retriever_uses_chunk_id_as_title(monkeypatch, tmp_path) -> None:
    import research_bench._kag_registry as kag_registry

    monkeypatch.setenv("KAG_BENCH_QUERY_DIR", str(tmp_path / "query"))
    monkeypatch.setenv("KAG_BENCH_QUESTION_ID", "q005")
    retriever = object.__new__(kag_registry.BenchmarkCompatibilityPprChunkRetriever)
    retriever.graph_api = types.SimpleNamespace(
        get_entity_prop_by_id=lambda label, biz_id: types.SimpleNamespace(items=lambda: {"content": "Chunk body"}.items())
    )
    retriever.schema_helper = types.SimpleNamespace(get_label_within_prefix=lambda label: "Kag123.Chunk")
    docs = kag_registry.BenchmarkCompatibilityPprChunkRetriever.get_all_docs_by_id(retriever, ["query"], [("chunk-9", 0.5)], 1)
    assert docs[0].title == "chunk-9"
    diag = json.loads((tmp_path / "query" / "_bench_diag" / "q005.json").read_text(encoding="utf-8"))
    assert diag["ppr_chunk_properties"]["chunk-9"]["title"] == "chunk-9"


def test_registry_record_retriever_result(monkeypatch, tmp_path) -> None:
    import research_bench._kag_registry as kag_registry

    monkeypatch.setenv("KAG_BENCH_QUERY_DIR", str(tmp_path / "query"))
    monkeypatch.setenv("KAG_BENCH_QUESTION_ID", "q006")
    graph = type("Graph", (), {"get_all_spo": lambda self: ["A-r-B"]})()
    output = type("Output", (), {"chunks": [1], "graphs": [graph], "err_msg": ""})()
    kag_registry._record_retriever_result("rc_open_spg", output)
    diag = json.loads((tmp_path / "query" / "_bench_diag" / "q006.json").read_text(encoding="utf-8"))
    assert diag["retriever_results"]["rc_open_spg"]["status"] == "ok"
    assert diag["retriever_results"]["rc_open_spg"]["chunks_count"] == 1
    assert diag["retriever_results"]["rc_open_spg"]["graph_spo_count"] == 1
