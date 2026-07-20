from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

import research_bench.frameworks as frameworks_mod


def test_kag_build_skips_neo4j_on_failed_subprocess(tmp_path: Path, monkeypatch) -> None:
    adapter = frameworks_mod.KAGAdapter()
    source_path = tmp_path / "source.txt"
    source_path.write_text("data", encoding="utf-8")
    run_dir = tmp_path / "run"
    (run_dir / "build").mkdir(parents=True)
    monkeypatch.setattr(frameworks_mod, "_normalize_kag_namespace", lambda _run_dir: "Kag123456789012")
    monkeypatch.setattr(
        adapter,
        "_create_supported_project",
        lambda run_dir, namespace, bootstrap_config_text, config_path: (
            {"id": "1", "namespace": namespace},
            {"official_cli_schema": {"types": [{"type_name": "Chunk", "properties": [{"name": "id", "index_type": None}, {"name": "name", "index_type": None}, {"name": "content", "index_type": "TEXT_AND_VECTOR"}]}, {"type_name": "Person", "spg_type": "Entity", "properties": [{"name": "id", "index_type": None}, {"name": "name", "index_type": None}, {"name": "desc", "index_type": "TEXT_AND_VECTOR"}]}]}},
        ),
    )
    monkeypatch.setattr(adapter, "_sync_project_config", lambda project_id, namespace, config_text: None)
    monkeypatch.setattr(frameworks_mod, "run_command", lambda *args, **kwargs: type("Proc", (), {"returncode": 9})())
    monkeypatch.setattr(adapter, "_load_schema_snapshot", lambda project_id: {"types": []})
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
    monkeypatch.setattr(
        adapter,
        "_create_supported_project",
        lambda run_dir, namespace, bootstrap_config_text, config_path: (
            {"id": "31", "namespace": namespace},
            {"official_cli_schema": {"types": [{"type_name": "Chunk", "properties": [{"name": "id", "index_type": None}, {"name": "name", "index_type": None}, {"name": "content", "index_type": "TEXT_AND_VECTOR"}]}, {"type_name": "Person", "spg_type": "Entity", "properties": [{"name": "id", "index_type": None}, {"name": "name", "index_type": None}, {"name": "desc", "index_type": "TEXT_AND_VECTOR"}]}]}},
        ),
    )
    sync_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(adapter, "_sync_project_config", lambda project_id, namespace, config_text: sync_calls.append((project_id, namespace)))
    monkeypatch.setattr(
        adapter,
        "_neo4j_summary",
        lambda namespace, run_dir=None: {
            "nodes_total": 1,
            "documents_count": 0,
            "backend_document_nodes_count": 0,
            "chunks_count": 1,
            "entity_rows": [],
            "relationship_rows": [],
            "communities_count": None,
        },
    )
    call: dict[str, Any] = {}

    def fake_run_command(*args, **kwargs):
        call["cwd"] = kwargs["cwd"]
        call["cmd"] = args[0]
        return type("Proc", (), {"returncode": 0})()

    monkeypatch.setattr(frameworks_mod, "run_command", fake_run_command)
    monkeypatch.setattr(adapter, "_load_schema_snapshot", lambda project_id: {"types": []})
    metrics, _ = adapter.build(source_path, run_dir)
    assert metrics.build_status == "success"
    assert metrics.documents_count == 1
    assert metrics.input_documents_count == 1
    assert metrics.backend_document_nodes_count == 0
    assert call["cwd"] == run_dir
    assert (run_dir / "kag_config.yaml").exists()
    assert sync_calls == [("31", "Kag123456789012")]


def test_validate_required_schema_detects_missing_chunk_content_and_vector_metadata() -> None:
    issues = frameworks_mod._validate_required_schema(
        {
            "types": [
                {"type_name": "Chunk", "properties": [{"name": "id", "index_type": None}, {"name": "name", "index_type": None}]},
                {"type_name": "Person", "spg_type": "Entity", "properties": [{"name": "id", "index_type": None}, {"name": "name", "index_type": "TEXT"}]},
            ]
        }
    )
    assert "missing schema property Chunk.content" in issues
    assert "missing vector index metadata for Entity.name" in issues


def test_validate_required_schema_accepts_official_entity_desc_vector() -> None:
    issues = frameworks_mod._validate_required_schema(
        {
            "types": [
                {
                    "type_name": "Chunk",
                    "properties": [
                        {"name": "id", "index_type": None},
                        {"name": "name", "index_type": None},
                        {"name": "content", "index_type": "TEXT_AND_VECTOR"},
                    ],
                },
                {
                    "type_name": "Person",
                    "spg_type": "Entity",
                    "properties": [
                        {"name": "id", "index_type": None},
                        {"name": "name", "index_type": None},
                        {"name": "desc", "index_type": "TEXT_AND_VECTOR"},
                    ],
                },
            ]
        }
    )
    assert issues == []


def test_schema_diff_tracks_direct_rest_vs_official_schema() -> None:
    diff = frameworks_mod._schema_diff(
        {"types": [{"type_name": "Chunk", "properties": [{"name": "id", "type": "Text", "index_type": None}], "relations": []}]},
        {"types": [{"type_name": "Chunk", "properties": [{"name": "id", "type": "Text", "index_type": None}, {"name": "content", "type": "Text", "index_type": "TEXT_AND_VECTOR"}], "relations": []}]},
    )
    assert diff["added_types"] == []
    assert diff["changed_types"][0]["type_name"] == "Chunk"
    assert diff["changed_types"][0]["added_properties"] == ["content"]


def test_kag_build_fails_before_subprocess_when_schema_invalid(tmp_path: Path, monkeypatch) -> None:
    adapter = frameworks_mod.KAGAdapter()
    source_path = tmp_path / "source.txt"
    source_path.write_text("data", encoding="utf-8")
    run_dir = tmp_path / "run"
    (run_dir / "build").mkdir(parents=True)
    monkeypatch.setattr(frameworks_mod, "_normalize_kag_namespace", lambda _run_dir: "Kag123456789012")
    monkeypatch.setattr(
        adapter,
        "_create_supported_project",
        lambda run_dir, namespace, bootstrap_config_text, config_path: (
            {"id": "31", "namespace": namespace},
            {"official_cli_schema": {"types": [{"type_name": "Chunk", "properties": [{"name": "id", "index_type": None}]}, {"type_name": "Entity", "properties": [{"name": "id", "index_type": None}]}]}},
        ),
    )
    monkeypatch.setattr(adapter, "_sync_project_config", lambda project_id, namespace, config_text: None)
    called = {"run_command": False}

    def fake_run_command(*args, **kwargs):
        called["run_command"] = True
        return type("Proc", (), {"returncode": 0})()

    monkeypatch.setattr(frameworks_mod, "run_command", fake_run_command)
    metrics, graph = adapter.build(source_path, run_dir)
    assert metrics.build_status == "failed"
    assert "missing schema property Chunk.name" in str(metrics.build_error)
    assert called["run_command"] is False
    assert graph == {}


def test_required_vector_indexes_online_filters_namespace() -> None:
    adapter = frameworks_mod.KAGAdapter()
    rows = [
        {"name": "other", "state": "ONLINE", "labelsOrTypes": ["OtherNs.Person"], "properties": ["_name_vector"]},
        {"name": "good", "state": "ONLINE", "labelsOrTypes": ["Kag123.Person"], "properties": ["_content_vector"]},
    ]
    assert adapter._required_vector_indexes_online(rows, {"Kag123.Person"}) is True
    assert adapter._required_vector_indexes_online(rows[:1], {"Kag123.Person"}) is False


def test_summarize_vector_probe_status() -> None:
    assert frameworks_mod._summarize_vector_probe_status([{"status": "WARN"}]) == "WARN"
    assert frameworks_mod._summarize_vector_probe_status([{"status": "PASS"}, {"status": "WARN"}]) == "PASS"
    assert frameworks_mod._summarize_vector_probe_status([{"status": "PASS"}, {"status": "FAIL"}]) == "FAIL"


def test_build_vector_search_probes_statuses(monkeypatch, tmp_path: Path) -> None:
    adapter = frameworks_mod.KAGAdapter()
    source_path = tmp_path / "source.txt"
    source_path.write_text("Орион-128\nАтлас\n", encoding="utf-8")

    class FakeSearchClient:
        def __init__(self, host_addr=None, project_id=None):
            self.calls = []

        def search_vector(self, label, property_key, query_vector, topk=3, ef_search=21):
            self.calls.append((label, property_key))
            if property_key == "name":
                return [{"score": 0.9, "node": {"id": "e1", "name": "Атлас"}}]
            return []

    monkeypatch.setattr(adapter, "_vectorize_probe_query", lambda query: [0.1, 0.2])
    monkeypatch.setitem(sys.modules, "knext.search.client", types.SimpleNamespace(SearchClient=FakeSearchClient))
    monkeypatch.setitem(sys.modules, "knext.schema.client", types.SimpleNamespace(CHUNK_TYPE="Chunk"))
    monkeypatch.setitem(
        sys.modules,
        "kag.interface.solver.model.schema_utils",
        types.SimpleNamespace(SchemaUtils=lambda config: types.SimpleNamespace(get_label_within_prefix=lambda label: "Kag123.Chunk")),
    )
    monkeypatch.setitem(sys.modules, "kag.common.config", types.SimpleNamespace(LogicFormConfiguration=lambda payload: payload))
    probes = adapter._build_vector_search_probes("36", source_path, "Kag123")
    assert probes[0]["status"] == "PASS"
    assert probes[1]["status"] == "WARN"


def test_build_vector_search_probes_fail_on_exception(monkeypatch, tmp_path: Path) -> None:
    adapter = frameworks_mod.KAGAdapter()
    source_path = tmp_path / "source.txt"
    source_path.write_text("Орион-128\n", encoding="utf-8")

    class FakeSearchClient:
        def __init__(self, host_addr=None, project_id=None):
            return None

        def search_vector(self, label, property_key, query_vector, topk=3, ef_search=21):
            raise RuntimeError("server boom")

    monkeypatch.setattr(adapter, "_vectorize_probe_query", lambda query: [0.1, 0.2])
    monkeypatch.setitem(sys.modules, "knext.search.client", types.SimpleNamespace(SearchClient=FakeSearchClient))
    monkeypatch.setitem(sys.modules, "knext.schema.client", types.SimpleNamespace(CHUNK_TYPE="Chunk"))
    monkeypatch.setitem(
        sys.modules,
        "kag.interface.solver.model.schema_utils",
        types.SimpleNamespace(SchemaUtils=lambda config: types.SimpleNamespace(get_label_within_prefix=lambda label: "Kag123.Chunk")),
    )
    monkeypatch.setitem(sys.modules, "kag.common.config", types.SimpleNamespace(LogicFormConfiguration=lambda payload: payload))
    probes = adapter._build_vector_search_probes("36", source_path, "Kag123")
    assert all(probe["status"] == "FAIL" for probe in probes)


def test_build_vector_search_probes_use_generated_run_config(monkeypatch, tmp_path: Path) -> None:
    adapter = frameworks_mod.KAGAdapter()
    source_path = tmp_path / "source.txt"
    source_path.write_text("Орион-128\n", encoding="utf-8")
    config_path = tmp_path / "generated_kag_config.yaml"
    config_path.write_text(
        """
vectorize_model:
  type: openai
  base_url: https://api.openai.com/v1
  model: text-embedding-3-small
  vector_dimensions: 1536
""",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeSearchClient:
        def __init__(self, host_addr=None, project_id=None):
            return None

        def search_vector(self, label, property_key, query_vector, topk=3, ef_search=21):
            return [{"score": 0.9}]

    class FakeVectorModel:
        def vectorize(self, query):
            return [0.1, 0.2, 0.3]

    def fake_from_config(config):
        captured["vectorize_model_config"] = config
        return FakeVectorModel()

    monkeypatch.setitem(sys.modules, "knext.search.client", types.SimpleNamespace(SearchClient=FakeSearchClient))
    monkeypatch.setitem(sys.modules, "knext.schema.client", types.SimpleNamespace(CHUNK_TYPE="Chunk"))
    monkeypatch.setitem(
        sys.modules,
        "kag.interface.solver.model.schema_utils",
        types.SimpleNamespace(SchemaUtils=lambda config: types.SimpleNamespace(get_label_within_prefix=lambda label: "Kag123.Chunk")),
    )
    monkeypatch.setitem(sys.modules, "kag.common.config", types.SimpleNamespace(LogicFormConfiguration=lambda payload: payload))
    monkeypatch.setitem(
        sys.modules,
        "kag.interface",
        types.SimpleNamespace(VectorizeModelABC=types.SimpleNamespace(from_config=fake_from_config)),
    )

    probes = adapter._build_vector_search_probes("36", source_path, "Kag123", config_path=config_path)
    assert len(probes) == 2
    assert captured["vectorize_model_config"]["model"] == "text-embedding-3-small"
    assert captured["vectorize_model_config"]["vector_dimensions"] == 1536


def test_provision_default_text_index_uses_namespace_labels_only(tmp_path: Path, monkeypatch) -> None:
    adapter = frameworks_mod.KAGAdapter()
    run_dir = tmp_path / "run"
    (run_dir / "build").mkdir(parents=True)
    monkeypatch.setenv("KAG_NEO4J_DATABASE", "neo4j")

    class FakeResult:
        def consume(self):
            return None

    class FakeSession:
        def __init__(self):
            self.queries: list[str] = []

        def run(self, query, **params):
            self.queries.append(query)
            if query == "SHOW FULLTEXT INDEXES":
                return [type("Record", (), {"data": lambda self: {"name": "_default_text_index", "state": "ONLINE"}})()]
            if "db.schema.nodeTypeProperties" in query:
                rows = [
                    {"nodeLabels": ["Kag123.Person"], "propertyName": "name", "propertyTypes": ["String"], "mandatory": False},
                    {"nodeLabels": ["Other.Person"], "propertyName": "name", "propertyTypes": ["String"], "mandatory": False},
                ]
                return [type("Record", (), {"data": lambda self, row=row: row})() for row in rows]
            if "RETURN labels(n) AS labels, properties(n) AS props" in query:
                rows = [{"labels": ["Kag123.Person"], "props": {"name": "Atlas"}}]
                return [type("Record", (), {"data": lambda self, row=row: row})() for row in rows]
            return FakeResult()

    session = FakeSession()
    payload = adapter._provision_default_text_index(session, "Kag123", {"Kag123.Person"}, run_dir)
    assert payload["method"] == "benchmark_post_build_provisioner"
    ddl = payload["ddl"]
    assert "Kag123.Person" in ddl
    assert "Other.Person" not in ddl


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


def test_kag_local_hosts_context_preserves_proxy_and_extends_no_proxy(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1081")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1081")
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:1080")
    monkeypatch.setenv("NO_PROXY", "example.com")
    monkeypatch.setenv("no_proxy", "internal.local")

    with frameworks_mod.kag_mod._without_proxy_for_kag_local_hosts():
        assert frameworks_mod.kag_mod.os.environ["HTTP_PROXY"] == "http://127.0.0.1:1081"
        assert frameworks_mod.kag_mod.os.environ["HTTPS_PROXY"] == "http://127.0.0.1:1081"
        assert frameworks_mod.kag_mod.os.environ["ALL_PROXY"] == "socks5://127.0.0.1:1080"

        no_proxy = frameworks_mod.kag_mod.os.environ["NO_PROXY"].split(",")
        no_proxy_lower = frameworks_mod.kag_mod.os.environ["no_proxy"].split(",")
        for host in ("127.0.0.1", "localhost", "::1", "host.docker.internal", "172.17.0.1"):
            assert host in no_proxy
            assert host in no_proxy_lower
        assert "example.com" in no_proxy
        assert "internal.local" in no_proxy_lower

    assert frameworks_mod.kag_mod.os.environ["HTTP_PROXY"] == "http://127.0.0.1:1081"
    assert frameworks_mod.kag_mod.os.environ["HTTPS_PROXY"] == "http://127.0.0.1:1081"
    assert frameworks_mod.kag_mod.os.environ["ALL_PROXY"] == "socks5://127.0.0.1:1080"
    assert frameworks_mod.kag_mod.os.environ["NO_PROXY"] == "example.com"
    assert frameworks_mod.kag_mod.os.environ["no_proxy"] == "internal.local"



def test_kag_runtime_default_llm_url_uses_loopback() -> None:
    kag_mod = frameworks_mod.kag_mod

    assert (
        kag_mod._kag_runtime_base_url(
            kag_mod.DEFAULT_OPENAI_BASE_URL,
            embedding=False,
        )
        == "http://127.0.0.1:8080/v1"
    )
