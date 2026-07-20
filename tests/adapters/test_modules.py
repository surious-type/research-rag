import sys
import types
import asyncio
from pathlib import Path

import yaml

from research_bench.adapters.kag import KAGAdapter
from research_bench.adapters.lightrag import LightRAGAdapter
from research_bench.adapters.msgraphrag import MsGraphRAGAdapter
from research_bench.runtime_config import ModelRuntimeConfig


def test_adapter_modules_export_expected_classes() -> None:
    assert MsGraphRAGAdapter.__name__ == "MsGraphRAGAdapter"
    assert LightRAGAdapter.__name__ == "LightRAGAdapter"
    assert KAGAdapter.__name__ == "KAGAdapter"


def test_lightrag_build_rag_uses_runtime_config(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeLightRAG:
        def __init__(self, **kwargs):
            captured["rag_kwargs"] = kwargs

    class FakeEmbeddingFunc:
        def __init__(self, **kwargs):
            captured["embedding_func_kwargs"] = kwargs

    async def fake_complete(**kwargs):
        captured["complete_kwargs"] = kwargs
        return "ok"

    class FakeOpenAIEmbed:
        max_token_size = 8192
        supports_asymmetric = False

        @staticmethod
        def func(**kwargs):
            captured["embed_kwargs"] = kwargs
            return [[0.1, 0.2]]

    runtime = ModelRuntimeConfig(
        base_url="https://api.openai.com/v1",
        embedding_base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5-nano",
        embedding_model="text-embedding-3-small",
        embedding_dimension=1536,
    )

    monkeypatch.setattr("research_bench.adapters.lightrag.load_model_runtime_config", lambda: runtime)
    monkeypatch.setitem(sys.modules, "lightrag", types.SimpleNamespace(LightRAG=FakeLightRAG))
    monkeypatch.setitem(sys.modules, "lightrag.llm.openai", types.SimpleNamespace(openai_complete_if_cache=fake_complete, openai_embed=FakeOpenAIEmbed))
    monkeypatch.setitem(sys.modules, "lightrag.utils", types.SimpleNamespace(EmbeddingFunc=FakeEmbeddingFunc))

    adapter = LightRAGAdapter()
    adapter._build_rag(tmp_path / "workspace")
    asyncio.run(captured["rag_kwargs"]["llm_model_func"]("probe"))

    assert captured["rag_kwargs"]["llm_model_name"] == "gpt-5-nano"
    assert captured["embedding_func_kwargs"]["model_name"] == "text-embedding-3-small"
    assert captured["embedding_func_kwargs"]["embedding_dim"] == 1536
    assert captured["rag_kwargs"]["default_llm_timeout"] == 240
    assert captured["complete_kwargs"]["temperature"] == 0.0


def test_lightrag_build_rag_uses_extended_timeout_for_local_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class FakeLightRAG:
        def __init__(self, **kwargs):
            captured["rag_kwargs"] = kwargs

    class FakeEmbeddingFunc:
        def __init__(self, **kwargs):
            captured["embedding_func_kwargs"] = kwargs

    class FakeOpenAIEmbed:
        max_token_size = 8192
        supports_asymmetric = False

        @staticmethod
        def func(**kwargs):
            return [[0.1, 0.2]]

    async def fake_complete(**kwargs):
        return "ok"

    runtime = ModelRuntimeConfig(
        base_url="http://127.0.0.1:8080/v1",
        embedding_base_url="http://127.0.0.1:8010/v1",
        api_key="local",
        model="/models/local.gguf",
        embedding_model="multilingual-e5-large",
        embedding_dimension=1024,
    )

    monkeypatch.setattr("research_bench.adapters.lightrag.load_model_runtime_config", lambda: runtime)
    monkeypatch.setitem(sys.modules, "lightrag", types.SimpleNamespace(LightRAG=FakeLightRAG))
    monkeypatch.setitem(sys.modules, "lightrag.llm.openai", types.SimpleNamespace(openai_complete_if_cache=fake_complete, openai_embed=FakeOpenAIEmbed))
    monkeypatch.setitem(sys.modules, "lightrag.utils", types.SimpleNamespace(EmbeddingFunc=FakeEmbeddingFunc))

    LightRAGAdapter()._build_rag(tmp_path / "workspace")

    assert captured["rag_kwargs"]["default_llm_timeout"] == 1800


def test_msgraphrag_apply_runtime_settings_uses_runtime_config() -> None:
    runtime = ModelRuntimeConfig(
        base_url="https://api.openai.com/v1",
        embedding_base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5-nano",
        embedding_model="text-embedding-3-small",
        embedding_dimension=1536,
    )

    rendered = MsGraphRAGAdapter()._apply_runtime_settings(
        """
completion_models:
  default_completion_model:
    model: /models/Qwen3.5-35B-A3B-Q4_K_M.gguf
    api_key: ${GRAPHRAG_API_KEY}
    api_base: http://127.0.0.1:8080/v1
embedding_models:
  default_embedding_model:
    model: multilingual-e5-large
    api_key: ${GRAPHRAG_API_KEY}
    api_base: http://127.0.0.1:8010/v1
""",
        runtime,
    )

    assert "gpt-5-nano" in rendered
    assert "text-embedding-3-small" in rendered
    assert "https://api.openai.com/v1" in rendered
    payload = yaml.safe_load(rendered)
    assert payload["completion_models"]["default_completion_model"]["call_args"]["temperature"] == 0.0
    assert payload["drift_search"]["local_search_temperature"] == 0.0
    assert payload["drift_search"]["reduce_temperature"] == 0.0


def test_msgraphrag_apply_runtime_settings_limits_concurrency_for_local_runtime() -> None:
    runtime = ModelRuntimeConfig(
        base_url="http://127.0.0.1:8080/v1",
        embedding_base_url="http://127.0.0.1:8010/v1",
        api_key="test-key",
        model="/models/Qwen3.5-35B-A3B-Q4_K_M.gguf",
        embedding_model="multilingual-e5-large",
        embedding_dimension=1024,
    )

    rendered = MsGraphRAGAdapter()._apply_runtime_settings(
        """
completion_models:
  default_completion_model:
    model: /models/Qwen3.5-35B-A3B-Q4_K_M.gguf
    api_key: ${GRAPHRAG_API_KEY}
    api_base: http://127.0.0.1:8080/v1
embedding_models:
  default_embedding_model:
    model: multilingual-e5-large
    api_key: ${GRAPHRAG_API_KEY}
    api_base: http://127.0.0.1:8010/v1
concurrent_requests: 25
extract_graph_nlp:
  concurrent_requests: 25
""",
        runtime,
    )

    payload = yaml.safe_load(rendered)
    assert payload["concurrent_requests"] == 1
    assert payload["extract_graph_nlp"]["concurrent_requests"] == 1


def test_msgraphrag_dummy_env_uses_runtime_api_key(monkeypatch) -> None:
    runtime = ModelRuntimeConfig(
        base_url="https://api.openai.com/v1",
        embedding_base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5-nano",
        embedding_model="text-embedding-3-small",
        embedding_dimension=1536,
    )
    monkeypatch.setattr("research_bench.adapters.msgraphrag.load_model_runtime_config", lambda: runtime)
    assert MsGraphRAGAdapter.__module__
    env = __import__("research_bench.adapters.msgraphrag", fromlist=["_dummy_env"])._dummy_env()
    assert env["GRAPHRAG_API_KEY"] == "test-key"


def test_lightrag_build_returns_failed_metrics_when_graph_flush_missing(tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_text("content", encoding="utf-8")
    run_dir = tmp_path / "run"
    work_dir = run_dir / "workspace"
    work_dir.mkdir(parents=True)
    (work_dir / "kv_store_full_docs.json").write_text("{}", encoding="utf-8")
    (work_dir / "kv_store_text_chunks.json").write_text("{}", encoding="utf-8")

    class FakeRag:
        async def initialize_storages(self):
            return None

        def insert(self, *args, **kwargs):
            return None

    monkeypatch.setattr(LightRAGAdapter, "_build_rag", lambda self, path: FakeRag())
    monkeypatch.setattr("research_bench.adapters.lightrag.parse_lightrag_outputs", lambda path: (_ for _ in ()).throw(FileNotFoundError("missing graphml")))

    metrics, graph = LightRAGAdapter().build(source_path, run_dir)
    assert metrics.build_status == "failed"
    assert "missing graphml" in str(metrics.build_error)
    assert graph == {}


def test_lightrag_build_surfaces_doc_status_failure_when_graph_missing(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "source.txt"
    source_path.write_text("content", encoding="utf-8")
    run_dir = tmp_path / "run"
    work_dir = run_dir / "workspace"
    work_dir.mkdir(parents=True)
    (work_dir / "kv_store_full_docs.json").write_text("{}", encoding="utf-8")
    (work_dir / "kv_store_doc_status.json").write_text(
        '{"doc-1":{"status":"failed","error_msg":"C[9/11]: extract LLM func: Worker execution timeout after 480s","chunks_count":11}}',
        encoding="utf-8",
    )

    class FakeRag:
        async def initialize_storages(self):
            return None

        def insert(self, *args, **kwargs):
            return None

    monkeypatch.setattr(LightRAGAdapter, "_build_rag", lambda self, path: FakeRag())
    monkeypatch.setattr(
        "research_bench.adapters.lightrag.parse_lightrag_outputs",
        lambda path: (_ for _ in ()).throw(FileNotFoundError("missing graphml")),
    )

    metrics, graph = LightRAGAdapter().build(source_path, run_dir)

    assert metrics.build_status == "failed"
    assert "Worker execution timeout after 480s" in str(metrics.build_error)
    assert "missing graphml" not in str(metrics.build_error)
    assert graph == {}


def test_lightrag_query_uses_lightrag_sync_query_helpers(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRag:
        async def initialize_storages(self):
            captured["initialized"] = True
            return None

        def query_data(self, question, params):
            captured["query_data"] = {"question": question, "params": params}
            return {"data": {"chunks": [{"content": "ctx", "file_path": "source.txt"}]}}

        def query(self, question, params):
            captured["query"] = {"question": question, "params": params}
            return "ok"

    monkeypatch.setattr(LightRAGAdapter, "_build_rag", lambda self, path: FakeRag())
    monkeypatch.setitem(sys.modules, "lightrag", types.SimpleNamespace(QueryParam=lambda **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "lightrag.utils", types.SimpleNamespace(always_get_an_event_loop=lambda: asyncio.new_event_loop()))

    answers = LightRAGAdapter().query(
        "run-1",
        tmp_path / "run",
        [{"id": "q1", "question": "Q?", "reference_answer": "A"}],
    )
    assert captured["initialized"] is True
    assert captured["query_data"]["question"] == "Q?"
    assert captured["query"]["question"] == "Q?"
    assert len(answers) == 1
