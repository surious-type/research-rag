from __future__ import annotations

import asyncio
from contextlib import contextmanager
import functools
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .models import BuildMetrics, ContextItem, QueryAnswer
from .parsers import parse_lightrag_outputs, parse_msgraphrag_outputs, parse_neo4j_summary
from .utils import (
    atomic_write_json,
    copy_file,
    ensure_dir,
    file_size,
    run_command,
)


ROOT = Path(__file__).resolve().parents[1]
KAG_ROOT = ROOT / "frameworks" / "kag"
KAG_NEO4J_EXCLUDED_TYPES = {"AtomicQuery", "Chunk", "Doc", "Document", "KnowledgeUnit", "Outline", "Summary"}
KAG_NEO4J_DEFAULT_URI = "bolt://127.0.0.1:17687"
KAG_NEO4J_DEFAULT_USER = "neo4j"
KAG_NEO4J_DEFAULT_PASSWORD = "neo4j@openspg"
KAG_NEO4J_DEFAULT_DATABASE = "neo4j"
KAG_PROXY_ENV_KEYS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "WS_PROXY",
    "ws_proxy",
    "WSS_PROXY",
    "wss_proxy",
)


def _dummy_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("GRAPHRAG_API_KEY", "local")
    return env


def _ensure_kag_imports() -> None:
    kag_root = str(KAG_ROOT)
    if kag_root not in sys.path:
        sys.path.insert(0, kag_root)


def get_kag_neo4j_config() -> dict[str, str]:
    return {
        "uri": os.getenv("KAG_NEO4J_URI", KAG_NEO4J_DEFAULT_URI),
        "user": os.getenv("KAG_NEO4J_USER", KAG_NEO4J_DEFAULT_USER),
        "password": os.getenv("KAG_NEO4J_PASSWORD", KAG_NEO4J_DEFAULT_PASSWORD),
        "database": os.getenv("KAG_NEO4J_DATABASE", KAG_NEO4J_DEFAULT_DATABASE),
    }


def _normalize_kag_namespace(run_dir: Path) -> str:
    digits = "".join(ch for ch in run_dir.name if ch.isdigit())
    suffix = digits[-12:] if digits else str(int(time.time()))[-12:]
    namespace = f"Kag{suffix}"
    return namespace[:16]


def _rewrite_kag_server_config(value: Any) -> Any:
    if isinstance(value, dict):
        updated = {key: _rewrite_kag_server_config(item) for key, item in value.items()}
        if updated.get("type") == "openai" and updated.get("base_url") == "http://127.0.0.1:8080/v1":
            updated["base_url"] = "http://host.docker.internal:8080/v1"
        return updated
    if isinstance(value, list):
        return [_rewrite_kag_server_config(item) for item in value]
    return value


def build_kag_server_project_config(config_text: str) -> dict[str, Any]:
    config = yaml.safe_load(config_text)
    config = _rewrite_kag_server_config(config)
    mock_vectorizer = {"type": "mock", "vector_dimensions": 1024}
    config["vectorize_model"] = mock_vectorizer
    config["vectorizer"] = mock_vectorizer
    return config


@contextmanager
def _without_proxy_for_kag_local_hosts():
    saved_values = {key: os.environ.get(key) for key in (*KAG_PROXY_ENV_KEYS, "NO_PROXY", "no_proxy")}
    try:
        for key in KAG_PROXY_ENV_KEYS:
            os.environ.pop(key, None)
        no_proxy_hosts = ["127.0.0.1", "localhost", "::1", "host.docker.internal", "172.17.0.1"]
        for key in ("NO_PROXY", "no_proxy"):
            current = saved_values.get(key) or ""
            parts = [item.strip() for item in current.split(",") if item.strip()]
            merged = ",".join(dict.fromkeys([*parts, *no_proxy_hosts]))
            os.environ[key] = merged
        yield
    finally:
        for key, value in saved_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _extract_kag_label_type(labels: list[str], namespace: str) -> str | None:
    prefix = f"{namespace}."
    for label in labels:
        if not isinstance(label, str) or not label.startswith(prefix):
            continue
        suffix = label[len(prefix):].strip(".")
        if not suffix:
            continue
        entity_type = suffix.split(".")[0].strip()
        if entity_type:
            return entity_type
    return None


def summarize_kag_namespace_graph(node_rows: list[dict[str, Any]], edge_rows: list[dict[str, Any]], namespace: str) -> dict[str, Any]:
    semantic_degree: Counter[str] = Counter()
    entity_rows: list[dict[str, Any]] = []
    relationship_rows: list[dict[str, Any]] = []
    documents_count = 0
    chunks_count = 0

    for row in node_rows:
        node_type = _extract_kag_label_type(row.get("labels", []), namespace)
        if node_type is None:
            continue
        if node_type in {"Doc", "Document"}:
            documents_count += 1
            continue
        if node_type == "Chunk":
            chunks_count += 1
            continue
        if node_type in KAG_NEO4J_EXCLUDED_TYPES:
            continue
        entity_rows.append({"type": node_type, "node_id": row.get("node_id")})

    for row in edge_rows:
        source_type = _extract_kag_label_type(row.get("source_labels", []), namespace)
        target_type = _extract_kag_label_type(row.get("target_labels", []), namespace)
        rel_type = str(row.get("type") or "").strip()
        if not source_type or not target_type or not rel_type:
            continue
        if source_type in KAG_NEO4J_EXCLUDED_TYPES or target_type in KAG_NEO4J_EXCLUDED_TYPES:
            continue
        relationship_rows.append({"type": rel_type, "source_type": source_type, "target_type": target_type})
        semantic_degree[str(row.get("source_id"))] += 1
        semantic_degree[str(row.get("target_id"))] += 1

    for row in entity_rows:
        row["degree"] = semantic_degree.get(str(row.get("node_id")), 0)

    return {
        "nodes_total": len(node_rows),
        "edges_total": len(edge_rows),
        "documents_count": documents_count,
        "chunks_count": chunks_count,
        "communities_count": None,
        "entity_rows": entity_rows,
        "relationship_rows": relationship_rows,
    }


class MsGraphRAGAdapter:
    name = "msgraphrag"

    def build(self, source_path: Path, run_dir: Path) -> tuple[BuildMetrics, dict[str, Any]]:
        project_dir = run_dir / "project"
        ensure_dir(project_dir / "input")
        copy_file(source_path, project_dir / "input" / source_path.name)
        copy_file(ROOT / "frameworks" / "msgraphrag" / "settings.yaml", project_dir / "settings.yaml")
        for prompt_path in (ROOT / "frameworks" / "msgraphrag" / "prompts").glob("*"):
            copy_file(prompt_path, project_dir / "prompts" / prompt_path.name)

        started = time.perf_counter()
        process = run_command(
            [str(ROOT / ".venv" / "bin" / "graphrag"), "index", "--root", str(project_dir)],
            cwd=ROOT,
            env=_dummy_env(),
            stdout_path=run_dir / "build" / "stdout.log",
            stderr_path=run_dir / "build" / "stderr.log",
        )
        duration = time.perf_counter() - started
        output_dir = project_dir / "output"
        metrics = BuildMetrics(
            build_time_seconds=duration,
            documents_count=None,
            chunks_count=None,
            index_size_bytes=file_size(output_dir),
            build_status="success" if process.returncode == 0 else "failed",
            build_error=None if process.returncode == 0 else f"graphrag index exit code {process.returncode}",
        )
        graph_metrics = parse_msgraphrag_outputs(output_dir) if process.returncode == 0 else {}
        if process.returncode == 0:
            documents = pd.read_parquet(output_dir / "documents.parquet")
            chunks = pd.read_parquet(output_dir / "text_units.parquet")
            metrics.documents_count = len(documents)
            metrics.chunks_count = len(chunks)
        return metrics, graph_metrics.to_dict() if hasattr(graph_metrics, "to_dict") else graph_metrics

    def query(self, run_id: str, run_dir: Path, questions: list[dict[str, Any]]) -> list[QueryAnswer]:
        from graphrag.cli.query import run_local_search

        project_dir = run_dir / "project"
        answers: list[QueryAnswer] = []
        os.environ.setdefault("GRAPHRAG_API_KEY", "local")
        for row in questions:
            started = time.perf_counter()
            response, context = run_local_search(
                None,
                project_dir,
                2,
                "Multiple Paragraphs",
                False,
                row["question"],
                False,
            )
            latency = time.perf_counter() - started
            contexts = _normalize_msgraphrag_context(context)
            answers.append(
                QueryAnswer(
                    run_id=run_id,
                    framework=self.name,
                    question_id=row["id"],
                    question_type=row.get("question_type"),
                    question=row["question"],
                    reference_answer=row["reference_answer"],
                    answer=response,
                    contexts=contexts,
                    latency_seconds=latency,
                    retrieval_time_seconds=None,
                    generation_time_seconds=None,
                    status="success",
                    error=None,
                )
            )
        return answers


class LightRAGAdapter:
    name = "lightrag"

    def _build_rag(self, work_dir: Path):
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc

        async def llm_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str:
            kwargs.pop("model", None)
            return await openai_complete_if_cache(
                model="/models/Qwen3.5-35B-A3B-Q4_K_M.gguf",
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url="http://127.0.0.1:8080/v1",
                api_key="local",
                temperature=0,
                **kwargs,
            )

        embedding_func = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=getattr(openai_embed, "max_token_size", None),
            func=functools.partial(
                openai_embed.func,
                model="multilingual-e5-large",
                base_url="http://127.0.0.1:8010/v1",
                api_key="local",
            ),
            model_name="multilingual-e5-large",
            supports_asymmetric=getattr(openai_embed, "supports_asymmetric", False),
        )
        return LightRAG(
            working_dir=str(work_dir),
            llm_model_func=llm_func,
            llm_model_name="/models/Qwen3.5-35B-A3B-Q4_K_M.gguf",
            embedding_func=embedding_func,
            embedding_batch_num=1,
            llm_model_max_async=1,
            embedding_func_max_async=1,
            vector_db_storage_cls_kwargs={},
        )

    def build(self, source_path: Path, run_dir: Path) -> tuple[BuildMetrics, dict[str, Any]]:
        work_dir = ensure_dir(run_dir / "workspace")
        started = time.perf_counter()
        rag = self._build_rag(work_dir)
        asyncio.run(rag.initialize_storages())
        rag.insert(source_path.read_text(encoding="utf-8"), file_paths=str(source_path))
        duration = time.perf_counter() - started
        graph_metrics = parse_lightrag_outputs(work_dir)
        metrics = BuildMetrics(
            build_time_seconds=duration,
            documents_count=len(json.loads((work_dir / "kv_store_full_docs.json").read_text(encoding="utf-8"))),
            chunks_count=len(json.loads((work_dir / "kv_store_text_chunks.json").read_text(encoding="utf-8"))),
            index_size_bytes=file_size(work_dir),
            build_status="success",
            build_error=None,
        )
        return metrics, graph_metrics.to_dict()

    def query(self, run_id: str, run_dir: Path, questions: list[dict[str, Any]]) -> list[QueryAnswer]:
        from lightrag import QueryParam
        from lightrag.operate import kg_query

        work_dir = run_dir / "workspace"
        rag = self._build_rag(work_dir)
        asyncio.run(rag.initialize_storages())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        answers: list[QueryAnswer] = []
        try:
            for row in questions:
                started = time.perf_counter()
                result = loop.run_until_complete(
                    kg_query(
                        row["question"],
                        rag.chunk_entity_relation_graph,
                        rag.entities_vdb,
                        rag.relationships_vdb,
                        rag.text_chunks,
                        QueryParam(mode="hybrid", stream=False),
                        rag._build_global_config(),
                        hashing_kv=rag.llm_response_cache,
                        system_prompt=None,
                        chunks_vdb=rag.chunks_vdb,
                    )
                )
                latency = time.perf_counter() - started
                raw_data = result.raw_data if result else {}
                contexts = _normalize_lightrag_context(raw_data)
                answers.append(
                    QueryAnswer(
                        run_id=run_id,
                        framework=self.name,
                        question_id=row["id"],
                        question_type=row.get("question_type"),
                        question=row["question"],
                        reference_answer=row["reference_answer"],
                        answer=result.content if result else "",
                        contexts=contexts,
                        latency_seconds=latency,
                        retrieval_time_seconds=None,
                        generation_time_seconds=None,
                        status="success" if result else "failed",
                        error=None if result else "LightRAG returned no result",
                    )
                )
        finally:
            loop.close()
        return answers


class KAGAdapter:
    name = "kag"

    def _py_env(self) -> dict[str, str]:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{KAG_ROOT}:{existing}" if existing else str(KAG_ROOT)
        env.setdefault("KAG_PROJECT_HOST_ADDR", "http://127.0.0.1:8887")
        return env

    def _create_project(self, namespace: str, config_text: str) -> dict[str, Any]:
        os.environ.setdefault("KAG_PROJECT_HOST_ADDR", "http://127.0.0.1:8887")
        _ensure_kag_imports()
        from knext.project.client import ProjectClient

        client = ProjectClient(host_addr="http://127.0.0.1:8887")
        project = client.get_by_namespace(namespace)
        if project is None:
            project = client.create(name=namespace, namespace=namespace, config=build_kag_server_project_config(config_text))
        return {"id": project.id, "namespace": project.namespace}

    def build(self, source_path: Path, run_dir: Path) -> tuple[BuildMetrics, dict[str, Any]]:
        config_template = (ROOT / "configs" / "kag" / "graph_config.template.yaml").read_text(encoding="utf-8")
        namespace = _normalize_kag_namespace(run_dir)
        config_text = (
            config_template.replace("__PROJECT_NAMESPACE__", namespace)
            .replace("__PROJECT_ID__", "0")
            .replace("__CKPT_DIR__", str(run_dir / "ckpt"))
            .replace("__KAG_REGISTER_PATH__", str(KAG_ROOT / "kag"))
        )
        project_info = self._create_project(namespace, config_text)
        os.environ["KAG_PROJECT_ID"] = str(project_info["id"])
        os.environ["KAG_PROJECT_NAMESPACE"] = project_info["namespace"]
        os.environ["KAG_PROJECT_HOST_ADDR"] = "http://127.0.0.1:8887"
        config_text = (
            config_template.replace("__PROJECT_NAMESPACE__", project_info["namespace"])
            .replace("__PROJECT_ID__", str(project_info["id"]))
            .replace("__CKPT_DIR__", str(run_dir / "ckpt"))
            .replace("__KAG_REGISTER_PATH__", str(KAG_ROOT / "kag"))
        )
        config_path = run_dir / "generated_kag_config.yaml"
        atomic_write_json(run_dir / "project.json", project_info)
        from .utils import atomic_write_text

        atomic_write_text(config_path, config_text)
        started = time.perf_counter()
        process = run_command(
            [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "kag" / "build.py"), "--config", str(config_path), "--input", str(source_path)],
            cwd=ROOT,
            env=self._py_env(),
            stdout_path=run_dir / "build" / "stdout.log",
            stderr_path=run_dir / "build" / "stderr.log",
        )
        duration = time.perf_counter() - started
        metrics = BuildMetrics(
            build_time_seconds=duration,
            documents_count=None,
            chunks_count=None,
            index_size_bytes=None,
            build_status="success" if process.returncode == 0 else "failed",
            build_error=None if process.returncode == 0 else f"kag build exit code {process.returncode}",
        )
        if process.returncode != 0:
            return metrics, {}
        summary = self._neo4j_summary(project_info["namespace"])
        metrics.documents_count = summary.get("documents_count")
        metrics.chunks_count = summary.get("chunks_count")
        return metrics, parse_neo4j_summary(summary).to_dict()

    def _neo4j_summary(self, namespace: str) -> dict[str, Any]:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            return {
                "nodes_total": None,
                "edges_total": None,
                "documents_count": None,
                "chunks_count": None,
                "communities_count": None,
                "entity_rows": [],
                "relationship_rows": [],
                "error": str(exc),
            }
        config = get_kag_neo4j_config()
        driver = GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))
        try:
            with driver.session(database=config["database"]) as session:
                node_rows = [
                    record.data()
                    for record in session.run(
                        """
                        MATCH (n)
                        WHERE any(label IN labels(n) WHERE label STARTS WITH $prefix)
                        RETURN elementId(n) AS node_id, labels(n) AS labels
                        """,
                        prefix=f"{namespace}.",
                    )
                ]
                edge_rows = [
                    record.data()
                    for record in session.run(
                        """
                        MATCH (a)-[r]->(b)
                        WHERE any(label IN labels(a) WHERE label STARTS WITH $prefix)
                          AND any(label IN labels(b) WHERE label STARTS WITH $prefix)
                        RETURN elementId(a) AS source_id,
                               elementId(b) AS target_id,
                               labels(a) AS source_labels,
                               labels(b) AS target_labels,
                               type(r) AS type
                        """,
                        prefix=f"{namespace}.",
                    )
                ]
        finally:
            driver.close()
        return summarize_kag_namespace_graph(node_rows, edge_rows, namespace)

    def query(self, run_id: str, run_dir: Path, questions: list[dict[str, Any]]) -> list[QueryAnswer]:
        _ensure_kag_imports()
        from kag.solver.main_solver import qa

        project = json.loads((run_dir / "project.json").read_text(encoding="utf-8"))
        config = yaml.safe_load((run_dir / "generated_kag_config.yaml").read_text(encoding="utf-8"))
        answers: list[QueryAnswer] = []
        for row in questions:
            started = time.perf_counter()
            with _without_proxy_for_kag_local_hosts():
                payload = asyncio.run(
                    qa(
                        task_id=run_id,
                        query=row["question"],
                        project_id=project["id"],
                        host_addr="http://127.0.0.1:8887",
                        app_id="kag-local",
                        params={"config": config},
                    )
                )
            latency = time.perf_counter() - started
            answer_text, trace = _normalize_kag_response(payload)
            answers.append(
                QueryAnswer(
                    run_id=run_id,
                    framework=self.name,
                    question_id=row["id"],
                    question_type=row.get("question_type"),
                    question=row["question"],
                    reference_answer=row["reference_answer"],
                    answer=answer_text,
                    contexts=_normalize_kag_context(trace),
                    latency_seconds=latency,
                    retrieval_time_seconds=None,
                    generation_time_seconds=None,
                    status="success",
                    error=None,
                )
            )
        return answers


def _normalize_msgraphrag_context(context: dict[str, Any]) -> list[ContextItem]:
    items: list[ContextItem] = []
    if isinstance(context, list):
        records = context
    elif isinstance(context, dict):
        records = []
        if "records" in context and isinstance(context["records"], list):
            records.extend(context["records"])
        else:
            for section, value in context.items():
                if not hasattr(value, "to_dict"):
                    continue
                for row in value.to_dict(orient="records"):
                    row = {"section": section, **row}
                    records.append(row)
    else:
        records = []

    for index, row in enumerate(records, start=1):
        text = str(row.get("text") or row.get("description") or row.get("content") or row)
        source = str(row.get("section") or row.get("source") or row.get("entity") or row.get("id") or "text_unit")
        items.append(ContextItem(rank=index, text=text, source=source, metadata=row))
    return items[:20]


def _normalize_lightrag_context(raw_data: dict[str, Any]) -> list[ContextItem]:
    data = raw_data.get("data", {})
    chunks = data.get("chunks", [])
    items = []
    for index, chunk in enumerate(chunks, start=1):
        items.append(
            ContextItem(
                rank=index,
                text=str(chunk.get("content") or chunk.get("text") or ""),
                source=str(chunk.get("file_path") or chunk.get("source") or "chunk"),
                score=chunk.get("score"),
                metadata=chunk,
            )
        )
    return items


def _normalize_kag_response(payload: Any) -> tuple[str, Any]:
    if isinstance(payload, tuple) and len(payload) == 2:
        return str(payload[0]), payload[1]
    if isinstance(payload, list) and payload:
        return str(payload[0]), payload[1] if len(payload) > 1 else {}
    return str(payload), {}


def _normalize_kag_context(trace: Any) -> list[ContextItem]:
    if not isinstance(trace, dict):
        return []
    contexts = trace.get("context") or trace.get("evidence") or trace.get("chunks") or []
    items = []
    for index, item in enumerate(contexts, start=1):
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("content") or item)
            source = str(item.get("source") or item.get("file_path") or "kag")
            items.append(ContextItem(rank=index, text=text, source=source, metadata=item))
        else:
            items.append(ContextItem(rank=index, text=str(item), source="kag"))
    return items


def get_adapter(name: str):
    mapping = {
        "msgraphrag": MsGraphRAGAdapter(),
        "lightrag": LightRAGAdapter(),
        "kag": KAGAdapter(),
    }
    return mapping[name]
