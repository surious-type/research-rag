from __future__ import annotations

import asyncio
import copy
from contextlib import contextmanager
import functools
import json
import logging
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
    load_json,
    run_command,
)


ROOT = Path(__file__).resolve().parents[1]
KAG_ROOT = ROOT / "frameworks" / "kag"
logger = logging.getLogger(__name__)
KAG_NEO4J_EXCLUDED_TYPES = {"AtomicQuery", "Chunk", "Doc", "Document", "Outline", "Summary"}
KAG_NEO4J_DEFAULT_URI = "bolt://127.0.0.1:17687"
KAG_NEO4J_DEFAULT_USER = "neo4j"
KAG_NEO4J_DEFAULT_PASSWORD = "neo4j@openspg"
KAG_NEO4J_DEFAULT_DATABASE = "neo4j"
KAG_LOCAL_HOST_ADDR = "http://127.0.0.1:8887"
KAG_SELECTED_PIPELINE = "kag_solver_pipeline"
KAG_EXPECTED_RETRIEVER_TYPES = ("kg_cs_open_spg", "kg_fr_open_spg", "rc_open_spg")
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


def _kag_register_path() -> str:
    return str(ROOT / "research_bench" / "_kag_registry")


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


def _resolve_kag_neo4j_database(namespace: str, configured_database: str) -> str:
    candidate = str(namespace or "").strip().lower()
    return candidate or configured_database


def _normalize_kag_namespace(run_dir: Path) -> str:
    digits = "".join(ch for ch in run_dir.name if ch.isdigit())
    suffix = digits[-12:] if digits else str(int(time.time()))[-12:]
    namespace = f"Kag{suffix}"
    return namespace[:16]


def _rewrite_kag_server_config(value: Any) -> Any:
    if isinstance(value, dict):
        updated = {key: _rewrite_kag_server_config(item) for key, item in value.items()}
        if updated.get("type") == "openai":
            if updated.get("base_url") == "http://127.0.0.1:8080/v1":
                updated["base_url"] = "http://host.docker.internal:8080/v1"
            elif updated.get("base_url") == "http://127.0.0.1:8010/v1":
                updated["base_url"] = "http://multilingual-e5-server/v1"
        return updated
    if isinstance(value, list):
        return [_rewrite_kag_server_config(item) for item in value]
    return value


def build_kag_server_project_config(config_text: str) -> dict[str, Any]:
    return _rewrite_kag_server_config(yaml.safe_load(config_text))


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
    knowledge_units_count = 0

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
        if node_type == "KnowledgeUnit":
            knowledge_units_count += 1
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
        "knowledge_units_count": knowledge_units_count,
        "communities_count": None,
        "entity_rows": entity_rows,
        "relationship_rows": relationship_rows,
    }


def _collect_kag_retriever_types(pipeline_config: dict[str, Any]) -> list[str]:
    retriever_types: list[str] = []
    for executor in pipeline_config.get("executors", []):
        if not isinstance(executor, dict):
            continue
        for retriever in executor.get("retrievers", []):
            retriever_type = retriever.get("type") if isinstance(retriever, dict) else None
            if retriever_type:
                retriever_types.append(str(retriever_type))
    return retriever_types


def _prepare_kag_query_environment(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    _ensure_kag_imports()
    from kag.common.conf import init_env
    from kag.common.registry import import_modules_from_path

    project = json.loads((run_dir / "project.json").read_text(encoding="utf-8"))
    config_path = run_dir / "generated_kag_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    os.environ["KAG_PROJECT_ID"] = str(project["id"])
    os.environ["KAG_PROJECT_NAMESPACE"] = str(project["namespace"])
    os.environ["KAG_PROJECT_HOST_ADDR"] = KAG_LOCAL_HOST_ADDR
    init_env(str(config_path))
    register_path = config.get("kag_builder_pipeline", {}).get("register_path")
    if register_path:
        import_modules_from_path(register_path)
    return project, config, config_path


def _resolve_kag_query_diagnostics(config: dict[str, Any], project: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_pipeline = KAG_SELECTED_PIPELINE
    if selected_pipeline not in config:
        raise RuntimeError(f"KAG pipeline {selected_pipeline} not found in generated config")
    resolved_pipeline = copy.deepcopy(config[selected_pipeline])
    resolved_retriever_types = _collect_kag_retriever_types(resolved_pipeline)
    diagnostics = {
        "project_id": str(project["id"]),
        "namespace": str(project["namespace"]),
        "selected_pipeline": selected_pipeline,
        "resolved_retriever_types": resolved_retriever_types,
        "expected_retriever_types": list(KAG_EXPECTED_RETRIEVER_TYPES),
        "reporter_type": "trace_log_reporter",
    }
    logger.info(
        "KAG query diagnostics project_id=%s namespace=%s selected_pipeline=%s resolved_retriever_types=%s",
        diagnostics["project_id"],
        diagnostics["namespace"],
        diagnostics["selected_pipeline"],
        diagnostics["resolved_retriever_types"],
    )
    if diagnostics["selected_pipeline"] != KAG_SELECTED_PIPELINE:
        raise RuntimeError(f"KAG selected_pipeline is {diagnostics['selected_pipeline']}, expected {KAG_SELECTED_PIPELINE}")
    if not resolved_retriever_types:
        raise RuntimeError("KAG resolved_retriever_types is empty")
    missing_retrievers = [name for name in KAG_EXPECTED_RETRIEVER_TYPES if name not in resolved_retriever_types]
    if missing_retrievers:
        raise RuntimeError(f"KAG resolved_retriever_types missing expected retrievers: {', '.join(missing_retrievers)}")
    return resolved_pipeline, diagnostics


async def _invoke_kag_query_with_trace(
    run_id: str,
    question: str,
    config: dict[str, Any],
    project: dict[str, Any],
) -> dict[str, Any]:
    from kag.common.conf import KAG_PROJECT_CONF
    from kag.solver.main_solver import do_qa_pipeline, is_chinese
    from kag.solver.reporter.trace_log_reporter import TraceLogReporter

    KAG_PROJECT_CONF.host_addr = KAG_LOCAL_HOST_ADDR
    KAG_PROJECT_CONF.language = "zh" if is_chinese(question) else "en"
    reporter = TraceLogReporter(
        host_addr=KAG_LOCAL_HOST_ADDR,
        project_id=project["id"],
        thinking_enabled=False,
        report_all_references=False,
    )
    await reporter.start()
    try:
        answer = await do_qa_pipeline(
            KAG_SELECTED_PIPELINE,
            question,
            copy.deepcopy(config),
            reporter,
            task_id=run_id,
            kb_project_ids=[],
        )
        reporter.add_report_line("answer", "Final Answer", answer, "FINISH")
    finally:
        await reporter.stop()
    trace_payload, reporter_status = reporter.generate_report_data()
    trace = trace_payload.to_dict() if hasattr(trace_payload, "to_dict") else trace_payload
    return {"answer": str(answer), "trace": trace, "reporter_status": reporter_status}


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
        env.setdefault("KAG_PROJECT_HOST_ADDR", KAG_LOCAL_HOST_ADDR)
        return env

    def _create_project(self, namespace: str, config_text: str) -> dict[str, Any]:
        os.environ.setdefault("KAG_PROJECT_HOST_ADDR", KAG_LOCAL_HOST_ADDR)
        _ensure_kag_imports()
        from knext.project.client import ProjectClient

        client = ProjectClient(host_addr=KAG_LOCAL_HOST_ADDR)
        project = client.get_by_namespace(namespace)
        if project is None:
            project = client.create(name=namespace, namespace=namespace, config=build_kag_server_project_config(config_text))
        return {"id": project.id, "namespace": project.namespace}

    def _sync_project_config(self, project_id: str, namespace: str, config_text: str) -> None:
        _ensure_kag_imports()
        from knext.project.client import ProjectClient

        client = ProjectClient(host_addr=KAG_LOCAL_HOST_ADDR)
        client.update(
            id=project_id,
            namespace=namespace,
            config=build_kag_server_project_config(config_text),
            visibility="PRIVATE",
            tag="LOCAL",
            userNo="openspg",
        )

    def build(self, source_path: Path, run_dir: Path) -> tuple[BuildMetrics, dict[str, Any]]:
        config_template = (ROOT / "configs" / "kag" / "graph_config.template.yaml").read_text(encoding="utf-8")
        namespace = _normalize_kag_namespace(run_dir)
        config_text = (
            config_template.replace("__PROJECT_NAMESPACE__", namespace)
            .replace("__PROJECT_ID__", "0")
            .replace("__CKPT_DIR__", str(run_dir / "ckpt"))
            .replace("__KAG_REGISTER_PATH__", _kag_register_path())
        )
        project_info = self._create_project(namespace, config_text)
        os.environ["KAG_PROJECT_ID"] = str(project_info["id"])
        os.environ["KAG_PROJECT_NAMESPACE"] = project_info["namespace"]
        os.environ["KAG_PROJECT_HOST_ADDR"] = KAG_LOCAL_HOST_ADDR
        config_text = (
            config_template.replace("__PROJECT_NAMESPACE__", project_info["namespace"])
            .replace("__PROJECT_ID__", str(project_info["id"]))
            .replace("__CKPT_DIR__", str(run_dir / "ckpt"))
            .replace("__KAG_REGISTER_PATH__", _kag_register_path())
        )
        config_path = run_dir / "generated_kag_config.yaml"
        server_config = build_kag_server_project_config(config_text)
        atomic_write_json(run_dir / "project.json", project_info)
        from .utils import atomic_write_text

        atomic_write_text(config_path, config_text)
        atomic_write_text(run_dir / "kag_config.yaml", config_text)
        atomic_write_json(run_dir / "server_project_config.json", server_config)
        atomic_write_json(run_dir / "runtime_project_config.json", yaml.safe_load(config_text))
        self._sync_project_config(str(project_info["id"]), project_info["namespace"], config_text)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            manifest["kag"] = {
                "project_id": str(project_info["id"]),
                "namespace": project_info["namespace"],
                "selected_pipeline": KAG_SELECTED_PIPELINE,
                "server_vectorize_model_base_url": server_config.get("vectorize_model", {}).get("base_url"),
                "runtime_vectorize_model_base_url": yaml.safe_load(config_text).get("vectorize_model", {}).get("base_url"),
            }
            atomic_write_json(manifest_path, manifest)
        started = time.perf_counter()
        process = run_command(
            [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "kag" / "build.py"), "--config", str(config_path), "--input", str(source_path)],
            cwd=run_dir,
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
        summary = self._neo4j_summary(project_info["namespace"], run_dir=run_dir)
        metrics.documents_count = summary.get("documents_count")
        metrics.chunks_count = summary.get("chunks_count")
        graph_metrics = parse_neo4j_summary(summary).to_dict()
        if not summary.get("nodes_total"):
            metrics.build_status = "failed"
            metrics.build_error = f"kag namespace {project_info['namespace']} has zero matched nodes in Neo4j"
        elif not summary.get("chunks_count"):
            metrics.build_status = "failed"
            metrics.build_error = f"kag namespace {project_info['namespace']} has zero matched chunks in Neo4j"
        return metrics, graph_metrics

    def _neo4j_summary(self, namespace: str, run_dir: Path | None = None) -> dict[str, Any]:
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
        database_name = _resolve_kag_neo4j_database(namespace, config["database"])
        driver = GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))
        try:
            with driver.session(database=database_name) as session:
                label_rows = [
                    record.data()
                    for record in session.run(
                        """
                        MATCH (n)
                        UNWIND labels(n) AS label
                        RETURN label, count(*) AS count
                        ORDER BY count DESC, label ASC
                        """
                    )
                ]
                relationship_type_rows = [
                    record.data()
                    for record in session.run(
                        """
                        MATCH ()-[r]->()
                        RETURN type(r) AS type, count(*) AS count
                        ORDER BY count DESC, type ASC
                        """
                    )
                ]
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
                matched_relationship_type_rows = [
                    record.data()
                    for record in session.run(
                        """
                        MATCH (a)-[r]->(b)
                        WHERE any(label IN labels(a) WHERE label STARTS WITH $prefix)
                          AND any(label IN labels(b) WHERE label STARTS WITH $prefix)
                        RETURN type(r) AS type, count(*) AS count
                        ORDER BY count DESC, type ASC
                        """,
                        prefix=f"{namespace}.",
                    )
                ]
        finally:
            driver.close()
        matched_label_rows = [row for row in label_rows if str(row.get("label", "")).startswith(f"{namespace}.")]
        if run_dir is not None:
            atomic_write_json(run_dir / "build" / "neo4j_labels_raw.json", {"labels": label_rows})
            atomic_write_json(run_dir / "build" / "neo4j_relationship_types_raw.json", {"relationship_types": relationship_type_rows})
            atomic_write_json(
                run_dir / "build" / "neo4j_namespace_match.json",
                {
                    "database": database_name,
                    "namespace_prefix": f"{namespace}.",
                    "matched_labels": matched_label_rows,
                    "matched_relationship_types": matched_relationship_type_rows,
                },
            )
        return summarize_kag_namespace_graph(node_rows, edge_rows, namespace)

    def query(self, run_id: str, run_dir: Path, questions: list[dict[str, Any]]) -> list[QueryAnswer]:
        project, config, _ = _prepare_kag_query_environment(run_dir)
        _, diagnostics = _resolve_kag_query_diagnostics(config, project)
        atomic_write_json(run_dir / "query" / "diagnostics.json", diagnostics)
        ensure_dir(run_dir / "query" / "traces")
        answers: list[QueryAnswer] = []
        for row in questions:
            started = time.perf_counter()
            try:
                with _without_proxy_for_kag_local_hosts():
                    payload = asyncio.run(
                        _invoke_kag_query_with_trace(
                            run_id=run_id,
                            question=row["question"],
                            config=config,
                            project=project,
                        )
                    )
                atomic_write_json(run_dir / "query" / "traces" / f"{row['id']}.json", payload)
                answer_text, trace = _normalize_kag_response(payload)
                contexts = _normalize_kag_context(trace)
                status = "success"
                error = None
            except Exception as exc:
                payload = {"answer": "", "trace": {}, "error": str(exc)}
                atomic_write_json(run_dir / "query" / "traces" / f"{row['id']}.json", payload)
                answer_text = ""
                contexts = []
                status = "failed"
                error = str(exc)
            latency = time.perf_counter() - started
            answers.append(
                QueryAnswer(
                    run_id=run_id,
                    framework=self.name,
                    question_id=row["id"],
                    question_type=row.get("question_type"),
                    question=row["question"],
                    reference_answer=row["reference_answer"],
                    answer=answer_text,
                    contexts=contexts,
                    latency_seconds=latency,
                    retrieval_time_seconds=None,
                    generation_time_seconds=None,
                    status=status,
                    error=error,
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
    if isinstance(payload, dict):
        return str(payload.get("answer", "")), payload.get("trace", {})
    if isinstance(payload, tuple) and len(payload) == 2:
        return str(payload[0]), payload[1]
    if isinstance(payload, list) and payload:
        return str(payload[0]), payload[1] if len(payload) > 1 else {}
    return str(payload), {}


def _normalize_kag_context(trace: Any) -> list[ContextItem]:
    if not isinstance(trace, dict):
        return []
    items: list[ContextItem] = []
    seen: set[tuple[str, str]] = set()

    def add_context(text: Any, source: Any, metadata: dict[str, Any] | None = None, score: Any = None) -> None:
        normalized_text = str(text or "").strip()
        normalized_source = str(source or "kag")
        if not normalized_text:
            return
        key = (normalized_source, normalized_text)
        if key in seen:
            return
        seen.add(key)
        items.append(
            ContextItem(
                rank=len(items) + 1,
                text=normalized_text,
                source=normalized_source,
                score=score if isinstance(score, (int, float)) else None,
                metadata=metadata or {},
            )
        )

    for block in trace.get("decompose", []):
        if not isinstance(block, dict):
            continue
        for chunk in block.get("chunk_datas", []):
            if not isinstance(chunk, dict):
                continue
            metadata = {
                "document_id": chunk.get("document_id") or chunk.get("chunk_id"),
                "document_name": chunk.get("document_name") or chunk.get("title"),
                **chunk,
            }
            add_context(
                chunk.get("content"),
                chunk.get("document_name") or chunk.get("title") or "chunk",
                metadata=metadata,
                score=chunk.get("score"),
            )
        for graph_item in block.get("graph_data", []):
            if isinstance(graph_item, dict):
                add_context(graph_item.get("content") or graph_item.get("text") or graph_item, "graph", metadata=graph_item)
            else:
                add_context(graph_item, "graph", metadata={"graph_data": graph_item})

    for reference_group in trace.get("reference", []):
        if not isinstance(reference_group, dict):
            continue
        for ref in reference_group.get("info", []):
            if not isinstance(ref, dict):
                continue
            add_context(
                ref.get("content"),
                ref.get("document_name") or ref.get("id") or "reference",
                metadata=ref,
            )

    for generator_item in trace.get("generator", []):
        if not isinstance(generator_item, dict):
            continue
        for ref in generator_item.get("reference", []):
            if not isinstance(ref, dict):
                continue
            add_context(
                ref.get("content"),
                ref.get("document_name") or ref.get("id") or "generator_reference",
                metadata=ref,
            )

    for item in trace.get("context", []) or trace.get("evidence", []) or trace.get("chunks", []) or []:
        if isinstance(item, dict):
            add_context(item.get("text") or item.get("content") or item, item.get("source") or item.get("file_path") or "kag", metadata=item)
        else:
            add_context(item, "kag")
    return items


def get_adapter(name: str):
    mapping = {
        "msgraphrag": MsGraphRAGAdapter(),
        "lightrag": LightRAGAdapter(),
        "kag": KAGAdapter(),
    }
    return mapping[name]
