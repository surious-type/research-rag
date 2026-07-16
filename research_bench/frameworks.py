from __future__ import annotations

import asyncio
import functools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

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


def _dummy_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("GRAPHRAG_API_KEY", "local")
    return env


def _ensure_kag_imports() -> None:
    kag_root = str(KAG_ROOT)
    if kag_root not in sys.path:
        sys.path.insert(0, kag_root)


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
            try:
                project = client.create(name=namespace, namespace=namespace, config=config_text)
            except Exception:
                for fallback in ("SmokeKagRuntimeRepair", "SmokeKagRuntimeRepair20260710112059", "SmokePreB2", "SmokePreB1"):
                    project = client.get_by_namespace(fallback)
                    if project is not None:
                        break
                if project is None:
                    raise
        return {"id": project.id, "namespace": project.namespace}

    def build(self, source_path: Path, run_dir: Path) -> tuple[BuildMetrics, dict[str, Any]]:
        config_template = (ROOT / "configs" / "kag" / "graph_config.template.yaml").read_text(encoding="utf-8")
        namespace = f"KagSmoke{run_dir.name.replace('_', '')}"
        project_info = self._create_project(namespace, config_template)
        os.environ["KAG_PROJECT_ID"] = str(project_info["id"])
        os.environ["KAG_PROJECT_NAMESPACE"] = project_info["namespace"]
        os.environ["KAG_PROJECT_HOST_ADDR"] = "http://127.0.0.1:8887"
        config_text = (
            config_template.replace("__PROJECT_NAMESPACE__", namespace)
            .replace("__PROJECT_ID__", str(project_info["id"]))
            .replace("__CKPT_DIR__", str(run_dir / "ckpt"))
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
        summary = self._neo4j_summary()
        metrics = BuildMetrics(
            build_time_seconds=duration,
            documents_count=summary.get("documents_count"),
            chunks_count=summary.get("chunks_count"),
            index_size_bytes=None,
            build_status="success" if process.returncode == 0 else "failed",
            build_error=None if process.returncode == 0 else f"kag build exit code {process.returncode}",
        )
        return metrics, parse_neo4j_summary(summary).to_dict()

    def _neo4j_summary(self) -> dict[str, Any]:
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
        driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "neo4j@openspg"))
        with driver.session(database="neo4j") as session:
            nodes_total = session.run("MATCH (n) RETURN count(n) AS value").single()["value"]
            edges_total = session.run("MATCH ()-[r]->() RETURN count(r) AS value").single()["value"]
            documents_count = session.run("MATCH (n:Document) RETURN count(n) AS value").single()["value"]
            chunks_count = session.run("MATCH (n:Chunk) RETURN count(n) AS value").single()["value"]
            entity_rows = [record.data() for record in session.run("MATCH (n:Entity) RETURN labels(n) AS type")]
            relationship_rows = [
                record.data()
                for record in session.run(
                    "MATCH (a:Entity)-[r]->(b:Entity) RETURN type(r) AS type, labels(a)[0] AS source_type, labels(b)[0] AS target_type"
                )
            ]
        driver.close()
        return {
            "nodes_total": nodes_total,
            "edges_total": edges_total,
            "documents_count": documents_count,
            "chunks_count": chunks_count,
            "communities_count": None,
            "entity_rows": entity_rows,
            "relationship_rows": relationship_rows,
        }

    def query(self, run_id: str, run_dir: Path, questions: list[dict[str, Any]]) -> list[QueryAnswer]:
        _ensure_kag_imports()
        from kag.solver.main_solver import qa

        project = json.loads((run_dir / "project.json").read_text(encoding="utf-8"))
        answers: list[QueryAnswer] = []
        for row in questions:
            started = time.perf_counter()
            payload = asyncio.run(
                qa(
                    task_id=run_id,
                    query=row["question"],
                    project_id=project["id"],
                    host_addr="http://127.0.0.1:8887",
                    app_id="kag-local",
                    params={},
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
