from __future__ import annotations

import asyncio
import functools
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from research_bench.models import BuildMetrics, ContextItem, QueryAnswer
from research_bench.parsers import parse_lightrag_outputs
from research_bench.runtime_config import (
    DEFAULT_TEMPERATURE,
    load_model_runtime_config,
    resolve_embedding_dimension,
)
from research_bench.shared.io import ensure_dir, file_size


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


class LightRAGAdapter:
    name = "lightrag"

    def _ensure_numpy_graphml_compatibility(self) -> None:
        import numpy as np

        if not hasattr(np, "float_"):
            np.float_ = np.float64

    @staticmethod
    def _uses_local_runtime(base_url: str) -> bool:
        hostname = (urlparse(base_url).hostname or "").lower()
        return hostname in {"127.0.0.1", "localhost", "host.docker.internal"}

    def _lightrag_llm_timeout_seconds(self, runtime) -> int:
        configured = os.getenv("LIGHTRAG_LLM_TIMEOUT_SECONDS")
        if configured:
            return int(configured)
        if self._uses_local_runtime(runtime.base_url):
            return 1800
        return 240

    def _build_diagnostics(self, work_dir: Path) -> dict[str, Any]:
        files = sorted(path.name for path in work_dir.iterdir()) if work_dir.exists() else []
        doc_status_path = work_dir / "kv_store_doc_status.json"
        doc_status = {}
        if doc_status_path.exists():
            try:
                doc_status = json.loads(doc_status_path.read_text(encoding="utf-8"))
            except Exception:
                doc_status = {}
        failed_docs = []
        if isinstance(doc_status, dict):
            for doc_id, payload in doc_status.items():
                if not isinstance(payload, dict):
                    continue
                if payload.get("status") != "failed":
                    continue
                failed_docs.append(
                    {
                        "doc_id": doc_id,
                        "error_msg": str(payload.get("error_msg") or "").strip() or None,
                        "chunks_count": payload.get("chunks_count"),
                    }
                )
        return {
            "workspace_files": files,
            "graphml_exists": (work_dir / "graph_chunk_entity_relation.graphml").exists(),
            "doc_status_failed": failed_docs,
        }

    def _build_error_message(self, exc: Exception, work_dir: Path) -> str:
        diagnostics = self._build_diagnostics(work_dir)
        if (
            isinstance(exc, FileNotFoundError)
            and not diagnostics["graphml_exists"]
            and diagnostics["doc_status_failed"]
        ):
            first_failure = diagnostics["doc_status_failed"][0]
            details = first_failure.get("error_msg") or "unknown LightRAG extraction failure"
            return f"LightRAG build failed before graph flush: {details}"
        return str(exc)

    def _build_rag(self, work_dir: Path):
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc

        runtime = load_model_runtime_config()
        embedding_dimension = resolve_embedding_dimension(runtime)
        llm_timeout_seconds = self._lightrag_llm_timeout_seconds(runtime)

        async def llm_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str:
            kwargs.pop("model", None)
            kwargs.setdefault("temperature", DEFAULT_TEMPERATURE)
            return await openai_complete_if_cache(
                model=runtime.model,
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=runtime.base_url,
                api_key=runtime.api_key,
                **kwargs,
            )

        embedding_func = EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=getattr(openai_embed, "max_token_size", None),
            func=functools.partial(
                openai_embed.func,
                model=runtime.embedding_model,
                base_url=runtime.embedding_base_url,
                api_key=runtime.api_key,
            ),
            model_name=runtime.embedding_model,
            supports_asymmetric=getattr(openai_embed, "supports_asymmetric", False),
        )
        return LightRAG(
            working_dir=str(work_dir),
            llm_model_func=llm_func,
            llm_model_name=runtime.model,
            default_llm_timeout=llm_timeout_seconds,
            embedding_func=embedding_func,
            embedding_batch_num=1,
            llm_model_max_async=1,
            embedding_func_max_async=1,
            vector_db_storage_cls_kwargs={},
        )

    def build(self, source_path: Path, run_dir: Path) -> tuple[BuildMetrics, dict[str, Any]]:
        work_dir = ensure_dir(run_dir / "workspace")
        started = time.perf_counter()
        self._ensure_numpy_graphml_compatibility()
        try:
            rag = self._build_rag(work_dir)
            asyncio.run(rag.initialize_storages())
            rag.insert(source_path.read_text(encoding="utf-8"), file_paths=str(source_path))
            duration = time.perf_counter() - started
            graph_metrics = parse_lightrag_outputs(work_dir)
            metrics = BuildMetrics(
                build_time_seconds=duration,
                documents_count=len(json.loads((work_dir / "kv_store_full_docs.json").read_text(encoding="utf-8"))),
                input_documents_count=len(json.loads((work_dir / "kv_store_full_docs.json").read_text(encoding="utf-8"))),
                backend_document_nodes_count=None,
                chunks_count=len(json.loads((work_dir / "kv_store_text_chunks.json").read_text(encoding="utf-8"))),
                index_size_bytes=file_size(work_dir),
                build_status="success",
                build_error=None,
            )
            return metrics, graph_metrics.to_dict()
        except Exception as exc:
            duration = time.perf_counter() - started
            full_docs_path = work_dir / "kv_store_full_docs.json"
            text_chunks_path = work_dir / "kv_store_text_chunks.json"
            documents_count = len(json.loads(full_docs_path.read_text(encoding="utf-8"))) if full_docs_path.exists() else None
            chunks_count = len(json.loads(text_chunks_path.read_text(encoding="utf-8"))) if text_chunks_path.exists() else None
            diagnostics = self._build_diagnostics(work_dir)
            (run_dir / "build").mkdir(parents=True, exist_ok=True)
            (run_dir / "build" / "lightrag_diagnostics.json").write_text(
                json.dumps(diagnostics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            metrics = BuildMetrics(
                build_time_seconds=duration,
                documents_count=documents_count,
                input_documents_count=documents_count,
                backend_document_nodes_count=None,
                chunks_count=chunks_count,
                index_size_bytes=file_size(work_dir),
                build_status="failed",
                build_error=self._build_error_message(exc, work_dir),
            )
            return metrics, {}

    def query(self, run_id: str, run_dir: Path, questions: list[dict[str, Any]]) -> list[QueryAnswer]:
        from lightrag import QueryParam
        from lightrag.utils import always_get_an_event_loop

        work_dir = run_dir / "workspace"
        rag = self._build_rag(work_dir)
        loop = always_get_an_event_loop()
        answers: list[QueryAnswer] = []
        loop.run_until_complete(rag.initialize_storages())
        for row in questions:
            started = time.perf_counter()
            params = QueryParam(mode="hybrid", stream=False)
            raw_data = rag.query_data(row["question"], params)
            answer_text = rag.query(row["question"], params)
            latency = time.perf_counter() - started
            contexts = _normalize_lightrag_context(raw_data if isinstance(raw_data, dict) else {})
            answer = answer_text if isinstance(answer_text, str) else ""
            answers.append(
                QueryAnswer(
                    run_id=run_id,
                    framework=self.name,
                    question_id=row["id"],
                    question_type=row.get("question_type"),
                    question=row["question"],
                    reference_answer=row["reference_answer"],
                    answer=answer,
                    contexts=contexts,
                    latency_seconds=latency,
                    retrieval_time_seconds=None,
                    generation_time_seconds=None,
                    status="success" if answer else "failed",
                    error=None if answer else "LightRAG returned no result",
                )
            )
        return answers


__all__ = ["LightRAGAdapter"]
