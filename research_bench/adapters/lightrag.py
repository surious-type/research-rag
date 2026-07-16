from __future__ import annotations

import asyncio
import functools
import json
import time
from pathlib import Path
from typing import Any

from research_bench.models import BuildMetrics, ContextItem, QueryAnswer
from research_bench.parsers import parse_lightrag_outputs
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
            input_documents_count=len(json.loads((work_dir / "kv_store_full_docs.json").read_text(encoding="utf-8"))),
            backend_document_nodes_count=None,
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


__all__ = ["LightRAGAdapter"]
