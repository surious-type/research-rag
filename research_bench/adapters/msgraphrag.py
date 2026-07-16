from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from research_bench.data import ROOT
from research_bench.models import BuildMetrics, ContextItem, QueryAnswer
from research_bench.parsers import parse_msgraphrag_outputs
from research_bench.shared.io import copy_file, ensure_dir, file_size
from research_bench.shared.subprocess import run_command


def _dummy_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("GRAPHRAG_API_KEY", "local")
    return env


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
                    records.append({"section": section, **row})
    else:
        records = []

    for index, row in enumerate(records, start=1):
        text = str(row.get("text") or row.get("description") or row.get("content") or row)
        source = str(row.get("section") or row.get("source") or row.get("entity") or row.get("id") or "text_unit")
        items.append(ContextItem(rank=index, text=text, source=source, metadata=row))
    return items[:20]


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
            progress_run_dir=run_dir,
            progress_stage="build",
            progress_framework=self.name,
            progress_run_id=run_dir.name,
            process_name="graphrag index",
        )
        duration = time.perf_counter() - started
        output_dir = project_dir / "output"
        metrics = BuildMetrics(
            build_time_seconds=duration,
            documents_count=None,
            input_documents_count=None,
            backend_document_nodes_count=None,
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


__all__ = ["MsGraphRAGAdapter"]
