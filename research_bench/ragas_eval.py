from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import aggregate_numeric, atomic_write_json, atomic_write_jsonl, write_csv


def prepare_ragas_rows(answer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for row in answer_rows:
        prepared.append(
            {
                "question_id": row["question_id"],
                "user_input": row["question"],
                "response": row["answer"],
                "reference": row["reference_answer"],
                "retrieved_contexts": [context["text"] for context in row.get("contexts", []) if context.get("text")],
            }
        )
    return prepared


def summarize_ragas_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        summary[metric] = aggregate_numeric([row.get(metric) for row in rows])
    return summary


def save_ragas_placeholder(base_dir: Path, answer_rows: list[dict[str, Any]], error: str) -> None:
    ensure_rows = []
    for row in answer_rows:
        ensure_rows.append(
            {
                "question_id": row["question_id"],
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
                "error": error,
            }
        )
    write_csv(base_dir / "scores.csv", ensure_rows)
    atomic_write_jsonl(base_dir / "scores.jsonl", ensure_rows)
    atomic_write_jsonl(base_dir / "errors.jsonl", [{"question_id": row["question_id"], "error": error} for row in answer_rows])
    atomic_write_json(base_dir / "summary.json", summarize_ragas_scores(ensure_rows))


def save_ragas_outputs(base_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(base_dir / "scores.csv", rows)
    atomic_write_jsonl(base_dir / "scores.jsonl", rows)
    errors = [{"question_id": row["question_id"], "error": row["error"]} for row in rows if row.get("error")]
    atomic_write_jsonl(base_dir / "errors.jsonl", errors)
    atomic_write_json(base_dir / "summary.json", summarize_ragas_scores(rows))
