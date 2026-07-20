from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import aggregate_numeric, atomic_write_json, atomic_write_jsonl, write_csv


def prepare_ragas_rows(answer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for row in answer_rows:
        if row.get("status") != "success" or not row.get("contexts"):
            continue
        prepared.append(
            {
                "question_id": row["question_id"],
                "user_input": row["question"],
                "response": row["answer"],
                "reference": row["reference_answer"],
                "retrieved_contexts": [
                    context["text"]
                    for context in row.get("contexts", [])
                    if context.get("text")
                ],
            }
        )
    return prepared


def summarize_ragas_scores(
    rows: list[dict[str, Any]], answer_rows: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ):
        summary[metric] = aggregate_numeric([row.get(metric) for row in rows])
    if answer_rows is not None:
        successful = 0
        degraded = 0
        failed = 0
        for row in answer_rows:
            status = str(row.get("status") or "").strip()
            if status == "success":
                successful += 1
            elif status == "degraded":
                degraded += 1
            else:
                failed += 1
        total = len(answer_rows)
        summary["successful_retrieval_questions"] = successful
        summary["degraded_questions"] = degraded
        summary["failed_questions"] = failed
        summary["retrieval_failure_rate"] = (
            (degraded + failed) / total if total else None
        )
    return summary


def save_ragas_placeholder(
    base_dir: Path, answer_rows: list[dict[str, Any]], error: str
) -> None:
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
    atomic_write_jsonl(
        base_dir / "errors.jsonl",
        [{"question_id": row["question_id"], "error": error} for row in answer_rows],
    )
    atomic_write_json(
        base_dir / "summary.json",
        summarize_ragas_scores(ensure_rows, answer_rows=answer_rows),
    )


def save_ragas_outputs(
    base_dir: Path,
    rows: list[dict[str, Any]],
    answer_rows: list[dict[str, Any]] | None = None,
) -> None:
    write_csv(base_dir / "scores.csv", rows)
    atomic_write_jsonl(base_dir / "scores.jsonl", rows)
    errors = [
        {"question_id": row["question_id"], "error": row["error"]}
        for row in rows
        if row.get("error")
    ]
    atomic_write_jsonl(base_dir / "errors.jsonl", errors)
    atomic_write_json(
        base_dir / "summary.json", summarize_ragas_scores(rows, answer_rows=answer_rows)
    )
