from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import REPORTS_DIR, RESULTS_DIR
from .models import FRAMEWORKS
from .parsers import load_answers, load_ragas_scores
from .utils import ensure_dir, load_json, write_csv


def build_report() -> Path:
    ensure_dir(REPORTS_DIR)
    latest = {}
    for framework in FRAMEWORKS:
        candidates = sorted((RESULTS_DIR).glob(f"{framework}_*"))
        for candidate in reversed(candidates):
            verification = candidate / "verification.json"
            if verification.exists() and load_json(verification).get("status") == "PASS":
                latest[framework] = candidate
                break
    if len(latest) != len(FRAMEWORKS):
        missing = sorted(set(FRAMEWORKS) - set(latest))
        raise RuntimeError(f"Missing successful runs for: {', '.join(missing)}")

    build_rows = []
    graph_rows = []
    query_rows = []
    ragas_rows = []
    per_question_rows = []
    ragas_by_type: dict[tuple[str, str], list[dict[str, Any]]] = {}
    summary_lines = ["# GraphRAG-Bench-inspired evaluation", ""]

    shared_fingerprints = None
    for framework, run_dir in latest.items():
        manifest = load_json(run_dir / "manifest.json")
        fingerprint = (
            manifest["source"]["sha256"],
            manifest["questions_sha256"],
        )
        if shared_fingerprints is None:
            shared_fingerprints = fingerprint
        elif shared_fingerprints != fingerprint:
            raise RuntimeError("Latest successful runs do not share the same source/questions hashes")

        build = load_json(run_dir / "build" / "metrics.json")
        graph = load_json(run_dir / "build" / "graph_metrics.json")
        query = load_json(run_dir / "query" / "metrics.json")
        ragas_summary = load_json(run_dir / "ragas" / "summary.json")
        answers = load_answers(run_dir / "query" / "answers.jsonl")
        ragas_scores = load_ragas_scores(run_dir / "ragas" / "scores.jsonl")

        build_rows.append({"framework": framework, **build})
        graph_rows.append({"framework": framework, **graph})
        query_rows.append({"framework": framework, **query})
        ragas_rows.append(
            {
                "framework": framework,
                "faithfulness": ragas_summary["faithfulness"]["mean"],
                "answer_relevancy": ragas_summary["answer_relevancy"]["mean"],
                "context_precision": ragas_summary["context_precision"]["mean"],
                "context_recall": ragas_summary["context_recall"]["mean"],
            }
        )
        score_by_question = {row["question_id"]: row for row in ragas_scores}
        for answer in answers:
            combined = {
                "framework": framework,
                "question_id": answer["question_id"],
                "question_type": answer.get("question_type"),
                "status": answer["status"],
                "latency_seconds": answer.get("latency_seconds"),
            }
            combined.update(score_by_question.get(answer["question_id"], {}))
            per_question_rows.append(combined)
            ragas_by_type.setdefault((framework, answer.get("question_type") or "unknown"), []).append(combined)

        summary_lines.append(f"## {framework}")
        summary_lines.append(f"- run_id: `{run_dir.name}`")
        summary_lines.append(f"- build_time_seconds: {build.get('build_time_seconds')}")
        summary_lines.append(f"- successful_questions: {query.get('successful_questions')}")
        summary_lines.append(f"- faithfulness: {ragas_summary['faithfulness']['mean']}")
        summary_lines.append("")

    ragas_type_rows = []
    for (framework, question_type), rows in sorted(ragas_by_type.items()):
        ragas_type_rows.append(
            {
                "framework": framework,
                "question_type": question_type,
                "faithfulness": _mean(rows, "faithfulness"),
                "answer_relevancy": _mean(rows, "answer_relevancy"),
                "context_precision": _mean(rows, "context_precision"),
                "context_recall": _mean(rows, "context_recall"),
            }
        )

    write_csv(REPORTS_DIR / "build_comparison.csv", build_rows)
    write_csv(REPORTS_DIR / "graph_comparison.csv", graph_rows)
    write_csv(REPORTS_DIR / "query_comparison.csv", query_rows)
    write_csv(REPORTS_DIR / "ragas_comparison.csv", ragas_rows)
    write_csv(REPORTS_DIR / "ragas_by_question_type.csv", ragas_type_rows)
    write_csv(REPORTS_DIR / "per_question_comparison.csv", per_question_rows)
    (REPORTS_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return REPORTS_DIR / "summary.md"


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)
