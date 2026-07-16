from __future__ import annotations

from pathlib import Path

from research_bench.shared.io import ensure_dir


def trace_path_for_question(run_dir: Path, question_id: str) -> Path:
    traces_dir = run_dir / "query" / "traces"
    ensure_dir(traces_dir)
    return traces_dir / f"{question_id}.json"
