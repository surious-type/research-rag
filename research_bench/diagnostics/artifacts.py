from __future__ import annotations

from pathlib import Path

from research_bench.shared.io import ensure_dir


def ensure_run_artifact_dirs(run_dir: Path) -> dict[str, Path]:
    paths = {
        "root": run_dir,
        "build": run_dir / "build",
        "query": run_dir / "query",
        "query_traces": run_dir / "query" / "traces",
        "ragas": run_dir / "ragas",
    }
    for path in paths.values():
        ensure_dir(path)
    return paths
