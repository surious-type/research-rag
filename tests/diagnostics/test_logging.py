import json
from pathlib import Path

from research_bench.diagnostics.artifacts import ensure_run_artifact_dirs
from research_bench.diagnostics.logging import log_stage_event


def test_log_stage_event_writes_progress_files(tmp_path: Path) -> None:
    log_stage_event(
        tmp_path,
        event="stage_started",
        stage="build",
        status="running",
        framework="msgraphrag",
        run_id="demo_1",
        message="build started",
    )

    assert (tmp_path / "progress.log").exists()
    lines = (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[0])
    assert payload["stage"] == "build"
    assert payload["status"] == "running"


def test_ensure_run_artifact_dirs_creates_expected_structure(tmp_path: Path) -> None:
    paths = ensure_run_artifact_dirs(tmp_path)

    assert paths["build"].exists()
    assert paths["query"].exists()
    assert paths["query_traces"].exists()
    assert paths["ragas"].exists()
