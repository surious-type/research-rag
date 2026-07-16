from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from research_bench.shared.io import ensure_dir


def log_stage_event(
    run_dir: Path,
    *,
    event: str,
    stage: str,
    status: str,
    framework: str,
    run_id: str,
    message: str,
    artifact_path: Path | None = None,
) -> None:
    ensure_dir(run_dir)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "framework": framework,
        "stage": stage,
        "event": event,
        "status": status,
        "message": message,
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
    }
    with (run_dir / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{payload['timestamp']}] framework={framework} run_id={run_id} stage={stage} event={event} status={status} message={message}\n"
        )
    with (run_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
