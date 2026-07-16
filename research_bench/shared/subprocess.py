from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path

from research_bench.diagnostics.logging import log_stage_event
from .io import ensure_dir


def run_command(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    progress_run_dir: Path | None = None,
    progress_stage: str | None = None,
    progress_framework: str | None = None,
    progress_run_id: str | None = None,
    process_name: str | None = None,
) -> subprocess.CompletedProcess[str]:
    ensure_dir(stdout_path.parent) if stdout_path else None
    ensure_dir(stderr_path.parent) if stderr_path else None
    if progress_run_dir and progress_stage and progress_framework and progress_run_id:
        log_stage_event(
            progress_run_dir,
            event="process_started",
            stage=progress_stage,
            status="running",
            framework=progress_framework,
            run_id=progress_run_id,
            message=f"{process_name or args[0]} started: {' '.join(args)}",
            artifact_path=stdout_path,
        )
    stdout_cm = stdout_path.open("w", encoding="utf-8") if stdout_path else contextlib.nullcontext(None)
    stderr_cm = stderr_path.open("w", encoding="utf-8") if stderr_path else contextlib.nullcontext(None)
    with stdout_cm as stdout_handle, stderr_cm as stderr_handle:
        result = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            text=True,
            stdout=stdout_handle if stdout_handle is not None else subprocess.DEVNULL,
            stderr=stderr_handle if stderr_handle is not None else subprocess.DEVNULL,
            check=False,
        )
    if progress_run_dir and progress_stage and progress_framework and progress_run_id:
        log_stage_event(
            progress_run_dir,
            event="process_finished",
            stage=progress_stage,
            status="success" if result.returncode == 0 else "failed",
            framework=progress_framework,
            run_id=progress_run_id,
            message=f"{process_name or args[0]} finished with exit code {result.returncode}",
            artifact_path=stderr_path or stdout_path,
        )
    return result
