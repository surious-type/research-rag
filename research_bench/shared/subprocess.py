from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path

from .io import ensure_dir


def run_command(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    ensure_dir(stdout_path.parent) if stdout_path else None
    ensure_dir(stderr_path.parent) if stderr_path else None
    stdout_cm = stdout_path.open("w", encoding="utf-8") if stdout_path else contextlib.nullcontext(None)
    stderr_cm = stderr_path.open("w", encoding="utf-8") if stderr_path else contextlib.nullcontext(None)
    with stdout_cm as stdout_handle, stderr_cm as stderr_handle:
        return subprocess.run(
            args,
            cwd=cwd,
            env=env,
            text=True,
            stdout=stdout_handle if stdout_handle is not None else subprocess.DEVNULL,
            stderr=stderr_handle if stderr_handle is not None else subprocess.DEVNULL,
            check=False,
        )
