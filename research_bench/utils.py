from __future__ import annotations

import csv
import contextlib
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def word_count(text: str) -> int:
    return len([part for part in text.split() if part.strip()])


def utc_run_id(prefix: str, now: datetime | None = None, existing: Iterable[str] = ()) -> str:
    now = now or datetime.now()
    base = f"{prefix}_{now.strftime('%Y%m%d_%H%M%S')}"
    existing_set = set(existing)
    if base not in existing_set:
        return base
    for index in range(1, 1000):
        candidate = f"{base}_{index:03d}"
        if candidate not in existing_set:
            return candidate
    raise RuntimeError(f"Could not generate unique run id for {prefix}")


def atomic_write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        handle.write(content)
        temp_name = handle.name
    os.replace(temp_name, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, content)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        temp_name = handle.name
    os.replace(temp_name, path)


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


def file_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copyfile(src, dst)


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def aggregate_numeric(values: Sequence[float | int | None]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"mean": None, "median": None, "standard_deviation": None, "valid_count": 0, "failed_count": len(values)}
    return {
        "mean": float(statistics.fmean(clean)),
        "median": float(statistics.median(clean)),
        "standard_deviation": float(statistics.pstdev(clean)) if len(clean) > 1 else 0.0,
        "valid_count": len(clean),
        "failed_count": len(values) - len(clean),
    }


def latency_summary(values: Sequence[float | None]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            "questions_count": len(values),
            "successful_questions": 0,
            "failed_questions": len(values),
            "mean_latency_seconds": None,
            "median_latency_seconds": None,
            "p95_latency_seconds": None,
            "min_latency_seconds": None,
            "max_latency_seconds": None,
        }
    return {
        "questions_count": len(values),
        "successful_questions": len(clean),
        "failed_questions": len(values) - len(clean),
        "mean_latency_seconds": float(statistics.fmean(clean)),
        "median_latency_seconds": float(statistics.median(clean)),
        "p95_latency_seconds": percentile(clean, 0.95),
        "min_latency_seconds": min(clean),
        "max_latency_seconds": max(clean),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
