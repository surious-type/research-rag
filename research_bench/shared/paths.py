from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
