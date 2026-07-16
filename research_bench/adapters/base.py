from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class FrameworkAdapter(ABC):
    @abstractmethod
    def check(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build(self, source_path: Path, run_dir: Path) -> Any:
        raise NotImplementedError

    @abstractmethod
    def query(self, run_id: str, run_dir: Path, question_rows: list[dict[str, Any]]) -> Any:
        raise NotImplementedError
