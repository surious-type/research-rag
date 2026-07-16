from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import QuestionRecord, SourceInfo
from .utils import sha256_file, word_count


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SOURCE_PATH = DATA_DIR / "corpus" / "source.txt"
QUESTIONS_PATH = DATA_DIR / "questions" / "questions.jsonl"
SMOKE_SOURCE_PATH = ROOT / "output" / "smoke_tests" / "data" / "smoke_document.txt"
SMOKE_QUESTIONS_PATH = ROOT / "output" / "smoke_tests" / "data" / "smoke_questions.json"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"


def canonical_source_path() -> Path:
    if SOURCE_PATH.exists():
        return SOURCE_PATH
    raise FileNotFoundError(f"Source corpus not found: {SOURCE_PATH}")


def load_source_info(path: Path | None = None) -> SourceInfo:
    path = path or canonical_source_path()
    text = path.read_text(encoding="utf-8")
    return SourceInfo(
        path=str(path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        characters_count=len(text),
        words_count=word_count(text),
    )


def load_questions(path: Path | None = None) -> list[QuestionRecord]:
    path = path or QUESTIONS_PATH
    rows: list[QuestionRecord] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        for key in ("id", "question", "reference_answer"):
            if not str(payload.get(key, "")).strip():
                raise ValueError(f"Missing required field {key!r} at line {line_number}")
        if "target_versions" in payload:
            raise ValueError("target_versions is forbidden")
        question_id = str(payload["id"])
        question_text = str(payload["question"])
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question id: {question_id}")
        if question_text in seen_questions:
            raise ValueError(f"Duplicate question text: {question_text}")
        seen_ids.add(question_id)
        seen_questions.add(question_text)
        rows.append(
            QuestionRecord(
                id=question_id,
                question=question_text,
                reference_answer=str(payload["reference_answer"]),
                payload=payload,
            )
        )
    if len(rows) != 100:
        raise ValueError(f"Expected 100 questions, found {len(rows)}")
    return rows


def questions_sha256(path: Path | None = None) -> str:
    return sha256_file(path or QUESTIONS_PATH)


def load_smoke_questions() -> list[dict[str, Any]]:
    payload = json.loads(SMOKE_QUESTIONS_PATH.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload.get("questions", [])
    return payload
