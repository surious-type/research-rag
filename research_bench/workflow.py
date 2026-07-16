from __future__ import annotations

"""High-level benchmark workflow orchestration.

Reading guide:
1. `check_environment()` validates local prerequisites.
2. `execute_run()` coordinates the benchmark stages for one framework.
3. `rerun_query_stage()` / `rerun_ragas_stage()` replay individual stages.
4. `verify_run()` validates the final artifacts.

The module intentionally keeps the public workflow contract stable for the CLI
and tests, while the internal helpers are organized by stage to make the flow
easier to follow.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import requests

from .data import (
    QUESTIONS_PATH,
    RESULTS_DIR,
    SOURCE_PATH,
    SMOKE_SOURCE_PATH,
    canonical_source_path,
    load_questions,
    load_smoke_questions,
    load_source_info,
    questions_sha256,
)
from .adapters.registry import get_adapter
from .models import CheckResult
from .ragas_eval import prepare_ragas_rows, save_ragas_outputs, save_ragas_placeholder
from .workflows import checks as checks_mod
from .workflows import ragas_verification as ragas_mod
from .utils import (
    atomic_write_json,
    atomic_write_jsonl,
    copy_file,
    ensure_dir,
    latency_summary,
    load_json,
    safe_float,
    sha256_file,
    utc_run_id,
)


def ensure_source_txt() -> Path:
    return canonical_source_path()


def find_run_dir(run_id: str, smoke: bool | None = None) -> tuple[Path, bool]:
    candidates = []
    if smoke is True:
        candidates.append((RESULTS_DIR / "_smoke" / run_id, True))
    elif smoke is False:
        candidates.append((RESULTS_DIR / run_id, False))
    else:
        candidates.extend(((RESULTS_DIR / run_id, False), (RESULTS_DIR / "_smoke" / run_id, True)))
    for path, detected_smoke in candidates:
        if path.exists():
            return path, detected_smoke
    raise FileNotFoundError(f"Run not found: {run_id}")


def _check_llm() -> list[CheckResult]:
    checks_mod.requests = requests
    return checks_mod._check_llm()


def _check_embeddings() -> list[CheckResult]:
    checks_mod.requests = requests
    return checks_mod._check_embeddings()


def _check_frameworks() -> list[CheckResult]:
    return checks_mod._check_frameworks()


def _check_kag_neo4j() -> list[CheckResult]:
    return checks_mod._check_kag_neo4j()


def _check_docker() -> list[CheckResult]:
    checks_mod.shutil = shutil
    checks_mod.subprocess = subprocess
    return checks_mod._check_docker()


def _check_storage() -> list[CheckResult]:
    checks_mod.ensure_dir = ensure_dir
    return checks_mod._check_storage()


def check_environment() -> list[CheckResult]:
    return checks_mod.build_check_results(
        ensure_source_txt_fn=ensure_source_txt,
        load_questions_fn=load_questions,
        questions_sha256_fn=questions_sha256,
        load_source_info_fn=load_source_info,
        check_llm_fn=_check_llm,
        check_embeddings_fn=_check_embeddings,
        check_frameworks_fn=_check_frameworks,
        check_kag_neo4j_fn=_check_kag_neo4j,
        check_docker_fn=_check_docker,
        check_storage_fn=_check_storage,
    )


def format_checks(rows: list[CheckResult]) -> str:
    return checks_mod.format_checks(rows)


def create_run_dir(framework: str, smoke: bool = False) -> tuple[str, Path]:
    root = RESULTS_DIR / "_smoke" if smoke else RESULTS_DIR
    ensure_dir(root)
    run_id = utc_run_id(framework, existing=[item.name for item in root.iterdir() if item.is_dir()])
    run_dir = root / run_id
    for part in ("build", "query", "ragas"):
        ensure_dir(run_dir / part)
    return run_id, run_dir


def _resolve_run_inputs(smoke: bool) -> tuple[Path, list[dict[str, Any]]]:
    source_path = SMOKE_SOURCE_PATH if smoke else ensure_source_txt()
    question_rows = load_smoke_questions() if smoke else [row.payload for row in load_questions()]
    return source_path, question_rows


def _build_manifest(run_id: str, framework: str, smoke: bool, source_path: Path) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "framework": framework,
        "smoke": smoke,
        "source": load_source_info(source_path).to_dict(),
        "questions_path": str(QUESTIONS_PATH),
        "questions_sha256": sha256_file(
            QUESTIONS_PATH if not smoke else Path("output/smoke_tests/data/smoke_questions.json")
        ),
    }


def _write_build_artifacts(run_dir: Path, build_metrics: Any, graph_metrics: dict[str, Any]) -> None:
    atomic_write_json(run_dir / "build" / "metrics.json", build_metrics.to_dict())
    atomic_write_json(run_dir / "build" / "graph_metrics.json", graph_metrics)


def _run_build_stage(adapter: Any, source_path: Path, run_dir: Path) -> tuple[Any, dict[str, Any]]:
    build_metrics, graph_metrics = adapter.build(source_path, run_dir)
    _write_build_artifacts(run_dir, build_metrics, graph_metrics)
    if build_metrics.build_status != "success":
        atomic_write_json(run_dir / "verification.json", {"status": "FAIL", "failed_stage": "build"})
        raise RuntimeError(build_metrics.build_error or "build failed")
    return build_metrics, graph_metrics


def _write_query_artifacts(run_dir: Path, answers: list[dict[str, Any]]) -> None:
    atomic_write_jsonl(run_dir / "query" / "answers.jsonl", answers)
    atomic_write_json(
        run_dir / "query" / "metrics.json",
        latency_summary([safe_float(row.get("latency_seconds")) for row in answers]),
    )


def _run_query_stage(
    adapter: Any,
    run_id: str,
    run_dir: Path,
    question_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        answers = [answer.to_dict() for answer in adapter.query(run_id, run_dir, question_rows)]
    except Exception as exc:
        atomic_write_json(
            run_dir / "verification.json",
            {"status": "FAIL", "failed_stage": "query", "error": str(exc)},
        )
        raise
    _write_query_artifacts(run_dir, answers)
    return answers


def _finalize_run(run_id: str, run_dir: Path, smoke: bool, answers: list[dict[str, Any]]) -> None:
    _run_ragas(run_dir, answers, limit=1 if smoke else None)
    verification = verify_run(run_id, smoke=smoke)
    atomic_write_json(run_dir / "verification.json", verification)


def execute_run(framework: str, smoke: bool = False) -> tuple[str, Path]:
    checks = check_environment()
    if any(row.status == "FAIL" for row in checks):
        raise RuntimeError(format_checks(checks))

    run_id, run_dir = create_run_dir(framework, smoke=smoke)
    source_path, question_rows = _resolve_run_inputs(smoke)
    adapter = get_adapter(framework)

    manifest = _build_manifest(run_id, framework, smoke, source_path)
    atomic_write_json(run_dir / "manifest.json", manifest)
    _run_build_stage(adapter, source_path, run_dir)
    answers = _run_query_stage(adapter, run_id, run_dir, question_rows)
    _finalize_run(run_id, run_dir, smoke, answers)
    return run_id, run_dir


def rerun_query_stage(run_id: str) -> Path:
    run_dir, smoke = find_run_dir(run_id)
    manifest = load_json(run_dir / "manifest.json")
    adapter = get_adapter(manifest["framework"])
    _, question_rows = _resolve_run_inputs(smoke)
    _run_query_stage(adapter, run_id, run_dir, question_rows)
    return run_dir


def rerun_ragas_stage(run_id: str, limit: int | None = None) -> Path:
    run_dir, _ = find_run_dir(run_id)
    answers = [json.loads(line) for line in (run_dir / "query" / "answers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    _run_ragas(run_dir, answers, limit=limit)
    return run_dir


def _run_ragas(run_dir: Path, answers: list[dict[str, Any]], limit: int | None = None) -> None:
    ragas_mod.save_ragas_outputs = save_ragas_outputs
    ragas_mod.save_ragas_placeholder = save_ragas_placeholder
    ragas_mod.prepare_ragas_rows = prepare_ragas_rows
    return ragas_mod._run_ragas_stage(
        run_dir,
        answers,
        limit=limit,
        prepare_ragas_rows_fn=prepare_ragas_rows,
        save_ragas_outputs_fn=save_ragas_outputs,
        save_ragas_placeholder_fn=save_ragas_placeholder,
        safe_float_fn=safe_float,
    )


def verify_run(run_id: str, smoke: bool | None = None) -> dict[str, Any]:
    run_dir, detected_smoke = find_run_dir(run_id, smoke=smoke)
    return ragas_mod.build_verification_result(
        run_dir,
        detected_smoke=detected_smoke,
        load_smoke_questions_fn=load_smoke_questions,
    )
