from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def create_run_dir(*, results_dir: Path, framework: str, smoke: bool, ensure_dir_fn, utc_run_id_fn) -> tuple[str, Path]:
    root = results_dir / "_smoke" if smoke else results_dir
    ensure_dir_fn(root)
    run_id = utc_run_id_fn(framework, existing=[item.name for item in root.iterdir() if item.is_dir()])
    run_dir = root / run_id
    for part in ("build", "query", "ragas"):
        ensure_dir_fn(run_dir / part)
    return run_id, run_dir


def resolve_run_inputs(*, smoke: bool, smoke_source_path: Path, ensure_source_txt_fn, load_smoke_questions_fn, load_questions_fn) -> tuple[Path, list[dict[str, Any]]]:
    source_path = smoke_source_path if smoke else ensure_source_txt_fn()
    question_rows = load_smoke_questions_fn() if smoke else [row.payload for row in load_questions_fn()]
    return source_path, question_rows


def build_manifest(
    *,
    run_id: str,
    framework: str,
    smoke: bool,
    source_path: Path,
    questions_path: Path,
    smoke_questions_path: Path,
    load_source_info_fn,
    sha256_file_fn,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "framework": framework,
        "smoke": smoke,
        "source": load_source_info_fn(source_path).to_dict(),
        "questions_path": str(questions_path),
        "questions_sha256": sha256_file_fn(questions_path if not smoke else smoke_questions_path),
    }


def write_build_artifacts(run_dir: Path, build_metrics: Any, graph_metrics: dict[str, Any], *, atomic_write_json_fn) -> None:
    atomic_write_json_fn(run_dir / "build" / "metrics.json", build_metrics.to_dict())
    atomic_write_json_fn(run_dir / "build" / "graph_metrics.json", graph_metrics)


def run_build_stage(adapter: Any, source_path: Path, run_dir: Path, *, write_build_artifacts_fn, atomic_write_json_fn) -> tuple[Any, dict[str, Any]]:
    build_metrics, graph_metrics = adapter.build(source_path, run_dir)
    write_build_artifacts_fn(run_dir, build_metrics, graph_metrics)
    if build_metrics.build_status != "success":
        atomic_write_json_fn(run_dir / "verification.json", {"status": "FAIL", "failed_stage": "build"})
        raise RuntimeError(build_metrics.build_error or "build failed")
    return build_metrics, graph_metrics


def write_query_artifacts(run_dir: Path, answers: list[dict[str, Any]], *, atomic_write_json_fn, atomic_write_jsonl_fn, latency_summary_fn, safe_float_fn) -> None:
    atomic_write_jsonl_fn(run_dir / "query" / "answers.jsonl", answers)
    atomic_write_json_fn(
        run_dir / "query" / "metrics.json",
        latency_summary_fn([safe_float_fn(row.get("latency_seconds")) for row in answers]),
    )


def run_query_stage(adapter: Any, run_id: str, run_dir: Path, question_rows: list[dict[str, Any]], *, write_query_artifacts_fn, atomic_write_json_fn) -> list[dict[str, Any]]:
    try:
        answers = [answer.to_dict() for answer in adapter.query(run_id, run_dir, question_rows)]
    except Exception as exc:
        atomic_write_json_fn(
            run_dir / "verification.json",
            {"status": "FAIL", "failed_stage": "query", "error": str(exc)},
        )
        raise
    write_query_artifacts_fn(run_dir, answers)
    return answers


def finalize_run(run_id: str, run_dir: Path, smoke: bool, answers: list[dict[str, Any]], *, run_ragas_fn, verify_run_fn, atomic_write_json_fn) -> None:
    run_ragas_fn(run_dir, answers, limit=1 if smoke else None)
    verification = verify_run_fn(run_id, smoke=smoke)
    atomic_write_json_fn(run_dir / "verification.json", verification)


def execute_run(
    framework: str,
    *,
    smoke: bool,
    check_environment_fn,
    format_checks_fn,
    create_run_dir_fn,
    resolve_run_inputs_fn,
    get_adapter_fn,
    build_manifest_fn,
    atomic_write_json_fn,
    run_build_stage_fn,
    run_query_stage_fn,
    finalize_run_fn,
) -> tuple[str, Path]:
    checks = check_environment_fn()
    if any(row.status == "FAIL" for row in checks):
        raise RuntimeError(format_checks_fn(checks))

    run_id, run_dir = create_run_dir_fn(framework, smoke=smoke)
    source_path, question_rows = resolve_run_inputs_fn(smoke)
    adapter = get_adapter_fn(framework)

    manifest = build_manifest_fn(run_id, framework, smoke, source_path)
    atomic_write_json_fn(run_dir / "manifest.json", manifest)
    run_build_stage_fn(adapter, source_path, run_dir)
    answers = run_query_stage_fn(adapter, run_id, run_dir, question_rows)
    finalize_run_fn(run_id, run_dir, smoke, answers)
    return run_id, run_dir


def rerun_query_stage(run_id: str, *, find_run_dir_fn, load_json_fn, get_adapter_fn, resolve_run_inputs_fn, run_query_stage_fn) -> Path:
    run_dir, smoke = find_run_dir_fn(run_id)
    manifest = load_json_fn(run_dir / "manifest.json")
    adapter = get_adapter_fn(manifest["framework"])
    _, question_rows = resolve_run_inputs_fn(smoke)
    run_query_stage_fn(adapter, run_id, run_dir, question_rows)
    return run_dir


def rerun_ragas_stage(run_id: str, *, limit: int | None, find_run_dir_fn, run_ragas_fn) -> Path:
    run_dir, _ = find_run_dir_fn(run_id)
    answers = [json.loads(line) for line in (run_dir / "query" / "answers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    run_ragas_fn(run_dir, answers, limit=limit)
    return run_dir
