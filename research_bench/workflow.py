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
    SMOKE_QUESTIONS_PATH,
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
from .workflows import run_stages as run_stages_mod
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
        candidates.extend(
            ((RESULTS_DIR / run_id, False), (RESULTS_DIR / "_smoke" / run_id, True))
        )
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
    return run_stages_mod.create_run_dir(
        results_dir=RESULTS_DIR,
        framework=framework,
        smoke=smoke,
        ensure_dir_fn=ensure_dir,
        utc_run_id_fn=utc_run_id,
    )


def _resolve_run_inputs(smoke: bool) -> tuple[Path, list[dict[str, Any]]]:
    return run_stages_mod.resolve_run_inputs(
        smoke=smoke,
        smoke_source_path=SMOKE_SOURCE_PATH,
        ensure_source_txt_fn=ensure_source_txt,
        load_smoke_questions_fn=load_smoke_questions,
        load_questions_fn=load_questions,
    )


def _build_manifest(
    run_id: str, framework: str, smoke: bool, source_path: Path
) -> dict[str, Any]:
    return run_stages_mod.build_manifest(
        run_id=run_id,
        framework=framework,
        smoke=smoke,
        source_path=source_path,
        questions_path=QUESTIONS_PATH,
        smoke_questions_path=SMOKE_QUESTIONS_PATH,
        load_source_info_fn=load_source_info,
        sha256_file_fn=sha256_file,
    )


def _write_build_artifacts(
    run_dir: Path, build_metrics: Any, graph_metrics: dict[str, Any]
) -> None:
    run_stages_mod.write_build_artifacts(
        run_dir,
        build_metrics,
        graph_metrics,
        atomic_write_json_fn=atomic_write_json,
    )


def _run_build_stage(
    adapter: Any, source_path: Path, run_dir: Path
) -> tuple[Any, dict[str, Any]]:
    return run_stages_mod.run_build_stage(
        adapter,
        source_path,
        run_dir,
        write_build_artifacts_fn=_write_build_artifacts,
        atomic_write_json_fn=atomic_write_json,
    )


def _write_query_artifacts(run_dir: Path, answers: list[dict[str, Any]]) -> None:
    run_stages_mod.write_query_artifacts(
        run_dir,
        answers,
        atomic_write_json_fn=atomic_write_json,
        atomic_write_jsonl_fn=atomic_write_jsonl,
        latency_summary_fn=latency_summary,
        safe_float_fn=safe_float,
    )


def _run_query_stage(
    adapter: Any,
    run_id: str,
    run_dir: Path,
    question_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return run_stages_mod.run_query_stage(
        adapter,
        run_id,
        run_dir,
        question_rows,
        write_query_artifacts_fn=_write_query_artifacts,
        atomic_write_json_fn=atomic_write_json,
    )


def _finalize_run(
    run_id: str, run_dir: Path, smoke: bool, answers: list[dict[str, Any]]
) -> None:
    run_stages_mod.finalize_run(
        run_id,
        run_dir,
        smoke,
        answers,
        run_ragas_fn=_run_ragas,
        verify_run_fn=verify_run,
        atomic_write_json_fn=atomic_write_json,
    )


def execute_run(framework: str, smoke: bool = False) -> tuple[str, Path]:
    return run_stages_mod.execute_run(
        framework,
        smoke=smoke,
        check_environment_fn=check_environment,
        format_checks_fn=format_checks,
        create_run_dir_fn=create_run_dir,
        resolve_run_inputs_fn=_resolve_run_inputs,
        get_adapter_fn=get_adapter,
        build_manifest_fn=_build_manifest,
        atomic_write_json_fn=atomic_write_json,
        run_build_stage_fn=_run_build_stage,
        run_query_stage_fn=_run_query_stage,
        finalize_run_fn=_finalize_run,
    )


def rerun_query_stage(run_id: str) -> Path:
    return run_stages_mod.rerun_query_stage(
        run_id,
        find_run_dir_fn=find_run_dir,
        load_json_fn=load_json,
        get_adapter_fn=get_adapter,
        resolve_run_inputs_fn=_resolve_run_inputs,
        run_query_stage_fn=_run_query_stage,
    )


def rerun_ragas_stage(run_id: str, limit: int | None = None) -> Path:
    return run_stages_mod.rerun_ragas_stage(
        run_id,
        limit=limit,
        find_run_dir_fn=find_run_dir,
        run_ragas_fn=_run_ragas,
    )


def _run_ragas(
    run_dir: Path, answers: list[dict[str, Any]], limit: int | None = None
) -> None:
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
        evaluate_prepared_row_fn=ragas_mod._evaluate_prepared_ragas_row,
    )


def verify_run(run_id: str, smoke: bool | None = None) -> dict[str, Any]:
    run_dir, detected_smoke = find_run_dir(run_id, smoke=smoke)
    return ragas_mod.build_verification_result(
        run_dir,
        detected_smoke=detected_smoke,
        load_smoke_questions_fn=load_smoke_questions,
    )
