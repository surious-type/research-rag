from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import requests

from .data import (
    QUESTIONS_PATH,
    REPORTS_DIR,
    RESULTS_DIR,
    SOURCE_PATH,
    SMOKE_SOURCE_PATH,
    canonical_source_path,
    load_questions,
    load_smoke_questions,
    load_source_info,
    questions_sha256,
)
from .frameworks import get_adapter
from .models import CheckResult, FRAMEWORKS
from .ragas_eval import prepare_ragas_rows, save_ragas_outputs, save_ragas_placeholder
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


def check_environment() -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        source_info = load_source_info(ensure_source_txt())
        results.append(CheckResult("source.txt", "PASS", f"sha256={source_info.sha256} size={source_info.size_bytes}"))
    except Exception as exc:
        results.append(CheckResult("source.txt", "FAIL", str(exc)))

    try:
        questions = load_questions()
        results.append(CheckResult("questions.jsonl", "PASS", f"records={len(questions)} sha256={questions_sha256()}"))
    except Exception as exc:
        results.append(CheckResult("questions.jsonl", "FAIL", str(exc)))

    results.extend(_check_llm())
    results.extend(_check_embeddings())
    results.extend(_check_frameworks())
    results.extend(_check_kag_neo4j())
    results.extend(_check_docker())
    results.extend(_check_storage())
    return results


def _check_llm() -> list[CheckResult]:
    results = []
    try:
        models = requests.get("http://127.0.0.1:8080/v1/models", timeout=30)
        models.raise_for_status()
        payload = models.json()
        model_id = payload["data"][0]["id"]
        results.append(CheckResult("llm endpoint", "PASS", "reachable"))
        results.append(CheckResult("llm model", "PASS", model_id))
        probe = requests.post(
            "http://127.0.0.1:8080/v1/chat/completions",
            timeout=60,
            json={
                "model": model_id,
                "temperature": 0,
                "messages": [{"role": "user", "content": "Ответь словом ГОТОВО"}],
            },
        )
        probe.raise_for_status()
        results.append(CheckResult("llm chat completion", "PASS", probe.json()["choices"][0]["message"]["content"].strip()))
    except Exception as exc:
        results.append(CheckResult("llm endpoint", "FAIL", str(exc)))
    return results


def _check_embeddings() -> list[CheckResult]:
    results = []
    try:
        probe = requests.post(
            "http://127.0.0.1:8010/v1/embeddings",
            timeout=60,
            json={"model": "multilingual-e5-large", "input": ["probe"]},
        )
        probe.raise_for_status()
        data = probe.json()["data"][0]["embedding"]
        results.append(CheckResult("embedding endpoint", "PASS", "reachable"))
        results.append(CheckResult("embedding dimension", "PASS", str(len(data))))
    except Exception as exc:
        results.append(CheckResult("embedding endpoint", "FAIL", str(exc)))
    return results


def _check_frameworks() -> list[CheckResult]:
    results = []
    for name in FRAMEWORKS:
        if name == "msgraphrag":
            exists = (Path(".venv/bin/graphrag")).exists()
        elif name == "lightrag":
            exists = (Path(".venv/lib/python3.12/site-packages/lightrag")).exists()
        else:
            exists = (Path("frameworks/kag")).exists()
        results.append(CheckResult(name, "PASS" if exists else "FAIL", "available" if exists else "missing"))
    return results


def _check_kag_neo4j() -> list[CheckResult]:
    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        return [CheckResult("kag neo4j", "FAIL", str(exc))]

    uri = os.getenv("KAG_NEO4J_URI", "bolt://127.0.0.1:17687")
    user = os.getenv("KAG_NEO4J_USER", "neo4j")
    password = os.getenv("KAG_NEO4J_PASSWORD", "neo4j@openspg")
    database = os.getenv("KAG_NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            value = session.run("RETURN 1 AS value").single()["value"]
    except Exception as exc:
        return [CheckResult("kag neo4j", "FAIL", str(exc))]
    finally:
        driver.close()
    return [CheckResult("kag neo4j", "PASS", f"{uri} database={database} probe={value}")]


def _check_docker() -> list[CheckResult]:
    results = []
    docker_bin = shutil.which("docker")
    results.append(CheckResult("docker", "PASS" if docker_bin else "FAIL", docker_bin or "docker not found"))
    if not docker_bin:
        return results
    try:
        process = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            text=True,
            capture_output=True,
            check=False,
        )
        payload = process.stdout.strip().splitlines()
        results.append(CheckResult("containers", "PASS", ", ".join(payload) if payload else "no running containers"))
        details = process.stderr.strip() or ", ".join(payload) or "not running"
        results.append(CheckResult("neo4j for kag", "PASS" if any("neo4j" in item for item in payload) else "WARN", details))
    except Exception as exc:
        results.append(CheckResult("containers", "WARN", str(exc)))
    return results


def _check_storage() -> list[CheckResult]:
    ensure_dir(RESULTS_DIR)
    free_gb = shutil.disk_usage(RESULTS_DIR).free / (1024 ** 3)
    return [CheckResult("results dir", "PASS", str(RESULTS_DIR)), CheckResult("free space", "PASS" if free_gb > 5 else "WARN", f"{free_gb:.2f} GiB")]


def format_checks(rows: list[CheckResult]) -> str:
    lines = ["CHECK | STATUS | DETAILS"]
    lines.extend(f"{row.name} | {row.status} | {row.details}" for row in rows)
    return "\n".join(lines)


def create_run_dir(framework: str, smoke: bool = False) -> tuple[str, Path]:
    root = RESULTS_DIR / "_smoke" if smoke else RESULTS_DIR
    ensure_dir(root)
    run_id = utc_run_id(framework, existing=[item.name for item in root.iterdir() if item.is_dir()])
    run_dir = root / run_id
    for part in ("build", "query", "ragas"):
        ensure_dir(run_dir / part)
    return run_id, run_dir


def execute_run(framework: str, smoke: bool = False) -> tuple[str, Path]:
    checks = check_environment()
    if any(row.status == "FAIL" for row in checks):
        raise RuntimeError(format_checks(checks))

    run_id, run_dir = create_run_dir(framework, smoke=smoke)
    source_path = SMOKE_SOURCE_PATH if smoke else ensure_source_txt()
    question_rows = load_smoke_questions() if smoke else [row.payload for row in load_questions()]
    adapter = get_adapter(framework)

    manifest = {
        "run_id": run_id,
        "framework": framework,
        "smoke": smoke,
        "source": load_source_info(source_path).to_dict(),
        "questions_path": str(QUESTIONS_PATH),
        "questions_sha256": sha256_file(QUESTIONS_PATH if not smoke else Path("output/smoke_tests/data/smoke_questions.json")),
    }
    atomic_write_json(run_dir / "manifest.json", manifest)

    build_metrics, graph_metrics = adapter.build(source_path, run_dir)
    atomic_write_json(run_dir / "build" / "metrics.json", build_metrics.to_dict())
    atomic_write_json(run_dir / "build" / "graph_metrics.json", graph_metrics)
    if build_metrics.build_status != "success":
        atomic_write_json(run_dir / "verification.json", {"status": "FAIL", "failed_stage": "build"})
        raise RuntimeError(build_metrics.build_error or "build failed")

    try:
        answers = [answer.to_dict() for answer in adapter.query(run_id, run_dir, question_rows)]
    except Exception as exc:
        atomic_write_json(run_dir / "verification.json", {"status": "FAIL", "failed_stage": "query", "error": str(exc)})
        raise
    atomic_write_jsonl(run_dir / "query" / "answers.jsonl", answers)
    atomic_write_json(run_dir / "query" / "metrics.json", latency_summary([safe_float(row.get("latency_seconds")) for row in answers]))
    _run_ragas(run_dir, answers, limit=1 if smoke else None)
    verification = verify_run(run_id, smoke=smoke)
    atomic_write_json(run_dir / "verification.json", verification)
    return run_id, run_dir


def rerun_query_stage(run_id: str) -> Path:
    run_dir, smoke = find_run_dir(run_id)
    manifest = load_json(run_dir / "manifest.json")
    adapter = get_adapter(manifest["framework"])
    question_rows = load_smoke_questions() if smoke else [row.payload for row in load_questions()]
    answers = [answer.to_dict() for answer in adapter.query(run_id, run_dir, question_rows)]
    atomic_write_jsonl(run_dir / "query" / "answers.jsonl", answers)
    atomic_write_json(run_dir / "query" / "metrics.json", latency_summary([safe_float(row.get("latency_seconds")) for row in answers]))
    return run_dir


def rerun_ragas_stage(run_id: str, limit: int | None = None) -> Path:
    run_dir, _ = find_run_dir(run_id)
    answers = [json.loads(line) for line in (run_dir / "query" / "answers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    _run_ragas(run_dir, answers, limit=limit)
    return run_dir


def _run_ragas(run_dir: Path, answers: list[dict[str, Any]], limit: int | None = None) -> None:
    ragas_dir = run_dir / "ragas"
    if not answers:
        save_ragas_placeholder(ragas_dir, answers, "no answers available")
        return

    eligible_answers = [row for row in answers if row.get("status") == "success" and row.get("contexts")]
    answers_to_score = eligible_answers[:limit] if limit is not None else eligible_answers
    prepared = prepare_ragas_rows(answers_to_score)
    score_rows_by_question: dict[str, dict[str, Any]] = {}
    for answer_row in answers:
        reason = None
        if answer_row.get("status") == "degraded":
            reason = "degraded answer is excluded from ragas"
        elif answer_row.get("status") == "failed":
            reason = "failed answer is excluded from ragas"
        elif not answer_row.get("contexts"):
            reason = "contextless answer is excluded from ragas"
        elif answer_row not in answers_to_score:
            reason = f"skipped after evaluating first {len(answers_to_score)} answer(s)"
        score_rows_by_question[answer_row["question_id"]] = {
            "question_id": answer_row["question_id"],
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
            "error": reason,
        }

    if not prepared:
        save_ragas_outputs(ragas_dir, list(score_rows_by_question.values()), answer_rows=answers)
        return
    try:
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from datasets import Dataset
    except Exception as exc:
        save_ragas_placeholder(ragas_dir, answers, str(exc))
        return

    dataset = Dataset.from_list(prepared)
    llm = LangchainLLMWrapper(
        ChatOpenAI(
            base_url="http://127.0.0.1:8080/v1",
            api_key="local",
            model="/models/Qwen3.5-35B-A3B-Q4_K_M.gguf",
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            base_url="http://127.0.0.1:8010/v1",
            api_key="local",
            model="multilingual-e5-large",
        )
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        batch_size=1,
        show_progress=False,
    )
    frame = result.to_pandas()
    rows = []
    for answer_row, ragas_row in zip(answers_to_score, frame.to_dict(orient="records"), strict=False):
        score_rows_by_question[answer_row["question_id"]] = {
            "question_id": answer_row["question_id"],
            "faithfulness": safe_float(ragas_row.get("faithfulness")),
            "answer_relevancy": safe_float(ragas_row.get("answer_relevancy")),
            "context_precision": safe_float(ragas_row.get("context_precision")),
            "context_recall": safe_float(ragas_row.get("context_recall")),
            "error": None,
        }
    for answer_row in answers:
        rows.append(score_rows_by_question[answer_row["question_id"]])
    save_ragas_outputs(ragas_dir, rows, answer_rows=answers)


def verify_run(run_id: str, smoke: bool | None = None) -> dict[str, Any]:
    base, detected_smoke = find_run_dir(run_id, smoke=smoke)
    manifest = load_json(base / "manifest.json")
    build = load_json(base / "build" / "metrics.json")
    graph = load_json(base / "build" / "graph_metrics.json")
    answers = [json.loads(line) for line in (base / "query" / "answers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ragas_summary = load_json(base / "ragas" / "summary.json")
    diagnostics_path = base / "query" / "diagnostics.json"
    diagnostics = load_json(diagnostics_path) if diagnostics_path.exists() else {}
    status = "PASS"
    issues = []

    if build.get("build_status") != "success":
        status = "FAIL"
        issues.append("build not successful")
    if not build.get("documents_count"):
        status = "FAIL"
        issues.append("documents_count is zero")
    if not build.get("chunks_count"):
        status = "FAIL"
        issues.append("chunks_count is zero")
    expected_questions = len(load_smoke_questions()) if detected_smoke else 100
    if len(answers) != expected_questions:
        status = "FAIL"
        issues.append(f"expected {expected_questions} answers, got {len(answers)}")
    if len({row["question_id"] for row in answers}) != len(answers):
        status = "FAIL"
        issues.append("duplicate question_id")
    if diagnostics:
        if diagnostics.get("selected_pipeline") != "kag_solver_pipeline":
            status = "FAIL"
            issues.append("selected_pipeline is not kag_solver_pipeline")
        if not diagnostics.get("resolved_retriever_types"):
            status = "FAIL"
            issues.append("resolved_retriever_types is empty")
        if diagnostics.get("reporter_type") != "trace_log_reporter":
            status = "FAIL"
            issues.append("reporter_type is not trace_log_reporter")
        if diagnostics.get("fulltext_index", {}).get("name") != "_default_text_index":
            status = "FAIL"
            issues.append("missing _default_text_index")
        if diagnostics.get("fulltext_index", {}).get("state") != "ONLINE":
            status = "FAIL"
            issues.append("_default_text_index is not ONLINE")
        vector_probe_status = str(diagnostics.get("vector_search_probe_status") or "").upper()
        if vector_probe_status == "FAIL":
            status = "FAIL"
            issues.append("vector search probes failed")
    for row in answers:
        if row["status"] == "success" and not str(row.get("answer", "")).strip():
            status = "FAIL"
            issues.append(f"empty answer for {row['question_id']}")
        if row["status"] == "success" and not row.get("contexts"):
            status = "FAIL"
            issues.append(f"missing contexts for {row['question_id']}")
        if row["status"] != "success":
            status = "FAIL"
            issues.append(f"query status is {row['status']} for {row['question_id']}")
    for metric_name, values in ragas_summary.items():
        if metric_name in {"successful_retrieval_questions", "degraded_questions", "failed_questions", "retrieval_failure_rate"}:
            continue
        for key in ("mean", "median"):
            value = values.get(key)
            if value is not None and not 0 <= value <= 1:
                status = "FAIL"
                issues.append(f"invalid ragas range for {metric_name}.{key}")
        if values.get("valid_count", 0) + values.get("failed_count", 0) != len(answers):
            status = "FAIL"
            issues.append(f"inconsistent counts for {metric_name}")
    ragas_metric_names = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    if detected_smoke and not any(ragas_summary.get(metric, {}).get("valid_count", 0) > 0 for metric in ragas_metric_names):
        status = "FAIL"
        issues.append("smoke run has no valid ragas scores")
    if not graph:
        status = "FAIL"
        issues.append("missing graph metrics")

    if status == "PASS" and issues:
        status = "WARN"
    return {"status": status, "issues": issues, "manifest": manifest}
