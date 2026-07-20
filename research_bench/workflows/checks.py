from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import requests

from research_bench.adapters.kag import get_kag_neo4j_config
from research_bench.data import RESULTS_DIR, ROOT, load_questions, load_source_info, questions_sha256
from research_bench.models import CheckResult, FRAMEWORKS
from research_bench.runtime_config import DEFAULT_TEMPERATURE, load_model_runtime_config
from research_bench.shared.io import ensure_dir


def check_environment() -> list[CheckResult]:
    return build_check_results(
        ensure_source_txt_fn=_ensure_source_txt,
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


def build_check_results(
    *,
    ensure_source_txt_fn,
    load_questions_fn,
    questions_sha256_fn,
    load_source_info_fn,
    check_llm_fn,
    check_embeddings_fn,
    check_frameworks_fn,
    check_kag_neo4j_fn,
    check_docker_fn,
    check_storage_fn,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        source_info = load_source_info_fn(ensure_source_txt_fn())
        results.append(CheckResult("source.txt", "PASS", f"sha256={source_info.sha256} size={source_info.size_bytes}"))
    except Exception as exc:
        results.append(CheckResult("source.txt", "FAIL", str(exc)))

    try:
        questions = load_questions_fn()
        results.append(CheckResult("questions.jsonl", "PASS", f"records={len(questions)} sha256={questions_sha256_fn()}"))
    except Exception as exc:
        results.append(CheckResult("questions.jsonl", "FAIL", str(exc)))

    results.extend(check_llm_fn())
    results.extend(check_embeddings_fn())
    results.extend(check_frameworks_fn())
    results.extend(check_kag_neo4j_fn())
    results.extend(check_docker_fn())
    results.extend(check_storage_fn())
    return results


def format_checks(rows: list[CheckResult]) -> str:
    lines = ["CHECK | STATUS | DETAILS"]
    lines.extend(f"{row.name} | {row.status} | {row.details}" for row in rows)
    return "\n".join(lines)


def _ensure_source_txt() -> Path:
    from research_bench.workflow import ensure_source_txt

    return ensure_source_txt()


def _check_llm() -> list[CheckResult]:
    results = []
    config = load_model_runtime_config()
    base_url = config.base_url.rstrip("/")
    try:
        models = requests.get(f"{base_url}/models", timeout=30, headers=config.auth_headers)
        models.raise_for_status()
        results.append(CheckResult("llm endpoint", "PASS", "reachable"))
        results.append(CheckResult("llm model", "PASS", config.model))
        probe = requests.post(
            f"{base_url}/chat/completions",
            timeout=60,
            headers=config.auth_headers,
            json={
                "model": config.model,
                "messages": [{"role": "user", "content": "Ответь словом ГОТОВО"}],
                "temperature": DEFAULT_TEMPERATURE,
            },
        )
        probe.raise_for_status()
        results.append(CheckResult("llm chat completion", "PASS", probe.json()["choices"][0]["message"]["content"].strip()))
    except Exception as exc:
        results.append(CheckResult("llm endpoint", "FAIL", str(exc)))
    return results


def _check_embeddings() -> list[CheckResult]:
    results = []
    config = load_model_runtime_config()
    base_url = config.embedding_base_url.rstrip("/")
    try:
        probe = requests.post(
            f"{base_url}/embeddings",
            timeout=60,
            headers=config.auth_headers,
            json={"model": config.embedding_model, "input": ["probe"]},
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
            exists = bool(importlib.util.find_spec("graphrag")) or bool(shutil.which("graphrag"))
        elif name == "lightrag":
            exists = bool(importlib.util.find_spec("lightrag"))
        else:
            exists = (ROOT / "frameworks" / "kag").exists()
        results.append(CheckResult(name, "PASS" if exists else "FAIL", "available" if exists else "missing"))
    return results


def _check_kag_neo4j() -> list[CheckResult]:
    try:
        from neo4j import GraphDatabase
    except Exception as exc:
        return [CheckResult("kag neo4j", "FAIL", str(exc))]

    config = get_kag_neo4j_config()
    driver = GraphDatabase.driver(config["uri"], auth=(config["user"], config["password"]))
    try:
        driver.verify_connectivity()
        with driver.session(database=config["database"]) as session:
            value = session.run("RETURN 1 AS value").single()["value"]
    except Exception as exc:
        return [CheckResult("kag neo4j", "FAIL", str(exc))]
    finally:
        driver.close()
    return [CheckResult("kag neo4j", "PASS", f"{config['uri']} database={config['database']} probe={value}")]


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
