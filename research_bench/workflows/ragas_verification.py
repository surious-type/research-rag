from __future__ import annotations

import json
from pathlib import Path
from types import MethodType
from typing import Any

from research_bench.data import load_smoke_questions
from research_bench.ragas_eval import (
    prepare_ragas_rows,
    save_ragas_outputs,
    save_ragas_placeholder,
)
from research_bench.runtime_config import DEFAULT_TEMPERATURE, load_model_runtime_config
from research_bench.shared.io import load_json
from research_bench.shared.text import safe_float

RAGAS_REQUEST_TIMEOUT_SECONDS = None
RAGAS_EVALUATION_TIMEOUT_SECONDS = None


def _patch_ragas_temperature_handling(wrapper, *, default_temperature: float) -> Any:
    def generate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks=None,
    ):
        llm = self.langchain_llm
        llm.temperature = default_temperature  # type: ignore[attr-defined]
        if self.multiple_completion_supported:
            return llm.generate_prompt(
                prompts=[prompt],
                n=n,
                stop=stop,
                callbacks=callbacks,
            )
        result = llm.generate_prompt(
            prompts=[prompt] * n,
            stop=stop,
            callbacks=callbacks,
        )
        result.generations = [[generation[0] for generation in result.generations]]
        return result

    async def agenerate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks=None,
    ):
        llm = self.langchain_llm
        llm.temperature = default_temperature  # type: ignore[attr-defined]
        if hasattr(llm, "n"):
            llm.n = n  # type: ignore[attr-defined]
            return await llm.agenerate_prompt(
                prompts=[prompt],
                stop=stop,
                callbacks=callbacks,
            )
        result = await llm.agenerate_prompt(
            prompts=[prompt] * n,
            stop=stop,
            callbacks=callbacks,
        )
        result.generations = [[generation[0] for generation in result.generations]]
        return result

    wrapper.generate_text = MethodType(generate_text, wrapper)
    wrapper.agenerate_text = MethodType(agenerate_text, wrapper)
    return wrapper


def run_ragas_stage(
    run_dir: Path, answers: list[dict[str, Any]], limit: int | None = None
) -> None:
    _run_ragas_stage(
        run_dir,
        answers,
        limit=limit,
        prepare_ragas_rows_fn=prepare_ragas_rows,
        save_ragas_outputs_fn=save_ragas_outputs,
        save_ragas_placeholder_fn=save_ragas_placeholder,
        safe_float_fn=safe_float,
        evaluate_prepared_row_fn=_evaluate_prepared_ragas_row,
    )


def _runtime_payload_from_config(config: Any) -> dict[str, Any]:
    return {
        "base_url": config.base_url,
        "embedding_base_url": config.embedding_base_url,
        "api_key": config.api_key,
        "model": config.model,
        "embedding_model": config.embedding_model,
    }


def _evaluate_prepared_ragas_row_direct(
    prepared_row: dict[str, Any], *, runtime_payload: dict[str, Any]
) -> dict[str, Any]:
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            base_url=runtime_payload["base_url"],
            api_key=runtime_payload["api_key"],
            model=runtime_payload["model"],
            temperature=DEFAULT_TEMPERATURE,
            timeout=RAGAS_REQUEST_TIMEOUT_SECONDS,
        )
    )
    llm = _patch_ragas_temperature_handling(
        llm, default_temperature=DEFAULT_TEMPERATURE
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            base_url=runtime_payload["embedding_base_url"],
            api_key=runtime_payload["api_key"],
            model=runtime_payload["embedding_model"],
            timeout=RAGAS_REQUEST_TIMEOUT_SECONDS,
        )
    )

    run_config = RunConfig(
        timeout=RAGAS_EVALUATION_TIMEOUT_SECONDS,
        max_retries=2,
        max_wait=10,
        max_workers=4,
        log_tenacity=True,
        seed=42,
    )
    result = evaluate(
        Dataset.from_list([prepared_row]),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        batch_size=1,
        show_progress=True,
        run_config=run_config,
        raise_exceptions=False,
    )
    result_rows = result.to_pandas().to_dict(orient="records")
    if not result_rows:
        return {"error": "ragas returned no rows"}
    return dict(result_rows[0])


def _evaluate_prepared_ragas_row(
    prepared_row: dict[str, Any],
    *,
    runtime_payload: dict[str, Any],
) -> dict[str, Any]:
    try:
        return _evaluate_prepared_ragas_row_direct(
            prepared_row,
            runtime_payload=runtime_payload,
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _run_ragas_stage(
    run_dir: Path,
    answers: list[dict[str, Any]],
    *,
    limit: int | None,
    prepare_ragas_rows_fn,
    save_ragas_outputs_fn,
    save_ragas_placeholder_fn,
    safe_float_fn,
    evaluate_prepared_row_fn,
) -> None:
    ragas_dir = run_dir / "ragas"
    if not answers:
        save_ragas_placeholder_fn(ragas_dir, answers, "no answers available")
        return

    eligible_answers = [
        row for row in answers if row.get("status") == "success" and row.get("contexts")
    ]
    answers_to_score = (
        eligible_answers[:limit] if limit is not None else eligible_answers
    )
    prepared = prepare_ragas_rows_fn(answers_to_score)
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
        save_ragas_outputs_fn(
            ragas_dir, list(score_rows_by_question.values()), answer_rows=answers
        )
        return
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except Exception as exc:
        save_ragas_placeholder_fn(ragas_dir, answers, str(exc))
        return

    config = load_model_runtime_config()
    runtime_payload = _runtime_payload_from_config(config)
    prepared_queue = list(prepared)
    prepared_by_question = {
        row["question_id"]: row for row in prepared if row.get("question_id")
    }

    def save_current_scores() -> None:
        rows = [
            score_rows_by_question[answer_row["question_id"]] for answer_row in answers
        ]
        save_ragas_outputs_fn(ragas_dir, rows, answer_rows=answers)

    for answer_row in answers_to_score:
        question_id = answer_row["question_id"]
        prepared_row = prepared_by_question.get(question_id)
        if prepared_row is None and prepared_queue:
            prepared_row = prepared_queue.pop(0)
        if prepared_row is None:
            score_rows_by_question[question_id]["error"] = "missing prepared ragas row"
            save_current_scores()
            continue
        try:
            ragas_row = evaluate_prepared_row_fn(
                prepared_row,
                runtime_payload=runtime_payload,
            )
            row_error = str(ragas_row.get("error") or "").strip() or None
            if row_error:
                score_rows_by_question[question_id]["error"] = row_error
            else:
                faithfulness_value = safe_float_fn(ragas_row.get("faithfulness"))
                answer_relevancy_value = safe_float_fn(
                    ragas_row.get("answer_relevancy")
                )
                context_precision_value = safe_float_fn(
                    ragas_row.get("context_precision")
                )
                context_recall_value = safe_float_fn(ragas_row.get("context_recall"))
                error = None
                if all(
                    value is None
                    for value in (
                        faithfulness_value,
                        answer_relevancy_value,
                        context_precision_value,
                        context_recall_value,
                    )
                ):
                    error = "ragas returned no metric values (likely timeout or evaluator failure)"
                score_rows_by_question[question_id] = {
                    "question_id": question_id,
                    "faithfulness": faithfulness_value,
                    "answer_relevancy": answer_relevancy_value,
                    "context_precision": context_precision_value,
                    "context_recall": context_recall_value,
                    "error": error,
                }
        except Exception as exc:
            score_rows_by_question[question_id]["error"] = (
                f"{type(exc).__name__}: {exc}"
            )
        save_current_scores()


def verify_run(run_id: str, smoke: bool | None = None) -> dict[str, Any]:
    run_dir, detected_smoke = _find_run_dir(run_id, smoke=smoke)
    return build_verification_result(
        run_dir,
        detected_smoke=detected_smoke,
        load_smoke_questions_fn=load_smoke_questions,
    )


def build_verification_result(
    run_dir: Path, *, detected_smoke: bool, load_smoke_questions_fn
) -> dict[str, Any]:
    manifest = load_json(run_dir / "manifest.json")
    build = load_json(run_dir / "build" / "metrics.json")
    graph = load_json(run_dir / "build" / "graph_metrics.json")
    answers = [
        json.loads(line)
        for line in (run_dir / "query" / "answers.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    ragas_summary = load_json(run_dir / "ragas" / "summary.json")
    diagnostics_path = run_dir / "query" / "diagnostics.json"
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
    expected_questions = len(load_smoke_questions_fn()) if detected_smoke else 100
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
        vector_probe_status = str(
            diagnostics.get("vector_search_probe_status") or ""
        ).upper()
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
        if metric_name in {
            "successful_retrieval_questions",
            "degraded_questions",
            "failed_questions",
            "retrieval_failure_rate",
        }:
            continue
        for key in ("mean", "median"):
            value = values.get(key)
            if value is not None and not 0 <= value <= 1:
                status = "FAIL"
                issues.append(f"invalid ragas range for {metric_name}.{key}")
        if values.get("valid_count", 0) + values.get("failed_count", 0) != len(answers):
            status = "FAIL"
            issues.append(f"inconsistent counts for {metric_name}")
    ragas_metric_names = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )
    if detected_smoke and not any(
        ragas_summary.get(metric, {}).get("valid_count", 0) > 0
        for metric in ragas_metric_names
    ):
        status = "FAIL"
        issues.append("smoke run has no valid ragas scores")
    if not graph:
        status = "FAIL"
        issues.append("missing graph metrics")

    if status == "PASS" and issues:
        status = "WARN"
    return {"status": status, "issues": issues, "manifest": manifest}


def _find_run_dir(run_id: str, smoke: bool | None = None) -> tuple[Path, bool]:
    from research_bench.workflow import find_run_dir

    return find_run_dir(run_id, smoke=smoke)
