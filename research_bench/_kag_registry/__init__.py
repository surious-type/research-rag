from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

from kag.common.tools.algorithm_tool.graph_retriever.kg_fr_retriever import (
    KgFreeRetrieverWithOpenSPGRetriever,
)
from kag.common.tools.algorithm_tool.ner import Ner
from kag.interface import RetrieverABC, RetrieverOutput, ToolABC
from kag.interface.solver.base_model import SPOEntity


def _query_dir() -> Path | None:
    value = os.getenv("KAG_BENCH_QUERY_DIR")
    return Path(value) if value else None


def _question_id() -> str:
    return os.getenv("KAG_BENCH_QUESTION_ID", "unknown")


def _diag_path() -> Path | None:
    query_dir = _query_dir()
    if query_dir is None:
        return None
    path = query_dir / "_bench_diag" / f"{_question_id()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _merge_diag(update: dict[str, Any]) -> None:
    path = _diag_path()
    if path is None:
        return
    payload = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(update)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_error_log(message: str) -> None:
    query_dir = _query_dir()
    if query_dir is None:
        return
    path = query_dir / "kg_fr_error.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _task_repr(task: Any) -> dict[str, Any]:
    arguments = getattr(task, "arguments", None)
    return {
        "task_type": type(task).__name__,
        "task_arguments_type": type(arguments).__name__,
        "task_arguments_repr": repr(arguments),
        "planner_output": str(task),
        "parsed_logical_form_action": repr(arguments.get("logic_form_node")) if isinstance(arguments, dict) else None,
    }


def _normalize_task_arguments(arguments: Any) -> tuple[Any, bool, str | None]:
    if isinstance(arguments, dict):
        return arguments, False, None
    if isinstance(arguments, str):
        try:
            payload = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return arguments, False, f"task.arguments is not a JSON object: {exc}"
        if isinstance(payload, dict):
            return payload, True, None
        return arguments, False, "task.arguments JSON payload is not an object"
    return arguments, False, f"unsupported task.arguments type: {type(arguments).__name__}"


def _normalize_ner_item(item: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(item, dict):
        return item, None
    if isinstance(item, str):
        try:
            payload = json.loads(item)
        except json.JSONDecodeError:
            payload = {"name": item, "category": "Others", "official_name": item}
            return payload, "normalized plain-string ner item as Others"
        if isinstance(payload, dict):
            return payload, "normalized JSON-string ner item"
        return None, "discarded non-object JSON ner item"
    return None, f"discarded unsupported ner item type: {type(item).__name__}"


@ToolABC.register("ner")
class BenchmarkCompatibilityNer(Ner):
    def invoke(self, query, **kwargs) -> list[SPOEntity]:
        res: list[SPOEntity] = []
        ner_list = self._parse_ner_list(query)
        diagnostics: list[str] = []
        for item in ner_list:
            normalized_item, note = _normalize_ner_item(item)
            if note:
                diagnostics.append(note)
            if not normalized_item:
                continue
            entity = str(normalized_item.get("name") or "").strip()
            category = str(normalized_item.get("category") or "").strip()
            official_name = str(normalized_item.get("official_name") or entity).strip()
            if not entity or not official_name:
                continue
            if category.lower() in ["works", "person", "other", "others"]:
                res.append(SPOEntity(entity_name=entity, un_std_entity_type=category or "Others"))
            else:
                res.append(SPOEntity(entity_name=official_name, un_std_entity_type=category))
        if diagnostics:
            _merge_diag({"kg_fr_error": "; ".join(diagnostics)})
        return res


@RetrieverABC.register("kg_fr_open_spg")
class BenchmarkCompatibilityKgFrRetriever(KgFreeRetrieverWithOpenSPGRetriever):
    def invoke(self, task, **kwargs) -> RetrieverOutput:
        task_info = _task_repr(task)
        normalized_arguments, normalized, normalization_error = _normalize_task_arguments(getattr(task, "arguments", None))
        _merge_diag(
            {
                "kg_fr_arguments_type": task_info["task_arguments_type"],
                "kg_fr_arguments_normalized": normalized,
                "kg_fr_error": normalization_error,
                "planner_output": task_info["planner_output"],
                "parsed_logical_form_action": task_info["parsed_logical_form_action"],
            }
        )
        if normalization_error:
            _append_error_log(normalization_error)
            return RetrieverOutput(retriever_method=self.name, err_msg=normalization_error, task=task)
        original_arguments = getattr(task, "arguments", None)
        task.arguments = normalized_arguments
        try:
            output = super().invoke(task, **kwargs)
            err_msg = str(getattr(output, "err_msg", "") or "").strip()
            if err_msg:
                _merge_diag({"kg_fr_error": err_msg})
                _append_error_log(err_msg)
            return output
        except Exception:
            tb = traceback.format_exc()
            _merge_diag({"kg_fr_error": tb})
            _append_error_log(tb)
            raise
        finally:
            task.arguments = original_arguments
