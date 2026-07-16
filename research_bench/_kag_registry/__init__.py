from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from kag.common.tools.algorithm_tool.chunk_retriever.ppr_chunk_retriever import PprChunkRetriever
from kag.common.tools.algorithm_tool.chunk_retriever.rc_retriever import RCRetrieverOnOpenSPG
from kag.common.tools.algorithm_tool.graph_retriever.kg_cs_retriever import KgConstrainRetrieverWithOpenSPGRetriever
from kag.common.tools.algorithm_tool.graph_retriever.kg_fr_retriever import KgFreeRetrieverWithOpenSPGRetriever
from kag.common.tools.algorithm_tool.ner import Ner
from kag.interface import ChunkData, RetrieverABC, RetrieverOutput, ToolABC
from kag.interface.solver.base_model import SPOEntity
from knext.schema.client import CHUNK_TYPE


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


def _sanitize_chunk_properties(properties: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in properties.items():
        if "vector" in str(key).lower():
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[str(key)] = value
        else:
            sanitized[str(key)] = type(value).__name__
    return sanitized


def _chunk_title_from_node(node_dict: dict[str, Any], chunk_id: str) -> str:
    for key in ("name", "title", "document_name"):
        value = str(node_dict.get(key) or "").strip()
        if value:
            return value.replace("_split_0", "")
    return str(chunk_id)


def _record_retriever_result(name: str, output: RetrieverOutput) -> None:
    retriever_results = {}
    path = _diag_path()
    if path and path.exists():
        retriever_results = json.loads(path.read_text(encoding="utf-8")).get("retriever_results", {})
    retriever_results[name] = {
        "status": "error" if str(getattr(output, "err_msg", "") or "").strip() else "ok",
        "chunks_count": len(getattr(output, "chunks", []) or []),
        "graphs_count": len(getattr(output, "graphs", []) or []),
        "error": str(getattr(output, "err_msg", "") or "").strip() or None,
    }
    _merge_diag({"retriever_results": retriever_results})


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


@RetrieverABC.register("benchmark_ppr_chunk_retriever")
class BenchmarkCompatibilityPprChunkRetriever(PprChunkRetriever):
    def get_all_docs_by_id(self, queries: list[str], doc_ids: list, top_k: int):
        requested_doc_ids: list[str] = []
        loaded_doc_ids: list[str] = []
        missing_doc_ids: list[str] = []
        chunk_properties: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        matched_docs: list[ChunkData] = []

        def normalize_doc_request(doc_ref: Any) -> tuple[str | None, float | None]:
            if isinstance(doc_ref, tuple) and doc_ref:
                return str(doc_ref[0]), float(doc_ref[1]) if len(doc_ref) > 1 and isinstance(doc_ref[1], (int, float)) else None
            if isinstance(doc_ref, str):
                return doc_ref, None
            return None, None

        def process_get_doc_id(doc_ref: Any):
            doc_id, doc_score = normalize_doc_request(doc_ref)
            if not doc_id:
                errors.append(f"ignored unsupported doc reference: {repr(doc_ref)}")
                return None
            requested_doc_ids.append(doc_id)
            try:
                node = self.graph_api.get_entity_prop_by_id(
                    label=self.schema_helper.get_label_within_prefix(CHUNK_TYPE),
                    biz_id=doc_id,
                )
                if node is None:
                    missing_doc_ids.append(doc_id)
                    return None
                node_dict = dict(node.items())
                content = str(node_dict.get("content") or "")
                if not content.strip():
                    missing_doc_ids.append(doc_id)
                    chunk_properties[doc_id] = {"property_keys": sorted(node_dict.keys())}
                    return None
                title = _chunk_title_from_node(node_dict, doc_id)
                chunk_properties[doc_id] = {
                    "property_keys": sorted(node_dict.keys()),
                    "title": title,
                    "content_length": len(content),
                }
                return doc_id, ChunkData(
                    content=content.replace("_split_0", ""),
                    title=title,
                    chunk_id=doc_id,
                    score=doc_score if doc_score is not None else 0.0,
                    properties=_sanitize_chunk_properties(node_dict),
                )
            except Exception as exc:
                errors.append(f"{doc_id} get_entity_prop_by_id failed: {exc}")
                return None

        limited_doc_refs = doc_ids[:top_k]
        doc_maps: dict[str, ChunkData] = {}
        with ThreadPoolExecutor(max_workers=20) as executor:
            for result in executor.map(process_get_doc_id, limited_doc_refs):
                if result is None:
                    continue
                doc_maps[result[0]] = result[1]

        for doc_ref in limited_doc_refs:
            doc_id, _ = normalize_doc_request(doc_ref)
            if not doc_id:
                continue
            chunk = doc_maps.get(doc_id)
            if chunk is None:
                if doc_id not in missing_doc_ids:
                    missing_doc_ids.append(doc_id)
                continue
            matched_docs.append(chunk)
            loaded_doc_ids.append(doc_id)

        _merge_diag(
            {
                "ppr_requested_doc_ids": requested_doc_ids,
                "ppr_loaded_doc_ids": loaded_doc_ids,
                "ppr_missing_doc_ids": sorted(set(missing_doc_ids)),
                "ppr_chunk_properties": chunk_properties,
                "ppr_errors": errors,
            }
        )
        return matched_docs


@RetrieverABC.register("kg_cs_open_spg")
class BenchmarkCompatibilityKgCsRetriever(KgConstrainRetrieverWithOpenSPGRetriever):
    def invoke(self, task, **kwargs) -> RetrieverOutput:
        output = super().invoke(task, **kwargs)
        _record_retriever_result("kg_cs_open_spg", output)
        return output


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
            output = RetrieverOutput(retriever_method=self.name, err_msg=normalization_error, task=task)
            _record_retriever_result("kg_fr_open_spg", output)
            return output
        original_arguments = getattr(task, "arguments", None)
        task.arguments = normalized_arguments
        try:
            output = super().invoke(task, **kwargs)
            err_msg = str(getattr(output, "err_msg", "") or "").strip()
            if err_msg:
                _merge_diag({"kg_fr_error": err_msg})
                _append_error_log(err_msg)
            _record_retriever_result("kg_fr_open_spg", output)
            return output
        except Exception:
            tb = traceback.format_exc()
            _merge_diag({"kg_fr_error": tb})
            _append_error_log(tb)
            raise
        finally:
            task.arguments = original_arguments


@RetrieverABC.register("rc_open_spg")
class BenchmarkCompatibilityRcRetriever(RCRetrieverOnOpenSPG):
    def invoke(self, task, **kwargs) -> RetrieverOutput:
        output = super().invoke(task, **kwargs)
        _record_retriever_result("rc_open_spg", output)
        return output
