from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kag.builder.component.extractor.knowledge_unit_extractor import KnowledgeUnitSchemaFreeExtractor
from kag.builder.component.extractor.schema_free_extractor import SchemaFreeExtractor
from kag.common.llm.openai_client import OpenAIClient
from kag.common.tools.algorithm_tool.chunk_retriever.ppr_chunk_retriever import PprChunkRetriever
from kag.common.tools.algorithm_tool.chunk_retriever.rc_retriever import RCRetrieverOnOpenSPG
from kag.common.tools.algorithm_tool.graph_retriever.kg_cs_retriever import KgConstrainRetrieverWithOpenSPGRetriever
from kag.common.tools.algorithm_tool.graph_retriever.kg_fr_retriever import KgFreeRetrieverWithOpenSPGRetriever
from kag.common.tools.algorithm_tool.ner import Ner
from kag.interface import ChunkData, ExtractorABC, RetrieverABC, RetrieverOutput, ToolABC
from kag.interface.solver.reporter_abc import do_report
from kag.interface.solver.base_model import SPOEntity
from kag.interface.common.llm_client import LLMClient
from knext.schema.client import CHUNK_TYPE
from openai import NOT_GIVEN


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


def _is_official_openai_base_url(base_url: str) -> bool:
    normalized = str(base_url or "").strip()
    if not normalized:
        return False
    parsed = urlparse(normalized)
    host = (parsed.netloc or parsed.path).lower()
    return host in {"api.openai.com", "api.openai.com:443"}


def _sanitize_openai_extra_body(base_url: str, extra_body: Any) -> dict[str, Any]:
    if not isinstance(extra_body, dict):
        return {}
    if not _is_official_openai_base_url(base_url):
        return dict(extra_body)
    sanitized = dict(extra_body)
    sanitized.pop("chat_template_kwargs", None)
    return sanitized


def _sanitize_openai_request_kwargs(base_url: str, request_kwargs: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(request_kwargs)
    if not _is_official_openai_base_url(base_url):
        return sanitized
    if "max_tokens" in sanitized:
        value = sanitized.pop("max_tokens")
        if value is not NOT_GIVEN:
            sanitized["max_completion_tokens"] = value
    if "temperature" in sanitized and sanitized["temperature"] not in (NOT_GIVEN, 0, 0.0):
        sanitized["temperature"] = 0
    return sanitized


def _normalize_builder_ner_result(ner_result: Any) -> list[dict[str, Any]]:
    if ner_result is None:
        return []
    if isinstance(ner_result, list):
        output: list[dict[str, Any]] = []
        for item in ner_result:
            normalized_item, _ = _normalize_ner_item(item)
            if normalized_item:
                output.append(normalized_item)
        return output
    normalized_item, _ = _normalize_ner_item(ner_result)
    return [normalized_item] if normalized_item else []


def _normalize_core_entities_map(core_entities_raw: Any) -> dict[str, str]:
    if isinstance(core_entities_raw, dict):
        normalized: dict[str, str] = {}
        for entity_name, entity_type in core_entities_raw.items():
            name = str(entity_name or "").strip()
            if not name:
                continue
            normalized[name] = str(entity_type or "Others").strip() or "Others"
        return normalized
    if isinstance(core_entities_raw, str):
        normalized = {}
        for item in core_entities_raw.split(","):
            name = str(item or "").strip()
            if not name:
                continue
            normalized[name] = "Others"
        return normalized
    return {}


def _chunk_title_from_node(node_dict: dict[str, Any], chunk_id: str) -> str:
    for key in ("name", "title", "document_name"):
        value = str(node_dict.get(key) or "").strip()
        if value:
            return value.replace("_split_0", "")
    return str(chunk_id)


def _record_retriever_result(name: str, output: RetrieverOutput) -> None:
    graph_spo: list[str] = []
    for graph in getattr(output, "graphs", []) or []:
        get_all_spo = getattr(graph, "get_all_spo", None)
        if callable(get_all_spo):
            try:
                graph_spo.extend(str(item) for item in (get_all_spo() or []))
            except Exception:
                continue
    retriever_results = {}
    path = _diag_path()
    if path and path.exists():
        retriever_results = json.loads(path.read_text(encoding="utf-8")).get("retriever_results", {})
    retriever_results[name] = {
        "status": "error" if str(getattr(output, "err_msg", "") or "").strip() else "ok",
        "chunks_count": len(getattr(output, "chunks", []) or []),
        "graphs_count": len(getattr(output, "graphs", []) or []),
        "graph_spo_count": len(set(graph_spo)),
        "graph_spo_sample": list(dict.fromkeys(graph_spo))[:5],
        "error": str(getattr(output, "err_msg", "") or "").strip() or None,
    }
    _merge_diag({"retriever_results": retriever_results})


@LLMClient.register("maas")
@LLMClient.register("openai")
@LLMClient.register("vllm")
class BenchmarkCompatibilityOpenAIClient(OpenAIClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_body = _sanitize_openai_extra_body(self.base_url, self.extra_body)

    def _request_kwargs(self, *, messages, tools):
        return _sanitize_openai_request_kwargs(
            self.base_url,
            {
                "model": self.model,
                "messages": messages,
                "stream": self.stream,
                "temperature": self.temperature,
                "timeout": self.timeout,
                "tools": tools,
                "max_tokens": self.max_tokens if self.max_tokens > 0 else NOT_GIVEN,
                "stop": self.stop,
                "seed": self.seed,
                "top_p": self.top_p,
                "extra_body": self.extra_body,
            },
        )

    def __call__(self, prompt: str = "", image_url: str = None, **kwargs):
        tools = kwargs.get("tools", NOT_GIVEN)
        messages = kwargs.get("messages", None)
        token_meter = LLMClient.get_token_meter()

        if messages is None:
            if image_url:
                messages = [
                    {"role": "system", "content": "you are a helpful assistant"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ]
            else:
                messages = [
                    {"role": "system", "content": "you are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ]
        response = self.client.chat.completions.create(**self._request_kwargs(messages=messages, tools=tools))
        usages = []
        if not self.stream:
            rsp = response.choices[0].message.content
            tool_calls = response.choices[0].message.tool_calls
            usages.append(response.usage)
        else:
            rsp = ""
            tool_calls = None
            for chunk in response:
                if not chunk.choices:
                    continue
                delta_content = getattr(chunk.choices[0].delta, "content", None)
                if delta_content is not None:
                    rsp += delta_content
                    do_report(rsp, "RUNNING", **kwargs)
                usages.append(chunk.usage)

        if token_meter and len(usages) > 0 and usages[-1]:
            try:
                usage = usages[-1]
                token_meter.update(
                    usage.completion_tokens,
                    usage.prompt_tokens,
                    usage.total_tokens,
                )
            except Exception:
                pass

        do_report(rsp, "FINISH", **kwargs)
        if tools and tool_calls:
            return response.choices[0].message
        return rsp

    async def acall(self, prompt: str = "", image_url: str = None, **kwargs):
        tools = kwargs.get("tools", NOT_GIVEN)
        messages = kwargs.get("messages", None)
        token_meter = LLMClient.get_token_meter()
        if messages is None:
            if image_url:
                messages = [
                    {"role": "system", "content": "you are a helpful assistant"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ]
            else:
                messages = [
                    {"role": "system", "content": "you are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ]
        response = await self.aclient.chat.completions.create(**self._request_kwargs(messages=messages, tools=tools))
        usages = []
        if not self.stream:
            rsp = response.choices[0].message.content
            tool_calls = response.choices[0].message.tool_calls
            usages.append(response.usage)
        else:
            rsp = ""
            tool_calls = None
            async for chunk in response:
                if not chunk.choices:
                    continue
                delta_content = getattr(chunk.choices[0].delta, "content", None)
                if delta_content is not None:
                    rsp += delta_content
                do_report(rsp, "RUNNING", **kwargs)
                usages.append(chunk.usage)
        if token_meter and len(usages) > 0 and usages[-1]:
            try:
                usage = usages[-1]
                token_meter.update(
                    usage.completion_tokens,
                    usage.prompt_tokens,
                    usage.total_tokens,
                )
            except Exception:
                pass

        do_report(rsp, "FINISH", **kwargs)
        if tools and tool_calls:
            return response.choices[0].message
        return rsp


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


@ExtractorABC.register("schema_free")
@ExtractorABC.register("schema_free_extractor")
class BenchmarkCompatibilitySchemaFreeExtractor(SchemaFreeExtractor):
    def _named_entity_recognition_process(self, passage, ner_result):
        if self.external_graph:
            extra_ner_result = self.external_graph.ner(passage)
        else:
            extra_ner_result = []
        output = []
        dedup = set()
        for item in extra_ner_result:
            name = item.name
            label = item.label
            spg_type = self.schema.get(label)
            if spg_type is None:
                label = "Others"
                item.label = label
            description = item.properties.get("desc", "")
            semantic_type = item.properties.get("semanticType", label)
            if name not in dedup:
                dedup.add(name)
                output.append(
                    {
                        "name": name,
                        "type": semantic_type,
                        "category": label,
                        "description": description,
                    }
                )
        for item in _normalize_builder_ner_result(ner_result):
            name = item.get("name", None)
            if name and name not in dedup:
                dedup.add(name)
                output.append(item)
        return output


@ExtractorABC.register("knowledge_unit_extractor")
class BenchmarkCompatibilityKnowledgeUnitExtractor(KnowledgeUnitSchemaFreeExtractor):
    def assemble_knowledge_unit(self, sub_graph, source_entities, input_knowledge_units, triples):
        knowledge_unit_nodes = []
        knowledge_units = dict(input_knowledge_units)

        def triple_to_knowledge_unit(triple):
            ret = {}
            name = " ".join(triple)
            ret["content"] = name
            ret["knowledgetype"] = "triple"
            ret["core_entities"] = ",".join(triple)
            return name, ret

        for tri in triples:
            knowledge_unit_name, knowledge_unit_value = triple_to_knowledge_unit(tri)
            if knowledge_unit_name not in knowledge_units:
                knowledge_units[knowledge_unit_name] = knowledge_unit_value

        for knowledge_name, knowledge_value in knowledge_units.items():
            if knowledge_value["knowledgetype"] == "triple":
                knowledge_id = knowledge_name
            else:
                from kag.common.utils import generate_hash_id

                knowledge_id = generate_hash_id(f"{knowledge_name}_{knowledge_value['content'].strip()[:100]}")
            self.assemble_sub_graph_with_spg_properties(
                sub_graph,
                knowledge_id,
                knowledge_name,
                "KnowledgeUnit",
                knowledge_value,
            )
            sub_graph.add_node(
                knowledge_id,
                knowledge_name,
                "KnowledgeUnit",
                knowledge_value,
            )
            knowledge_unit_nodes.append({"name": knowledge_id, "category": "KnowledgeUnit"})
            core_entities = _normalize_core_entities_map(knowledge_value.get("core_entities", ""))

            for core_entity, ent_type in core_entities.items():
                if core_entity == "":
                    continue
                found_in_source_entity = None
                for source_entity in source_entities:
                    if core_entity == source_entity.get("name", ""):
                        found_in_source_entity = source_entity
                        break
                ent_type = self.get_stand_schema(ent_type)
                if found_in_source_entity is None:
                    found_in_source_entity = {"name": core_entity, "category": ent_type}
                    sub_graph.add_node(core_entity, core_entity, ent_type, {})
                sub_graph.add_edge(
                    found_in_source_entity.get("name"),
                    found_in_source_entity.get("category"),
                    "source",
                    knowledge_id,
                    "KnowledgeUnit",
                )
        return knowledge_unit_nodes


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


# The benchmark contains document-grounded questions only.
# Disable KAG's chatbot self-cognition route, which is not part of RAG
# evaluation and is unreliable with reasoning-model responses.
from kag.interface import PromptABC as _BenchmarkPromptABC
from kag.solver.prompt.self_cognition import (
    SelfCognitionPrompt as _UpstreamSelfCognitionPrompt,
)


@_BenchmarkPromptABC.register("default_self_cognition")
class BenchmarkNoSelfCognitionPrompt(_UpstreamSelfCognitionPrompt):
    def parse_response(self, response: str, **kwargs):
        _merge_diag(
            {
                "self_cognition_raw_response": str(response),
                "self_cognition_result": False,
                "self_cognition_policy": (
                    "disabled_for_document_qa_benchmark"
                ),
            }
        )
        return False
