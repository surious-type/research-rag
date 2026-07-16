from __future__ import annotations

from .adapters import kag as kag_mod
from .adapters.kag import (
    KAG_LOCAL_HOST_ADDR,
    _ensure_kag_imports,
    _classify_kag_query_answer,
    _extract_kag_label_type,
    _extract_kag_retriever_errors,
    _invoke_kag_query_with_trace,
    _normalize_kag_context,
    _normalize_kag_namespace,
    _prepare_kag_query_environment,
    _resolve_kag_neo4j_database,
    _resolve_kag_query_diagnostics,
    _schema_diff,
    _summarize_vector_probe_status,
    _validate_required_schema,
    build_kag_server_project_config,
    get_kag_neo4j_config,
    summarize_kag_namespace_graph,
)
from .adapters.lightrag import LightRAGAdapter
from .adapters.msgraphrag import MsGraphRAGAdapter
from .models import ContextItem
from .adapters.registry import get_adapter
from .shared.subprocess import run_command


class KAGAdapter(kag_mod.KAGAdapter):
    def _sync_framework_helper_overrides(self) -> None:
        kag_mod._ensure_kag_imports = _ensure_kag_imports
        kag_mod._normalize_kag_namespace = _normalize_kag_namespace
        kag_mod.run_command = run_command

    def build(self, source_path, run_dir):
        self._sync_framework_helper_overrides()
        return super().build(source_path, run_dir)

    def query(self, run_id, run_dir, questions):
        self._sync_framework_helper_overrides()
        return super().query(run_id, run_dir, questions)

    def _sync_project_config(self, project_id, namespace, config_text):
        self._sync_framework_helper_overrides()
        return super()._sync_project_config(project_id, namespace, config_text)


__all__ = [
    "ContextItem",
    "KAGAdapter",
    "KAG_LOCAL_HOST_ADDR",
    "LightRAGAdapter",
    "MsGraphRAGAdapter",
    "_ensure_kag_imports",
    "_classify_kag_query_answer",
    "_extract_kag_label_type",
    "_extract_kag_retriever_errors",
    "_invoke_kag_query_with_trace",
    "_normalize_kag_context",
    "_normalize_kag_namespace",
    "_prepare_kag_query_environment",
    "_resolve_kag_neo4j_database",
    "_resolve_kag_query_diagnostics",
    "_schema_diff",
    "_summarize_vector_probe_status",
    "_validate_required_schema",
    "build_kag_server_project_config",
    "get_adapter",
    "get_kag_neo4j_config",
    "run_command",
    "summarize_kag_namespace_graph",
]
