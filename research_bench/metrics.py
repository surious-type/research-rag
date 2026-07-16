from __future__ import annotations

from collections import Counter
from typing import Any

from .models import GraphMetrics, TECHNICAL_NODE_LABELS, TECHNICAL_REL_TYPES


def normalize_graph_metrics(
    *,
    nodes_total: int | None,
    edges_total: int | None,
    documents_count: int | None,
    input_documents_count: int | None = None,
    backend_document_nodes_count: int | None = None,
    chunks_count: int | None,
    communities_count: int | None,
    entity_rows: list[dict[str, Any]],
    relationship_rows: list[dict[str, Any]],
) -> GraphMetrics:
    entity_types: Counter[str] = Counter()
    relationship_types: Counter[str] = Counter()
    connected_entities = 0
    isolated_entities = 0

    semantic_entities = []
    for row in entity_rows:
        label = str(row.get("type") or row.get("label") or row.get("entity_type") or "").strip()
        if label in TECHNICAL_NODE_LABELS:
            continue
        semantic_entities.append(row)
        entity_types[label or "unknown"] += 1
        degree = int(row.get("degree") or row.get("connected_edges") or row.get("connections") or 0)
        if degree > 0:
            connected_entities += 1
        else:
            isolated_entities += 1

    semantic_relationships = []
    for row in relationship_rows:
        rel_type = str(row.get("type") or row.get("label") or row.get("relationship_type") or "").strip()
        source_type = str(row.get("source_type") or row.get("source_label") or "").strip()
        target_type = str(row.get("target_type") or row.get("target_label") or "").strip()
        if rel_type in TECHNICAL_REL_TYPES:
            continue
        if source_type in TECHNICAL_NODE_LABELS or target_type in TECHNICAL_NODE_LABELS:
            continue
        semantic_relationships.append(row)
        relationship_types[rel_type or "unknown"] += 1

    entities_count = len(semantic_entities)
    return GraphMetrics(
        nodes_total=nodes_total,
        edges_total=edges_total,
        entities_count=entities_count,
        relationships_count=len(semantic_relationships),
        documents_count=documents_count,
        input_documents_count=input_documents_count if input_documents_count is not None else documents_count,
        backend_document_nodes_count=backend_document_nodes_count,
        chunks_count=chunks_count,
        communities_count=communities_count,
        isolated_entities_count=isolated_entities,
        connected_entities_count=connected_entities,
        connected_entities_ratio=(connected_entities / entities_count) if entities_count else None,
        entity_types=dict(entity_types),
        relationship_types=dict(relationship_types),
        notes={
            "excluded_numeric_relationship_types": "Numeric relationship labels such as `128` are treated as opaque OpenSPG/internal identifiers unless mapped explicitly."
        }
        if any(str(row.get("type") or "").strip().isdigit() for row in relationship_rows)
        else {},
    )
