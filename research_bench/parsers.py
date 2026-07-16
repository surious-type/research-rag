from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import normalize_graph_metrics
from .models import GraphMetrics
from .utils import load_json, safe_float


def parse_msgraphrag_outputs(output_dir: Path) -> GraphMetrics:
    entities = pd.read_parquet(output_dir / "entities.parquet").to_dict(orient="records")
    relationships = pd.read_parquet(output_dir / "relationships.parquet").to_dict(orient="records")
    text_units = pd.read_parquet(output_dir / "text_units.parquet").to_dict(orient="records")
    documents = pd.read_parquet(output_dir / "documents.parquet").to_dict(orient="records")
    communities_path = output_dir / "communities.parquet"
    communities = pd.read_parquet(communities_path).to_dict(orient="records") if communities_path.exists() else []
    return normalize_graph_metrics(
        nodes_total=len(entities) + len(documents) + len(text_units) + len(communities),
        edges_total=len(relationships),
        documents_count=len(documents),
        chunks_count=len(text_units),
        communities_count=len(communities),
        entity_rows=entities,
        relationship_rows=relationships,
    )


def parse_lightrag_outputs(work_dir: Path) -> GraphMetrics:
    graphml_path = work_dir / "graph_chunk_entity_relation.graphml"
    tree = ET.parse(graphml_path)
    root = tree.getroot()
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    nodes = root.findall(".//g:node", ns)
    edges = root.findall(".//g:edge", ns)
    entity_rows: list[dict[str, Any]] = []
    relationship_rows: list[dict[str, Any]] = []
    for node in nodes:
        labels = [data.text for data in node.findall("g:data", ns) if data.text]
        label = next((item for item in labels if item in {"Chunk", "Document", "Community"}), "Entity")
        entity_rows.append({"type": label})
    for edge in edges:
        relationship_rows.append({"type": edge.attrib.get("label") or edge.attrib.get("id") or "related_to"})

    full_docs = load_json(work_dir / "kv_store_full_docs.json")
    text_chunks = load_json(work_dir / "kv_store_text_chunks.json")
    return normalize_graph_metrics(
        nodes_total=len(nodes),
        edges_total=len(edges),
        documents_count=len(full_docs),
        chunks_count=len(text_chunks),
        communities_count=None,
        entity_rows=entity_rows,
        relationship_rows=relationship_rows,
    )


def parse_neo4j_summary(summary: dict[str, Any]) -> GraphMetrics:
    entity_rows = summary.get("entity_rows", [])
    relationship_rows = summary.get("relationship_rows", [])
    return normalize_graph_metrics(
        nodes_total=summary.get("nodes_total"),
        edges_total=summary.get("edges_total"),
        documents_count=summary.get("documents_count"),
        chunks_count=summary.get("chunks_count"),
        communities_count=summary.get("communities_count"),
        entity_rows=entity_rows,
        relationship_rows=relationship_rows,
    )


def load_answers(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_ragas_scores(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
                payload[key] = safe_float(payload.get(key))
            rows.append(payload)
    return rows

