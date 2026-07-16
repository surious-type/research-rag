from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


FRAMEWORKS = ("msgraphrag", "lightrag", "kag")
SMOKE_FRAMEWORKS = ("msgraphrag", "lightrag", "kag", "all")
TECHNICAL_NODE_LABELS = {"Chunk", "Document", "TextUnit", "Community"}
TECHNICAL_REL_TYPES = {
    "HAS_CHUNK",
    "HAS_ENTITY",
    "HAS_RELATIONSHIP",
    "HAS_REPORT",
    "BELONGS_TO",
    "PART_OF",
    "IN_COMMUNITY",
    "MENTIONS",
    "SOURCE",
    "DOCUMENT_CHUNK",
}


@dataclass
class CheckResult:
    name: str
    status: str
    details: str


@dataclass
class SourceInfo:
    path: str
    sha256: str
    size_bytes: int
    characters_count: int
    words_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QuestionRecord:
    id: str
    question: str
    reference_answer: str
    payload: dict[str, Any]


@dataclass
class GraphMetrics:
    nodes_total: int | None = None
    edges_total: int | None = None
    entities_count: int | None = None
    relationships_count: int | None = None
    documents_count: int | None = None
    chunks_count: int | None = None
    communities_count: int | None = None
    isolated_entities_count: int | None = None
    connected_entities_count: int | None = None
    connected_entities_ratio: float | None = None
    entity_types: dict[str, int] = field(default_factory=dict)
    relationship_types: dict[str, int] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuildMetrics:
    build_time_seconds: float | None
    documents_count: int | None
    chunks_count: int | None
    index_size_bytes: int | None
    build_status: str
    build_error: str | None
    peak_ram_mb: float | None = None
    peak_gpu_memory_mb: float | None = None
    mean_cpu_percent: float | None = None
    llm_calls: int | None = None
    embedding_calls: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    unavailable_reasons: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContextItem:
    rank: int
    text: str
    source: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryAnswer:
    run_id: str
    framework: str
    question_id: str
    question_type: str | None
    question: str
    reference_answer: str
    answer: str
    contexts: list[ContextItem]
    latency_seconds: float | None
    retrieval_time_seconds: float | None
    generation_time_seconds: float | None
    status: str
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contexts"] = [context.to_dict() for context in self.contexts]
        return payload

