from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from research_bench.data import ROOT

DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_OPENAI_EMBEDDING_BASE_URL = "http://127.0.0.1:8010/v1"
DEFAULT_OPENAI_API_KEY = "local"
DEFAULT_OPENAI_MODEL = "/models/Qwen3.5-35B-A3B-Q4_K_M.gguf"
DEFAULT_OPENAI_EMBEDDING_MODEL = "multilingual-e5-large"
DEFAULT_EMBEDDING_DIMENSION = 1024
DEFAULT_TEMPERATURE = 0.0
KNOWN_EMBEDDING_DIMENSIONS = {
    "multilingual-e5-large": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


@dataclass(frozen=True)
class ModelRuntimeConfig:
    base_url: str
    embedding_base_url: str
    api_key: str
    model: str
    embedding_model: str
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION

    @property
    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


def load_model_runtime_config() -> ModelRuntimeConfig:
    _load_dotenv(ROOT / ".env")
    configured_base_url = os.getenv("OPENAI_BASE_URL")
    configured_embedding_base_url = os.getenv("OPENAI_EMBEDDING_BASE_URL")
    base_url = configured_base_url or DEFAULT_OPENAI_BASE_URL
    embedding_base_url = (
        configured_embedding_base_url
        or (
            DEFAULT_OPENAI_EMBEDDING_BASE_URL
            if configured_base_url in {None, "", DEFAULT_OPENAI_BASE_URL}
            else configured_base_url
        )
    )
    return ModelRuntimeConfig(
        base_url=base_url,
        embedding_base_url=embedding_base_url,
        api_key=os.getenv("OPENAI_API_KEY", DEFAULT_OPENAI_API_KEY),
        model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL),
    )


def infer_embedding_dimension(config: ModelRuntimeConfig) -> int:
    return KNOWN_EMBEDDING_DIMENSIONS.get(config.embedding_model, config.embedding_dimension)


@lru_cache(maxsize=16)
def probe_embedding_dimension(base_url: str, api_key: str, embedding_model: str) -> int:
    import requests

    response = requests.post(
        f"{base_url.rstrip('/')}/embeddings",
        timeout=60,
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        json={"model": embedding_model, "input": ["probe"]},
    )
    response.raise_for_status()
    return len(response.json()["data"][0]["embedding"])


def resolve_embedding_dimension(config: ModelRuntimeConfig) -> int:
    try:
        return probe_embedding_dimension(config.embedding_base_url, config.api_key, config.embedding_model)
    except Exception:
        return infer_embedding_dimension(config)


__all__ = [
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_OPENAI_API_KEY",
    "DEFAULT_OPENAI_BASE_URL",
    "DEFAULT_OPENAI_EMBEDDING_BASE_URL",
    "DEFAULT_OPENAI_EMBEDDING_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_TEMPERATURE",
    "KNOWN_EMBEDDING_DIMENSIONS",
    "ModelRuntimeConfig",
    "infer_embedding_dimension",
    "load_model_runtime_config",
    "probe_embedding_dimension",
    "resolve_embedding_dimension",
]
