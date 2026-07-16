from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import FrameworkAdapter


ADAPTER_NAMES = ("msgraphrag", "lightrag", "kag")


def get_adapter(framework_name: str) -> "FrameworkAdapter":
    import research_bench.frameworks as legacy_frameworks

    mapping = {
        "msgraphrag": legacy_frameworks.MsGraphRAGAdapter,
        "lightrag": legacy_frameworks.LightRAGAdapter,
        "kag": legacy_frameworks.KAGAdapter,
    }
    return mapping[framework_name]()
