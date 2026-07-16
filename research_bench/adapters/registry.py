from __future__ import annotations

from typing import TYPE_CHECKING

from .kag import KAGAdapter
from .lightrag import LightRAGAdapter
from .msgraphrag import MsGraphRAGAdapter

if TYPE_CHECKING:
    from .base import FrameworkAdapter


ADAPTER_NAMES = ("msgraphrag", "lightrag", "kag")


def get_adapter(framework_name: str) -> "FrameworkAdapter":
    mapping = {
        "msgraphrag": MsGraphRAGAdapter,
        "lightrag": LightRAGAdapter,
        "kag": KAGAdapter,
    }
    return mapping[framework_name]()
