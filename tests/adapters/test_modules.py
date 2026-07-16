from research_bench.adapters.kag import KAGAdapter
from research_bench.adapters.lightrag import LightRAGAdapter
from research_bench.adapters.msgraphrag import MsGraphRAGAdapter


def test_adapter_modules_export_expected_classes() -> None:
    assert MsGraphRAGAdapter.__name__ == "MsGraphRAGAdapter"
    assert LightRAGAdapter.__name__ == "LightRAGAdapter"
    assert KAGAdapter.__name__ == "KAGAdapter"
