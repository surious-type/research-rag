from research_bench.adapters.registry import ADAPTER_NAMES, get_adapter


def test_registry_contains_supported_frameworks() -> None:
    assert ADAPTER_NAMES == ("msgraphrag", "lightrag", "kag")


def test_get_adapter_returns_expected_legacy_instances() -> None:
    assert get_adapter("msgraphrag").__class__.__name__ == "MsGraphRAGAdapter"
    assert get_adapter("lightrag").__class__.__name__ == "LightRAGAdapter"
    assert get_adapter("kag").__class__.__name__ == "KAGAdapter"
