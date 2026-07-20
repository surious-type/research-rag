from __future__ import annotations

import os
import sys

from research_bench.data import ROOT


KAG_ROOT = ROOT / "frameworks" / "kag"
if str(KAG_ROOT) not in sys.path:
    sys.path.insert(0, str(KAG_ROOT))
os.environ.setdefault("KAG_PROJECT_HOST_ADDR", "http://127.0.0.1:8887")
os.environ.setdefault("KAG_PROJECT_ID", "1")


def test_schema_free_extractor_named_entity_recognition_process_tolerates_none_result() -> None:
    import research_bench._kag_registry as kag_registry

    extractor = object.__new__(kag_registry.BenchmarkCompatibilitySchemaFreeExtractor)
    extractor.external_graph = None
    extractor.schema = {}

    result = kag_registry.BenchmarkCompatibilitySchemaFreeExtractor._named_entity_recognition_process(
        extractor,
        "passage",
        None,
    )

    assert result == []


def test_knowledge_unit_extractor_assemble_knowledge_unit_accepts_dict_core_entities() -> None:
    import research_bench._kag_registry as kag_registry

    extractor = object.__new__(kag_registry.BenchmarkCompatibilityKnowledgeUnitExtractor)
    extractor.get_stand_schema = lambda type_name: type_name
    extractor.assemble_sub_graph_with_spg_properties = lambda *args, **kwargs: None

    class GraphStub:
        def __init__(self) -> None:
            self.nodes: list[tuple[str, str, str, dict]] = []
            self.edges: list[tuple[str, str, str, str, str]] = []

        def add_node(self, node_id, name, category, properties):
            self.nodes.append((node_id, name, category, properties))

        def add_edge(self, source_name, source_category, edge_type, target_name, target_category):
            self.edges.append((source_name, source_category, edge_type, target_name, target_category))

    graph = GraphStub()
    knowledge_units = {
        "ku-1": {
            "content": "Atlas is the main module.",
            "knowledgetype": "fact",
            "core_entities": {"Atlas": "ArtificialObject", "Orion-128": "ArtificialObject"},
        }
    }
    source_entities = [{"name": "Atlas", "category": "ArtificialObject"}]

    nodes = kag_registry.BenchmarkCompatibilityKnowledgeUnitExtractor.assemble_knowledge_unit(
        extractor,
        graph,
        source_entities,
        knowledge_units,
        [],
    )

    assert len(nodes) == 1
    assert any(edge[0] == "Atlas" and edge[2] == "source" for edge in graph.edges)
    assert any(edge[0] == "Orion-128" and edge[2] == "source" for edge in graph.edges)
    assert any(node[0] == "Orion-128" and node[2] == "ArtificialObject" for node in graph.nodes)
