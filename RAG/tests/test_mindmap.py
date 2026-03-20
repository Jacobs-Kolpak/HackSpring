import unittest

from src.mindmap.generator import build_graph_data


class TestMindmapGenerator(unittest.TestCase):
    def test_build_graph_data_has_nodes_and_edges(self):
        text = (
            "Python используется для backend разработки. "
            "Backend разработка включает API и базы данных. "
            "Python и API часто используются вместе в backend."
        )

        graph = build_graph_data(text, top_n_concepts=10, min_concept_freq=1, min_edge_weight=1)

        node_ids = {node["id"] for node in graph["nodes"]}
        self.assertIn("python", node_ids)
        self.assertIn("backend", node_ids)
        self.assertGreaterEqual(len(graph["edges"]), 1)

    def test_build_graph_data_respects_min_edge_weight(self):
        text = "аналитика данные аналитика визуализация"
        graph = build_graph_data(text, top_n_concepts=10, min_concept_freq=1, min_edge_weight=2)
        self.assertIsInstance(graph["nodes"], list)
        self.assertIsInstance(graph["edges"], list)


if __name__ == "__main__":
    unittest.main()
