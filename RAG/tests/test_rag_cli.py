import unittest
import sys
import types


def _install_test_stubs() -> None:
    docx_mod = types.ModuleType("docx")
    docx_mod.Document = object
    sys.modules.setdefault("docx", docx_mod)

    pypdf_mod = types.ModuleType("pypdf")
    pypdf_mod.PdfReader = object
    sys.modules.setdefault("pypdf", pypdf_mod)

    rank_mod = types.ModuleType("rank_bm25")
    rank_mod.BM25Okapi = object
    sys.modules.setdefault("rank_bm25", rank_mod)

    class MatchValue:
        def __init__(self, value):
            self.value = value

    class FieldCondition:
        def __init__(self, key, match):
            self.key = key
            self.match = match

    class Filter:
        def __init__(self, must):
            self.must = must

    qdrant_mod = types.ModuleType("qdrant_client")
    qdrant_mod.QdrantClient = object
    qdrant_mod.models = types.SimpleNamespace(
        MatchValue=MatchValue,
        FieldCondition=FieldCondition,
        Filter=Filter,
    )
    sys.modules.setdefault("qdrant_client", qdrant_mod)

    httpx_mod = types.ModuleType("httpx")
    httpx_mod.request = lambda *args, **kwargs: None
    sys.modules.setdefault("httpx", httpx_mod)


_install_test_stubs()

from src.rag_cli import build_parser, build_qdrant_filter, fuse_candidates, parse_metadata_pairs


class TestMetadataParsing(unittest.TestCase):
    def test_parse_metadata_pairs_parses_types(self):
        meta = parse_metadata_pairs(["team=ml", "version=2", "temp=0.75", "active=true"])
        self.assertEqual(meta["team"], "ml")
        self.assertEqual(meta["version"], 2)
        self.assertEqual(meta["temp"], 0.75)
        self.assertEqual(meta["active"], True)

    def test_parse_metadata_pairs_invalid_value(self):
        with self.assertRaises(SystemExit):
            parse_metadata_pairs(["broken"])


class TestFusion(unittest.TestCase):
    def test_fuse_candidates_combines_dense_and_opensearch(self):
        qdrant = [
            {
                "rank": 0,
                "dense_score_raw": 0.9,
                "chunk_id": "a",
                "source_name": "f1",
                "source_path": "/tmp/f1",
                "chunk_index": 0,
                "text": "alpha",
            },
            {
                "rank": 0,
                "dense_score_raw": 0.1,
                "chunk_id": "b",
                "source_name": "f2",
                "source_path": "/tmp/f2",
                "chunk_index": 0,
                "text": "beta",
            },
        ]
        opensearch = [
            {
                "rank": 0,
                "chunk_id": "b",
                "source_name": "f2",
                "source_path": "/tmp/f2",
                "chunk_index": 0,
                "text": "beta",
                "opensearch_score_raw": 2.0,
            }
        ]

        fused = fuse_candidates("beta навыки", qdrant, opensearch, dense_weight=0.3)
        self.assertEqual(fused[0]["chunk_id"], "b")
        self.assertIn("score", fused[0])
        self.assertIn("dense_score", fused[0])
        self.assertIn("bm25_score", fused[0])
        self.assertIn("opensearch_score", fused[0])
        self.assertIn("local_lexical_score", fused[0])


class TestQdrantFilter(unittest.TestCase):
    def test_build_qdrant_filter_supports_source_and_metadata(self):
        filter_obj = build_qdrant_filter(
            source_name="resume.pdf",
            metadata={"department": "risk", "version": 3},
        )
        self.assertIsNotNone(filter_obj)
        self.assertEqual(len(filter_obj.must), 3)


class TestMindmapArgs(unittest.TestCase):
    def test_mindmap_accepts_inputs(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "mindmap",
                "--inputs",
                "/tmp/file.pdf",
                "--output",
                "mindmap/output/out.html",
            ]
        )
        self.assertEqual(args.command, "mindmap")
        self.assertEqual(args.inputs, ["/tmp/file.pdf"])
        self.assertEqual(args.output, "mindmap/output/out.html")


if __name__ == "__main__":
    unittest.main()
