import unittest

from src.flash.generator import build_study_context


class TestFlashGenerator(unittest.TestCase):
    def test_build_study_context_respects_limit(self):
        chunks = [
            {"source_name": "a.txt", "chunk_index": 0, "text": "alpha " * 30},
            {"source_name": "b.txt", "chunk_index": 1, "text": "beta " * 30},
        ]
        context = build_study_context(chunks, max_chars=180)
        self.assertIn("a.txt", context)
        self.assertNotIn("b.txt", context)

    def test_build_study_context_skips_empty_text(self):
        chunks = [
            {"source_name": "a.txt", "chunk_index": 0, "text": ""},
            {"source_name": "b.txt", "chunk_index": 1, "text": "valid text"},
        ]
        context = build_study_context(chunks, max_chars=1000)
        self.assertIn("b.txt", context)
        self.assertNotIn("a.txt", context)


if __name__ == "__main__":
    unittest.main()

