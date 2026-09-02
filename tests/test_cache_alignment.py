import json
import tempfile
import unittest
from pathlib import Path

from common.cache_utils import (
    index_keyword_sentences,
    load_jsonl_records,
    validate_rewrite_cache_prefix,
)


class CacheAlignmentTests(unittest.TestCase):
    def test_partial_rewrite_cache_is_a_valid_resumable_prefix(self):
        records = [
            {"session_1": {"sentence": []}},
            {"session_2": {"sentence": []}},
        ]

        validate_rewrite_cache_prefix(
            records,
            ["session_1", "session_2", "session_3"],
            "rewrite.json",
        )

    def test_out_of_order_rewrite_cache_is_not_resumed_by_position(self):
        records = [{"session_2": {"sentence": []}}]

        with self.assertRaisesRegex(ValueError, "expected 'session_1'"):
            validate_rewrite_cache_prefix(
                records,
                ["session_1", "session_2"],
                "rewrite.json",
            )

    def test_keyword_records_are_indexed_by_sentence_id_across_lines(self):
        records = [
            {"sentence": [{"sentence_id": "D1:1-1", "keyword": ["one"]}]},
            {"sentence": [{"sentence_id": "D2:1-1", "keyword": ["two"]}]},
            {"sentence": [{"sentence_id": "stale", "keyword": ["ignored"]}]},
        ]

        indexed = index_keyword_sentences(records)

        self.assertEqual(indexed["D1:1-1"]["keyword"], ["one"])
        self.assertEqual(indexed["D2:1-1"]["keyword"], ["two"])

    def test_loader_reports_truncated_jsonl_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            path.write_text(
                json.dumps({"session_1": {}}) + "\n{" + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "at line 2"):
                load_jsonl_records(path)


if __name__ == "__main__":
    unittest.main()
