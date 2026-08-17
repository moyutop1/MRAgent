import unittest

from memory.keyword_matching import (
    append_keyword_if_missing,
    keyword_gate_allowed_ids,
    keyword_entries_match,
    matched_query_keyword_count,
    normalize_keyword,
    normalize_keyword_entries,
    select_keyword_gated_memory_ids,
)
from memory.system import EAESMemoryNote, MemorySystem


class KeywordNormalizationTests(unittest.TestCase):
    def test_normalization_uses_requested_separator_rules(self):
        normalized, tokens = normalize_keyword("Lake-Sunrise/View:Point!")
        self.assertEqual(normalized, "lake_sunrise_view_point")
        self.assertEqual(tokens, {"lake", "sunrise", "view", "point"})

    def test_exact_subset_and_jaccard_matching(self):
        self.assertTrue(keyword_entries_match(
            normalize_keyword("Lake-Sunrise"),
            normalize_keyword("lake sunrise"),
        ))
        self.assertTrue(keyword_entries_match(
            normalize_keyword("sunrise"),
            normalize_keyword("lake sunrise"),
        ))
        self.assertTrue(keyword_entries_match(
            normalize_keyword("alpha beta gamma delta"),
            normalize_keyword("alpha beta gamma epsilon"),
        ))
        self.assertFalse(keyword_entries_match(
            normalize_keyword("alpha beta delta"),
            normalize_keyword("alpha beta epsilon"),
        ))

    def test_speaker_is_added_only_when_normalized_value_is_missing(self):
        keywords = ["Caroline!"]
        self.assertFalse(append_keyword_if_missing(keywords, "caroline"))
        self.assertTrue(append_keyword_if_missing(keywords, "Nate"))
        self.assertEqual(keywords, ["Caroline!", "Nate"])


class KeywordGateTests(unittest.TestCase):
    query_plan = {
        "keywords": ["Caroline", "visit"],
        "keyword_groups": [
            {"keyword": "Caroline", "alternatives": []},
            {
                "keyword": "visit",
                "alternatives": ["visited", "see", "seeing"],
            },
        ],
    }

    def test_memory_needs_two_distinct_original_keyword_groups(self):
        index = {
            "M_two": normalize_keyword_entries(["Caroline", "visited"]),
            "M_one": normalize_keyword_entries(["Caroline", "mountain"]),
        }
        self.assertEqual(
            select_keyword_gated_memory_ids(index, self.query_plan),
            {"M_two"},
        )

    def test_alternatives_of_one_original_group_count_only_once(self):
        memory_keywords = normalize_keyword_entries(["visited", "seeing"])
        self.assertEqual(
            matched_query_keyword_count(
                {
                    "keyword_groups": [{
                        "keyword": "visit",
                        "alternatives": ["visited", "see", "seeing"],
                    }]
                },
                memory_keywords,
            ),
            1,
        )

    def test_zero_eligible_matches_signal_full_memory_fallback(self):
        index = {
            "M_one": normalize_keyword_entries(["Caroline", "visited"]),
        }
        self.assertIsNone(keyword_gate_allowed_ids(
            index,
            self.query_plan,
            excluded_memory_ids={"M_one"},
        ))
        self.assertIsNone(keyword_gate_allowed_ids(
            {"M_unrelated": normalize_keyword_entries(["mountain"])},
            self.query_plan,
        ))

    def test_memory_system_indexes_internal_keywords(self):
        memory = MemorySystem()
        note = EAESMemoryNote(
            memory_id="M_D1_1",
            event_id="D1:1",
            entities=[],
            attribute_paths=[],
            raw_text="Caroline: I visited the lake.",
            rewrite_content="Caroline visited the lake.",
            conversation_time="2023-05-08",
            event_lifecycle="historical",
            origin="D1:1",
            keywords=["Caroline", "Lake-Visit"],
        )
        memory.add_eaes_memory_note(note)

        self.assertEqual(
            memory.eaes_keyword_index["M_D1_1"],
            normalize_keyword_entries(["Caroline", "Lake-Visit"]),
        )
        self.assertNotIn("keywords", note.to_dict())


if __name__ == "__main__":
    unittest.main()
