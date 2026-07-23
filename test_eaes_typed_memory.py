import sys
import types
import unittest
from unittest.mock import patch

# Keep tests independent from the optional python-dotenv dependency while
# retaining the real common.config module for cross-test compatibility.
dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_module)

from agent.eaes import EAESMixin
from common import config
from memory.system import (
    EAESMemoryNote,
    MemorySystem,
    eaes_persistence_compatibility,
    eaes_type_compatibility,
)
from prompts.prompts import Prompts

try:
    import numpy as np
    from memory.controller import MemoryController
except ModuleNotFoundError:
    np = None
    MemoryController = None


def _note(memory_id, memory_types, persistence):
    return EAESMemoryNote(
        memory_id=memory_id,
        event_id=memory_id.replace("M_", ""),
        entities=["Caroline"],
        attribute_paths=["relationship.status: Caroline's social relationship."],
        raw_text="Caroline described the relationship.",
        rewrite_content="Caroline described the relationship.",
        time_interval={"start": "2023-01-01", "end": "2023-01-01"},
        event_lifecycle="current",
        origin="D1:1",
        retrieval_embedding=(
            np.asarray([1.0, 0.0], dtype=np.float32) if np is not None else None
        ),
        memory_types=memory_types,
        persistence=persistence,
    )


class _QueryLLM:
    def __init__(self, output):
        self.output = output
        self.system_prompt = None

    def chat_text(self, messages, **_kwargs):
        self.system_prompt = messages[0]["content"]
        return self.output


class _QueryAgent(EAESMixin):
    def __init__(self, output):
        self.llm = _QueryLLM(output)

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        return value if isinstance(value, list) else [value]


class TypedMemorySchemaTests(unittest.TestCase):
    def test_typed_index_prompt_is_isolated_from_legacy_prompt(self):
        self.assertNotIn("memory_types", Prompts.EAES_INDEX_SYSTEM_PROMPT)
        self.assertIn("memory_types", Prompts.EAES_TYPED_INDEX_SYSTEM_PROMPT)
        self.assertIn("persistence", Prompts.EAES_TYPED_INDEX_SYSTEM_PROMPT)

    def test_note_serializes_typed_fields_and_can_hide_them(self):
        note = _note("M_D1_1_1", ["relation_social"], "durable")

        typed = note.to_dict()
        legacy_shape = note.to_dict(include_typed=False)

        self.assertEqual(typed["memory_types"], ["relation_social"])
        self.assertEqual(typed["persistence"], "durable")
        self.assertNotIn("memory_types", legacy_shape)
        self.assertNotIn("persistence", legacy_shape)

    def test_legacy_note_attributes_fall_back_to_unknown(self):
        note = _note("M_D1_1_1", ["relation_social"], "durable")
        del note.memory_types
        del note.persistence

        data = note.to_dict()

        self.assertEqual(data["memory_types"], [])
        self.assertEqual(data["persistence"], "unknown")

    def test_type_normalization_deduplicates_and_drops_unknown_labels(self):
        values = EAESMixin._eaes_normalize_memory_types([
            {"value": "relation_social"},
            {"value": "relation_social"},
            "event_action",
            "not_a_type",
        ])

        self.assertEqual(values, ["relation_social", "event_action"])

    def test_heuristic_fallback_separates_type_and_persistence(self):
        types_out, persistence = EAESMixin._eaes_infer_semantic_properties(
            "Caroline's friend Melanie supports her counseling career interest."
        )

        self.assertIn("relation_social", types_out)
        self.assertIn("profile_preference", types_out)
        self.assertEqual(persistence, "durable")

    def test_unknown_persistence_is_neutral_not_a_conflict(self):
        query_plan = {
            "required_memory_types": ["relation_social"],
            "preferred_persistence": ["durable"],
        }
        note = _note("M_D1_1_1", ["relation_social"], "unknown")

        type_score, matched_types, type_state = eaes_type_compatibility(
            query_plan, note
        )
        persistence_score, matched_persistence, persistence_state = (
            eaes_persistence_compatibility(query_plan, note)
        )

        self.assertEqual(type_score, 1.0)
        self.assertEqual(matched_types, ["relation_social"])
        self.assertEqual(type_state, "exact")
        self.assertEqual(persistence_score, 0.5)
        self.assertEqual(matched_persistence, "durable")
        self.assertEqual(persistence_state, "unknown")


class TypedQueryPlanTests(unittest.TestCase):
    def test_typed_query_prompt_and_plan_are_enabled_only_by_flag(self):
        output = {
            "entities": ["Caroline"],
            "query_attributes": ["relationship.status: Caroline's relationship"],
            "answer_type": "fact",
            "temporal_intent": "none",
            "required_lifecycle": "unknown",
            "keywords": ["relationship"],
            "required_memory_types": ["relation_social", "invalid"],
            "preferred_persistence": ["durable"],
        }
        agent = _QueryAgent(output)

        with patch.object(config, "EAES_TYPED_MEMORY", True):
            plan = agent.parse_eaes_query("What is Caroline's relationship?")

        self.assertIs(agent.llm.system_prompt, Prompts.EAES_TYPED_QUERY_SYSTEM_PROMPT)
        self.assertEqual(
            plan["required_memory_types"],
            ["relation_social"],
        )
        self.assertEqual(
            plan["preferred_persistence"],
            ["durable"],
        )

        legacy_agent = _QueryAgent(output)
        with patch.object(config, "EAES_TYPED_MEMORY", False):
            legacy_plan = legacy_agent.parse_eaes_query(
                "What is Caroline's relationship?"
            )

        self.assertIs(
            legacy_agent.llm.system_prompt, Prompts.EAES_QUERY_SYSTEM_PROMPT
        )
        self.assertNotIn("required_memory_types", legacy_plan)
        self.assertNotIn("preferred_persistence", legacy_plan)


@unittest.skipUnless(np is not None, "numpy is not installed in this test environment")
class TypedCandidateScoreTests(unittest.TestCase):
    def setUp(self):
        memory = MemorySystem()
        memory.add_eaes_memory_note(
            _note("M_durable", ["relation_social"], "durable")
        )
        memory.add_eaes_memory_note(
            _note("M_unknown", ["relation_social"], "unknown")
        )
        memory.add_eaes_memory_note(
            _note("M_unmatched", ["event_action"], "episodic")
        )
        self.controller = MemoryController(memory)
        self.query_plan = {
            "query_attributes": ["relationship.status: Caroline's relationship"],
            "entities": [],
            "keywords": [],
            "required_lifecycle": "unknown",
            "required_memory_types": ["relation_social"],
            "preferred_persistence": ["durable"],
        }

    def _score(self, typed):
        with (
            patch.object(config, "EAES_TYPED_MEMORY", typed),
            patch.object(config, "EAES_TYPE_WEIGHT", 0.15),
            patch.object(config, "EAES_PERSISTENCE_WEIGHT", 0.05),
            patch.object(
                self.controller,
                "_eaes_query_embeddings",
                return_value=(
                    np.asarray([[1.0, 0.0]], dtype=np.float32),
                    ["relationship.status: Caroline's relationship"],
                ),
            ),
        ):
            rows = self.controller.score_eaes_candidates(
                self.query_plan, include_rank=True
            )
        return {row["memory_id"]: row for row in rows}

    def test_exact_and_unknown_persistence_are_soft_positive_signals(self):
        typed = self._score(True)
        baseline = self._score(False)

        self.assertGreater(
            typed["M_durable"]["score"], typed["M_unknown"]["score"]
        )
        self.assertGreater(
            typed["M_unknown"]["score"], typed["M_unmatched"]["score"]
        )
        self.assertGreaterEqual(
            typed["M_unknown"]["score"], baseline["M_unknown"]["score"]
        )
        self.assertEqual(
            typed["M_unknown"]["persistence_match_state"], "unknown"
        )
        self.assertEqual(typed["M_unknown"]["score_parts"]["type"], 1.0)
        self.assertEqual(
            typed["M_unknown"]["score_parts"]["persistence"], 0.5
        )

    def test_disabled_mode_preserves_legacy_candidate_shape(self):
        baseline = self._score(False)
        row = baseline["M_durable"]

        self.assertNotIn("memory_types", row)
        self.assertNotIn("persistence", row)
        self.assertNotIn("type", row["score_parts"])
        self.assertNotIn("type_match_state", row)


if __name__ == "__main__":
    unittest.main()
