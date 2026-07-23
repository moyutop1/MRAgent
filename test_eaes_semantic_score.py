import copy
import importlib.util
import json
import sys
import types
import unittest
from unittest.mock import patch

# Keep this unit test independent from the optional python-dotenv dependency.
dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_module)

from agent.eaes import EAESMixin
from common import config
from memory.system import EAESMemoryNote, EpisodeEvent, MemorySystem
from prompts.prompts import Prompts
from prompts.schema import check_rewrite_json

HAS_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None
HAS_RUNTIME_DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "nltk", "openai", "sentence_transformers")
)
if HAS_RUNTIME_DEPENDENCIES:
    import numpy as np

    from agent.agent import Agent
    from memory.controller import MemoryController
else:
    np = None
    Agent = None
    MemoryController = None


def _valid_rewrite():
    return {
        "conversation_time": "2023-07-22",
        "sentence": [{
            "id": "D1:1-1",
            "text": "Caroline owns a dog and works as a counselor.",
            "tag": "Personal Profile",
            "origin": "D1:1",
            "topic": [],
            "time": "2023-07-22",
            "semantic_properties": ["personal_profile", "durable"],
        }],
        "topics": {},
        "personal_sentences": [],
    }


class SemanticRewriteSchemaTests(unittest.TestCase):
    dialogue = "time:2023-07-22\ndia_id:D1:1 Caroline: I own a dog."

    @unittest.skipUnless(HAS_JSONSCHEMA, "jsonschema is not installed")
    def test_valid_semantic_properties_pass_schema(self):
        ok, error = check_rewrite_json(_valid_rewrite(), self.dialogue)
        self.assertTrue(ok, error)

    @unittest.skipUnless(HAS_JSONSCHEMA, "jsonschema is not installed")
    def test_legacy_labels_duplicates_and_invalid_axis_counts_are_rejected(self):
        cases = []

        legacy = _valid_rewrite()
        legacy["sentence"][0]["semantic_properties"] = [
            "profile_preference", "durable"
        ]
        cases.append(legacy)

        duplicate = _valid_rewrite()
        duplicate["sentence"][0]["semantic_properties"] = [
            "personal_profile", "personal_profile", "durable"
        ]
        cases.append(duplicate)

        no_persistence = _valid_rewrite()
        no_persistence["sentence"][0]["semantic_properties"] = [
            "personal_profile"
        ]
        cases.append(no_persistence)

        two_persistence = _valid_rewrite()
        two_persistence["sentence"][0]["semantic_properties"] = [
            "personal_profile", "episodic", "durable"
        ]
        cases.append(two_persistence)

        four_content = _valid_rewrite()
        four_content["sentence"][0]["semantic_properties"] = [
            "event_action", "state_opinion", "personal_profile",
            "relation_social", "durable",
        ]
        cases.append(four_content)

        for payload in cases:
            with self.subTest(properties=payload["sentence"][0]["semantic_properties"]):
                ok, _ = check_rewrite_json(payload, self.dialogue)
                self.assertFalse(ok)

    def test_rewrite_prompt_defines_the_approved_labels(self):
        prompt = Prompts.REWRITE_SYSTEM_PROMPT
        self.assertIn('"personal_profile"', prompt)
        self.assertIn('"relation_social"', prompt)
        self.assertIn('never "profile_preference" or "fact_background"', prompt)


class SemanticPersistenceTests(unittest.TestCase):
    @unittest.skipUnless(
        HAS_RUNTIME_DEPENDENCIES,
        "full retrieval runtime dependencies are not installed",
    )
    def test_store_event_new_persists_properties_without_reembedding(self):
        memory = MemorySystem()
        stored_embedding = np.array([0.25, 0.75], dtype=np.float32)
        memory.embeddings = {"D1:1-1": stored_embedding}
        memory.raw_text = {"D1": {"D1:1": "Caroline: I own a dog."}}

        agent = Agent.__new__(Agent)
        agent.memory = memory
        agent.episode_link_num = 0
        events = _valid_rewrite()

        with patch.object(config, "EAES_MODE", False):
            agent.store_event_new(events, {"sentence": []}, session_id=1)

        event = memory.episode_events["D1:1-1"]
        self.assertEqual(event.semantic_properties, ["personal_profile", "durable"])
        self.assertIs(event.embedding, stored_embedding)

    def test_episode_event_owns_the_persisted_properties(self):
        event = EpisodeEvent(
            "D1:1-1",
            "Caroline owns a dog.",
            "D1:1",
            embedding=object(),
            semantic_properties=["personal_profile", "durable"],
        )
        self.assertEqual(
            event.semantic_properties,
            ["personal_profile", "durable"],
        )

    def test_eaes_memory_note_remains_unchanged(self):
        note = EAESMemoryNote(
            memory_id="M_D1_1_1",
            event_id="D1:1-1",
            entities=["Caroline"],
            attribute_paths=["profile.pet: Caroline owns a dog."],
            raw_text="Caroline: I own a dog.",
            rewrite_content="Caroline owns a dog.",
            time_interval={"start": "2023-07-22", "end": "2023-07-22"},
            event_lifecycle="current",
            origin="D1:1",
        )
        self.assertFalse(hasattr(note, "semantic_properties"))
        self.assertNotIn("semantic_properties", note.to_dict())


class _QueryLLM:
    def __init__(self, output):
        self.output = output
        self.messages = None

    def chat_text(self, messages, **_kwargs):
        self.messages = messages
        return copy.deepcopy(self.output)


class _QueryAgent(EAESMixin):
    def __init__(self, output):
        self.llm = _QueryLLM(output)

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        return value if isinstance(value, list) else [value]


def _query_output():
    return {
        "entities": ["Caroline"],
        "query_attributes": ["profile.pet: pets Caroline owns"],
        "answer_type": "fact",
        "temporal_intent": "none",
        "required_lifecycle": "unknown",
        "keywords": ["dog"],
        "required_semantic_properties": [
            "personal_profile", "durable", "unknown", "personal_profile",
            "profile_preference", "fact_background",
        ],
    }


class SemanticQueryTests(unittest.TestCase):
    def test_disabled_flag_keeps_original_prompt_and_query_plan(self):
        agent = _QueryAgent(_query_output())
        with patch.object(config, "EAES_SEMANTIC_SCORE", False):
            plan = agent.parse_eaes_query("What pet does Caroline own?")

        self.assertEqual(
            agent.llm.messages[0]["content"],
            Prompts.EAES_QUERY_SYSTEM_PROMPT,
        )
        self.assertNotIn("required_semantic_properties", plan)

    def test_enabled_flag_filters_and_deduplicates_query_properties(self):
        agent = _QueryAgent(_query_output())
        with patch.object(config, "EAES_SEMANTIC_SCORE", True):
            plan = agent.parse_eaes_query("What pet does Caroline own?")

        self.assertIn(
            Prompts.EAES_SEMANTIC_QUERY_EXTENSION,
            agent.llm.messages[0]["content"],
        )
        self.assertEqual(
            plan["required_semantic_properties"],
            ["personal_profile", "durable"],
        )


@unittest.skipUnless(
    HAS_RUNTIME_DEPENDENCIES,
    "full retrieval runtime dependencies are not installed",
)
class SemanticScoringTests(unittest.TestCase):
    @staticmethod
    def _score(memory_properties, required_properties, enabled=True):
        memory = MemorySystem()
        event_id = "D1:1-1"
        memory.episode_events[event_id] = EpisodeEvent(
            event_id,
            "Caroline owns a dog.",
            "D1:1",
            semantic_properties=memory_properties,
        )
        note = EAESMemoryNote(
            memory_id="M_D1_1_1",
            event_id=event_id,
            entities=["Caroline"],
            attribute_paths=["profile.pet: Caroline owns a dog."],
            raw_text="Caroline: I own a dog.",
            rewrite_content="Caroline owns a dog.",
            time_interval={"start": "2023-07-22", "end": "2023-07-22"},
            event_lifecycle="current",
            origin="D1:1",
            retrieval_embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
        memory.add_eaes_memory_note(note)
        controller = MemoryController(memory)
        query_plan = {
            "entities": [],
            "keywords": [],
            "required_lifecycle": "unknown",
            "required_semantic_properties": required_properties,
        }

        with (
            patch.object(config, "EAES_SEMANTIC_SCORE", enabled),
            patch("memory.controller.get_embedding") as get_embedding,
        ):
            rows = controller.score_eaes_candidates(
                query_plan,
                question_emb=np.array([1.0, 0.0], dtype=np.float32),
            )

        get_embedding.assert_not_called()
        return rows[0]

    def test_match_bonus_has_four_positive_tiers_and_caps_at_three(self):
        required = [
            "event_action", "state_opinion", "personal_profile", "durable"
        ]
        memories = [
            ["relation_social", "unknown"],
            ["event_action", "unknown"],
            ["event_action", "state_opinion", "unknown"],
            ["event_action", "state_opinion", "personal_profile", "unknown"],
            ["event_action", "state_opinion", "personal_profile", "durable"],
        ]
        expected = [
            (0, 0.0), (1, 0.1), (2, 0.2), (3, 0.3), (4, 0.3)
        ]

        for properties, (match_count, bonus) in zip(memories, expected):
            with self.subTest(properties=properties):
                row = self._score(properties, required)
                parts = row["score_parts"]
                self.assertEqual(parts["semantic_match_count"], match_count)
                self.assertAlmostEqual(parts["semantic_bonus"], bonus)

    def test_mismatch_and_disabled_flag_never_reduce_the_baseline_score(self):
        mismatch = self._score(
            ["relation_social", "unknown"],
            ["personal_profile", "durable"],
            enabled=True,
        )
        disabled = self._score(
            ["personal_profile", "durable"],
            ["personal_profile", "durable"],
            enabled=False,
        )
        self.assertEqual(mismatch["score_parts"]["semantic_bonus"], 0.0)
        self.assertEqual(disabled["score_parts"]["semantic_bonus"], 0.0)
        self.assertEqual(mismatch["score"], disabled["score"])


if __name__ == "__main__":
    unittest.main()
