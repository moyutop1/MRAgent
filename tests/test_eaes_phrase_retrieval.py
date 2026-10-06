import json
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_module)

from agent.eaes import EAESMixin
from common import config
from prompts.prompts import Prompts

HAS_RETRIEVAL_RUNTIME = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "nltk", "openai", "sentence_transformers")
)
if HAS_RETRIEVAL_RUNTIME:
    import numpy as np

    from memory.controller import MemoryController
    from memory.system import EAESMemoryNote, EpisodeEvent, MemorySystem
else:
    np = None
    MemoryController = None
    EAESMemoryNote = None
    EpisodeEvent = None
    MemorySystem = None


class _TestEAES(EAESMixin):
    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        return value if isinstance(value, list) else [value]


def _query_output(phrases):
    return {
        "entities": ["Caroline"],
        "query_attributes": ["event.attendance: event Caroline attended"],
        "answer_type": "event_list",
        "keywords": ["Caroline"],
        "retrieval_breadth": "several",
        "detail_need": "exact",
        "retrieval_phrases": phrases,
    }


class _QueuedLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.inputs = []

    def chat_text(self, messages, **_kwargs):
        self.inputs.append({
            "system": messages[0]["content"],
            "user": json.loads(messages[-1]["content"]),
        })
        return self.outputs.pop(0)


class PhrasePlanTests(unittest.TestCase):
    def test_normalizer_requires_six_but_accepts_duplicate_views_by_default(self):
        self.assertIsNone(
            EAESMixin._normalize_eaes_retrieval_phrases([
                "Caroline event", "event attendance", "joined event",
            ])
        )
        self.assertIsNone(
            EAESMixin._normalize_eaes_retrieval_phrases(
                [
                    "Caroline event", "event attendance",
                    None, "joined event",
                ]
            )
        )
        self.assertIsNone(
            EAESMixin._normalize_eaes_retrieval_phrases(
                [
                    "Caroline event", "event attendance", "joined event",
                    "fourth", "fifth", "one two three four five six seven eight nine ten eleven",
                ]
            )
        )
        duplicate_phrases = [
            "Caroline support group",
            "Caroline support group",
            "career interest",
            "pottery class",
            "camping location",
            "reading collection",
            "ignored seventh phrase",
        ]
        self.assertEqual(
            EAESMixin._normalize_eaes_retrieval_phrases(duplicate_phrases),
            duplicate_phrases[:6],
        )
        phrases = [
            "Caroline event attendance",
            "Caroline joined support group",
            "Caroline community gathering",
            "Caroline participation history",
            "Caroline attended local event",
            "Caroline group involvement",
        ]
        self.assertEqual(
            EAESMixin._normalize_eaes_retrieval_phrases(phrases),
            phrases,
        )

    def test_strict_phrase_validator_remains_available_for_ablation(self):
        phrases = [
            "Caroline support group",
            "Caroline support group",
            "Caroline career interest",
            "Caroline pottery class",
            "Caroline camping location",
            "Caroline reading collection",
        ]
        with patch.object(config, "EAES_STRICT_PHRASE_VALIDATION", True):
            self.assertIsNone(
                EAESMixin._normalize_eaes_retrieval_phrases(phrases)
            )

    def test_parse_repairs_wrong_count_or_overlong_phrase_once(self):
        mixin = _TestEAES()
        mixin.llm = _QueuedLLM([
            _query_output([
                "Caroline support group",
                "career interest",
                "pottery class",
                "four word phrase invalid",
            ]),
            {
                "retrieval_phrases": [
                    "Caroline event attendance",
                    "Caroline joined support group",
                    "Caroline community gathering",
                    "Caroline participation history",
                    "Caroline attended local event",
                    "Caroline group involvement",
                ]
            },
        ])

        plan = mixin.parse_eaes_query("What event did Caroline attend?")

        self.assertEqual(
            plan["retrieval_phrases"],
            [
                "Caroline event attendance",
                "Caroline joined support group",
                "Caroline community gathering",
                "Caroline participation history",
                "Caroline attended local event",
                "Caroline group involvement",
            ],
        )
        self.assertEqual(plan["retrieval_phrase_source"], "regenerated")
        self.assertEqual(len(mixin.llm.inputs), 2)

    def test_plain_retrieval_phrases_have_ten_word_limit(self):
        phrases = [
            "Caroline favorite books",
            "book ownership",
            "reading collection",
            "children's literature",
            "Caroline book preferences",
            "books Caroline enjoys reading with her family",
        ]

        self.assertEqual(
            EAESMixin._normalize_eaes_retrieval_phrases(phrases),
            phrases,
        )
        self.assertIsNone(EAESMixin._normalize_eaes_retrieval_phrases([
            "Caroline favorite books",
            "book ownership",
            "reading collection",
            "Caroline children's literature collection",
            "fifth phrase",
            "one two three four five six seven eight nine ten eleven",
        ]))

    def test_parse_repairs_only_once_then_raises_with_validation_error(self):
        question = "What event did Caroline attend?"
        mixin = _TestEAES()
        mixin.llm = _QueuedLLM([
            _query_output(["one"]),
            {"retrieval_phrases": ["one", "two", "three"]},
        ])

        with self.assertRaisesRegex(
                ValueError, "failed after exactly one repair attempt"):
            mixin.parse_eaes_query(question)

        self.assertEqual(len(mixin.llm.inputs), 2)
        self.assertIn(
            "must contain at least 6",
            mixin.llm.inputs[1]["user"]["validation_error"],
        )

    def test_deprecated_temporal_fields_are_never_kept_in_query_plan(self):
        output = _query_output([
            "Caroline event attendance",
            "Caroline joined support group",
            "Caroline community gathering",
            "Caroline participation history",
            "Caroline attended local event",
            "Caroline group involvement",
        ])
        output.update({
            "temporal_intent": "historical_event",
            "required_lifecycle": "historical",
            "no_time_limit": False,
        })
        mixin = _TestEAES()
        mixin.llm = _QueuedLLM([output])

        plan = mixin.parse_eaes_query("What event did Caroline attend?")

        self.assertNotIn("temporal_intent", plan)
        self.assertNotIn("required_lifecycle", plan)
        self.assertNotIn("no_time_limit", plan)

    def test_query_prompts_require_shared_six_ten_word_phrases(self):
        for prompt in (
                Prompts.EAES_QUERY_SYSTEM_PROMPT,
                Prompts.EAES_RETRIEVAL_PHRASE_REPAIR_PROMPT):
            self.assertIn("normal retrieval expression", prompt)
            self.assertIn(
                'do not use the child-memory "prefix.facet" format',
                prompt,
            )
            self.assertIn("no more than ten whitespace-separated words", prompt)
            self.assertIn("controlled redundancy", prompt)
            self.assertIn("relation-complete", prompt)

    def test_breadth_and_detail_labels_are_normalized_for_routing_only(self):
        mixin = _TestEAES()
        mixin.llm = _QueuedLLM([_query_output([
            "Caroline event attendance",
            "Caroline joined support group",
            "Caroline community gathering",
            "Caroline participation history",
            "Caroline attended local event",
            "Caroline group involvement",
        ])])

        plan = mixin.parse_eaes_query("What events did Caroline join?")
        child_plan = mixin._eaes_child_query_plan(plan)

        self.assertEqual(plan["retrieval_breadth"], "several")
        self.assertEqual(plan["breadth_value"], 0.5)
        self.assertEqual(plan["detail_need"], "exact")
        self.assertEqual(plan["detail_value"], 1.0)
        for key in (
                "retrieval_breadth", "breadth_value",
                "detail_need", "detail_value"):
            self.assertNotIn(key, child_plan)


@unittest.skipUnless(HAS_RETRIEVAL_RUNTIME, "retrieval runtime dependencies are unavailable")
class PhraseRankingTests(unittest.TestCase):
    def test_ranker_uses_each_childs_maximum_tag_similarity(self):
        store = MemorySystem()
        for index, (tags, vector) in enumerate(
                [
                    (["Caroline profile.alpha"], np.array([1.0, 0.0])),
                    ([
                        "Caroline activity.other topic",
                        "Caroline activity.beta",
                    ], np.array([0.0, 1.0])),
                ],
                start=1):
            event_id = f"D1:{index}-1"
            event = EpisodeEvent(event_id, f"memory {index}", f"D1:{index}", vector)
            event.tag_t = tags
            store.episode_events[event_id] = event
            note = EAESMemoryNote(
                memory_id=f"M_{index}",
                event_id=event_id,
                entities=[],
                attribute_paths=[],
                raw_text="",
                rewrite_content=f"memory {index}",
                conversation_time="2023-01-01",
                event_lifecycle="historical",
                origin=f"D1:{index}",
            )
            store.add_eaes_memory_note(note)

        vectors = {
            "Caroline profile.unrelated": np.array([-1.0, 0.0]),
            "Caroline profile.alpha": np.array([1.0, 0.0]),
            "Caroline activity.other topic": np.array([0.0, -1.0]),
            "Caroline activity.beta": np.array([0.0, 1.0]),
            "Caroline profile.alpha query": np.array([1.0, 0.0]),
            "Caroline activity.beta query": np.array([0.0, 1.0]),
        }

        embedding_inputs = []

        def fake_embedding(texts):
            embedding_inputs.append(list(texts))
            return np.vstack([vectors[text] for text in texts])

        controller = MemoryController(store)
        phrases = [
            "Caroline profile.alpha query",
            "Caroline activity.beta query",
            "Caroline profile.alpha query",
            "Caroline activity.beta query",
        ]
        with patch("memory.controller.get_embedding", side_effect=fake_embedding):
            rankings = controller.rank_eaes_children_per_phrase(phrases, top_k=2)

        self.assertEqual(rankings[0][0]["memory_id"], "M_1")
        self.assertEqual(
            rankings[0][0]["matched_tag"], "Caroline profile.alpha"
        )
        self.assertEqual(rankings[0][0]["matched_tag_index"], 0)
        self.assertEqual(
            rankings[0][0]["tag"], ["Caroline profile.alpha"]
        )
        self.assertEqual(rankings[1][0]["memory_id"], "M_2")
        self.assertEqual(
            rankings[1][0]["matched_tag"], "Caroline activity.beta"
        )
        self.assertEqual(rankings[2][0]["memory_id"], "M_1")
        self.assertEqual(rankings[3][0]["memory_id"], "M_2")
        self.assertEqual(embedding_inputs[0], [
            "Caroline profile.alpha",
            "Caroline activity.other topic",
            "Caroline activity.beta",
        ])


class PhraseRerankerTests(unittest.TestCase):
    def test_reranker_payload_hides_phrase_and_rrf_information(self):
        mixin = _TestEAES()
        mixin.llm = _QueuedLLM([{"ranked_memory_ids": ["M_2", "M_1"]}])
        candidates = [
            {
                "memory_id": "M_1",
                "origin": "D1:1",
                "tag": [
                    "Person profile.alpha", "Person profile.alpha topic",
                ],
                "rewrite_content": "Alpha memory",
                "_internal_score": 0.9,
                "phrase": "hidden",
                "phrase_rank": 1,
            },
            {
                "memory_id": "M_2",
                "origin": "D1:2",
                "tag": [
                    "Person profile.beta", "Person profile.beta topic",
                ],
                "rewrite_content": "Beta memory",
                "_internal_score": 0.4,
            },
        ]

        result = mixin.rerank_eaes_phrase_candidates(
            "Which memory answers the question?", candidates, top_k=2
        )

        payload = mixin.llm.inputs[0]["user"]
        self.assertEqual([item["memory_id"] for item in result], ["M_2", "M_1"])
        self.assertNotIn("query_plan", payload)
        self.assertEqual(
            set(payload["candidates"][0]),
            {"memory_id", "origin", "tag", "rewrite_content"},
        )
        for item in result:
            self.assertNotIn("_internal_score", item)


if __name__ == "__main__":
    unittest.main()
