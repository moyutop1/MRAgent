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
    def test_normalizer_requires_four_but_does_not_deduplicate(self):
        self.assertIsNone(
            EAESMixin._normalize_eaes_retrieval_phrases(["a", "b", "c"])
        )
        self.assertIsNone(
            EAESMixin._normalize_eaes_retrieval_phrases(
                ["a", "b", None, "d"]
            )
        )
        self.assertIsNone(
            EAESMixin._normalize_eaes_retrieval_phrases(
                ["one", "two", "three", "four word phrase invalid"]
            )
        )
        self.assertEqual(
            EAESMixin._normalize_eaes_retrieval_phrases(
                [
                    "support group", "support group", "career interest",
                    "pottery class", "ignored phrase",
                ]
            ),
            ["support group", "support group", "career interest", "pottery class"],
        )

    def test_parse_repairs_wrong_count_or_overlong_phrase_once(self):
        mixin = _TestEAES()
        mixin.llm = _QueuedLLM([
            _query_output([
                "support group", "career interest", "pottery class",
                "four word phrase invalid",
            ]),
            {
                "retrieval_phrases": [
                    "support group", "career interest",
                    "pottery class", "camping location",
                ]
            },
        ])

        plan = mixin.parse_eaes_query("What event did Caroline attend?")

        self.assertEqual(
            plan["retrieval_phrases"],
            [
                "support group", "career interest",
                "pottery class", "camping location",
            ],
        )
        self.assertEqual(plan["retrieval_phrase_source"], "regenerated")
        self.assertEqual(len(mixin.llm.inputs), 2)

    def test_parse_repairs_only_once_then_uses_question_fallback(self):
        question = "What event did Caroline attend?"
        mixin = _TestEAES()
        mixin.llm = _QueuedLLM([
            _query_output(["one"]),
            {"retrieval_phrases": ["one", "two", "three"]},
        ])

        plan = mixin.parse_eaes_query(question)

        self.assertEqual(
            plan["retrieval_phrases"], ["event Caroline attend"] * 4
        )
        self.assertEqual(plan["retrieval_phrase_source"], "question_fallback")
        self.assertEqual(len(mixin.llm.inputs), 2)

    def test_deprecated_temporal_fields_are_never_kept_in_query_plan(self):
        output = _query_output([
            "support group", "career interest",
            "pottery class", "camping location",
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

    def test_query_prompts_require_tag_style_three_word_noun_phrases(self):
        for prompt in (
                Prompts.EAES_QUERY_SYSTEM_PROMPT,
                Prompts.EAES_RETRIEVAL_PHRASE_REPAIR_PROMPT):
            self.assertIn("short concrete noun phrase", prompt)
            self.assertIn("no more than three words", prompt)
            self.assertIn("access wording", prompt)

    def test_breadth_and_detail_labels_are_normalized_for_routing_only(self):
        mixin = _TestEAES()
        mixin.llm = _QueuedLLM([_query_output([
            "support group", "career interest",
            "pottery class", "camping location",
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


def _ranking(phrase_index):
    items = []
    for rank in range(1, 16):
        memory_id = "SHARED" if rank == 15 else f"P{phrase_index}_{rank}"
        items.append({
            "memory_id": memory_id,
            "event_id": memory_id,
            "origin": memory_id,
            "tag": [memory_id, f"alternate {memory_id}"],
            "rewrite_content": memory_id,
            "phrase_index": phrase_index,
            "phrase": f"phrase {phrase_index}",
            "phrase_rank": rank,
            "phrase_similarity": 1.0 / rank,
        })
    return items


@unittest.skipUnless(HAS_RETRIEVAL_RUNTIME, "retrieval runtime dependencies are unavailable")
class PhraseFusionTests(unittest.TestCase):
    def test_fusion_keeps_dynamic_phrase_lists_and_uses_rrf_as_soft_feature(self):
        rankings = [_ranking(index) for index in range(4)]

        fused, diagnostics = MemoryController.fuse_eaes_phrase_rankings(
            rankings,
            rrf_k=10,
        )

        self.assertEqual(len(fused), 57)
        self.assertIn("SHARED", diagnostics["global_candidate_ids"])
        by_id = {item["memory_id"]: item for item in fused}
        self.assertGreater(
            by_id["SHARED"]["rrf_score"],
            by_id["P0_6"]["rrf_score"],
        )
        self.assertTrue(all("candidate_score" in item for item in fused))

    def test_dynamic_phrase_topk_uses_top30_probability_mass(self):
        sharp = [
            {"memory_id": f"S{i}", "phrase_rank": i + 1,
             "phrase_similarity": 1.0 if i == 0 else 0.0}
            for i in range(30)
        ]
        flat = [
            {"memory_id": f"F{i}", "phrase_rank": i + 1,
             "phrase_similarity": 0.0}
            for i in range(30)
        ]

        selected, diagnostics = (
            MemoryController.select_eaes_dynamic_phrase_rankings(
                [sharp, flat, sharp, flat]
            )
        )

        self.assertEqual([len(items) for items in selected], [15, 24, 15, 24])
        self.assertEqual(
            [item["selected_k"] for item in diagnostics],
            [15, 24, 15, 24],
        )

    def test_ranker_uses_each_childs_maximum_tag_similarity(self):
        store = MemorySystem()
        for index, (tags, vector) in enumerate(
                [
                    (["unrelated", "alpha"], np.array([1.0, 0.0])),
                    (["other topic", "beta"], np.array([0.0, 1.0])),
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
            "unrelated": np.array([-1.0, 0.0]),
            "alpha": np.array([1.0, 0.0]),
            "other topic": np.array([0.0, -1.0]),
            "beta": np.array([0.0, 1.0]),
            "alpha query": np.array([1.0, 0.0]),
            "beta query": np.array([0.0, 1.0]),
        }

        def fake_embedding(texts):
            return np.vstack([vectors[text] for text in texts])

        controller = MemoryController(store)
        phrases = ["alpha query", "beta query", "alpha query", "beta query"]
        with patch("memory.controller.get_embedding", side_effect=fake_embedding):
            rankings = controller.rank_eaes_children_per_phrase(phrases, top_k=2)

        self.assertEqual(rankings[0][0]["memory_id"], "M_1")
        self.assertEqual(rankings[0][0]["matched_tag"], "alpha")
        self.assertEqual(rankings[0][0]["matched_tag_index"], 1)
        self.assertEqual(rankings[0][0]["tag"], ["unrelated", "alpha"])
        self.assertEqual(rankings[1][0]["memory_id"], "M_2")
        self.assertEqual(rankings[1][0]["matched_tag"], "beta")
        self.assertEqual(rankings[2][0]["memory_id"], "M_1")
        self.assertEqual(rankings[3][0]["memory_id"], "M_2")


class PhraseRerankerTests(unittest.TestCase):
    def test_reranker_payload_hides_phrase_and_rrf_information(self):
        mixin = _TestEAES()
        mixin.llm = _QueuedLLM([{"ranked_memory_ids": ["M_2", "M_1"]}])
        candidates = [
            {
                "memory_id": "M_1",
                "origin": "D1:1",
                "tag": ["alpha", "alpha topic"],
                "rewrite_content": "Alpha memory",
                "_rrf_score": 0.9,
                "_rrf_rank": 1,
                "phrase": "hidden",
                "phrase_rank": 1,
            },
            {
                "memory_id": "M_2",
                "origin": "D1:2",
                "tag": ["beta", "beta topic"],
                "rewrite_content": "Beta memory",
                "_rrf_score": 0.4,
                "_rrf_rank": 2,
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
            self.assertNotIn("_rrf_score", item)
            self.assertNotIn("_rrf_rank", item)


if __name__ == "__main__":
    unittest.main()
