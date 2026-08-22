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
        "temporal_intent": "historical_event",
        "required_lifecycle": "historical",
        "keywords": ["Caroline"],
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
        self.assertEqual(
            EAESMixin._normalize_eaes_retrieval_phrases(
                ["same", "same", "another", "fourth", "ignored"]
            ),
            ["same", "same", "another", "fourth"],
        )

    def test_parse_repairs_short_phrase_list_once(self):
        mixin = _TestEAES()
        mixin.llm = _QueuedLLM([
            _query_output(["one", "two"]),
            {"retrieval_phrases": ["one", "two", "three", "four"]},
        ])

        plan = mixin.parse_eaes_query("What event did Caroline attend?")

        self.assertEqual(plan["retrieval_phrases"], ["one", "two", "three", "four"])
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

        self.assertEqual(plan["retrieval_phrases"], [question] * 4)
        self.assertEqual(plan["retrieval_phrase_source"], "question_fallback")
        self.assertEqual(len(mixin.llm.inputs), 2)


def _ranking(phrase_index):
    items = []
    for rank in range(1, 16):
        memory_id = "SHARED" if rank == 15 else f"P{phrase_index}_{rank}"
        items.append({
            "memory_id": memory_id,
            "event_id": memory_id,
            "origin": memory_id,
            "tag": memory_id,
            "rewrite_content": memory_id,
            "phrase_index": phrase_index,
            "phrase": f"phrase {phrase_index}",
            "phrase_rank": rank,
            "phrase_similarity": 1.0 / rank,
        })
    return items


@unittest.skipUnless(HAS_RETRIEVAL_RUNTIME, "retrieval runtime dependencies are unavailable")
class PhraseFusionTests(unittest.TestCase):
    def test_fusion_protects_five_then_uses_rrf_for_five_supplements(self):
        rankings = [_ranking(index) for index in range(4)]

        fused, diagnostics = MemoryController.fuse_eaes_phrase_rankings(
            rankings,
            rrf_k=10,
            protected_top_k=5,
            final_per_phrase=10,
        )

        for phrase_index, final_ids in enumerate(
                diagnostics["phrase_final_top10"]):
            self.assertEqual(len(final_ids), 10)
            self.assertIn(f"P{phrase_index}_5", final_ids)
            self.assertIn("SHARED", final_ids)
        self.assertLessEqual(len(fused), 40)
        self.assertGreater(
            diagnostics["rrf_scores"]["SHARED"],
            diagnostics["rrf_scores"]["P0_6"],
        )

    def test_ranker_scores_each_phrase_against_stored_tags_independently(self):
        store = MemorySystem()
        for index, (tag, vector) in enumerate(
                [("alpha", np.array([1.0, 0.0])), ("beta", np.array([0.0, 1.0]))],
                start=1):
            event_id = f"D1:{index}-1"
            event = EpisodeEvent(event_id, f"memory {index}", f"D1:{index}", vector)
            event.tag_t = tag
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
            "alpha": np.array([1.0, 0.0]),
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
        self.assertEqual(rankings[1][0]["memory_id"], "M_2")
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
                "tag": "alpha",
                "rewrite_content": "Alpha memory",
                "_rrf_score": 0.9,
                "_rrf_rank": 1,
                "phrase": "hidden",
                "phrase_rank": 1,
            },
            {
                "memory_id": "M_2",
                "origin": "D1:2",
                "tag": "beta",
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
