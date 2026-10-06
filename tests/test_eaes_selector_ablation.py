import json
import sys
import types
import unittest
from unittest.mock import patch

# Keep this unit test independent from the optional python-dotenv dependency
# while retaining the real common.config module for cross-test compatibility.
dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_module)

from agent.eaes import EAESMixin
from common import config


class _FakeLLM:
    def __init__(self, answer="test answer"):
        self.inputs = []
        self.answer = answer

    def chat_text(self, messages, **_kwargs):
        self.inputs.append(json.loads(messages[-1]["content"]))
        return {
            "mode": "answer",
            "answer": self.answer,
            "supports": ["M_1"],
            "confidence": 1.0,
        }


class _FakeMemory:
    @staticmethod
    def get_eaes_support_origin(memory_ids):
        if len(memory_ids) == 1 and str(memory_ids[0]).startswith("1-"):
            parent_id = memory_ids[0]
            index = int(parent_id.split("-", 1)[1])
            return [parent_id, f"D2:{index}", f"D2:{index + 10}"]
        return list(memory_ids)


class _FakeController:
    def __init__(self, candidates):
        self.candidates = candidates
        self.child_retrieval_phrases = []
        self.parent_query_plans = []

    def retrieve_eaes_candidates(self, query_plan, *_args, **_kwargs):
        return list(self.candidates)

    def route_eaes_parent_candidates(self, query_plan, *_args, **_kwargs):
        self.parent_query_plans.append(dict(query_plan))
        parents = [{
            "parent_id": f"1-{i}",
            "rewrite_content": f"Parent memory {i} about Caroline and dogs.",
            "score": 1.0 / i,
            "posterior_score": 1.0 / i,
            "rank": i,
            "matched_keyword": "dog",
        } for i in range(1, 5)]
        return parents, {
            "breadth_value": 0.5,
            "detail_value": 0.5,
            "parent_candidates": parents,
        }

    def retrieve_eaes_adaptive_views(
            self, retrieval_phrases, question_emb=None, question_text=None
    ):
        self.child_retrieval_phrases.append(list(retrieval_phrases))
        children = []
        for rank, candidate in enumerate(self.candidates, start=1):
            children.append({
                **candidate,
                "joint_relevance": 1.0 / rank,
                "fused_score": 1.0 / rank,
                "fused_score_ratio": 1.0 / rank,
                "fused_rank": rank,
                "prefilter_rank": rank,
                "inside_adaptive_prefix": rank <= config.EAES_RERANK_LIMIT,
                "adaptive_k": min(
                    config.EAES_RERANK_LIMIT, len(self.candidates)
                ),
            })
        adaptive = children[:config.EAES_RERANK_LIMIT]
        parents = []
        if config.SEMANTIC_HIERARCHY:
            parents = [{
                "parent_id": f"1-{i}",
                "rewrite_content": f"Parent memory {i} about Caroline and dogs.",
                "joint_relevance": 1.0 / i,
                "fused_score": 1.0 / i,
                "fused_score_ratio": 1.0 / i,
                "fused_rank": i,
                "rank": i,
                "score": 1.0 / (10 + i),
                "inside_adaptive_prefix": True,
                "adaptive_k": 4,
            } for i in range(1, 5)]
        return {
            "child_probe_candidates": children,
            "parent_probe_candidates": parents,
            "selected_child_candidates": children,
            "selected_parent_candidates": parents,
            "adaptive_children": adaptive,
            "adaptive_parents": parents,
            "child_rankings": [],
            "parent_rankings": [],
            "phrase_retrieval": {
                "phrases": [],
                "child_selected_phrase_indices": [0],
                "parent_selected_phrase_indices": [0],
            },
        }

class _AblationAgent(EAESMixin):
    def __init__(self, candidates, reader_answer="test answer"):
        self.llm = _FakeLLM(reader_answer)
        self.memory = _FakeMemory()
        self.memory_controller = _FakeController(candidates)
        self.selector_calls = 0
        self.selector_query_plans = []
        self.rollback_calls = 0

    @staticmethod
    def parse_eaes_query(_question, _question_emb):
        return {
            "answer_type": "fact",
            "query_attributes": ["profile.pet: pet owned by Caroline"],
            "keywords": ["dog"],
            "retrieval_phrases": [
                "Caroline possession.pet ownership",
                "Caroline possession.owned animal",
                "Caroline profile.animal companion",
                "Caroline possession.dog ownership",
                "Caroline pet ownership",
                "animal companion owned by Caroline",
            ],
            "breadth_value": 0.5,
            "detail_value": 0.5,
        }

    @staticmethod
    def _as_list(value):
        return value if isinstance(value, list) else []

    @staticmethod
    def rerank_eaes_phrase_candidates(_question, candidates, top_k=15):
        return list(candidates)[:top_k]

    def select_eaes_evidence(self, _question, query_plan, candidates):
        self.selector_calls += 1
        self.selector_query_plans.append(dict(query_plan))
        return self._fallback_eaes_package(candidates[:1], reason="selector enabled")

    def apply_eaes_rollback_check(
            self, _question, _query_plan, candidates, parents, _question_emb
    ):
        self.rollback_calls += 1
        return candidates, parents, {"enabled": True}


def _candidates(count=20):
    return [
        {
            "memory_id": f"M_{i}",
            "event_id": f"D1:{i}-1",
            "origin": f"D1:{i}",
            "rewrite_content": f"Memory {i}",
            "conversation_time": "2023-05-08",
            "score": 1.0 / i,
        }
        for i in range(1, count + 1)
    ]


class EvidenceSelectorAblationTests(unittest.TestCase):
    def test_disabled_selector_passes_every_reranked_candidate_to_reader(self):
        agent = _AblationAgent(_candidates())

        with (
            patch.object(config, "DISABLE_EVIDENCE_SELECTOR", True),
            patch.object(config, "SEMANTIC_HIERARCHY", False),
            patch.object(config, "EAES_ROLLBACK_CHECK", False),
            patch.object(config, "EAES_RERANK_LIMIT", 16),
        ):
            answer, prediction_context = agent.answer_question_eaes(
                "question", category=1
            )

        self.assertEqual(answer, "test answer")
        self.assertEqual(
            prediction_context,
            [f"D1:{i}" for i in range(1, 17)],
        )
        self.assertEqual(agent.selector_calls, 0)
        package = agent.llm.inputs[0]["evidence_package"]
        self.assertNotIn("backup_candidates", agent.llm.inputs[0])
        self.assertEqual(len(package["answer_items"]), 16)
        self.assertEqual(
            [item["evidence"][0]["memory_id"] for item in package["answer_items"]],
            [f"M_{i}" for i in range(1, 17)],
        )
        self.assertTrue(all(
            set(item["evidence"][0]) == {
                "memory_id", "conversation_time", "rewrite_content",
                "retrieval_score", "rerank_rank",
            }
            for item in package["answer_items"]
        ))
        self.assertTrue(all(set(item) == {"evidence"} for item in package["answer_items"]))

    def test_default_hierarchy_budget_is_sixteen_children_plus_four_parents(self):
        agent = _AblationAgent(_candidates(30))

        with (
            patch.object(config, "DISABLE_EVIDENCE_SELECTOR", True),
            patch.object(config, "SEMANTIC_HIERARCHY", True),
            patch.object(config, "EAES_ROLLBACK_CHECK", False),
            patch.object(config, "EAES_RERANK_LIMIT", 16),
            patch.object(config, "PARENT_TOP_K", 4),
        ):
            _, prediction_context = agent.answer_question_eaes(
                "question", category=4
            )

        reader_input = agent.llm.inputs[0]
        self.assertEqual(len(reader_input["evidence_package"]["answer_items"]), 16)
        self.assertEqual(len(reader_input["parent_memories"]), 4)
        self.assertEqual(
            len(reader_input["evidence_package"]["answer_items"])
            + len(reader_input["parent_memories"]),
            20,
        )
        self.assertEqual(len(prediction_context), 20)
        self.assertEqual(
            prediction_context[:16],
            [f"D1:{i}" for i in range(1, 17)],
        )
        self.assertEqual(prediction_context[16], "D2:1,D2:11")

    def test_enabled_selector_keeps_existing_path(self):
        agent = _AblationAgent(_candidates())

        with (
            patch.object(config, "DISABLE_EVIDENCE_SELECTOR", False),
            patch.object(config, "SEMANTIC_HIERARCHY", False),
            patch.object(config, "EAES_ROLLBACK_CHECK", False),
        ):
            agent.answer_question_eaes("question", category=1)

        self.assertEqual(agent.selector_calls, 1)
        reader_input = agent.llm.inputs[0]
        package = reader_input["evidence_package"]
        self.assertEqual(len(package["answer_items"]), 1)
        self.assertEqual(
            [item["memory_id"] for item in reader_input["backup_candidates"]],
            [f"M_{i}" for i in range(1, 13)],
        )
        self.assertTrue(all(
            set(item) == {
                "memory_id", "conversation_time", "rewrite_content",
                "retrieval_score", "rerank_rank",
            }
            for item in reader_input["backup_candidates"]
        ))

    def test_parent_and_child_use_the_same_phrase_pool(self):
        agent = _AblationAgent(_candidates())

        with (
            patch.object(config, "DISABLE_EVIDENCE_SELECTOR", False),
            patch.object(config, "SEMANTIC_HIERARCHY", True),
            patch.object(config, "EAES_ROLLBACK_CHECK", False),
        ):
            agent.answer_question_eaes("question", category=1)

        self.assertEqual(
            agent.memory_controller.child_retrieval_phrases[0],
            [
                "Caroline possession.pet ownership",
                "Caroline possession.owned animal",
                "Caroline profile.animal companion",
                "Caroline possession.dog ownership",
                "Caroline pet ownership",
                "animal companion owned by Caroline",
            ],
        )
        self.assertNotIn("keywords", agent.selector_query_plans[0])
        self.assertNotIn("retrieval_phrases", agent.selector_query_plans[0])
        self.assertNotIn("keywords", agent.llm.inputs[0]["query_plan"])
        self.assertNotIn(
            "matched_keyword", agent.llm.inputs[0]["parent_memories"][0]
        )

    def test_answer_path_runs_enabled_rollback_check_with_selector_disabled(self):
        agent = _AblationAgent(
            _candidates(30), reader_answer="no information available"
        )

        with (
            patch.object(config, "DISABLE_EVIDENCE_SELECTOR", True),
            patch.object(config, "SEMANTIC_HIERARCHY", True),
            patch.object(config, "EAES_ROLLBACK_CHECK", True),
            patch.object(config, "EAES_RERANK_LIMIT", 16),
            patch.object(config, "PARENT_TOP_K", 4),
        ):
            agent.answer_question_eaes("question", category=4)

        self.assertEqual(agent.rollback_calls, 1)
        self.assertEqual(agent.selector_calls, 0)


if __name__ == "__main__":
    unittest.main()
