import sys
import types
import unittest
from unittest.mock import patch


try:
    import numpy  # noqa: F401
except ImportError:
    numpy_module = types.ModuleType("numpy")
    numpy_module.vstack = lambda values: values
    sys.modules["numpy"] = numpy_module

dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_module)

utils_module = types.ModuleType("common.utils")
utils_module.topk_answers_by_similarity = lambda *_args, **_kwargs: ([], [], [], [])
sys.modules["common.utils"] = utils_module

from agent.eaes import EAESMixin
from agent.retrieval import RetrievalMixin
from common import config
from eval.retrieval_metrics import retrieval_metrics


def _children(count=24):
    return [{
        "memory_id": f"M_{index}",
        "event_id": f"D1:{index}-1",
        "origin": f"D1:{index}",
        "rewrite_content": f"Child memory {index}",
        "rank": index,
    } for index in range(1, count + 1)]


class _Memory:
    episode_events = {}

    @staticmethod
    def get_eaes_support_origin(parent_ids):
        parent_id = parent_ids[0]
        index = int(parent_id.rsplit("t", 1)[1])
        return [parent_id, f"D2:{index}", f"D2:{index + 10}"]


class _Controller:
    def __init__(self):
        self.child_plans = []
        self.parent_plans = []

    def retrieve_eaes_candidates(self, query_plan, *_args, **_kwargs):
        return _children()

    def retrieve_eaes_phrase_candidates(
            self, retrieval_phrases, include_diagnostics=False, **_kwargs
    ):
        self.child_plans.append(list(retrieval_phrases))
        candidates = _children()
        diagnostics = {
            "phrase_top15": [],
            "phrase_final_top10": [],
            "fused_candidate_ids": [item["memory_id"] for item in candidates],
            "rrf_scores": {},
        }
        return (candidates, diagnostics) if include_diagnostics else candidates

    def retrieve_eaes_parent_candidates(self, query_plan, *_args, **kwargs):
        self.parent_plans.append(dict(query_plan))
        limit = kwargs.get("limit", 4)
        return [{
            "parent_id": f"D1:t{index}",
            "rewrite_content": f"Parent memory {index}",
            "rank": index,
            "score": 1.0 / index,
            "matched_keyword": "dog",
        } for index in range(1, limit + 1)]


class _RetrievalAgent(EAESMixin, RetrievalMixin):
    def __init__(self):
        self.memory = _Memory()
        self.memory_controller = _Controller()
        self.rollback_calls = 0
        self.retained_rollback_child = None
        self.reader_answer = "no information available"
        self.reader_calls = 0

    @staticmethod
    def parse_eaes_query(_question, _question_emb=None):
        return {
            "query_attributes": ["profile.pet: pet owned by Caroline"],
            "keywords": ["dog"],
            "retrieval_phrases": [
                "Caroline pet", "pet ownership",
                "Caroline animal", "Caroline companion",
            ],
        }

    @staticmethod
    def _eaes_child_query_plan(query_plan):
        child_plan = dict(query_plan)
        child_plan.pop("keywords", None)
        child_plan.pop("retrieval_phrases", None)
        return child_plan

    @staticmethod
    def rerank_eaes_phrase_candidates(_question, candidates, top_k=15):
        return list(candidates)[:top_k]

    def _read_eaes_candidates(
            self, _question, _child_query_plan, _candidates, _parents,
            _category=0, _lm_current_date=None
    ):
        self.reader_calls += 1
        return self.reader_answer, [], self.reader_answer

    def apply_eaes_rollback_check(
            self, _question, _query_plan, candidates, parents, _question_emb
    ):
        self.rollback_calls += 1
        if self.retained_rollback_child is not None:
            candidates = list(candidates)
            candidates[-1] = self.retained_rollback_child
        return candidates, parents, {"enabled": True}


class RetrievalTopTwentyTests(unittest.TestCase):
    def test_retrieval_only_matches_fifteen_child_four_parent_budget(self):
        agent = _RetrievalAgent()

        with (
            patch.object(config, "EAES_MODE", True),
            patch.object(config, "SEMANTIC_HIERARCHY", True),
            patch.object(config, "EAES_CANDIDATE_LIMIT", 120),
            patch.object(config, "EAES_PHRASE_RERANK_LIMIT", 15),
            patch.object(config, "PARENT_TOP_K", 4),
        ):
            result = agent.retrieve_question_evidence("What pet does Caroline own?")

        self.assertEqual(result["retrieval_k"], 19)
        self.assertEqual(result["child_k"], 15)
        self.assertEqual(result["parent_k"], 4)
        self.assertEqual(len(result["candidates"]), 15)
        self.assertEqual(len(result["parent_candidates"]), 4)
        self.assertEqual(len(result["retrieved_origin_groups"]), 19)
        self.assertEqual(len(result["retrieved_memory_ids"]), 19)
        self.assertEqual(
            agent.memory_controller.parent_plans[0]["keywords"], ["dog"]
        )
        self.assertEqual(len(agent.memory_controller.child_plans[0]), 4)
        self.assertIn("D2:1", result["parent_origins"])

    def test_retrieval_only_runs_enabled_rollback_check(self):
        agent = _RetrievalAgent()

        with (
            patch.object(config, "EAES_MODE", True),
            patch.object(config, "SEMANTIC_HIERARCHY", True),
            patch.object(config, "EAES_ROLLBACK_CHECK", True),
            patch.object(config, "EAES_CANDIDATE_LIMIT", 120),
            patch.object(config, "EAES_PHRASE_RERANK_LIMIT", 15),
            patch.object(config, "PARENT_TOP_K", 4),
        ):
            result = agent.retrieve_question_evidence("What pet does Caroline own?")

        self.assertEqual(agent.rollback_calls, 1)
        self.assertEqual(agent.reader_calls, 1)
        self.assertTrue(result["rollback_check"]["enabled"])
        self.assertNotIn("applied", result["rollback_check"])
        self.assertTrue(
            result["rollback_check"]["reader_gate"][
                "returned_no_information_available"
            ]
        )
        self.assertEqual(
            len(result["rollback_check"]["first_prefilter"]["child_ids"]), 24
        )
        self.assertEqual(
            len(result["rollback_check"]["first_prefilter"]["parent_ids"]), 4
        )

    def test_retrieval_only_discards_normal_internal_answer_and_skips_rollback(self):
        agent = _RetrievalAgent()
        agent.reader_answer = "Caroline owns a dog."

        with (
            patch.object(config, "EAES_MODE", True),
            patch.object(config, "SEMANTIC_HIERARCHY", True),
            patch.object(config, "EAES_ROLLBACK_CHECK", True),
            patch.object(config, "EAES_CANDIDATE_LIMIT", 120),
            patch.object(config, "EAES_PHRASE_RERANK_LIMIT", 15),
            patch.object(config, "PARENT_TOP_K", 4),
        ):
            result = agent.retrieve_question_evidence(
                "What pet does Caroline own?"
            )

        self.assertEqual(agent.reader_calls, 1)
        self.assertEqual(agent.rollback_calls, 0)
        self.assertFalse(
            result["rollback_check"]["reader_gate"][
                "returned_no_information_available"
            ]
        )
        self.assertNotIn("answer", result)
        self.assertNotIn("prediction", result)

    def test_retained_rollback_node_contributes_to_final_hit_and_mrr(self):
        agent = _RetrievalAgent()
        agent.retained_rollback_child = {
            "memory_id": "M_ROLLBACK_HIT",
            "event_id": "D9:99-1",
            "origin": "D9:99",
            "rewrite_content": "Rollback evidence retained in final Top20.",
            "rank": 15,
        }

        with (
            patch.object(config, "EAES_MODE", True),
            patch.object(config, "SEMANTIC_HIERARCHY", True),
            patch.object(config, "EAES_ROLLBACK_CHECK", True),
            patch.object(config, "EAES_CANDIDATE_LIMIT", 120),
            patch.object(config, "EAES_PHRASE_RERANK_LIMIT", 15),
            patch.object(config, "PARENT_TOP_K", 4),
        ):
            result = agent.retrieve_question_evidence("What happened?")

        metrics = retrieval_metrics(
            ["D9:99"],
            result["retrieved_origins"],
            result["retrieved_origin_groups"],
        )
        self.assertEqual(result["retrieval_k"], 19)
        self.assertEqual(len(result["retrieved_origin_groups"]), 19)
        self.assertEqual(result["retrieved_origin_groups"][14], ["D9:99"])
        self.assertEqual(metrics["hit"], 1)
        self.assertEqual(metrics["mrr"], 1 / 15)


if __name__ == "__main__":
    unittest.main()
