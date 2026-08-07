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

from agent.retrieval import RetrievalMixin
from common import config


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
        self.child_plans.append(dict(query_plan))
        return _children()

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


class _RetrievalAgent(RetrievalMixin):
    def __init__(self):
        self.memory = _Memory()
        self.memory_controller = _Controller()

    @staticmethod
    def parse_eaes_query(_question, _question_emb=None):
        return {
            "query_attributes": ["profile.pet: pet owned by Caroline"],
            "keywords": ["dog"],
        }

    @staticmethod
    def _eaes_child_query_plan(query_plan):
        child_plan = dict(query_plan)
        child_plan.pop("keywords", None)
        return child_plan

    @staticmethod
    def rerank_eaes_candidates(_question, _query_plan, candidates):
        return list(candidates)[:config.EAES_RERANK_LIMIT]


class RetrievalTopTwentyTests(unittest.TestCase):
    def test_retrieval_only_matches_sixteen_child_four_parent_budget(self):
        agent = _RetrievalAgent()

        with (
            patch.object(config, "EAES_MODE", True),
            patch.object(config, "SEMANTIC_HIERARCHY", True),
            patch.object(config, "EAES_CANDIDATE_LIMIT", 120),
            patch.object(config, "EAES_RERANK_LIMIT", 16),
            patch.object(config, "PARENT_TOP_K", 4),
        ):
            result = agent.retrieve_question_evidence("What pet does Caroline own?")

        self.assertEqual(result["retrieval_k"], 20)
        self.assertEqual(result["child_k"], 16)
        self.assertEqual(result["parent_k"], 4)
        self.assertEqual(len(result["candidates"]), 16)
        self.assertEqual(len(result["parent_candidates"]), 4)
        self.assertEqual(len(result["retrieved_origin_groups"]), 20)
        self.assertEqual(len(result["retrieved_memory_ids"]), 20)
        self.assertEqual(
            agent.memory_controller.parent_plans[0]["keywords"], ["dog"]
        )
        self.assertNotIn("keywords", agent.memory_controller.child_plans[0])
        self.assertIn("D2:1", result["parent_origins"])


if __name__ == "__main__":
    unittest.main()
