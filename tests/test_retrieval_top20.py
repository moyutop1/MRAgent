import json
import re
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
from agent.retrieval import RetrievalMixin, compact_eaes_retrieval
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
        index = int(parent_id.split("-", 1)[1])
        return [parent_id, f"D2:{index}", f"D2:{index + 10}"]

    @staticmethod
    def get_eaes_note(_memory_id):
        return None


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
            "phrases": [{"selected_k": len(candidates)}] * 4,
            "prefilter_candidate_ids": [item["memory_id"] for item in candidates],
        }
        return (candidates, diagnostics) if include_diagnostics else candidates

    def route_eaes_parent_candidates(self, query_plan, *_args, **kwargs):
        self.parent_plans.append(dict(query_plan))
        parents = [{
            "parent_id": f"1-{index}",
            "rewrite_content": f"Parent memory {index}",
            "rank": index,
            "score": 1.0 / index,
            "posterior_score": 1.0 / index,
            "matched_keyword": "dog",
        } for index in range(1, 5)]
        return parents, {
            "breadth_value": 0.5,
            "detail_value": 0.5,
            "parent_candidates": parents,
        }

class _RetrievalAgent(EAESMixin, RetrievalMixin):
    def __init__(self):
        self.memory = _Memory()
        self.memory_controller = _Controller()
        self.rollback_calls = 0
        self.retained_rollback_child = None
        self.retained_rollback_parent = None
        self.reader_answer = "no information available"
        self.reader_calls = 0

    @staticmethod
    def parse_eaes_query(_question, _question_emb=None):
        return {
            "query_attributes": ["profile.pet: pet owned by Caroline"],
            "keywords": ["dog"],
            "retrieval_phrases": [
                "Caroline possession.pet ownership",
                "Caroline possession.owned animal",
                "Caroline profile.animal companion",
                "Caroline possession.dog ownership",
            ],
            "breadth_value": 0.5,
            "detail_value": 0.5,
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
            candidates.append(self.retained_rollback_child)
        if self.retained_rollback_parent is not None:
            parents = list(parents)
            parents.append(self.retained_rollback_parent)
        return candidates, parents, {
            "enabled": True,
            "terminal_reason": "no_need_more",
            "rollback_count": int(
                self.retained_rollback_child is not None
                or self.retained_rollback_parent is not None
            ),
        }


class _RollbackLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.inputs = []

    def chat_text(self, messages, **_kwargs):
        self.inputs.append(json.loads(messages[-1]["content"]))
        return self.responses.pop(0)


class _RealRollbackController(_Controller):
    def retrieve_eaes_rollback_children(
            self, *, exclude_memory_ids=None, **_kwargs
    ):
        candidate = {
            "memory_id": "M_REAL_ROLLBACK",
            "event_id": "D9:99-1",
            "origin": "D9:99",
            "rewrite_content": "Caroline adopted a cat.",
            "score": 3.4,
            "matched_query_phase": "Caroline adopted pet",
        }
        return [] if candidate["memory_id"] in set(
            exclude_memory_ids or []
        ) else [candidate]

    @staticmethod
    def retrieve_eaes_rollback_parents(**_kwargs):
        return []


class _RealRollbackRetrievalAgent(_RetrievalAgent):
    apply_eaes_rollback_check = EAESMixin.apply_eaes_rollback_check

    def __init__(self, responses):
        super().__init__()
        self.llm = _RollbackLLM(responses)
        self.memory_controller = _RealRollbackController()

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        return value if isinstance(value, list) else [value]


class RetrievalTopTwentyTests(unittest.TestCase):
    def test_retrieval_only_matches_fifteen_child_four_parent_budget(self):
        agent = _RetrievalAgent()

        with (
            patch.object(config, "EAES_MODE", True),
            patch.object(config, "SEMANTIC_HIERARCHY", True),
            patch.object(config, "EAES_ROLLBACK_CHECK", False),
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
        self.assertEqual(set(result["stage_origins"]), {
            "prefilter_child", "initial_child", "final_child",
            "selected_parent", "final_combined",
        })
        self.assertEqual(result["counts"]["prefilter_child_k"], 24)
        self.assertEqual(result["counts"]["initial_child_k"], 24)
        self.assertNotIn("global_child", result["stage_origins"])

    def test_prefilter_initial_final_names_keep_existing_stage_logic(self):
        agent = _RetrievalAgent()

        with (
            patch.object(config, "EAES_MODE", True),
            patch.object(config, "SEMANTIC_HIERARCHY", True),
            patch.object(config, "EAES_ROLLBACK_CHECK", False),
            patch.object(config, "EAES_PHRASE_UNION_LIMIT", 10),
            patch.object(config, "EAES_PHRASE_RERANK_LIMIT", 5),
        ):
            result = agent.retrieve_question_evidence(
                "What pet does Caroline own?"
            )

        self.assertEqual(len(result["prefilter_candidates"]), 24)
        self.assertEqual(len(result["initial_candidates"]), 10)
        self.assertEqual(len(result["candidates"]), 5)
        self.assertEqual(result["counts"]["prefilter_child_k"], 24)
        self.assertEqual(result["counts"]["initial_child_k"], 10)
        self.assertEqual(result["counts"]["final_child_k"], 5)

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
        self.assertEqual(agent.reader_calls, 0)
        self.assertTrue(result["rollback_check"]["enabled"])
        self.assertNotIn("applied", result["rollback_check"])
        self.assertNotIn("reader_gate", result["rollback_check"])
        self.assertEqual(
            len(result["rollback_check"]["initial_retrieval_pool"]["child_ids"]), 24
        )
        self.assertEqual(
            len(result["rollback_check"]["initial_retrieval_pool"]["parent_ids"]), 4
        )

    def test_retrieval_only_never_calls_reader_and_still_runs_rollback(self):
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

        self.assertEqual(agent.reader_calls, 0)
        self.assertEqual(agent.rollback_calls, 1)
        self.assertNotIn("reader_gate", result["rollback_check"])
        self.assertNotIn("answer", result)
        self.assertNotIn("prediction", result)

    def test_retrieval_only_runs_real_two_stage_s2g_loop_without_reader(self):
        agent = _RealRollbackRetrievalAgent([
            {
                "state": "need_more",
                "semantic_properties": [],
                "query_phase": [
                    "Caroline adopted pet",
                    "Caroline gained animal",
                    "pet adoption event",
                    "animal joined Caroline",
                ],
            },
            {
                "ranked_nodes": [{
                    "node_type": "child",
                    "node_id": "M_REAL_ROLLBACK",
                }],
            },
            {
                "state": "no_need_more",
                "semantic_properties": [],
                "query_phase": [],
            },
        ])

        with (
            patch.object(config, "EAES_MODE", True),
            patch.object(config, "SEMANTIC_HIERARCHY", True),
            patch.object(config, "EAES_ROLLBACK_CHECK", True),
            patch.object(config, "EAES_PHRASE_RERANK_LIMIT", 15),
            patch.object(config, "PARENT_TOP_K", 4),
        ):
            result = agent.retrieve_question_evidence(
                "What pet did Caroline adopt?"
            )

        self.assertEqual(agent.reader_calls, 0)
        self.assertEqual(len(agent.llm.inputs), 3)
        self.assertEqual(result["rollback_check"]["rollback_count"], 1)
        self.assertEqual(
            result["rollback_check"]["terminal_reason"], "no_need_more"
        )
        self.assertEqual(result["child_k"], 16)
        self.assertEqual(
            result["candidates"][-1]["memory_id"], "M_REAL_ROLLBACK"
        )

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
        self.assertEqual(result["retrieval_k"], 20)
        self.assertEqual(len(result["retrieved_origin_groups"]), 20)
        self.assertEqual(result["retrieved_origin_groups"][15], ["D9:99"])
        self.assertEqual(metrics["hit"], 1)
        self.assertEqual(metrics["mrr"], 1 / 16)

    def test_compact_retrieval_schema_removes_deprecated_and_duplicate_fields(self):
        agent = _RetrievalAgent()
        with (
            patch.object(config, "EAES_MODE", True),
            patch.object(config, "SEMANTIC_HIERARCHY", True),
            patch.object(config, "EAES_ROLLBACK_CHECK", False),
            patch.object(config, "EAES_PHRASE_RERANK_LIMIT", 15),
        ):
            internal = agent.retrieve_question_evidence("What pet does Caroline own?")
        internal["prefilter_candidates"][0].update({
            "temporal_intent": "deprecated",
            "required_lifecycle": "deprecated",
            "event_lifecycle": "historical",
            "entities": ["duplicate index field"],
        })
        compact = compact_eaes_retrieval(internal)
        serialized = json.dumps(compact)

        self.assertEqual(set(compact), {
            "mode", "query_plan", "routing", "phrase_retrieval",
            "parent_candidates", "child_candidates",
            "final_child_candidates", "final_parent_candidates",
            "final_child_ids", "final_parent_ids", "counts",
            "rollback_check",
        })
        self.assertNotIn("retrieved_origins", compact)
        self.assertNotIn("prefilter_candidates", compact)
        self.assertNotIn("candidates", compact)
        for deprecated in (
                "temporal_intent", "required_lifecycle",
                "event_lifecycle", "entities"):
            self.assertNotIn(deprecated, serialized)
        self.assertEqual(len(compact["child_candidates"]), 24)
        self.assertEqual(len(compact["final_child_candidates"]), 15)
        self.assertEqual(len(compact["final_parent_candidates"]), 4)
        self.assertEqual(len(compact["final_child_ids"]), 15)
        self.assertEqual(len(compact["final_parent_ids"]), 4)
        self.assertTrue(all(
            re.fullmatch(r"\d+-\d+", item["parent_id"])
            for item in compact["parent_candidates"]
        ))

    def test_compact_schema_preserves_rollback_supplement_contents(self):
        agent = _RetrievalAgent()
        agent.retained_rollback_child = {
            "memory_id": "M_ROLLBACK",
            "event_id": "D9:99-1",
            "origin": "D9:99",
            "rewrite_content": "Caroline adopted a cat.",
            "score": 3.4,
            "matched_query_phase": "Caroline adopted pet",
        }
        agent.retained_rollback_parent = {
            "parent_id": "2-99",
            "rewrite_content": "Caroline's pet adoption history.",
            "score": 0.8,
            "matched_query_phase": "pet adoption history",
        }

        with (
            patch.object(config, "EAES_MODE", True),
            patch.object(config, "SEMANTIC_HIERARCHY", True),
            patch.object(config, "EAES_ROLLBACK_CHECK", True),
            patch.object(config, "EAES_PHRASE_RERANK_LIMIT", 15),
            patch.object(config, "PARENT_TOP_K", 4),
        ):
            internal = agent.retrieve_question_evidence(
                "What pet does Caroline own?"
            )
        compact = compact_eaes_retrieval(internal)

        self.assertNotIn(
            "M_ROLLBACK",
            [item["memory_id"] for item in compact["child_candidates"]],
        )
        final_child = next(
            item for item in compact["final_child_candidates"]
            if item["memory_id"] == "M_ROLLBACK"
        )
        final_parent = next(
            item for item in compact["final_parent_candidates"]
            if item["parent_id"] == "2-99"
        )
        self.assertEqual(
            final_child["rewrite_content"], "Caroline adopted a cat."
        )
        self.assertEqual(final_child["score"], 3.4)
        self.assertEqual(
            final_parent["rewrite_content"],
            "Caroline's pet adoption history.",
        )
        self.assertIn("M_ROLLBACK", compact["final_child_ids"])
        self.assertIn("2-99", compact["final_parent_ids"])


if __name__ == "__main__":
    unittest.main()
