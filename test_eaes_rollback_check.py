import json
import sys
import types
import unittest
from unittest.mock import patch


dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_module)

from agent.eaes import EAESMixin
from common import config


def _children(prefix, count):
    return [
        {
            "memory_id": f"{prefix}{index}",
            "rewrite_content": f"Child memory {prefix}{index}",
            "conversation_time": "2023-05-08",
            "attribute_paths": [f"profile.fact.{index}"],
            "event_lifecycle": "current",
            "rank": index,
            "score": 1.0 / index,
        }
        for index in range(1, count + 1)
    ]


def _parents(prefix, count):
    return [
        {
            "parent_id": f"{prefix}{index}",
            "rewrite_content": f"Parent memory {prefix}{index}",
            "rank": index,
            "score": 1.0 / index,
        }
        for index in range(1, count + 1)
    ]


class _RollbackLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.inputs = []

    def chat_text(self, messages, **_kwargs):
        self.inputs.append(json.loads(messages[-1]["content"]))
        return self.responses.pop(0)


class _RollbackController:
    def __init__(self):
        self.child_calls = []
        self.parent_calls = []

    def retrieve_eaes_candidates(self, query_plan, _question_emb, **kwargs):
        self.child_calls.append((dict(query_plan), kwargs))
        return _children("RC", 27)

    def retrieve_eaes_parent_candidates(self, query_plan, _question_emb, **kwargs):
        self.parent_calls.append((dict(query_plan), kwargs))
        return _parents("RP", 3)


class _RollbackAgent(EAESMixin):
    def __init__(self, responses):
        self.llm = _RollbackLLM(responses)
        self.memory_controller = _RollbackController()

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        return value if isinstance(value, list) else [value]


def _rollback_plan():
    return {
        "entities": ["Caroline"],
        "query_attributes": ["profile.pet: animal owned by Caroline"],
        "answer_type": "fact",
        "temporal_intent": "none",
        "required_lifecycle": "unknown",
        "keywords": ["animal companion"],
    }


class EAESRollbackCheckTests(unittest.TestCase):
    def test_excludes_first_top20_and_preserves_sixteen_plus_four(self):
        initial_children = _children("C", 16)
        initial_parents = _parents("P", 4)
        final_child_ids = ["RC1", "RC2"] + [
            f"C{index}" for index in range(1, 15)
        ]
        final_parent_ids = ["RP1", "P1", "P2", "P3"]
        agent = _RollbackAgent([
            _rollback_plan(),
            {
                "ranked_nodes": [
                    {"node_type": "child", "node_id": "RC1"},
                    {"node_type": "parent", "node_id": "RP1"},
                    {"node_type": "child", "node_id": "RC2"},
                ]
            },
            {
                "ranked_child_ids": final_child_ids,
                "ranked_parent_ids": final_parent_ids,
            },
        ])

        with (
            patch.object(config, "EAES_SEMANTIC_SCORE", False),
            patch.object(config, "EAES_RERANK_LIMIT", 16),
            patch.object(config, "PARENT_TOP_K", 4),
            patch.object(config, "EAES_ROLLBACK_CHILD_PREFILTER_LIMIT", 27),
            patch.object(config, "EAES_ROLLBACK_PARENT_PREFILTER_LIMIT", 3),
            patch.object(config, "EAES_ROLLBACK_SUPPLEMENT_LIMIT", 3),
        ):
            children, parents, metadata = agent.apply_eaes_rollback_check(
                "What pet does Caroline own?",
                {
                    "entities": ["Caroline"],
                    "query_attributes": ["profile.pet: pet owned by Caroline"],
                    "keywords": ["pet"],
                },
                initial_children,
                initial_parents,
                question_emb="question-embedding",
            )

        self.assertEqual([item["memory_id"] for item in children], final_child_ids)
        self.assertEqual([item["parent_id"] for item in parents], final_parent_ids)
        self.assertEqual(len(children), 16)
        self.assertEqual(len(parents), 4)
        self.assertNotIn("applied", metadata)
        self.assertEqual(metadata["first_query_plan"]["keywords"], ["pet"])
        self.assertEqual(
            metadata["rollback_query_plan"]["keywords"], ["animal companion"]
        )
        self.assertEqual(
            metadata["selected_supplements"]["child_ids"], ["RC1", "RC2"]
        )
        self.assertEqual(
            metadata["selected_supplements"]["parent_ids"], ["RP1"]
        )
        self.assertEqual(len(
            metadata["rollback_prefilter"]["child_candidates"]
        ), 27)
        self.assertEqual(len(
            metadata["rollback_prefilter"]["parent_candidates"]
        ), 3)
        self.assertEqual(metadata["final"]["child_ids"], final_child_ids)
        self.assertEqual(metadata["final"]["parent_ids"], final_parent_ids)

        child_plan, child_kwargs = agent.memory_controller.child_calls[0]
        parent_plan, parent_kwargs = agent.memory_controller.parent_calls[0]
        self.assertNotIn("keywords", child_plan)
        self.assertEqual(parent_plan["keywords"], ["animal companion"])
        self.assertEqual(child_kwargs["limit"], 27)
        self.assertEqual(parent_kwargs["limit"], 3)
        self.assertEqual(
            child_kwargs["exclude_memory_ids"],
            {f"C{index}" for index in range(1, 17)},
        )
        self.assertEqual(
            parent_kwargs["exclude_parent_ids"],
            {f"P{index}" for index in range(1, 5)},
        )

        planner_input = agent.llm.inputs[0]
        self.assertNotIn("current_top_memories", planner_input)
        self.assertEqual(len(planner_input["current_top_rewrite_contents"]), 20)
        self.assertTrue(all(
            isinstance(value, str)
            for value in planner_input["current_top_rewrite_contents"]
        ))
        supplement_input = agent.llm.inputs[1]
        self.assertEqual(len(supplement_input["child_candidates"]), 27)
        self.assertEqual(len(supplement_input["parent_candidates"]), 3)
        final_input = agent.llm.inputs[2]
        self.assertEqual(len(final_input["child_candidates"]), 18)
        self.assertEqual(len(final_input["parent_candidates"]), 5)
        self.assertNotIn("first_pass_child_candidates", final_input)
        self.assertNotIn("supplemental_child_candidates", final_input)

    def test_invalid_query_plan_keeps_first_pass_top20(self):
        initial_children = _children("C", 16)
        initial_parents = _parents("P", 4)
        agent = _RollbackAgent(["invalid plan"])

        with patch.object(config, "EAES_SEMANTIC_SCORE", False):
            children, parents, metadata = agent.apply_eaes_rollback_check(
                "What pet does Caroline own?",
                {"query_attributes": ["profile.pet"], "keywords": ["pet"]},
                initial_children,
                initial_parents,
            )

        self.assertIs(children, initial_children)
        self.assertIs(parents, initial_parents)
        self.assertNotIn("applied", metadata)
        self.assertEqual(metadata["failure_reason"], "invalid_rollback_query_plan")
        self.assertEqual(agent.memory_controller.child_calls, [])
        self.assertEqual(agent.memory_controller.parent_calls, [])

    def test_incomplete_final_rerank_keeps_first_pass_top20(self):
        initial_children = _children("C", 16)
        initial_parents = _parents("P", 4)
        agent = _RollbackAgent([
            _rollback_plan(),
            {
                "ranked_nodes": [
                    {"node_type": "child", "node_id": "RC1"},
                    {"node_type": "child", "node_id": "RC2"},
                    {"node_type": "parent", "node_id": "RP1"},
                ]
            },
            {
                "ranked_child_ids": ["RC1"],
                "ranked_parent_ids": ["RP1"],
            },
        ])

        with (
            patch.object(config, "EAES_SEMANTIC_SCORE", False),
            patch.object(config, "EAES_RERANK_LIMIT", 16),
            patch.object(config, "PARENT_TOP_K", 4),
        ):
            children, parents, metadata = agent.apply_eaes_rollback_check(
                "What pet does Caroline own?",
                {"query_attributes": ["profile.pet"], "keywords": ["pet"]},
                initial_children,
                initial_parents,
            )

        self.assertEqual(children, initial_children)
        self.assertEqual(parents, initial_parents)
        self.assertNotIn("applied", metadata)
        self.assertEqual(
            metadata["final"]["child_ids"],
            [f"C{index}" for index in range(1, 17)],
        )
        self.assertEqual(
            metadata["final"]["parent_ids"],
            [f"P{index}" for index in range(1, 5)],
        )


if __name__ == "__main__":
    unittest.main()
