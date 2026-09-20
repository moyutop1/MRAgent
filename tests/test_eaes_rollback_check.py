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


def _decision(state="need_more", semantic_properties=None, query_phase=None):
    if state == "no_need_more":
        return {
            "state": state,
            "semantic_properties": [],
            "query_phase": [],
        }
    return {
        "state": state,
        "semantic_properties": (
            ["event_action", "episodic"]
            if semantic_properties is None else semantic_properties
        ),
        "query_phase": query_phase or [
            "Caroline adopted pet",
            "Caroline gained animal",
            "pet adoption event",
            "animal joined Caroline",
        ],
    }


def _query_plan():
    return {
        "entities": ["Caroline"],
        "query_attributes": ["profile.pet: animal owned by Caroline"],
        "keywords": ["animal companion"],
        "retrieval_phrases": [
            "Caroline pet",
            "owned animal",
            "animal companion",
            "pet identity",
        ],
    }


class _RollbackLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.inputs = []
        self.system_prompts = []

    def chat_text(self, messages, **_kwargs):
        self.system_prompts.append(messages[0]["content"])
        self.inputs.append(json.loads(messages[-1]["content"]))
        return self.responses.pop(0)


class _RollbackController:
    def __init__(self):
        self.child_calls = []
        self.parent_calls = []

    def retrieve_eaes_rollback_children(self, **kwargs):
        call = {
            **kwargs,
            "exclude_memory_ids": set(kwargs.get("exclude_memory_ids") or []),
        }
        self.child_calls.append(call)
        return _children(f"R{len(self.child_calls)}C", kwargs["limit"])

    def retrieve_eaes_rollback_parents(self, **kwargs):
        call = {
            **kwargs,
            "exclude_parent_ids": set(kwargs.get("exclude_parent_ids") or []),
        }
        self.parent_calls.append(call)
        return _parents(f"R{len(self.parent_calls)}P", kwargs["limit"])


class _RollbackAgent(EAESMixin):
    def __init__(self, responses):
        self.llm = _RollbackLLM(responses)
        self.memory_controller = _RollbackController()

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        return value if isinstance(value, list) else [value]


class _AnswerAgent(_RollbackAgent):
    def __init__(self, responses):
        super().__init__(responses)
        self.reader_inputs = []

    @staticmethod
    def _retrieve_eaes_first_pass(_question, _question_emb=None):
        return {
            "query_plan": _query_plan(),
            "final_children": _children("C", 15),
            "selected_parents": _parents("P", 4),
        }

    def _read_eaes_candidates(
            self, _question, _child_query_plan, candidates, parents,
            _category=0, _lm_current_date=None,
    ):
        self.reader_inputs.append((list(candidates), list(parents)))
        context = [
            candidate["memory_id"] for candidate in candidates
        ] + [parent["parent_id"] for parent in parents]
        return "A cat", context, {"answer": "A cat"}


class _EmptyRollbackController(_RollbackController):
    def retrieve_eaes_rollback_children(self, **kwargs):
        super().retrieve_eaes_rollback_children(**kwargs)
        return []

    def retrieve_eaes_rollback_parents(self, **kwargs):
        super().retrieve_eaes_rollback_parents(**kwargs)
        return []


class _EmptyAnswerAgent(_AnswerAgent):
    def __init__(self, responses):
        super().__init__(responses)
        self.memory_controller = _EmptyRollbackController()

    @staticmethod
    def _retrieve_eaes_first_pass(_question, _question_emb=None):
        return {
            "query_plan": _query_plan(),
            "final_children": [],
            "selected_parents": [],
        }


class EAESRollbackCheckTests(unittest.TestCase):
    def test_validator_accepts_multiple_or_empty_semantic_properties(self):
        parsed = EAESMixin._validate_eaes_rollback_decision(_decision())
        self.assertEqual(
            parsed["semantic_properties"], ["event_action", "episodic"]
        )

        parsed = EAESMixin._validate_eaes_rollback_decision(
            _decision(semantic_properties=[])
        )
        self.assertEqual(parsed["semantic_properties"], [])

    def test_validator_rejects_invalid_state_properties_and_phases(self):
        invalid_outputs = [
            _decision(state="unknown"),
            _decision(semantic_properties=["unknown"]),
            _decision(query_phase=["one", "two", "three"]),
            _decision(query_phase=["same", "Same", "three", "four"]),
            _decision(query_phase=[
                "one two three four", "two", "three", "four"
            ]),
            {
                "state": "no_need_more",
                "semantic_properties": ["durable"],
                "query_phase": [],
            },
        ]
        for output in invalid_outputs:
            with self.subTest(output=output), self.assertRaises(ValueError):
                EAESMixin._validate_eaes_rollback_decision(output)

    def test_invalid_decision_gets_one_error_informed_retry(self):
        invalid = _decision(query_phase=["only one"])
        valid = _decision(state="no_need_more")
        agent = _RollbackAgent([invalid, valid])

        result = agent._request_eaes_rollback_decision(
            "What pet does Caroline own?",
            _children("C", 1),
            _parents("P", 1),
            _query_plan()["retrieval_phrases"],
            [],
            2,
        )

        self.assertEqual(result, valid)
        self.assertEqual(len(agent.llm.inputs), 2)
        self.assertNotIn("validation_error", agent.llm.inputs[0])
        self.assertEqual(
            agent.llm.inputs[1]["previous_invalid_output"], invalid
        )
        self.assertIn("exactly four", agent.llm.inputs[1]["validation_error"])
        self.assertEqual(
            agent.llm.system_prompts[0], agent.llm.system_prompts[1]
        )

    def test_second_invalid_decision_raises(self):
        agent = _RollbackAgent([{"bad": 1}, {"still_bad": 2}])

        with self.assertRaisesRegex(ValueError, "repair attempt"):
            agent._request_eaes_rollback_decision(
                "Question", [], [], [], [], 2
            )

        self.assertEqual(len(agent.llm.inputs), 2)

    def test_first_no_need_more_stops_without_retrieval(self):
        initial_children = _children("C", 15)
        initial_parents = _parents("P", 4)
        agent = _RollbackAgent([_decision(state="no_need_more")])

        children, parents, metadata = agent.apply_eaes_rollback_check(
            "What pet does Caroline own?",
            _query_plan(),
            initial_children,
            initial_parents,
        )

        self.assertEqual(children, initial_children)
        self.assertEqual(parents, initial_parents)
        self.assertEqual(metadata["terminal_reason"], "no_need_more")
        self.assertEqual(metadata["rollback_count"], 0)
        self.assertEqual(agent.memory_controller.child_calls, [])
        self.assertEqual(agent.memory_controller.parent_calls, [])

    def test_second_s2g_sees_selected_supplements_and_both_phase_sets(self):
        first_decision = _decision()
        agent = _RollbackAgent([
            first_decision,
            {
                "ranked_nodes": [
                    {"node_type": "child", "node_id": "R1C1"},
                    {"node_type": "parent", "node_id": "R1P1"},
                    {"node_type": "child", "node_id": "R1C2"},
                ]
            },
            _decision(state="no_need_more"),
        ])

        with (
            patch.object(config, "EAES_ROLLBACK_CHILD_PREFILTER_LIMIT", 27),
            patch.object(config, "EAES_ROLLBACK_PARENT_PREFILTER_LIMIT", 3),
            patch.object(config, "EAES_ROLLBACK_SUPPLEMENT_LIMIT", 3),
        ):
            children, parents, metadata = agent.apply_eaes_rollback_check(
                "What pet does Caroline own?",
                _query_plan(),
                _children("C", 15),
                _parents("P", 4),
                question_emb="question-embedding",
            )

        self.assertEqual(len(children), 17)
        self.assertEqual(len(parents), 5)
        self.assertEqual(metadata["rollback_count"], 1)
        self.assertEqual(metadata["terminal_reason"], "no_need_more")
        second_s2g_input = agent.llm.inputs[2]
        self.assertEqual(agent.llm.inputs[0]["remaining_rollbacks"], 2)
        self.assertEqual(second_s2g_input["remaining_rollbacks"], 1)
        self.assertEqual(len(second_s2g_input["current_evidence"]), 22)
        self.assertTrue(all(
            isinstance(value, str)
            for value in second_s2g_input["current_evidence"]
        ))
        self.assertTrue({
            "Child memory R1C1",
            "Child memory R1C2",
            "Parent memory R1P1",
        }.issubset(second_s2g_input["current_evidence"]))
        self.assertEqual(
            second_s2g_input["initial_query_phase"],
            _query_plan()["retrieval_phrases"],
        )
        self.assertEqual(
            second_s2g_input["rollback_history"][0]["query_phase"],
            first_decision["query_phase"],
        )

    def test_two_rounds_are_additive_and_exclude_only_selected_nodes(self):
        first_decision = _decision()
        second_decision = _decision(
            semantic_properties=[],
            query_phase=[
                "pet name", "animal type", "Caroline companion", "owned pet"
            ],
        )
        agent = _RollbackAgent([
            first_decision,
            {
                "ranked_nodes": [
                    {"node_type": "child", "node_id": "R1C1"},
                    {"node_type": "parent", "node_id": "R1P1"},
                ]
            },
            second_decision,
            {
                "ranked_nodes": [
                    {"node_type": "child", "node_id": "R2C1"},
                ]
            },
        ])

        with (
            patch.object(config, "EAES_ROLLBACK_CHILD_PREFILTER_LIMIT", 27),
            patch.object(config, "EAES_ROLLBACK_PARENT_PREFILTER_LIMIT", 3),
            patch.object(config, "EAES_ROLLBACK_SUPPLEMENT_LIMIT", 3),
        ):
            children, parents, metadata = agent.apply_eaes_rollback_check(
                "What pet does Caroline own?",
                _query_plan(),
                _children("C", 15),
                _parents("P", 4),
                question_emb="question-embedding",
            )

        self.assertEqual(len(agent.llm.inputs), 4)
        self.assertEqual(len(agent.memory_controller.child_calls), 2)
        self.assertEqual(len(agent.memory_controller.parent_calls), 2)
        self.assertEqual(
            agent.memory_controller.child_calls[0]["exclude_memory_ids"],
            {f"C{index}" for index in range(1, 16)},
        )
        self.assertEqual(
            agent.memory_controller.child_calls[1]["exclude_memory_ids"],
            {f"C{index}" for index in range(1, 16)} | {"R1C1"},
        )
        self.assertNotIn(
            "R1C2",
            agent.memory_controller.child_calls[1]["exclude_memory_ids"],
        )
        self.assertEqual(
            agent.memory_controller.parent_calls[1]["exclude_parent_ids"],
            {f"P{index}" for index in range(1, 5)} | {"R1P1"},
        )
        self.assertEqual(
            [item["memory_id"] for item in children[-2:]],
            ["R1C1", "R2C1"],
        )
        self.assertEqual(parents[-1]["parent_id"], "R1P1")
        self.assertEqual(metadata["rollback_count"], 2)
        self.assertEqual(
            metadata["terminal_reason"], "max_rollbacks_completed"
        )
        self.assertEqual(
            metadata["post_second_rollback_sufficiency"], "not_checked"
        )

    def test_two_rounds_can_add_full_three_plus_three_without_truncation(self):
        agent = _RollbackAgent([
            _decision(),
            {
                "ranked_nodes": [
                    {"node_type": "child", "node_id": f"R1C{index}"}
                    for index in range(1, 4)
                ]
            },
            _decision(query_phase=["a", "b", "c", "d"]),
            {
                "ranked_nodes": [
                    {"node_type": "parent", "node_id": f"R2P{index}"}
                    for index in range(1, 4)
                ]
            },
        ])

        children, parents, metadata = agent.apply_eaes_rollback_check(
            "Question",
            _query_plan(),
            _children("C", 15),
            _parents("P", 4),
        )

        self.assertEqual(len(children), 18)
        self.assertEqual(len(parents), 7)
        self.assertEqual(
            [item["memory_id"] for item in children[-3:]],
            ["R1C1", "R1C2", "R1C3"],
        )
        self.assertEqual(
            [item["parent_id"] for item in parents[-3:]],
            ["R2P1", "R2P2", "R2P3"],
        )
        self.assertEqual(metadata["rollback_count"], 2)
        self.assertEqual(
            len(metadata["selected_supplements"]["child_ids"])
            + len(metadata["selected_supplements"]["parent_ids"]),
            6,
        )

    def test_empty_supplements_still_consume_both_rounds(self):
        agent = _RollbackAgent([
            _decision(),
            {"ranked_nodes": []},
            _decision(query_phase=["a", "b", "c", "d"]),
            {"ranked_nodes": []},
        ])
        initial_children = _children("C", 15)
        initial_parents = _parents("P", 4)

        children, parents, metadata = agent.apply_eaes_rollback_check(
            "Question",
            _query_plan(),
            initial_children,
            initial_parents,
        )

        self.assertEqual(children, initial_children)
        self.assertEqual(parents, initial_parents)
        self.assertEqual(metadata["rollback_count"], 2)
        self.assertEqual(len(metadata["rounds"]), 2)
        self.assertEqual(
            [item["selected_supplement_count"] for item in metadata["rounds"]],
            [0, 0],
        )

    def test_answer_path_rolls_back_before_one_reader_call(self):
        agent = _AnswerAgent([
            _decision(),
            {
                "ranked_nodes": [
                    {"node_type": "child", "node_id": "R1C1"},
                ]
            },
            _decision(state="no_need_more"),
        ])

        with patch.object(config, "EAES_ROLLBACK_CHECK", True):
            answer, context = agent.answer_question_eaes(
                "What pet does Caroline own?"
            )

        self.assertEqual(answer, "A cat")
        self.assertEqual(len(agent.reader_inputs), 1)
        self.assertEqual(len(agent.reader_inputs[0][0]), 16)
        self.assertEqual(agent.reader_inputs[0][0][-1]["memory_id"], "R1C1")
        self.assertIn("R1C1", context)

    def test_answer_path_calls_reader_after_two_empty_rollback_rounds(self):
        agent = _EmptyAnswerAgent([
            _decision(),
            _decision(query_phase=["a", "b", "c", "d"]),
        ])

        with patch.object(config, "EAES_ROLLBACK_CHECK", True):
            answer, context = agent.answer_question_eaes("Question")

        self.assertEqual(answer, "A cat")
        self.assertEqual(context, [])
        self.assertEqual(len(agent.llm.inputs), 2)
        self.assertEqual(len(agent.reader_inputs), 1)
        self.assertEqual(agent.reader_inputs[0], ([], []))


if __name__ == "__main__":
    unittest.main()
