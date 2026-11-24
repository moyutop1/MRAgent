import json
import unittest
import sys
import types
from unittest.mock import MagicMock, patch

import common


test_config = types.SimpleNamespace(
    CHILD_MAX_TURNS=8,
    UTILITY_TOKEN_BUDGET=2048,
    UTILITY_MODEL="test-utility",
    DISABLE_EVIDENCE_SELECTOR=False,
    SEMANTIC_HIERARCHY=True,
    EAES_CHILD_CANDIDATE_LIMIT=30,
    EAES_PARENT_CHILD_ROUTE_LIMIT=30,
    PARENT_TOP_K=10,
    RE_MODEL="test-model",
)
common.config = test_config
sys.modules["common.config"] = test_config

try:
    import jsonschema  # noqa: F401
except ModuleNotFoundError:
    jsonschema = types.ModuleType("jsonschema")

    class _ValidationError(Exception):
        message = ""

    class _Validator:
        def __init__(self, schema):
            self.schema = schema

        def validate(self, value):
            return None

    jsonschema.Draft202012Validator = _Validator
    jsonschema.ValidationError = _ValidationError
    sys.modules["jsonschema"] = jsonschema

from agent.eaes import EAESMixin
from agent.rewrite_memory import (
    _keep_new_children,
    normalize_rewrite_temporal_granularity,
    rewrite_semantic_hierarchy_session,
)
from agent.semantic_segmentation import ChildWindow, TurnRecord
from memory.system import EAESMemoryNote, EAESParentNode, MemorySystem
from llm.unified_utility_judge import UnifiedUtilityJudge
from prompts.schema import check_child_window_rewrite_json


class _Judge:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = []

    def estimate(
            self, operation, source, selected_set, candidate, context=None
    ):
        self.calls.append((operation, len(selected_set), candidate))
        keep = next(self.decisions)
        return {
            "new_target_unit": keep,
            "necessary_constraint": False,
            "marginal_utility": int(keep),
            "keep": keep,
        }

    @staticmethod
    def count_tokens(value):
        return len(str(value).split())


class UnifiedUtilityPipelineTests(unittest.TestCase):
    def test_select_utility_is_computed_from_structured_components(self):
        judge = UnifiedUtilityJudge.__new__(UnifiedUtilityJudge)
        judge.client = MagicMock()
        judge.client.chat.completions.create.return_value = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(
                content=(
                    '{"new_target_unit":false,'
                    '"necessary_constraint":true}'
                )
            ))]
        )
        utility = judge.estimate(
            "SELECT",
            {"rewrite_content": "May 2"},
            [],
            {"rewrite_content": "May 2"},
            {"query": "when?"},
        )
        self.assertEqual(utility["marginal_utility"], 1)
        self.assertTrue(utility["keep"])

    def test_child_window_may_produce_no_memory(self):
        turns = [TurnRecord("D1:1", "user: hello dia_id: D1:1", 0)]
        output = {
            "conversation_time": "2025-01-01",
            "sentence": [],
            "personal_sentences": [],
        }
        valid, error = check_child_window_rewrite_json(
            output,
            ChildWindow("D1:1", "D1:1"),
            turns,
            "time:2025-01-01\nuser: hello dia_id: D1:1",
        )
        self.assertTrue(valid, error)

    def test_write_utility_is_conditioned_on_kept_session_children(self):
        judge = _Judge([True, False])
        candidates = [
            {"id": "draft-1", "origin": "D1:1", "text": "new detail"},
            {"id": "draft-2", "origin": "D1:2", "text": "duplicate"},
        ]
        kept = []
        with patch(
            "llm.unified_utility_judge.get_unified_utility_judge",
            return_value=judge,
        ):
            accepted = _keep_new_children(candidates, kept, "source dialogue")
        self.assertEqual(accepted, candidates[:1])
        self.assertEqual([call[:2] for call in judge.calls], [("WRITE", 0), ("WRITE", 1)])

    def test_missing_facts_use_frozen_baseline_and_are_appended_once(self):
        prior = {"id": "D1:1-1", "origin": "D1:1", "text": "prior fact"}
        draft = {"id": "D1:2-1", "origin": "D1:2", "text": "Nate dyed his hair."}
        alias = {"id": "D1:2-1", "origin": "D1:2", "text": "Nate calls Joanna Jo."}
        repeat = {"id": "D1:2-2", "origin": "D1:2", "text": "prior fact"}
        judge = _Judge([True, True, True, False])
        judge.missing_fact_indices = MagicMock(side_effect=[set(), {1, 2}])
        with patch("agent.rewrite_memory.plan_child_windows", return_value=[
            ChildWindow("D1:1", "D1:1"), ChildWindow("D1:2", "D1:2"),
        ]), patch("agent.rewrite_memory._rewrite_child_window", side_effect=[
            {"sentence": [prior.copy()]}, {"sentence": [prior]},
            {"sentence": [draft.copy(), alias, repeat]}, {"sentence": [draft]},
        ]), patch("agent.rewrite_memory._build_session_parents", return_value=[]), patch(
            "llm.unified_utility_judge.get_unified_utility_judge", return_value=judge
        ):
            result = rewrite_semantic_hierarchy_session(
                None, "time:2022-04-15\ndia_id:D1:1 Nate: Hi.\ndia_id:D1:2 Nate: Hey Jo, I dyed my hair."
            )
        self.assertEqual(
            [(call[1], call[2]["text"]) for call in judge.calls],
            [(0, "prior fact"), (1, draft["text"]), (1, alias["text"]), (1, repeat["text"])],
        )
        self.assertEqual([item["text"] for item in result["sentence"]],
                         [prior["text"], draft["text"], alias["text"]])
        self.assertEqual(result["sentence"][-1]["id"], "D1:2-2")

    def test_relative_time_stays_with_the_fact_that_mentions_it(self):
        output = {"sentence": [
            {"origin": "D16:6", "text": "Seven people attended the party."},
            {"origin": "D16:6", "text": "Six wanted to do it again next month."},
        ]}
        normalize_rewrite_temporal_granularity(
            output, "time:2022-06-24\ndia_id:D16:6 Nate: Seven attended. Six would do it again next month."
        )
        self.assertEqual([item["text"] for item in output["sentence"]], [
            "Seven people attended the party.",
            "Six wanted to do it again July 2022.",
        ])

    def test_session_view_and_global_indices_share_nodes(self):
        memory = MemorySystem()
        child = EAESMemoryNote(
            "M_D1_1_1", "D1:1-1", "child", "2025-01-01", "D1:1"
        )
        parent = EAESParentNode("D1:P1", "parent", [child.memory_id])
        memory.add_eaes_memory_note(child, 1)
        memory.add_eaes_parent_node(parent, 1)
        session = memory.eaes_sessions["session_1"]
        self.assertIs(session["children"][child.memory_id], memory.eaes_notes[child.memory_id])
        self.assertIs(session["parents"][parent.parent_id], memory.eaes_parent_nodes[parent.parent_id])

    def test_select_uses_backward_marginal_elimination(self):
        judge = _Judge([False, True])
        mixin = EAESMixin()
        children = [{
            "memory_id": "M1", "score": 0.9,
            "rewrite_content": "child", "conversation_time": "2025-01-01",
        }]
        parents = [{"parent_id": "P1", "score": 0.8, "rewrite_content": "parent"}]
        with patch.object(test_config, "RETRIEVAL_ONLY", True, create=True), patch(
            "llm.unified_utility_judge.get_unified_utility_judge",
            return_value=judge,
        ):
            selected = mixin.select_eaes_evidence("query", {}, children, parents)
        self.assertEqual(len(selected["answer_items"]), 1)
        self.assertEqual(selected["selected_parent_memories"], [])
        self.assertEqual([call[:2] for call in judge.calls], [("SELECT", 1), ("SELECT", 0)])
        trace = selected["selector_diagnostics"]
        self.assertEqual(trace["selection_direction"], "backward_elimination")
        self.assertEqual(trace["deletion_order"], [
            {"step": 1, "kind": "parent", "id": "P1"},
        ])
        self.assertEqual(trace["decisions"][0]["comparison_set"], [
            {"kind": "child", "id": "M1"},
        ])
        self.assertEqual(trace["decisions"][1]["comparison_set"], [])

    def test_parent_bypasses_child_selector_and_reaches_reader(self):
        judge = _Judge([True])
        mixin = EAESMixin()
        mixin.llm = MagicMock()
        mixin.llm.chat_text.return_value = {"answer": "answer"}
        children = [{"memory_id": "M1", "rewrite_content": "child evidence"}]
        parents = [{"parent_id": "P1", "rewrite_content": "parent summary"}]
        with patch.object(
            test_config, "PARENT_READER_PASSTHROUGH", True, create=True
        ), patch(
            "llm.unified_utility_judge.get_unified_utility_judge", return_value=judge
        ), patch.object(mixin, "_eaes_prediction_context", return_value=[]):
            _, _, _, trace = mixin._read_eaes_candidates(
                "query", {}, children, parents, return_details=True
            )

        self.assertEqual(judge.calls, [
            ("SELECT", 0, mixin._eaes_reader_child_memory(children[0])),
        ])
        self.assertEqual(trace["bypassed_parent_count"], 1)
        messages = mixin.llm.chat_text.call_args.kwargs["messages"]
        self.assertEqual(json.loads(messages[-1]["content"])["parent_memories"], parents)

    def test_disabled_selector_passes_every_child_and_parent_without_judge(self):
        mixin = EAESMixin()
        children = [
            {
                "memory_id": "M1", "score": 0.9,
                "rewrite_content": "first child",
                "conversation_time": "2025-01-01",
            },
            {
                "memory_id": "M2", "score": 0.8,
                "rewrite_content": "second child",
                "conversation_time": "2025-01-02",
            },
        ]
        parents = [
            {"parent_id": "P1", "score": 0.7, "rewrite_content": "parent"}
        ]
        with patch.object(test_config, "DISABLE_EVIDENCE_SELECTOR", True), patch(
            "llm.unified_utility_judge.get_unified_utility_judge"
        ) as get_judge:
            selected = mixin.select_eaes_evidence(
                "query", {}, children, parents
            )

        get_judge.assert_not_called()
        self.assertEqual(
            [item["evidence"][0] for item in selected["answer_items"]],
            children,
        )
        self.assertEqual(selected["selected_parent_memories"], parents)
        diagnostics = selected["selector_diagnostics"]
        self.assertFalse(diagnostics["enabled"])
        self.assertEqual(diagnostics["selection_direction"], "passthrough")
        self.assertEqual(diagnostics["candidate_count"], 3)
        self.assertEqual(diagnostics["selected_count"], 3)
        self.assertEqual(diagnostics["decisions"], [])


if __name__ == "__main__":
    unittest.main()
