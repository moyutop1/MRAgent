import unittest
import sys
import types
from unittest.mock import MagicMock, patch

import common


test_config = types.SimpleNamespace(
    CHILD_MAX_TURNS=8,
    UTILITY_TOKEN_BUDGET=2048,
    UTILITY_MODEL="test-utility",
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
from agent.rewrite_memory import _keep_new_children
from agent.semantic_segmentation import ChildWindow, TurnRecord
from memory.system import EAESMemoryNote, EAESParentNode, MemorySystem
from llm.unified_utility_judge import UnifiedUtilityJudge
from prompts.schema import check_child_window_rewrite_json


class _Judge:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = []

    def keep(self, operation, condition, selected_set, candidate, source=None):
        self.calls.append((operation, len(selected_set), candidate))
        return next(self.decisions)

    @staticmethod
    def count_tokens(value):
        return len(str(value).split())


class UnifiedUtilityPipelineTests(unittest.TestCase):
    def test_flash_cue_uses_child_tag_schema(self):
        judge = UnifiedUtilityJudge.__new__(UnifiedUtilityJudge)
        judge.client = MagicMock()
        judge.client.chat.completions.create.return_value = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(
                content='{"prefix":"Nate recommendation",'
                        '"facet":"books for Joanna"}'
            ))]
        )
        self.assertEqual(
            judge.generate_cue("What books did Nate recommend to Joanna?"),
            "Nate recommendation.books for Joanna",
        )

    def test_flash_cue_retries_validation_then_falls_back(self):
        judge = UnifiedUtilityJudge.__new__(UnifiedUtilityJudge)
        judge.client = MagicMock()
        invalid = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(
                content='{"prefix":"person","facet":"too many words in this facet"}'
            ))]
        )
        judge.client.chat.completions.create.return_value = invalid
        self.assertIsNone(judge.generate_cue("What happened?"))
        self.assertEqual(judge.client.chat.completions.create.call_count, 3)
        retry_messages = (
            judge.client.chat.completions.create.call_args_list[1]
            .kwargs["messages"]
        )
        self.assertIn("Validation error:", retry_messages[-1]["content"])

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
            accepted = _keep_new_children(candidates, kept)
        self.assertEqual(accepted, candidates[:1])
        self.assertEqual([call[:2] for call in judge.calls], [("WRITE", 0), ("WRITE", 1)])

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

    def test_select_uses_one_ordered_marginal_utility_pass(self):
        judge = _Judge([True, False])
        mixin = EAESMixin()
        children = [{
            "memory_id": "M1", "score": 0.9,
            "rewrite_content": "child", "conversation_time": "2025-01-01",
        }]
        parents = [{"parent_id": "P1", "score": 0.8, "rewrite_content": "parent"}]
        with patch(
            "llm.unified_utility_judge.get_unified_utility_judge",
            return_value=judge,
        ):
            selected = mixin.select_eaes_evidence("query", {}, children, parents)
        self.assertEqual(len(selected["answer_items"]), 1)
        self.assertEqual(selected["selected_parent_memories"], [])
        self.assertEqual([call[:2] for call in judge.calls], [("SELECT", 0), ("SELECT", 1)])


if __name__ == "__main__":
    unittest.main()
