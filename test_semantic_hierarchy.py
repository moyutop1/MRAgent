import copy
import sys
import types
import unittest
from unittest.mock import patch


common_module = types.ModuleType("common")
common_module.config = types.SimpleNamespace(
    REWRITE_WINDOW_SIZE=40,
    REWRITE_OVERLAP_SIZE=2,
    REWRITE_PREVIOUS_LIMIT=3,
    SEMANTIC_HIERARCHY=True,
    PARENT_MIN_TURNS=4,
    PARENT_MAX_TURNS=10,
    PARENT_CONTEXT_TURNS=2,
    CHILD_MAX_TURNS=8,
    CHILD_REWRITE_BATCH_SIZE=15,
)
sys.modules.setdefault("common", common_module)
sys.modules.setdefault("common.config", common_module.config)

jsonschema_module = types.ModuleType("jsonschema")


class _NoOpValidator:
    def __init__(self, schema):
        self.schema = schema

    def validate(self, value):
        return None


jsonschema_module.Draft202012Validator = _NoOpValidator
jsonschema_module.ValidationError = ValueError
sys.modules.setdefault("jsonschema", jsonschema_module)

from agent.rewrite_memory import rewrite_semantic_hierarchy_session
from agent.semantic_segmentation import (
    ChildSegment,
    TurnRecord,
    attach_child_ids_to_parents,
    iter_child_batches,
    parse_session_turns,
    plan_child_segments,
    plan_parent_segments,
)
from common import config
from memory.system import EAESMemoryNote, EAESParentNode, MemorySystem


class SequenceLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def chat_text(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        if not self.outputs:
            raise AssertionError("unexpected LLM call")
        return copy.deepcopy(self.outputs.pop(0))


def _dialogue(turn_count=5):
    return "time:2023-05-08\n" + "\n".join(
        f"dia_id:D1:{index} speaker:turn {index}"
        for index in range(1, turn_count + 1)
    )


class SemanticHierarchyTests(unittest.TestCase):
    def test_parent_retry_explains_required_split_for_twenty_one_turns(self):
        turns = parse_session_turns(_dialogue(21))
        llm = SequenceLLM([
            {"parent_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:21"}
            ]},
            {"parent_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:7"},
                {"start_origin": "D1:8", "end_origin": "D1:14"},
                {"start_origin": "D1:15", "end_origin": "D1:21"},
            ]},
        ])

        parents = plan_parent_segments(llm, turns, "2023-05-08")

        self.assertEqual(len(parents), 3)
        initial_payload = llm.calls[0][1]["content"]
        self.assertIn('"maximum_turns": 10', initial_payload)
        self.assertIn('"minimum_segment_count": 3', initial_payload)
        self.assertIn('"position": 21', initial_payload)
        repair_payload = llm.calls[1][1]["content"]
        self.assertIn("must return at least 3 parent segments", repair_payload)
        self.assertIn('"end_origin": "D1:21"', repair_payload)

    def test_independent_plans_and_first_origin_parent_ownership(self):
        turns = parse_session_turns(_dialogue(6))
        parent_llm = SequenceLLM([{
            "parent_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:4"},
                {"start_origin": "D1:5", "end_origin": "D1:6"},
            ]
        }])
        child_llm = SequenceLLM([{
            "child_segments": [
                {"source_origins": ["D1:2"], "focus": "first fact"},
                {"source_origins": ["D1:2"], "focus": "second fact"},
                {"source_origins": ["D1:4", "D1:5"], "focus": "cross-boundary fact"},
            ]
        }])

        parents = plan_parent_segments(parent_llm, turns, "2023-05-08")
        children = plan_child_segments(child_llm, turns, "2023-05-08")
        attach_child_ids_to_parents(parents, children)

        self.assertEqual([child.child_id for child in children], [
            "D1:2-1", "D1:2-2", "D1:4"
        ])
        self.assertEqual(
            parents[0].child_ids,
            ["D1:2-1", "D1:2-2", "D1:4"],
        )
        self.assertEqual(parents[1].child_ids, [])
        self.assertNotIn("parent_segments", child_llm.calls[0][1]["content"])

    def test_child_batches_never_exceed_fifteen(self):
        turn = TurnRecord("D1:1", "dia_id:D1:1 speaker:text", 0)
        children = [
            ChildSegment(f"D1:1-{index}", ["D1:1"], str(index), [turn], [])
            for index in range(1, 17)
        ]
        with patch.object(config, "CHILD_REWRITE_BATCH_SIZE", 15):
            self.assertEqual(
                [len(batch) for batch in iter_child_batches(children)],
                [15, 1],
            )

    def test_hierarchical_rewrite_is_one_sentence_per_child(self):
        llm = SequenceLLM([
            {"parent_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:5"}
            ]},
            {"child_segments": [
                {"source_origins": ["D1:2"], "focus": "first fact"},
                {"source_origins": ["D1:2"], "focus": "second fact"},
                {"source_origins": ["D1:4"], "focus": "unique fact"},
            ]},
            {
                "parent_id": "D1:t1",
                "rewrite_content": "A coarse parent memory.",
            },
            {
                "conversation_time": "2023-05-08",
                "sentence": [
                    {
                        "id": "D1:2-1",
                        "text": "The speaker stated the first fact.",
                        "tag": "First Fact",
                        "origin": "D1:2",
                        "topic": [],
                        "semantic_properties": ["event_action", "episodic"],
                    },
                    {
                        "id": "D1:2-2",
                        "text": "The speaker stated the second fact.",
                        "tag": "Second Fact",
                        "origin": "D1:2",
                        "topic": [],
                        "semantic_properties": ["state_opinion", "transient"],
                    },
                    {
                        "id": "D1:4",
                        "text": "The speaker stated the unique fact.",
                        "tag": "Unique Fact",
                        "origin": "D1:4",
                        "topic": [],
                        "semantic_properties": ["state_opinion", "durable"],
                    },
                ],
                "topics": {},
                "personal_sentences": [],
            },
        ])

        output = rewrite_semantic_hierarchy_session(llm, _dialogue(5))

        self.assertEqual(
            [item["id"] for item in output["sentence"]],
            ["D1:2-1", "D1:2-2", "D1:4"],
        )
        self.assertEqual(output["parent_nodes"], [{
            "parent_id": "D1:t1",
            "rewrite_content": "A coarse parent memory.",
            "child_ids": ["D1:2-1", "D1:2-2", "D1:4"],
        }])
        self.assertEqual(
            len(llm.calls[-1][1]["content"].split('"child_id"')), 4
        )
        parent_rewrite_call = llm.calls[2]
        self.assertNotIn("conversation_time", parent_rewrite_call[1]["content"])
        self.assertIn(
            "Do not record any temporal information",
            parent_rewrite_call[0]["content"],
        )
        self.assertIn(
            "PERSON PROFILE MEMORY",
            parent_rewrite_call[0]["content"],
        )
        self.assertIn(
            "INVALID event-summary style",
            parent_rewrite_call[0]["content"],
        )
        child_rewrite_call = llm.calls[-1]
        self.assertIn("conversation_time", child_rewrite_call[1]["content"])

    def test_child_rewrite_cannot_redefine_planner_provenance(self):
        llm = SequenceLLM([
            {"parent_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:4"}
            ]},
            {"child_segments": [{
                "source_origins": ["D1:2", "D1:3"],
                "focus": "a fact supported by two turns",
            }]},
            {
                "parent_id": "D1:t1",
                "rewrite_content": "A coarse parent memory.",
            },
            {
                "conversation_time": "2023-05-08",
                "sentence": [{
                    "id": "D1:2",
                    "text": "A fact supported by two turns.",
                    "tag": "Supported Fact",
                    # Reproduce the production failure: the rewrite model
                    # drops the planner's first source origin.
                    "origin": "D1:3",
                    "topic": [],
                    "semantic_properties": ["event_action", "episodic"],
                }],
                "topics": {},
                "personal_sentences": [],
            },
        ])

        output = rewrite_semantic_hierarchy_session(llm, _dialogue(4))

        self.assertEqual(output["sentence"][0]["id"], "D1:2")
        self.assertEqual(output["sentence"][0]["origin"], "D1:2,D1:3")
        self.assertEqual(len(llm.calls), 4)

    def test_parent_reader_payload_hides_child_attributes_and_support_expands(self):
        memory = MemorySystem()
        note = EAESMemoryNote(
            memory_id="M_D1_2_1",
            event_id="D1:2-1",
            entities=["speaker"],
            attribute_paths=["fact.detail: first fact"],
            raw_text="speaker:turn 2",
            rewrite_content="The speaker stated the first fact.",
            conversation_time="2023-05-08",
            event_lifecycle="historical",
            origin="D1:2",
        )
        memory.add_eaes_memory_note(note)
        parent = EAESParentNode(
            parent_id="D1:t1",
            rewrite_content="A coarse parent memory.",
            child_ids=[note.memory_id],
            child_attributes=[{
                "child_id": note.memory_id,
                "attributes": list(note.attribute_paths),
            }],
            retrieval_embedding=[1.0, 0.0],
        )
        memory.add_eaes_parent_node(parent)

        self.assertEqual(parent.to_reader_dict(), {
            "parent_id": "D1:t1",
            "rewrite_content": "A coarse parent memory.",
        })
        self.assertEqual(
            memory.get_eaes_support_origin(["D1:t1"]),
            ["D1:t1", "D1:2"],
        )


if __name__ == "__main__":
    unittest.main()
