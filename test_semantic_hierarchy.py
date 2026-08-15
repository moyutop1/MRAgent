import copy
import sys
import types
import unittest


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

from agent.rewrite_memory import (
    _child_window_source_text,
    _previous_child_rewrite_context,
    _rewrite_child_window,
    rewrite_semantic_hierarchy_session,
)
from agent.semantic_segmentation import (
    ChildWindow,
    attach_child_memory_ids_to_parents,
    child_window_turns,
    parse_session_turns,
    plan_child_windows,
    plan_parent_segments,
)
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


def _dialogue_with_prefix(prefix, turn_count):
    return "time:2023-05-08\n" + "\n".join(
        f"dia_id:{prefix}:{index} speaker:turn {index}"
        for index in range(1, turn_count + 1)
    )


def _sentence(origin, text, sentence_id=None, **extra):
    item = {
        "id": sentence_id or origin.split(",", 1)[0],
        "text": text,
        "tag": "Turn Memory",
        "origin": origin,
        "topic": [],
        "semantic_properties": ["event_action", "episodic"],
    }
    item.update(extra)
    return item


def _rewrite_output(*sentences):
    return {
        "conversation_time": "2023-05-08",
        "sentence": list(sentences),
        "topics": {},
        "personal_sentences": [],
    }


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

    def test_parent_retry_reports_exact_inclusive_invalid_range(self):
        turns = parse_session_turns(_dialogue_with_prefix("D13", 37))
        llm = SequenceLLM([
            {"parent_segments": [
                {"start_origin": "D13:1", "end_origin": "D13:7"},
                {"start_origin": "D13:8", "end_origin": "D13:14"},
                {"start_origin": "D13:15", "end_origin": "D13:20"},
                {"start_origin": "D13:21", "end_origin": "D13:26"},
                {"start_origin": "D13:27", "end_origin": "D13:37"},
            ]},
            {"parent_segments": [
                {"start_origin": "D13:1", "end_origin": "D13:7"},
                {"start_origin": "D13:8", "end_origin": "D13:14"},
                {"start_origin": "D13:15", "end_origin": "D13:20"},
                {"start_origin": "D13:21", "end_origin": "D13:26"},
                {"start_origin": "D13:27", "end_origin": "D13:31"},
                {"start_origin": "D13:32", "end_origin": "D13:37"},
            ]},
        ])

        parents = plan_parent_segments(llm, turns, "2023-05-08")

        self.assertEqual(len(parents), 6)
        repair_system = llm.calls[1][0]["content"]
        self.assertIn("D13:27 through D13:37", repair_system)
        self.assertIn("37 - 27 + 1 = 11 turns", repair_system)
        self.assertIn("hard maximum of 10", repair_system)

    def test_parent_and_child_window_plans_are_independent(self):
        turns = parse_session_turns(_dialogue(6))
        parent_llm = SequenceLLM([{
            "parent_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:4"},
                {"start_origin": "D1:5", "end_origin": "D1:6"},
            ]
        }])
        child_llm = SequenceLLM([{
            "child_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:2"},
                {"start_origin": "D1:3", "end_origin": "D1:6"},
            ]
        }])

        parents = plan_parent_segments(parent_llm, turns, "2023-05-08")
        windows = plan_child_windows(child_llm, turns, "2023-05-08")

        self.assertEqual(len(parents), 2)
        self.assertEqual(windows, [
            ChildWindow("D1:1", "D1:2"),
            ChildWindow("D1:3", "D1:6"),
        ])
        self.assertIsInstance(windows[0].start_origin, str)
        self.assertFalse(hasattr(windows[0], "window_id"))
        self.assertNotIn("parent_segments", child_llm.calls[0][1]["content"])
        self.assertEqual(
            [turn.origin for turn in child_window_turns(windows[1], turns)],
            ["D1:3", "D1:4", "D1:5", "D1:6"],
        )

    def test_child_window_plan_retries_gap_and_requires_full_coverage(self):
        turns = parse_session_turns(_dialogue(6))
        llm = SequenceLLM([
            {"child_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:2"},
                {"start_origin": "D1:4", "end_origin": "D1:6"},
            ]},
            {"child_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:2"},
                {"start_origin": "D1:3", "end_origin": "D1:6"},
            ]},
        ])

        windows = plan_child_windows(llm, turns, "2023-05-08")

        self.assertEqual(windows[-1].end_origin, "D1:6")
        self.assertIn(
            "contiguous ordered coverage",
            llm.calls[1][0]["content"],
        )

    def test_child_window_plan_allows_one_turn_and_enforces_maximum(self):
        single_turns = parse_session_turns(_dialogue(1))
        single = plan_child_windows(
            SequenceLLM([{"child_segments": [{
                "start_origin": "D1:1", "end_origin": "D1:1"
            }]}]),
            single_turns,
            "2023-05-08",
        )
        self.assertEqual(single, [ChildWindow("D1:1", "D1:1")])

        turns = parse_session_turns(_dialogue(9))
        llm = SequenceLLM([
            {"child_segments": [{
                "start_origin": "D1:1", "end_origin": "D1:9"
            }]},
            {"child_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:4"},
                {"start_origin": "D1:5", "end_origin": "D1:9"},
            ]},
        ])

        windows = plan_child_windows(llm, turns, "2023-05-08")

        self.assertEqual(len(windows), 2)
        self.assertIn("hard maximum is 8", llm.calls[1][0]["content"])

    def test_child_window_plan_rejects_segment_after_full_coverage(self):
        turns = parse_session_turns(_dialogue(2))
        llm = SequenceLLM([
            {"child_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:2"},
                {"start_origin": "D1:1", "end_origin": "D1:1"},
            ]},
            {"child_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:2"},
            ]},
        ])

        windows = plan_child_windows(llm, turns, "2023-05-08")

        self.assertEqual(windows, [ChildWindow("D1:1", "D1:2")])
        self.assertIn("already fully covered", llm.calls[1][0]["content"])

    def test_previous_reference_contains_only_last_two_rewrite_texts(self):
        memories = [
            _sentence("D1:1", "first"),
            _sentence("D1:2", "second"),
            _sentence("D1:3", "third"),
        ]

        context = _previous_child_rewrite_context(memories)

        self.assertEqual(context, [
            {"rewrite_content": "second"},
            {"rewrite_content": "third"},
        ])
        self.assertNotIn("origin", context[0])
        self.assertNotIn("id", context[0])

    def test_child_window_rewrite_retries_missing_turn_and_separates_reference(self):
        turns = parse_session_turns(_dialogue(3))
        window = ChildWindow("D1:2", "D1:3")
        llm = SequenceLLM([
            _rewrite_output(_sentence("D1:1", "copied reference")),
            _rewrite_output(_sentence("D1:2", "turn two")),
            _rewrite_output(
                _sentence("D1:2", "turn two"),
                _sentence("D1:3", "turn three"),
            ),
        ])

        output = _rewrite_child_window(
            llm,
            window,
            turns,
            "2023-05-08",
            previous_rewrites=[{"rewrite_content": "reference fact"}],
        )

        self.assertEqual(
            [item["id"] for item in output["sentence"]],
            ["D1:2-1", "D1:3-1"],
        )
        self.assertIn("origin ids not found", llm.calls[1][0]["content"])
        self.assertIn("missing origins", llm.calls[2][0]["content"])
        user_prompt = llm.calls[0][1]["content"]
        reference_section = user_prompt.split("CURRENT_CHILD_WINDOW:", 1)[0]
        self.assertIn("reference fact", reference_section)
        self.assertNotIn('"origin"', reference_section)
        self.assertNotIn('"id"', reference_section)
        self.assertNotIn("D1:1", _child_window_source_text(
            window, turns, "2023-05-08"
        ))

    def test_one_turn_can_produce_repeated_memories_without_deduplication(self):
        turns = parse_session_turns(_dialogue(1))
        window = ChildWindow("D1:1", "D1:1")
        llm = SequenceLLM([_rewrite_output(
            _sentence("D1:1", "The same stated information."),
            _sentence("D1:1", "The same stated information."),
        )])

        output = _rewrite_child_window(
            llm, window, turns, "2023-05-08"
        )

        self.assertEqual(len(output["sentence"]), 2)
        self.assertEqual(
            [item["id"] for item in output["sentence"]],
            ["D1:1-1", "D1:1-2"],
        )
        self.assertEqual(
            output["sentence"][0]["text"],
            output["sentence"][1]["text"],
        )

    def test_child_window_rewrite_rejects_raw_storage_fields(self):
        turns = parse_session_turns(_dialogue(1))
        window = ChildWindow("D1:1", "D1:1")
        invalid = _rewrite_output(_sentence(
            "D1:1", "turn one", raw_content="speaker:turn 1"
        ))
        valid = _rewrite_output(_sentence("D1:1", "turn one"))
        llm = SequenceLLM([invalid, valid])

        output = _rewrite_child_window(
            llm, window, turns, "2023-05-08"
        )

        self.assertNotIn("raw_content", output["sentence"][0])
        self.assertIn("forbidden raw fields", llm.calls[1][0]["content"])

    def test_hierarchical_rewrite_is_sequential_and_links_final_memory_ids(self):
        first_window = _rewrite_output(
            _sentence("D1:1", "first turn"),
            _sentence("D1:2", "first fact in turn two"),
            _sentence("D1:2", "second fact in turn two"),
            _sentence("D1:3", "third turn"),
        )
        second_window = _rewrite_output(
            _sentence("D1:4", "repeated information"),
            _sentence("D1:5", "repeated information"),
        )
        llm = SequenceLLM([
            {"parent_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:5"}
            ]},
            {"child_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:3"},
                {"start_origin": "D1:4", "end_origin": "D1:5"},
            ]},
            first_window,
            second_window,
            {
                "parent_id": "D1:t1",
                "rewrite_content": "A coarse parent memory.",
            },
        ])

        output = rewrite_semantic_hierarchy_session(llm, _dialogue(5))

        expected_ids = [
            "D1:1-1", "D1:2-1", "D1:2-2",
            "D1:3-1", "D1:4-1", "D1:5-1",
        ]
        self.assertEqual(
            [item["id"] for item in output["sentence"]], expected_ids
        )
        self.assertEqual(output["parent_nodes"], [{
            "parent_id": "D1:t1",
            "rewrite_content": "A coarse parent memory.",
            "child_ids": expected_ids,
        }])
        self.assertEqual(len(llm.calls), 5)
        second_prompt = llm.calls[3][1]["content"]
        reference_section = second_prompt.split("CURRENT_CHILD_WINDOW:", 1)[0]
        self.assertIn("second fact in turn two", reference_section)
        self.assertIn("third turn", reference_section)
        self.assertNotIn("first fact in turn two", reference_section)
        self.assertIn(
            "every repeated occurrence",
            llm.calls[3][0]["content"],
        )
        parent_rewrite_call = llm.calls[4]
        self.assertIn("PERSON PROFILE MEMORY", parent_rewrite_call[0]["content"])

    def test_memory_ids_attach_to_parent_by_first_origin(self):
        turns = parse_session_turns(_dialogue(6))
        parent_llm = SequenceLLM([{
            "parent_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:4"},
                {"start_origin": "D1:5", "end_origin": "D1:6"},
            ]
        }])
        parents = plan_parent_segments(parent_llm, turns, "2023-05-08")
        memories = [
            _sentence("D1:2", "first", "D1:2-1"),
            _sentence("D1:4,D1:5", "cross boundary", "D1:4-1"),
            _sentence("D1:6", "last", "D1:6-1"),
        ]

        attach_child_memory_ids_to_parents(parents, memories)

        self.assertEqual(parents[0].child_ids, ["D1:2-1", "D1:4-1"])
        self.assertEqual(parents[1].child_ids, ["D1:6-1"])

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
