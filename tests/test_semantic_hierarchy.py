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
    CHILD_DUPLICATE_SIMILARITY_THRESHOLD=0.55,
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
    _extract_session_tag_prefix_pool,
    _fuse_adjacent_duplicate_child_memories,
    _previous_child_rewrite_context,
    _rewrite_child_window,
    inherit_adjacent_question_origins,
    normalize_rewrite_semantic_properties,
    normalize_rewrite_tag_lengths,
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
from prompts.prompts import Prompts


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
        "tag": [
            "Speaker activity.turn memory",
            "Speaker activity.dialogue detail",
        ],
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

    def test_child_window_plan_rejects_boundary_between_question_and_answer(self):
        turns = parse_session_turns("\n".join([
            "time:2023-05-08",
            "dia_id:D1:1 Alex: I have an update.",
            "dia_id:D1:2 Alex: What are your summer plans?",
            "dia_id:D1:3 Morgan: I am researching adoption agencies.",
            "dia_id:D1:4 Alex: That sounds promising.",
        ]))
        llm = SequenceLLM([
            {"child_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:2"},
                {"start_origin": "D1:3", "end_origin": "D1:4"},
            ]},
            {"child_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:3"},
                {"start_origin": "D1:4", "end_origin": "D1:4"},
            ]},
        ])

        windows = plan_child_windows(llm, turns, "2023-05-08")

        self.assertEqual(windows[0], ChildWindow("D1:1", "D1:3"))
        self.assertIn("question/answer pair", llm.calls[1][0]["content"])

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

    def test_adjacent_question_origins_are_inherited_by_answer_memories(self):
        turns = parse_session_turns("\n".join([
            "time:2023-05-08",
            "dia_id:D1:6 Melanie: What's it done for you?",
            "dia_id:D1:7 Caroline: It made me feel accepted.",
            "dia_id:D1:8 Melanie: What now?",
            "dia_id:D1:9 Caroline: I will continue my education.",
        ]))
        output = _rewrite_output(
            _sentence("D1:7", "The group made Caroline feel accepted."),
            _sentence("D1:9", "Caroline will continue her education."),
        )

        changed = inherit_adjacent_question_origins(output, turns)

        self.assertEqual(changed, 2)
        self.assertEqual(
            [item["origin"] for item in output["sentence"]],
            ["D1:6,D1:7", "D1:8,D1:9"],
        )

    def test_child_window_accepts_answer_memories_missing_question_origins(self):
        turns = parse_session_turns("\n".join([
            "time:2023-05-08",
            "dia_id:D1:6 Melanie: What's it done for you?",
            "dia_id:D1:7 Caroline: It made me feel accepted.",
            "dia_id:D1:8 Melanie: What now?",
            "dia_id:D1:9 Caroline: I will continue my education.",
        ]))
        window = ChildWindow("D1:6", "D1:9")
        llm = SequenceLLM([_rewrite_output(
            _sentence("D1:7", "The group made Caroline feel accepted."),
            _sentence("D1:9", "Caroline will continue her education."),
        )])

        output = _rewrite_child_window(
            llm, window, turns, "2023-05-08"
        )

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(
            [item["origin"] for item in output["sentence"]],
            ["D1:6,D1:7", "D1:8,D1:9"],
        )
        self.assertEqual(
            [item["id"] for item in output["sentence"]],
            ["D1:6-1", "D1:8-1"],
        )

    def test_overlong_child_tag_is_compounded_without_regenerating_window(self):
        turns = parse_session_turns(_dialogue(1))
        window = ChildWindow("D1:1", "D1:1")
        llm = SequenceLLM([_rewrite_output(_sentence(
            "D1:1",
            "The family admired a lake sunrise last year.",
            tag=[
                "Speaker activity.lake sunrise",
                "Speaker activity.lake sunrise last year",
            ],
        ))])

        output = _rewrite_child_window(
            llm, window, turns, "2023-05-08"
        )

        self.assertEqual(
            output["sentence"][0]["tag"],
            [
                "Speaker activity.lake sunrise",
                "Speaker activity.lake sunrise last-year",
            ],
        )
        self.assertEqual(len(llm.calls), 1)

    def test_tag_length_normalizer_leaves_non_string_for_validation(self):
        output = _rewrite_output(_sentence(
            "D1:1", "memory", tag=["Speaker activity.valid tag", 123]
        ))

        changed = normalize_rewrite_tag_lengths(output)

        self.assertEqual(changed, 0)
        self.assertEqual(
            output["sentence"][0]["tag"],
            ["Speaker activity.valid tag", 123],
        )

    def test_tag_heads_leaked_into_semantic_properties_are_normalized(self):
        output = _rewrite_output(_sentence(
            "D1:1",
            "Caroline plans to attend a workshop.",
            semantic_properties=["plan", "episodic", "activity"],
        ))

        changed = normalize_rewrite_semantic_properties(output)

        self.assertEqual(changed, 3)
        self.assertEqual(
            output["sentence"][0]["semantic_properties"],
            ["event_action", "episodic"],
        )

    def test_child_window_repairs_plan_property_without_llm_retry(self):
        turns = parse_session_turns(_dialogue(1))
        window = ChildWindow("D1:1", "D1:1")
        llm = SequenceLLM([_rewrite_output(_sentence(
            "D1:1",
            "The speaker plans to attend a workshop.",
            semantic_properties=["plan", "episodic"],
        ))])

        output = _rewrite_child_window(
            llm, window, turns, "2023-05-08"
        )

        self.assertEqual(
            output["sentence"][0]["semantic_properties"],
            ["event_action", "episodic"],
        )
        self.assertEqual(len(llm.calls), 1)

    def test_prefix_pool_retries_generic_prefix_and_reads_all_parents(self):
        parents = [
            types.SimpleNamespace(
                parent_id="1-1", rewrite_content="Caroline shared her journey."
            ),
            types.SimpleNamespace(
                parent_id="1-2", rewrite_content="Caroline joined mentoring."
            ),
        ]
        llm = SequenceLLM([
            {"tag_prefix_pool": ["Caroline activity"]},
            {"tag_prefix_pool": ["Caroline advocacy activity"]},
        ])

        pool = _extract_session_tag_prefix_pool(llm, parents)

        self.assertEqual(pool, ["Caroline advocacy activity"])
        self.assertIn("must contain a person", llm.calls[1][0]["content"])
        payload = llm.calls[0][1]["content"]
        self.assertIn("Caroline shared her journey.", payload)
        self.assertIn("Caroline joined mentoring.", payload)

    def test_child_tags_use_exact_pool_prefix_or_two_word_fallback(self):
        turns = parse_session_turns(_dialogue(1))
        window = ChildWindow("D1:1", "D1:1")
        llm = SequenceLLM([_rewrite_output(_sentence(
            "D1:1",
            "Caroline delivered a school speech.",
            tag=[
                "Caroline advocacy activity.school speech",
                "Caroline advocacy activity.journey sharing",
            ],
        ))])

        output = _rewrite_child_window(
            llm,
            window,
            turns,
            "2023-05-08",
            tag_prefix_pool=["Caroline advocacy activity"],
        )

        self.assertEqual(
            output["sentence"][0]["tag"][0],
            "Caroline advocacy activity.school speech",
        )
        self.assertIn(
            '"Caroline advocacy activity"', llm.calls[0][1]["content"]
        )

    def test_duplicate_model_outputs_are_fused_after_window_validation(self):
        turns = parse_session_turns(_dialogue(1))
        window = ChildWindow("D1:1", "D1:1")
        llm = SequenceLLM([
            _rewrite_output(
                _sentence("D1:1", "The same stated information."),
                _sentence("D1:1", "The same stated information."),
            ),
            {"rewrite_content": "The same stated information."},
        ])

        output = _rewrite_child_window(
            llm, window, turns, "2023-05-08"
        )
        retained = []
        with patch(
                "agent.rewrite_memory._embed_child_memory_texts",
                side_effect=lambda texts: [[1.0, 0.0] for _ in texts],
        ):
            kept, _ = _fuse_adjacent_duplicate_child_memories(
                llm, retained, output["sentence"], threshold=0.55
            )

        self.assertEqual(kept, [retained[0]])
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0]["id"], "D1:1-1")
        self.assertEqual(retained[0]["origin"], "D1:1")
        self.assertIn(
            "highly similar adjacent child memories",
            llm.calls[1][0]["content"],
        )

    def test_similarity_must_be_strictly_above_threshold(self):
        previous = _sentence("D1:1", "previous", "D1:1-1")
        current = _sentence("D1:2", "current", "D1:2-1")
        retained = [previous]
        llm = SequenceLLM([])

        with patch(
                "agent.rewrite_memory._embed_child_memory_texts",
                return_value=[[1.0, 0.0]],
        ), patch(
                "agent.rewrite_memory._child_memory_cosine_similarity",
                return_value=0.55,
        ):
            kept, _ = _fuse_adjacent_duplicate_child_memories(
                llm,
                retained,
                [current],
                previous_embedding=[1.0, 0.0],
                threshold=0.55,
            )

        self.assertEqual(kept, [current])
        self.assertEqual(retained, [previous, current])
        self.assertEqual(llm.calls, [])

    def test_child_prompt_no_longer_requires_independent_duplicates(self):
        prompt = Prompts.CHILD_WINDOW_REWRITE_SYSTEM_PROMPT

        self.assertNotIn("Preserve every repeated occurrence", prompt)
        self.assertNotIn("every repeated occurrence", prompt)
        self.assertIn(
            "Do not create multiple sentence objects that merely restate",
            prompt,
        )
        self.assertIn("directly answers the immediately preceding", prompt)
        self.assertIn(
            "origin must begin with the question origin followed by the answer origin",
            prompt,
        )

    def test_threshold_one_bypasses_child_similarity_embeddings(self):
        retained = [_sentence("D1:1", "previous", "D1:1-1")]
        current = _sentence("D1:2", "current", "D1:2-1")

        with patch(
                "agent.rewrite_memory._embed_child_memory_texts",
                side_effect=AssertionError("embedding must not be called"),
        ):
            kept, last_embedding = _fuse_adjacent_duplicate_child_memories(
                SequenceLLM([]), retained, [current], threshold=1.0
            )

        self.assertEqual(kept, [current])
        self.assertEqual(retained[-1], current)
        self.assertIsNone(last_embedding)

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
            _sentence(
                "D1:5",
                "repeated information",
                tag=[
                    "Speaker activity.different tag",
                    "Speaker activity.alternate detail",
                ],
                semantic_properties=["state_opinion", "transient"],
            ),
        )
        llm = SequenceLLM([
            {"parent_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:5"}
            ]},
            {"child_segments": [
                {"start_origin": "D1:1", "end_origin": "D1:3"},
                {"start_origin": "D1:4", "end_origin": "D1:5"},
            ]},
            {
                "parent_id": "1-1",
                "rewrite_content": "A coarse parent memory.",
            },
            {"tag_prefix_pool": ["Speaker conversation activity"]},
            first_window,
            second_window,
            {
                "rewrite_content": "The repeated information was stated once."
            },
        ])

        embedding_by_text = {
            "first turn": [1.0, 0.0],
            "first fact in turn two": [0.0, 1.0],
            "second fact in turn two": [-1.0, 0.0],
            "third turn": [0.0, -1.0],
            "repeated information": [1.0, 0.0],
            "The repeated information was stated once.": [1.0, 0.0],
        }
        with patch(
                "agent.rewrite_memory._embed_child_memory_texts",
                side_effect=lambda texts: [
                    embedding_by_text[text] for text in texts
                ],
        ):
            output = rewrite_semantic_hierarchy_session(llm, _dialogue(5))

        expected_ids = [
            "D1:1-1", "D1:2-1", "D1:2-2",
            "D1:3-1", "D1:4-1",
        ]
        self.assertEqual(
            [item["id"] for item in output["sentence"]], expected_ids
        )
        self.assertEqual(output["parent_nodes"], [{
            "parent_id": "1-1",
            "rewrite_content": "A coarse parent memory.",
            "child_ids": expected_ids,
        }])
        self.assertEqual(
            output["tag_prefix_pool"], ["Speaker conversation activity"]
        )
        self.assertTrue(all(
            item["parent_id"] == "1-1" for item in output["sentence"]
        ))
        fused = output["sentence"][-1]
        self.assertEqual(fused["text"], "The repeated information was stated once.")
        self.assertEqual(fused["origin"], "D1:4,D1:5")
        self.assertEqual(fused["tag"], [
            "Speaker activity.turn memory",
            "Speaker activity.dialogue detail",
        ])
        self.assertEqual(
            fused["semantic_properties"], ["event_action", "episodic"]
        )
        self.assertEqual(len(llm.calls), 7)
        parent_rewrite_call = llm.calls[2]
        self.assertIn("PERSON PROFILE MEMORY", parent_rewrite_call[0]["content"])
        prefix_pool_call = llm.calls[3]
        self.assertIn("topic-prefix pool", prefix_pool_call[0]["content"])
        second_prompt = llm.calls[5][1]["content"]
        reference_section = second_prompt.split("CURRENT_CHILD_WINDOW:", 1)[0]
        self.assertIn("second fact in turn two", reference_section)
        self.assertIn("third turn", reference_section)
        self.assertNotIn("first fact in turn two", reference_section)
        self.assertNotIn(
            "every repeated occurrence", llm.calls[5][0]["content"]
        )
        fusion_call = llm.calls[6]
        self.assertIn(
            "highly similar adjacent child memories",
            fusion_call[0]["content"],
        )

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
        self.assertEqual(
            [memory["parent_id"] for memory in memories],
            ["1-1", "1-1", "1-2"],
        )

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
            parent_id="1-1",
        )
        memory.add_eaes_memory_note(note)
        parent = EAESParentNode(
            parent_id="1-1",
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
            "parent_id": "1-1",
            "rewrite_content": "A coarse parent memory.",
        })
        self.assertEqual(
            memory.get_eaes_support_origin(["1-1"]),
            ["1-1", "D1:2"],
        )


if __name__ == "__main__":
    unittest.main()
