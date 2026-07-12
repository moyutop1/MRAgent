import unittest
import sys
import types
import inspect

common_module = types.ModuleType("common")
common_module.config = types.SimpleNamespace(dataset="locomo")
sys.modules.setdefault("common", common_module)
sys.modules.setdefault("common.config", common_module.config)

from agent.eaes import EAESMixin
from prompts.prompts import Prompts


class TemporalAnswerPolicyTests(unittest.TestCase):
    def test_extracts_relative_wording_before_absolute_parenthetical(self):
        text = "Caroline attended the workshop last Friday (2023-06-23)."
        self.assertEqual(EAESMixin._eaes_extract_temporal_expression(text), "last Friday")

    def test_recovers_original_relative_wording_from_dialogue(self):
        mixin = EAESMixin()
        mixin.memory = types.SimpleNamespace(
            raw_text={"D8": {"D8:9": "Caroline: Last Friday I went to a council meeting."}}
        )
        event = types.SimpleNamespace(
            origin="D8:9", text="Caroline attended the meeting on 2023-07-07.")
        source = mixin._eaes_raw_source_text(event)
        self.assertIn("Last Friday", source)

    def test_named_weekday_uses_anchor_instead_of_computed_date(self):
        answer = EAESMixin._eaes_anchor_relative_answer("last Friday", "2023-07-15")
        self.assertEqual(answer, "The Friday before 15 July 2023")

    def test_week_uses_anchor_instead_of_computed_date(self):
        answer = EAESMixin._eaes_anchor_relative_answer("last week", "2023-07-03")
        self.assertEqual(answer, "The week before 3 July 2023")

    def test_day_distance_uses_anchored_relative_style(self):
        answer = EAESMixin._eaes_anchor_relative_answer("a few days ago", "2023-11-22")
        self.assertEqual(answer, "A few days before 22 November 2023")

    def test_exact_day_distance_is_left_for_absolute_date_resolution(self):
        answer = EAESMixin._eaes_anchor_relative_answer("two days ago", "2023-07-12")
        self.assertIsNone(answer)

    def test_exact_day_distance_uses_normalized_event_date(self):
        answer = EAESMixin._eaes_precise_temporal_answer("two days ago", "2023-07-10")
        self.assertEqual(answer, "10 July 2023")

    def test_last_year_uses_normalized_event_year(self):
        answer = EAESMixin._eaes_precise_temporal_answer("last year", "2022-01-01")
        self.assertEqual(answer, "2022")

    def test_last_week_of_month_is_preserved_for_model(self):
        answer = EAESMixin._eaes_anchor_relative_answer("last week of July 2023", "2023-08-01")
        self.assertIsNone(answer)

    def test_policy_is_generic_and_forbids_iso_for_coarse_expressions(self):
        policy = Prompts.TEMPORAL_ANSWER_POLICY
        self.assertIn("Do not output an ISO date", policy)
        self.assertIn("The Friday before 15 July 2023", policy)
        self.assertNotIn("locomo", policy.lower())
        self.assertNotIn("category", policy.lower())

    def test_memory_note_does_not_add_an_anchor_field(self):
        source = inspect.getsource(EAESMixin._eaes_build_notes_for_session)
        self.assertNotIn('"anchor"', source)
        self.assertNotIn('"anchor_type"', source)

    def test_supported_source_expression_overrides_absolute_prediction(self):
        mixin = EAESMixin()
        mixin.memory = types.SimpleNamespace(
            raw_text={
                "D8": {"D8:9": "Caroline: Last Friday I went to a council meeting."},
            },
            episode_events={
                "D8:9-1": types.SimpleNamespace(
                    origin="D8:9",
                    text="Caroline attended the meeting on 2023-07-07.",
                    conversation_time="2023-07-15",
                    time="2023-07-07",
                ),
            },
        )
        evidence = {
            "answer_items": [{
                "evidence": [{
                    "memory_id": "M_D8_9_1",
                    "event_id": "D8:9-1",
                    "time_interval": {"start": "2023-07-15"},
                }],
            }],
        }
        answer = mixin._eaes_temporal_answer(
            "2023-07-07", ["M_D8_9_1"], evidence, [])
        self.assertEqual(answer, "The Friday before 15 July 2023")

    def test_unsupported_backup_cue_does_not_override_supported_answer(self):
        mixin = EAESMixin()
        mixin.memory = types.SimpleNamespace(
            raw_text={
                "D7": {"D7:1": "Caroline attended the conference two days ago."},
                "D8": {"D8:9": "Caroline attended a meeting last Friday."},
            },
            episode_events={
                "supported-event": types.SimpleNamespace(
                    origin="D7:1", text="Caroline attended the conference.",
                    conversation_time="2023-07-12", time="2023-07-10"),
                "unrelated-event": types.SimpleNamespace(
                    origin="D8:9", text="Caroline attended a meeting.",
                    conversation_time="2023-07-15", time="2023-07-07"),
            },
        )
        evidence = {
            "answer_items": [{
                "evidence": [{
                    "memory_id": "supported",
                    "event_id": "supported-event",
                    "time_interval": {"start": "2023-07-12"},
                }],
            }],
        }
        candidates = [{
            "memory_id": "unrelated",
            "event_id": "unrelated-event",
            "time_interval": {"start": "2023-07-15"},
        }]
        answer = mixin._eaes_temporal_answer(
            "10 July 2023", ["supported"], evidence, candidates)
        self.assertEqual(answer, "10 July 2023")


if __name__ == "__main__":
    unittest.main()
