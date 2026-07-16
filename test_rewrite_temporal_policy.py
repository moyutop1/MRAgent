import sys
import types
import unittest
from unittest.mock import patch


common_module = types.ModuleType("common")
common_module.config = types.SimpleNamespace(
    REWRITE_WINDOW_SIZE=40,
    REWRITE_OVERLAP_SIZE=2,
    REWRITE_PREVIOUS_LIMIT=3,
)
sys.modules.setdefault("common", common_module)
sys.modules.setdefault("common.config", common_module.config)

from agent.rewrite_memory import (
    _validate_current_window_origins,
    _window_turns,
    normalize_rewrite_temporal_granularity,
    rewrite_windowed_session,
)
from common import config
from prompts.prompts import Prompts


class RewriteTemporalPolicyTests(unittest.TestCase):
    @staticmethod
    def _normalize(cue, model_text="Caroline described the event {cue}."):
        output = {
            "sentence": [{
                "id": "D1:1-1",
                "text": model_text.format(cue=cue),
                "origin": "D1:1",
                "time": "1999-01-01",
            }],
        }
        dialogue = (
            "time:2023-07-22\n"
            f"dia_id:D1:1 Caroline: I described the event {cue}."
        )
        normalize_rewrite_temporal_granularity(output, dialogue)
        return output["sentence"][0]

    def test_named_weekday_stays_anchored_but_indexes_actual_day(self):
        sentence = self._normalize("last Friday")
        self.assertIn("the Friday before 22 July 2023", sentence["text"])
        self.assertEqual(sentence["time"], "2023-07-21")

    def test_last_week_uses_anchored_text_and_period_start_index(self):
        sentence = self._normalize("last week")
        self.assertIn("the week before 22 July 2023", sentence["text"])
        self.assertEqual(sentence["time"], "2023-07-10")

    def test_last_weekend_uses_anchored_text_and_period_start_index(self):
        sentence = self._normalize("last weekend")
        self.assertIn("the weekend before 22 July 2023", sentence["text"])
        self.assertEqual(sentence["time"], "2023-07-15")

    def test_exact_day_cues_become_absolute_dates(self):
        yesterday = self._normalize("yesterday")
        two_days = self._normalize("two days ago")
        self.assertIn("21 July 2023", yesterday["text"])
        self.assertEqual(yesterday["time"], "2023-07-21")
        self.assertIn("20 July 2023", two_days["text"])
        self.assertEqual(two_days["time"], "2023-07-20")

    def test_last_month_keeps_month_precision_and_indexes_month_start(self):
        sentence = self._normalize("last month")
        self.assertIn("June 2023", sentence["text"])
        self.assertEqual(sentence["time"], "2023-06-01")

    def test_absolute_model_month_is_reduced_to_source_month_precision(self):
        sentence = self._normalize(
            "last month", model_text="Caroline described the event in 2023-06-01.")
        self.assertEqual(
            sentence["text"],
            "Caroline described the event in June 2023.",
        )

    def test_last_year_keeps_year_precision_and_indexes_year_start(self):
        sentence = self._normalize("last year")
        self.assertIn("2022", sentence["text"])
        self.assertNotIn("2022-01-01", sentence["text"])
        self.assertEqual(sentence["time"], "2022-01-01")

    def test_absolute_model_year_is_reduced_to_source_year_precision(self):
        sentence = self._normalize(
            "last year", model_text="Caroline described the event in 2022-01-01.")
        self.assertEqual(sentence["text"], "Caroline described the event in 2022.")

    def test_absolute_model_output_is_replaced_with_anchored_weekday(self):
        sentence = self._normalize(
            "last Friday", model_text="Caroline described the event on 2023-07-21.")
        self.assertEqual(
            sentence["text"],
            "Caroline described the event on the Friday before 22 July 2023.",
        )

    def test_prompt_documents_text_and_index_granularity_separately(self):
        prompt = Prompts.REWRITE_SYSTEM_PROMPT
        self.assertIn('"last month" -> "June 2023"', prompt)
        self.assertIn('"last year" -> "2022"', prompt)
        self.assertIn('"time" = "2022-01-01"', prompt)


class RewriteWindowContextTests(unittest.TestCase):
    def setUp(self):
        self.original_window_size = config.REWRITE_WINDOW_SIZE
        self.original_overlap_size = config.REWRITE_OVERLAP_SIZE
        config.REWRITE_WINDOW_SIZE = 4
        config.REWRITE_OVERLAP_SIZE = 2

    def tearDown(self):
        config.REWRITE_WINDOW_SIZE = self.original_window_size
        config.REWRITE_OVERLAP_SIZE = self.original_overlap_size

    def test_each_base_window_gets_up_to_overlap_previous_turns(self):
        turns = [f"dia_id:D1:{i} turn {i}" for i in range(1, 8)]

        windows = _window_turns(turns)

        self.assertEqual(windows[0].previous_context, [])
        self.assertEqual(windows[0].current_turns, turns[:4])
        self.assertEqual(windows[1].previous_context, turns[2:4])
        self.assertEqual(windows[1].current_turns, turns[4:])
        self.assertTrue(all(
            len(window.previous_context) + len(window.current_turns) <= 6
            for window in windows
        ))

    def test_context_only_memory_is_rejected_but_cross_window_memory_is_valid(self):
        current_text = (
            "time:2023-07-22\n"
            "dia_id:D1:5 Morgan: I went to the national park with my kids."
        )
        context_only = {
            "sentence": [{"origin": "D1:4"}],
            "personal_sentences": [],
        }
        cross_window = {
            "sentence": [{"origin": "D1:4,D1:5"}],
            "personal_sentences": [],
        }

        valid, _ = _validate_current_window_origins(context_only, current_text)
        cross_valid, _ = _validate_current_window_origins(cross_window, current_text)

        self.assertFalse(valid)
        self.assertTrue(cross_valid)

    def test_boundary_question_supplies_time_to_current_answer(self):
        class BoundaryLLM:
            def __init__(self):
                self.calls = []

            def chat_text(self, messages, **_kwargs):
                self.calls.append(messages)
                if len(self.calls) == 1:
                    return {
                        "conversation_time": "2023-07-22",
                        "sentence": [],
                        "topics": {},
                        "personal_sentences": [],
                    }
                return {
                    "conversation_time": "2023-07-22",
                    "sentence": [{
                        "id": "D1:4-1",
                        "text": "Morgan went to the national park with Morgan's kids.",
                        "tag": "Park Visit",
                        "origin": "D1:4,D1:5",
                        "topic": [],
                        "time": "2023-07-22",
                    }],
                    "topics": {},
                    "personal_sentences": [],
                }

        dialogue = "\n".join([
            "time:2023-07-22",
            "dia_id:D1:1 Morgan: We had breakfast.",
            "dia_id:D1:2 Alex: That sounds nice.",
            "dia_id:D1:3 Morgan: The kids were excited.",
            "dia_id:D1:4 Alex: Where did you go last week?",
            "dia_id:D1:5 Morgan: I went to the national park with my kids.",
        ])
        llm = BoundaryLLM()

        with patch(
                "agent.rewrite_memory.json_scheme.check_rewrite_json",
                return_value=(True, ""),
        ):
            output = rewrite_windowed_session(llm, dialogue)

        self.assertEqual(len(llm.calls), 2)
        second_prompt = llm.calls[1][1]["content"]
        context_section, current_section = second_prompt.split(
            "CURRENT_DIALOGUE_WINDOW", maxsplit=1)
        self.assertIn("D1:4", context_section)
        self.assertNotIn("D1:5", context_section)
        self.assertIn("D1:5", current_section)
        self.assertNotIn("D1:4", current_section)
        self.assertEqual(output["sentence"][0]["origin"], "D1:4,D1:5")
        self.assertIn(
            "the week before 22 July 2023",
            output["sentence"][0]["text"],
        )
        self.assertEqual(output["sentence"][0]["time"], "2023-07-10")


if __name__ == "__main__":
    unittest.main()
