import sys
import types
import unittest


common_module = types.ModuleType("common")
common_module.config = types.SimpleNamespace(
    REWRITE_WINDOW_SIZE=40,
    REWRITE_OVERLAP_SIZE=2,
    REWRITE_PREVIOUS_LIMIT=3,
)
sys.modules.setdefault("common", common_module)
sys.modules.setdefault("common.config", common_module.config)

from agent.rewrite_memory import normalize_rewrite_temporal_granularity
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


if __name__ == "__main__":
    unittest.main()
