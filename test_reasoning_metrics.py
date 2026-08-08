import unittest

from eval.reasoning_metrics import (
    compute_wrong_given_hit20,
    format_wrong_given_hit20,
    hit_at_20,
)


class ReasoningConditionalMetricTests(unittest.TestCase):
    def test_wrong_given_hit20_requires_all_gold_evidence(self):
        judged = [
            {
                "category": 4,
                "llm_score": 1,
                "evidence": ["D1:1", "D1:2"],
                "prediction_context": ["D1:1", "D1:2", "D1:9"],
            },
            {
                "category": 4,
                "llm_score": 0,
                "evidence": ["D1:3", "D1:4"],
                "prediction_context": ["D1:3,D1:4", "D1:10"],
            },
            {
                "category": 4,
                "llm_score": 0,
                "evidence": ["D1:5", "D1:6"],
                "prediction_context": ["D1:5", "D1:6"],
            },
            {
                "category": 4,
                "llm_score": 0,
                "evidence": ["D1:7", "D1:8"],
                "prediction_context": ["D1:7"],
            },
        ]

        stats = compute_wrong_given_hit20(judged)

        category = stats["by_category"][4]
        self.assertEqual(category["hit20"], 3)
        self.assertEqual(category["correct_and_hit20"], 1)
        self.assertEqual(category["wrong_and_hit20"], 2)
        self.assertEqual(category["probability"], 2 / 3)
        rendered = format_wrong_given_hit20(stats)
        self.assertIn("P(wrong | Hit@20)=0.6667", rendered)
        self.assertIn("OVERALL: 2/3", rendered)

    def test_only_first_twenty_context_nodes_count(self):
        context = [f"D1:{index}" for index in range(1, 22)]

        self.assertEqual(hit_at_20({
            "evidence": ["D1:20"],
            "prediction_context": context,
        }), 1)
        self.assertEqual(hit_at_20({
            "evidence": ["D1:21"],
            "prediction_context": context,
        }), 0)

    def test_missing_gold_evidence_does_not_enter_denominator(self):
        stats = compute_wrong_given_hit20([{
            "category": 4,
            "llm_score": 1,
            "evidence": [],
            "prediction_context": ["D1:1"],
        }])

        self.assertEqual(stats["overall"]["judged"], 1)
        self.assertEqual(stats["overall"]["hit20_metric_available"], 0)
        self.assertIsNone(stats["overall"]["probability"])


if __name__ == "__main__":
    unittest.main()
