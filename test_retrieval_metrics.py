import unittest
from contextlib import redirect_stdout
from io import StringIO

from eval.evaluate_retrieval import print_group
from eval.retrieval_metrics import retrieval_metrics


class RetrievalMetricTests(unittest.TestCase):
    def test_mrr_uses_memory_node_rank_for_grouped_parent_provenance(self):
        metrics = retrieval_metrics(
            ["D2:11"],
            ["D1:1", "D1:t1", "D2:1", "D2:11"],
            origin_groups=[
                ["D1:1"],
                ["D1:t1", "D2:1", "D2:11"],
            ],
        )

        self.assertEqual(metrics["hit"], 1)
        self.assertEqual(metrics["exact_cover"], 1)
        self.assertEqual(metrics["mrr"], 0.5)

    def test_grouped_metric_preserves_multi_origin_coverage(self):
        metrics = retrieval_metrics(
            ["D1:2", "D1:3"],
            ["D1:2,D1:3"],
            origin_groups=[["D1:2,D1:3"]],
        )

        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["exact_cover"], 1)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_evaluator_prints_explicit_twenty_node_cutoff(self):
        output = StringIO()
        with redirect_stdout(output):
            print_group("OVERALL", [{
                "hit": 1,
                "recall": 1.0,
                "exact_cover": 1,
                "mrr": 0.5,
                "retrieval_k": 20,
            }])

        rendered = output.getvalue()
        self.assertIn("Hit@20=1.0000", rendered)
        self.assertIn("MRR@20=0.5000", rendered)


if __name__ == "__main__":
    unittest.main()
