"""Tests for deterministic RAG retrieval metrics."""

import unittest

from matagent.evaluation import RetrievalEvaluationCase, evaluate_retrieval


class RetrievalEvaluationTests(unittest.TestCase):
    def test_metrics_use_doi_identity_and_rank(self) -> None:
        cases = [
            RetrievalEvaluationCase(
                query="Why is SiC useful at high temperature?",
                expected_dois=frozenset({"10.1/sic", "10.1/review"}),
                retrieved_dois=("10.1/noise", "https://doi.org/10.1/SIC", "10.1/x"),
            ),
            RetrievalEvaluationCase(
                query="What limits beta-Ga2O3 thermal performance?",
                expected_dois=frozenset({"10.1/ga2o3"}),
                retrieved_dois=("10.1/ga2o3", "10.1/noise"),
            ),
        ]

        metrics = evaluate_retrieval(cases, k=3)

        self.assertEqual(metrics.case_count, 2)
        self.assertAlmostEqual(metrics.precision_at_k, 1 / 3)
        self.assertAlmostEqual(metrics.recall_at_k, 0.75)
        self.assertAlmostEqual(metrics.mean_reciprocal_rank, 0.75)
        self.assertEqual(metrics.hit_rate_at_k, 1.0)

    def test_empty_cases_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            evaluate_retrieval([], k=5)


if __name__ == "__main__":
    unittest.main()
