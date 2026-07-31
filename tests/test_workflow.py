"""Tests for the deterministic local workflow."""

import unittest

from matagent.graph import build_graph
from matagent.nodes import parse_requirements


class RequirementParsingTests(unittest.TestCase):
    def test_chinese_power_query_is_structured(self) -> None:
        update = parse_requirements(
            {
                "user_query": "寻找适合高温功率电子器件的材料",
                "tool_history": [],
            }
        )

        requirements = update["requirements"]
        self.assertEqual(requirements["application"], "power electronics")
        self.assertTrue(requirements["prefer_high_thermal_conductivity"])
        self.assertTrue(requirements["prefer_high_breakdown_field"])


class WorkflowTests(unittest.TestCase):
    def test_full_graph_returns_ranked_mock_candidates(self) -> None:
        graph = build_graph()
        result = graph.invoke(
            {
                "user_query": "寻找适合高温功率电子器件的宽禁带半导体材料",
                "tool_history": [],
            }
        )

        self.assertEqual(len(result["candidates"]), 5)
        self.assertNotIn(
            "Silicon reference",
            [candidate["name"] for candidate in result["candidates"]],
        )
        self.assertEqual(result["ranked_candidates"][0]["name"], "Diamond")
        self.assertIn("illustrative mock data", result["final_report"])
        self.assertEqual(
            [entry["step"] for entry in result["tool_history"]],
            ["parse_requirements", "search_materials", "rank_candidates"],
        )


if __name__ == "__main__":
    unittest.main()
