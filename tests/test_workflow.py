"""Tests for the deterministic local workflow."""

import json
import tempfile
import unittest
from pathlib import Path

from matagent.graph import build_graph
from matagent.nodes import parse_requirements, route_after_search


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
    @staticmethod
    def initial_state() -> dict:
        return {
            "user_query": "寻找适合高温功率电子器件的宽禁带半导体材料",
            "tool_history": [],
            "errors": [],
            "status": "started",
        }

    def test_full_graph_returns_ranked_mock_candidates(self) -> None:
        graph = build_graph()
        result = graph.invoke(self.initial_state())

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
        self.assertEqual(result["status"], "completed")

    def test_no_candidates_skips_ranking_and_generates_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "materials.json"
            data_path.write_text(
                json.dumps(
                    [
                        {
                            "name": "Silicon reference",
                            "band_gap_ev": 1.12,
                            "thermal_conductivity_w_mk": 150,
                            "breakdown_field_mv_cm": 0.3,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            result = build_graph(data_path).invoke(self.initial_state())

        self.assertEqual(result["candidates"], [])
        self.assertNotIn("ranked_candidates", result)
        self.assertIn("No mock candidates found", result["final_report"])
        self.assertEqual(
            [entry["step"] for entry in result["tool_history"]],
            ["parse_requirements", "search_materials"],
        )
        self.assertEqual(result["status"], "completed")

    def test_missing_dataset_becomes_reported_tool_error(self) -> None:
        missing_path = Path("tests") / "file-that-does-not-exist.json"
        result = build_graph(missing_path).invoke(self.initial_state())

        self.assertEqual(result["candidates"], [])
        self.assertTrue(result["errors"])
        self.assertIn("failed", result["final_report"])
        self.assertEqual(result["tool_history"][-1]["status"], "error")
        self.assertEqual(result["status"], "completed")


class RoutingTests(unittest.TestCase):
    def test_candidates_route_to_ranking(self) -> None:
        self.assertEqual(route_after_search({"candidates": [{"name": "GaN"}]}), "rank")

    def test_empty_candidates_route_to_report(self) -> None:
        self.assertEqual(route_after_search({"candidates": []}), "report")


if __name__ == "__main__":
    unittest.main()
