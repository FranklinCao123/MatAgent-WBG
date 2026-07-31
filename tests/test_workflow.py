"""Tests for the deterministic local workflow."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from matagent.graph import build_graph
from matagent.llm import DeepSeekRequirementParser, RequirementParsingError
from matagent.nodes import parse_requirements, plan_screening, route_after_search
from matagent.schemas import RankingWeights, ScreeningRequirements


class FakeDeepSeekClient:
    def __init__(self, content: str) -> None:
        self.last_request = None
        self._content = content
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._content),
                )
            ]
        )


class FailingRequirementParser:
    name = "failing_test_parser"

    def parse(self, user_query: str) -> ScreeningRequirements:
        raise RequirementParsingError("Test parser failure.")


class StaticPowerRequirementParser:
    name = "static_power_test_parser"

    def parse(self, user_query: str) -> ScreeningRequirements:
        return ScreeningRequirements(
            application="high-temperature power devices",
            minimum_band_gap_ev=3.0,
            prefer_high_thermal_conductivity=True,
            prefer_high_breakdown_field=False,
            assumptions=[],
        )


class DeepSeekRequirementParserTests(unittest.TestCase):
    def test_json_response_becomes_validated_requirements(self) -> None:
        client = FakeDeepSeekClient(
            json.dumps(
                {
                    "application": "high-temperature power electronics",
                    "minimum_band_gap_ev": 3.0,
                    "prefer_high_thermal_conductivity": True,
                    "prefer_high_breakdown_field": True,
                    "assumptions": [],
                }
            )
        )
        parser = DeepSeekRequirementParser(
            api_key="test-key-not-real",
            client=client,
        )

        requirements = parser.parse("Find materials with band gap above 3 eV")

        self.assertEqual(requirements.minimum_band_gap_ev, 3.0)
        self.assertEqual(
            client.last_request["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(client.last_request["model"], "deepseek-v4-flash")

    def test_invalid_json_is_reported_as_parsing_error(self) -> None:
        parser = DeepSeekRequirementParser(
            api_key="test-key-not-real",
            client=FakeDeepSeekClient("not-json"),
        )

        with self.assertRaises(RequirementParsingError):
            parser.parse("Find a material")


class ScreeningRequirementsTests(unittest.TestCase):
    def test_invalid_band_gap_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ScreeningRequirements(minimum_band_gap_ev=-1)

    def test_unexpected_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ScreeningRequirements(
                minimum_band_gap_ev=2.0,
                unsupported_property="unexpected",  # type: ignore[call-arg]
            )

    def test_ranking_weights_must_sum_to_one(self) -> None:
        with self.assertRaises(ValidationError):
            RankingWeights(
                band_gap_ev=0.5,
                thermal_conductivity_w_mk=0.5,
                breakdown_field_mv_cm=0.5,
            )


class ScientificPlanningTests(unittest.TestCase):
    def test_power_application_infers_breakdown_priority(self) -> None:
        requirements = ScreeningRequirements(
            application="high-temperature power devices",
            minimum_band_gap_ev=3.0,
            prefer_high_thermal_conductivity=True,
            prefer_high_breakdown_field=False,
            assumptions=[],
        )

        update = plan_screening(
            {
                "requirements": requirements,
                "tool_history": [],
            }
        )
        plan = update["ranking_plan"]

        self.assertGreater(
            plan.weights.breakdown_field_mv_cm,
            plan.weights.band_gap_ev,
        )
        self.assertTrue(
            any(
                "power-device" in item
                for item in plan.inferred_requirements
            )
        )

    def test_inferred_requirement_reaches_final_report(self) -> None:
        result = build_graph(
            requirement_parser=StaticPowerRequirementParser()
        ).invoke(
            {
                "user_query": "Find high-temperature power materials",
                "tool_history": [],
                "errors": [],
                "status": "started",
            }
        )

        self.assertIn(
            "High breakdown field was inferred from the power-device application.",
            result["final_report"],
        )


class RequirementParsingTests(unittest.TestCase):
    def test_chinese_power_query_is_structured(self) -> None:
        update = parse_requirements(
            {
                "user_query": "寻找适合高温功率电子器件的材料",
                "tool_history": [],
            }
        )

        requirements = update["requirements"]
        self.assertIsInstance(requirements, ScreeningRequirements)
        self.assertEqual(requirements.application, "power electronics")
        self.assertTrue(requirements.prefer_high_thermal_conductivity)
        self.assertTrue(requirements.prefer_high_breakdown_field)


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
            [
                "parse_requirements",
                "plan_screening",
                "search_materials",
                "rank_candidates",
            ],
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
            ["parse_requirements", "plan_screening", "search_materials"],
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

    def test_requirement_parser_failure_skips_material_tool(self) -> None:
        result = build_graph(
            requirement_parser=FailingRequirementParser()
        ).invoke(self.initial_state())

        self.assertNotIn("candidates", result)
        self.assertIn("Test parser failure", result["final_report"])
        self.assertEqual(
            [entry["step"] for entry in result["tool_history"]],
            ["parse_requirements"],
        )
        self.assertEqual(result["status"], "completed")


class RoutingTests(unittest.TestCase):
    def test_candidates_route_to_ranking(self) -> None:
        self.assertEqual(route_after_search({"candidates": [{"name": "GaN"}]}), "rank")

    def test_empty_candidates_route_to_report(self) -> None:
        self.assertEqual(route_after_search({"candidates": []}), "report")


if __name__ == "__main__":
    unittest.main()
