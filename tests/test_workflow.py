"""Tests for the deterministic local workflow."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic import ValidationError

from matagent.graph import build_graph
from matagent.llm import (
    DeepSeekRequirementParser,
    DeepSeekToolSelector,
    RequirementParsingError,
    RuleBasedRequirementParser,
    ToolSelectionError,
)
from matagent.workflow import (
    plan_screening,
    route_after_tool_decision,
    route_after_tool_execution,
)
from matagent.schemas import RankingWeights, ScreeningRequirements
from matagent.tools import (
    MaterialSearchArguments,
    MockMaterialSearchTool,
    ToolCallRequest,
    ToolExecutionError,
    ToolRegistry,
)


class FakeDeepSeekClient:
    def __init__(self, content: str | None = None, tool_calls=None) -> None:
        self.last_request = None
        self._content = content
        self._tool_calls = tool_calls
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )

    def _create(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self._content,
                        tool_calls=self._tool_calls,
                    ),
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
            band_gap_operator=">",
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
                    "band_gap_operator": ">",
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
        self.assertEqual(requirements.band_gap_operator, ">")
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


class DeepSeekToolSelectorTests(unittest.TestCase):
    def test_deepseek_tool_call_is_converted_without_execution(self) -> None:
        requirements = ScreeningRequirements(
            application="power devices",
            minimum_band_gap_ev=3.0,
            band_gap_operator=">",
            prefer_high_thermal_conductivity=True,
            prefer_high_breakdown_field=True,
            assumptions=[],
        )
        ranking_plan = plan_screening(
            {"requirements": requirements, "tool_history": []}
        )["ranking_plan"]
        raw_call = SimpleNamespace(
            id="call-search-1",
            function=SimpleNamespace(
                name="search_materials",
                arguments=json.dumps(
                    {
                        "band_gap_threshold_ev": 3.0,
                        "band_gap_operator": ">",
                    }
                ),
            ),
        )
        client = FakeDeepSeekClient(tool_calls=[raw_call])
        selector = DeepSeekToolSelector(
            api_key="test-key-not-real",
            client=client,
        )

        calls = selector.select(
            user_query="Find materials above 3 eV",
            requirements=requirements,
            ranking_plan=ranking_plan,
            tool_specs=[
                {
                    "type": "function",
                    "function": {
                        "name": "search_materials",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )

        self.assertEqual(calls[0].name, "search_materials")
        self.assertEqual(calls[0].arguments["band_gap_operator"], ">")
        self.assertEqual(client.last_request["tool_choice"], "auto")
        self.assertIn("tools", client.last_request)

    def test_invalid_tool_argument_json_is_rejected(self) -> None:
        raw_call = SimpleNamespace(
            id="call-invalid-1",
            function=SimpleNamespace(
                name="search_materials",
                arguments="not-json",
            ),
        )
        selector = DeepSeekToolSelector(
            api_key="test-key-not-real",
            client=FakeDeepSeekClient(tool_calls=[raw_call]),
        )
        requirements = ScreeningRequirements(
            application="power devices",
            minimum_band_gap_ev=3.0,
            band_gap_operator=">",
        )
        ranking_plan = plan_screening(
            {"requirements": requirements, "tool_history": []}
        )["ranking_plan"]

        with self.assertRaises(ToolSelectionError):
            selector.select(
                user_query="Find materials",
                requirements=requirements,
                ranking_plan=ranking_plan,
                tool_specs=[],
            )


class MaterialSearchToolTests(unittest.TestCase):
    def test_strict_and_inclusive_thresholds_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "boundary-materials.json"
            data_path.write_text(
                json.dumps(
                    [
                        {"name": "Equal", "band_gap_ev": 3.0},
                        {"name": "Above", "band_gap_ev": 3.1},
                    ]
                ),
                encoding="utf-8",
            )
            tool = MockMaterialSearchTool(data_path)
            strict = tool.search(
                MaterialSearchArguments(
                    band_gap_threshold_ev=3.0,
                    band_gap_operator=">",
                )
            )
            inclusive = tool.search(
                MaterialSearchArguments(
                    band_gap_threshold_ev=3.0,
                    band_gap_operator=">=",
                )
            )

        self.assertEqual(
            [item["name"] for item in strict["candidates"]],
            ["Above"],
        )
        self.assertEqual(
            [item["name"] for item in inclusive["candidates"]],
            ["Equal", "Above"],
        )


class ToolRegistryTests(unittest.TestCase):
    def test_unregistered_tool_is_rejected(self) -> None:
        registry = ToolRegistry()

        with self.assertRaises(ToolExecutionError):
            registry.execute(
                ToolCallRequest(
                    id="unknown-1",
                    name="delete_files",
                    arguments={},
                )
            )


class ScreeningRequirementsTests(unittest.TestCase):
    def test_invalid_band_gap_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ScreeningRequirements(
                minimum_band_gap_ev=-1,
                band_gap_operator=">=",
            )

    def test_unexpected_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ScreeningRequirements(
                minimum_band_gap_ev=2.0,
                band_gap_operator=">=",
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
            band_gap_operator=">",
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
    def test_chinese_numeric_band_gap_constraint_is_preserved(self) -> None:
        requirements = RuleBasedRequirementParser().parse(
            "寻找带隙大于3 eV、适合高温功率器件的材料"
        )

        self.assertEqual(requirements.minimum_band_gap_ev, 3.0)
        self.assertEqual(requirements.band_gap_operator, ">")
        self.assertEqual(requirements.application, "power electronics")
        self.assertTrue(requirements.prefer_high_thermal_conductivity)

    def test_chinese_power_query_is_structured(self) -> None:
        requirements = RuleBasedRequirementParser().parse(
            "寻找适合高温功率电子器件的材料"
        )

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
                "decide_tools",
                "execute_tool",
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
        self.assertEqual(result["ranked_candidates"], [])
        self.assertIn("No mock candidates found", result["final_report"])
        self.assertEqual(
            [entry["step"] for entry in result["tool_history"]],
            [
                "parse_requirements",
                "plan_screening",
                "decide_tools",
                "execute_tool",
            ],
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

        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["ranked_candidates"], [])
        self.assertIn("Test parser failure", result["final_report"])
        self.assertEqual(
            [entry["step"] for entry in result["tool_history"]],
            ["parse_requirements"],
        )
        self.assertEqual(result["status"], "completed")


class RoutingTests(unittest.TestCase):
    def test_selected_call_routes_to_execution(self) -> None:
        self.assertEqual(
            route_after_tool_decision(
                {"pending_tool_calls": [{"name": "search_materials"}]}
            ),
            "execute",
        )

    def test_candidates_route_to_ranking(self) -> None:
        self.assertEqual(
            route_after_tool_execution({"candidates": [{"name": "GaN"}]}),
            "rank",
        )

    def test_empty_candidates_route_to_report(self) -> None:
        self.assertEqual(route_after_tool_execution({"candidates": []}), "report")


if __name__ == "__main__":
    unittest.main()
