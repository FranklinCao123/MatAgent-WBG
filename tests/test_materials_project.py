"""Tests for the lightweight Materials Project REST backend."""

import json
import unittest
from urllib.parse import parse_qs, urlparse

from matagent.graph import build_graph
from matagent.tools import MaterialSearchArguments, MaterialsProjectSearchTool


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class RecordingOpener:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.request = None
        self.timeout = None

    def __call__(self, request, *, timeout):
        self.request = request
        self.timeout = timeout
        return FakeHTTPResponse(self.payload)


def sample_payload() -> dict:
    return {
        "data": [
            {
                "material_id": "mp-equal",
                "formula_pretty": "Equal3",
                "elements": ["Si"],
                "band_gap": 3.0,
                "is_stable": True,
                "is_metal": False,
                "theoretical": False,
                "energy_above_hull": 0.0,
                "formation_energy_per_atom": -1.0,
            },
            {
                "material_id": "mp-stable",
                "formula_pretty": "StableX",
                "elements": ["Al", "N"],
                "band_gap": 3.5,
                "is_stable": True,
                "is_metal": False,
                "theoretical": False,
                "energy_above_hull": 0.0,
                "formation_energy_per_atom": -2.0,
            },
            {
                "material_id": "mp-meta",
                "formula_pretty": "MetaY",
                "elements": ["Ga", "O"],
                "band_gap": 4.2,
                "is_stable": False,
                "is_metal": False,
                "theoretical": False,
                "energy_above_hull": 0.05,
                "formation_energy_per_atom": -1.5,
            },
            {
                "material_id": "mp-radioactive",
                "formula_pretty": "AcF3",
                "elements": ["Ac", "F"],
                "band_gap": 6.0,
                "is_stable": True,
                "is_metal": False,
                "theoretical": False,
                "energy_above_hull": 0.0,
                "formation_energy_per_atom": -4.0,
            },
        ]
    }


class MaterialsProjectSearchTests(unittest.TestCase):
    def test_request_is_small_authenticated_and_strictly_filtered(self) -> None:
        opener = RecordingOpener(sample_payload())
        tool = MaterialsProjectSearchTool(
            api_key="test-secret",
            max_results=7,
            timeout_seconds=4,
            opener=opener,
        )

        candidates = tool.search(
            MaterialSearchArguments(
                band_gap_threshold_ev=3.0,
                band_gap_operator=">",
                exclude_elements=["Ac"],
                require_nonmetal=True,
                require_experimental=True,
                maximum_element_count=2,
                maximum_energy_above_hull_ev_atom=0.1,
            )
        )

        self.assertEqual(
            [candidate["material_id"] for candidate in candidates["candidates"]],
            ["mp-stable", "mp-meta"],
        )
        self.assertEqual(candidates["retrieved_count"], 4)
        self.assertEqual(candidates["excluded"][1]["material_id"], "mp-radioactive")
        parsed = urlparse(opener.request.full_url)
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.path, "/materials/summary/")
        self.assertEqual(query["band_gap_min"], ["3.0"])
        self.assertEqual(query["_limit"], ["7"])
        self.assertEqual(
            query["_fields"][0].split(","),
            [
                "material_id",
                "formula_pretty",
                "elements",
                "band_gap",
                "is_stable",
                "is_metal",
                "theoretical",
                "energy_above_hull",
                "formation_energy_per_atom",
            ],
        )
        self.assertEqual(query["exclude_elements"], ["Ac"])
        self.assertEqual(query["is_metal"], ["false"])
        self.assertEqual(query["theoretical"], ["false"])
        self.assertEqual(query["nelements_max"], ["2"])
        self.assertEqual(query["energy_above_hull_max"], ["0.1"])
        self.assertEqual(query["_sort_fields"], ["energy_above_hull"])
        self.assertNotIn("test-secret", opener.request.full_url)
        headers = dict(opener.request.header_items())
        self.assertIn("test-secret", headers.values())
        self.assertIn("MatAgent-WBG", headers["User-agent"])
        self.assertEqual(opener.timeout, 4)

    def test_inclusive_operator_keeps_equal_band_gap(self) -> None:
        tool = MaterialsProjectSearchTool(
            api_key="test-secret",
            opener=RecordingOpener(sample_payload()),
        )

        candidates = tool.search(
            MaterialSearchArguments(
                band_gap_threshold_ev=3.0,
                band_gap_operator=">=",
            )
        )

        self.assertEqual(candidates["candidates"][0]["material_id"], "mp-equal")

    def test_graph_uses_stability_ranking_without_mock_properties(self) -> None:
        tool = MaterialsProjectSearchTool(
            api_key="test-secret",
            opener=RecordingOpener(sample_payload()),
        )
        graph = build_graph(
            material_backend="materials-project",
            material_search_tool=tool,
        )

        result = graph.invoke(
            {"user_query": "Find wide-bandgap power semiconductor materials"}
        )

        self.assertEqual(result["material_backend"], "materials-project")
        self.assertEqual(
            result["ranking_plan"].strategy,
            "materials_project_stability",
        )
        self.assertEqual(result["ranked_candidates"][0]["material_id"], "mp-stable")
        self.assertNotIn(
            "mp-radioactive",
            [candidate["material_id"] for candidate in result["candidates"]],
        )
        decided_call = next(
            item for item in result["tool_history"] if item["step"] == "decide_tools"
        )["calls"][0]
        self.assertIn("Ac", decided_call["arguments"]["exclude_elements"])
        self.assertIn("Ar", decided_call["arguments"]["exclude_elements"])
        self.assertTrue(decided_call["arguments"]["require_nonmetal"])
        self.assertTrue(decided_call["arguments"]["require_experimental"])
        self.assertEqual(decided_call["arguments"]["maximum_element_count"], 2)
        self.assertIn("Materials Project screening ranking", result["final_report"])
        self.assertIn("Candidate filtering", result["final_report"])
        self.assertNotIn("Demo score", result["final_report"])

    def test_malformed_response_is_rejected(self) -> None:
        tool = MaterialsProjectSearchTool(
            api_key="test-secret",
            opener=RecordingOpener({"unexpected": []}),
        )

        with self.assertRaisesRegex(ValueError, "data list"):
            tool.search(
                MaterialSearchArguments(
                    band_gap_threshold_ev=3.0,
                    band_gap_operator=">",
                )
            )

    def test_server_exclusion_subset_respects_api_character_limit(self) -> None:
        elements = [f"E{index}" for index in range(30)]

        selected = MaterialsProjectSearchTool._server_excluded_elements(elements)

        self.assertLessEqual(len(",".join(selected)), 60)
        self.assertEqual(selected, elements[: len(selected)])

    def test_local_validation_rejects_theoretical_and_complex_entries(self) -> None:
        payload = sample_payload()
        payload["data"] = [
            {
                **payload["data"][1],
                "material_id": "mp-theory",
                "theoretical": True,
            },
            {
                **payload["data"][1],
                "material_id": "mp-complex",
                "elements": ["Al", "Ga", "N", "O"],
            },
        ]
        tool = MaterialsProjectSearchTool(
            api_key="test-secret",
            opener=RecordingOpener(payload),
        )

        result = tool.search(
            MaterialSearchArguments(
                band_gap_threshold_ev=3.0,
                band_gap_operator=">",
                require_experimental=True,
                maximum_element_count=2,
            )
        )

        self.assertEqual(result["candidates"], [])
        reasons = {
            item["material_id"]: " ".join(item["reasons"])
            for item in result["excluded"]
        }
        self.assertIn("theoretical", reasons["mp-theory"])
        self.assertIn("more than 2", reasons["mp-complex"])


if __name__ == "__main__":
    unittest.main()
