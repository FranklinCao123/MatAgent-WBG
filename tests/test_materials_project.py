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
                "band_gap": 3.0,
                "is_stable": True,
                "energy_above_hull": 0.0,
                "formation_energy_per_atom": -1.0,
            },
            {
                "material_id": "mp-stable",
                "formula_pretty": "StableX",
                "band_gap": 3.5,
                "is_stable": True,
                "energy_above_hull": 0.0,
                "formation_energy_per_atom": -2.0,
            },
            {
                "material_id": "mp-meta",
                "formula_pretty": "MetaY",
                "band_gap": 4.2,
                "is_stable": False,
                "energy_above_hull": 0.05,
                "formation_energy_per_atom": -1.5,
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
            )
        )

        self.assertEqual(
            [candidate["material_id"] for candidate in candidates],
            ["mp-stable", "mp-meta"],
        )
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
                "band_gap",
                "is_stable",
                "energy_above_hull",
                "formation_energy_per_atom",
            ],
        )
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

        self.assertEqual(candidates[0]["material_id"], "mp-equal")

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
        self.assertIn("Materials Project screening ranking", result["final_report"])
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


if __name__ == "__main__":
    unittest.main()
