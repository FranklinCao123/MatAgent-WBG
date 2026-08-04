"""Tests for the thin FastAPI layer around the existing Agent graph."""

import asyncio
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from matagent.web import ScreenRequest, app, health, index, screen


class WebInterfaceTests(unittest.TestCase):
    def test_health_and_page_are_available(self) -> None:
        self.assertEqual(health()["status"], "ok")
        self.assertIn(b"MatAgent-WBG", index().body)
        self.assertIn("/api/screen", {route.path for route in app.routes})

    @patch("matagent.web.build_runtime_graph")
    def test_screen_returns_agent_result_without_secrets(self, build_graph) -> None:
        graph = Mock()
        graph.invoke.return_value = {
            "status": "completed",
            "final_report": "# Report",
            "ranked_candidates": [{"name": "AlN", "band_gap_ev": 4.05}],
            "candidate_evidence": {"AlN": [{"doi": "10.1/aln"}]},
            "tool_history": [{"step": "retrieve_evidence", "status": "success"}],
        }
        build_graph.return_value = graph

        response = asyncio.run(
            screen(
                ScreenRequest(
                    query="screen high-temperature materials",
                    use_rag=True,
                )
            )
        )

        self.assertEqual(response.status, "completed")
        self.assertEqual(response.candidates[0]["name"], "AlN")
        self.assertEqual(response.evidence["AlN"][0]["doi"], "10.1/aln")
        graph.invoke.assert_called_once_with(
            {"user_query": "screen high-temperature materials"}
        )
        self.assertNotIn("api_key", response.model_dump_json().casefold())

    def test_invalid_request_is_rejected_by_pydantic(self) -> None:
        with self.assertRaises(ValidationError):
            ScreenRequest(query="", use_rag=True, unexpected="field")


if __name__ == "__main__":
    unittest.main()
