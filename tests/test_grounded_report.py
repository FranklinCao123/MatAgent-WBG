"""Tests for citation-bounded LLM report synthesis."""

import json
import unittest
from types import SimpleNamespace

from matagent.graph import build_graph
from matagent.llm import DeepSeekReportSynthesizer, ReportSynthesisError
from matagent.schemas import (
    CandidateAssessment,
    GroundedReport,
    ScreeningRequirements,
)


class FakeClient:
    def __init__(self, payload, *, fenced=False) -> None:
        self.payload = payload
        self.fenced = fenced
        self.last_request = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    def create(self, **kwargs):
        self.last_request = kwargs
        content = json.dumps(self.payload)
        if self.fenced:
            content = f"```json\n{content}\n```"
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                )
            ]
        )


def requirements() -> ScreeningRequirements:
    return ScreeningRequirements(
        application="high-temperature power devices",
        minimum_band_gap_ev=3.0,
        band_gap_operator=">",
        prefer_high_thermal_conductivity=True,
    )


class DeepSeekReportTests(unittest.TestCase):
    def test_valid_candidate_and_doi_are_accepted(self) -> None:
        client = FakeClient(
            {
                "executive_summary": "SiC is supported by limited evidence.",
                "candidate_assessments": [
                    {
                        "material": "4H-SiC",
                        "assessment": "The retrieved passage supports thermal use.",
                        "confidence": "moderate",
                        "evidence_dois": ["10.1/example"],
                    }
                ],
                "caveats": ["Only one abstract was retrieved."],
            },
            fenced=True,
        )
        synthesizer = DeepSeekReportSynthesizer(
            api_key="test-key",
            client=client,
        )

        report = synthesizer.synthesize(
            user_query="screen SiC",
            requirements=requirements(),
            ranked_candidates=[
                {"name": "4H-SiC", "band_gap_ev": 3.26},
                {"name": "GaN", "band_gap_ev": 3.4},
                {"name": "AlN", "band_gap_ev": 6.0},
                {"name": "Diamond", "band_gap_ev": 5.47},
            ],
            candidate_evidence={
                "4H-SiC": [
                    {
                        "doi": "10.1/example",
                        "title": "SiC evidence",
                        "content": "Measured thermal transport.",
                        "similarity": 0.8,
                    }
                ]
            },
        )

        self.assertEqual(report.candidate_assessments[0].material, "4H-SiC")
        self.assertEqual(client.last_request["response_format"], {"type": "json_object"})
        context = json.loads(client.last_request["messages"][1]["content"])
        self.assertEqual(len(context["ranked_candidates"]), 3)

    def test_evidence_supported_candidate_enters_llm_context(self) -> None:
        client = FakeClient(
            {
                "executive_summary": "AlN has retrieved evidence.",
                "candidate_assessments": [
                    {
                        "material": "AlN",
                        "assessment": "Evidence is available.",
                        "confidence": "moderate",
                        "evidence_dois": ["10.1/aln"],
                    }
                ],
                "caveats": [],
            }
        )
        synthesizer = DeepSeekReportSynthesizer(
            api_key="test-key",
            client=client,
        )

        synthesizer.synthesize(
            user_query="screen materials",
            requirements=requirements(),
            ranked_candidates=[
                {"name": "CO2", "band_gap_ev": 6.6},
                {"name": "CsN3", "band_gap_ev": 4.2},
                {"name": "RbN3", "band_gap_ev": 4.1},
                {"name": "AlN", "band_gap_ev": 4.0},
            ],
            candidate_evidence={
                "AlN": [
                    {
                        "doi": "10.1/aln",
                        "title": "AlN evidence",
                        "content": "High-temperature device evidence.",
                        "similarity": 0.8,
                    }
                ]
            },
        )

        context = json.loads(client.last_request["messages"][1]["content"])
        materials = [item["material"] for item in context["ranked_candidates"]]
        self.assertEqual(materials, ["AlN", "CO2", "CsN3"])

    def test_invented_doi_is_rejected(self) -> None:
        client = FakeClient(
            {
                "executive_summary": "Unsupported citation.",
                "candidate_assessments": [
                    {
                        "material": "4H-SiC",
                        "assessment": "Claim.",
                        "confidence": "low",
                        "evidence_dois": ["10.1/invented"],
                    }
                ],
                "caveats": [],
            }
        )
        synthesizer = DeepSeekReportSynthesizer(
            api_key="test-key",
            client=client,
        )

        with self.assertRaises(ReportSynthesisError):
            synthesizer.synthesize(
                user_query="screen SiC",
                requirements=requirements(),
                ranked_candidates=[{"name": "4H-SiC"}],
                candidate_evidence={"4H-SiC": []},
            )


class FakeSynthesizer:
    name = "fake"

    def synthesize(self, **kwargs):
        return GroundedReport(
            executive_summary="Grounded summary.",
            candidate_assessments=[
                CandidateAssessment(
                    material=kwargs["ranked_candidates"][0]["name"],
                    assessment="Uses validated candidate data.",
                    confidence="low",
                    evidence_dois=[],
                )
            ],
            caveats=["No literature supplied in this test."],
        )


class GraphSynthesisTests(unittest.TestCase):
    def test_graph_appends_grounded_section_and_trace(self) -> None:
        result = build_graph(report_synthesizer=FakeSynthesizer()).invoke(
            {"user_query": "寻找带隙大于3 eV的高温功率材料"}
        )

        self.assertIn("Grounded LLM assessment", result["final_report"])
        self.assertEqual(result["tool_history"][-1]["step"], "synthesize_report")
        self.assertEqual(result["tool_history"][-1]["status"], "success")


if __name__ == "__main__":
    unittest.main()
