"""Tests for the query-side RAG tool and LangGraph integration."""

import unittest

from matagent.graph import build_graph
from matagent.rag.retriever import EMBEDDING_DIMENSION, EvidenceChunk
from matagent.tools import (
    CandidateEvidenceArguments,
    ScientificEvidenceTool,
)


class FakeEmbeddingProvider:
    model_name = "BAAI/bge-m3"
    dimension = EMBEDDING_DIMENSION

    def __init__(self) -> None:
        self.inputs = []

    def embed(self, texts):
        self.inputs.append(texts)
        return [[0.0] * EMBEDDING_DIMENSION for _ in texts]


class FakeRetriever:
    def __init__(self) -> None:
        self.searches = []

    def search(self, search):
        self.searches.append(search)
        if search.material_filter is not None and search.material_filter != "SiC":
            return []
        return [
            EvidenceChunk(
                chunk_id=1,
                document_id="12345678-1234-5678-1234-567812345678",
                title="Wide-bandgap power semiconductor review",
                content="SiC combines a wide bandgap with useful thermal transport.",
                source_url="https://example.org/paper",
                doi="10.0000/example",
                publication_year=2025,
                material_names=["SiC"],
                metadata={},
                similarity=0.88,
            )
        ]


class FakeGraphEvidenceTool:
    def search_candidates(self, arguments):
        return {
            "queries": {
                candidate: f"{arguments.user_query}. Focus material: {candidate}"
                for candidate in arguments.candidates
            },
            "candidate_evidence": {
                candidate: [
                    {
                        "chunk_id": 1,
                        "document_id": "12345678-1234-5678-1234-567812345678",
                        "title": "Wide-bandgap power semiconductor review",
                        "content": f"Evidence relevant to {candidate}.",
                        "source_url": "https://example.org/paper",
                        "doi": "10.0000/example",
                        "publication_year": 2025,
                        "material_names": [candidate],
                        "metadata": {},
                        "similarity": 0.88,
                    }
                ]
                for candidate in arguments.candidates
            },
        }


class ScientificEvidenceToolTests(unittest.TestCase):
    def test_candidate_queries_share_one_embedding_batch(self) -> None:
        embedding_provider = FakeEmbeddingProvider()
        retriever = FakeRetriever()
        tool = ScientificEvidenceTool(
            embedding_provider=embedding_provider,
            retriever=retriever,
        )

        result = tool.search_candidates(
            CandidateEvidenceArguments(
                user_query="high-temperature power devices",
                candidates=["4H-SiC", "GaN"],
                evidence_per_candidate=2,
            )
        )

        self.assertEqual(len(embedding_provider.inputs), 1)
        self.assertEqual(len(embedding_provider.inputs[0]), 2)
        self.assertEqual(len(retriever.searches), 3)
        self.assertEqual(
            [search.material_filter for search in retriever.searches],
            ["4H-SiC", "SiC", "GaN"],
        )
        self.assertEqual(set(result["candidate_evidence"]), {"4H-SiC", "GaN"})
        self.assertEqual(len(result["candidate_evidence"]["4H-SiC"]), 1)
        self.assertEqual(result["candidate_evidence"]["GaN"], [])

    def test_graph_adds_attributable_evidence_to_report_and_trace(self) -> None:
        graph = build_graph(scientific_evidence_tool=FakeGraphEvidenceTool())

        result = graph.invoke(
            {
                "user_query": (
                    "寻找带隙大于3 eV、适合高温功率器件并优先考虑高热导率的材料"
                )
            }
        )

        self.assertTrue(result["rag_enabled"])
        self.assertEqual(len(result["candidate_evidence"]), 5)
        self.assertIn("Candidate-specific scientific evidence", result["final_report"])
        self.assertIn("10.0000/example", result["final_report"])
        self.assertEqual(result["tool_history"][-1]["step"], "retrieve_evidence")


if __name__ == "__main__":
    unittest.main()
