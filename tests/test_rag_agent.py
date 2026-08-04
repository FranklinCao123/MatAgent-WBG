"""Tests for the query-side RAG tool and LangGraph integration."""

import unittest

from matagent.graph import build_graph
from matagent.rag.retriever import EMBEDDING_DIMENSION, EvidenceChunk
from matagent.tools import ScientificEvidenceArguments, ScientificEvidenceTool


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
    def search(self, arguments):
        return {
            "query": arguments.query,
            "evidence": [
                {
                    "chunk_id": 1,
                    "document_id": "12345678-1234-5678-1234-567812345678",
                    "title": "Wide-bandgap power semiconductor review",
                    "content": "SiC supports high-temperature power conversion.",
                    "source_url": "https://example.org/paper",
                    "doi": "10.0000/example",
                    "publication_year": 2025,
                    "material_names": ["SiC"],
                    "metadata": {},
                    "similarity": 0.88,
                }
            ],
        }


class ScientificEvidenceToolTests(unittest.TestCase):
    def test_query_is_embedded_once_and_retrieval_is_bounded(self) -> None:
        embedding_provider = FakeEmbeddingProvider()
        retriever = FakeRetriever()
        tool = ScientificEvidenceTool(
            embedding_provider=embedding_provider,
            retriever=retriever,
        )

        result = tool.search(
            ScientificEvidenceArguments(
                query="high-temperature SiC power devices",
                top_k=3,
                minimum_similarity=0.4,
            )
        )

        self.assertEqual(
            embedding_provider.inputs,
            [["high-temperature SiC power devices"]],
        )
        self.assertEqual(retriever.searches[0].match_count, 3)
        self.assertEqual(retriever.searches[0].match_threshold, 0.4)
        self.assertEqual(result["evidence"][0]["doi"], "10.0000/example")

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
        self.assertEqual(len(result["scientific_evidence"]), 1)
        self.assertIn("Retrieved scientific evidence", result["final_report"])
        self.assertIn("10.0000/example", result["final_report"])
        self.assertEqual(result["tool_history"][-1]["step"], "retrieve_evidence")


if __name__ == "__main__":
    unittest.main()
