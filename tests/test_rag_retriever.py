"""Tests for validated scientific-evidence retrieval."""

import unittest

from pydantic import ValidationError

from matagent.rag.retriever import (
    EMBEDDING_DIMENSION,
    EvidenceRetriever,
    EvidenceSearch,
)


class FakeClient:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = []

    def post(self, path, payload):
        self.calls.append((path, payload))
        return self.response


class EvidenceRetrieverTests(unittest.TestCase):
    def test_search_calls_rpc_and_validates_evidence(self) -> None:
        client = FakeClient(
            [
                {
                    "chunk_id": 7,
                    "document_id": "12345678-1234-5678-1234-567812345678",
                    "title": "Thermal transport in 4H-SiC",
                    "content": "The measured thermal conductivity was ...",
                    "source_url": "https://example.org/paper",
                    "doi": "10.0000/example",
                    "publication_year": 2025,
                    "material_names": ["4H-SiC"],
                    "metadata": {"section": "results"},
                    "similarity": 0.91,
                }
            ]
        )
        search = EvidenceSearch(
            query_embedding=[0.0] * EMBEDDING_DIMENSION,
            match_count=3,
            material_filter="4H-SiC",
        )

        evidence = EvidenceRetriever(client).search(search)

        self.assertEqual(evidence[0].chunk_id, 7)
        self.assertEqual(evidence[0].similarity, 0.91)
        self.assertEqual(client.calls[0][0], "/rest/v1/rpc/match_rag_chunks")
        self.assertEqual(client.calls[0][1]["match_count"], 3)

    def test_embedding_dimension_is_enforced(self) -> None:
        with self.assertRaises(ValidationError):
            EvidenceSearch(query_embedding=[0.0] * (EMBEDDING_DIMENSION - 1))

    def test_non_finite_embedding_is_rejected(self) -> None:
        embedding = [0.0] * EMBEDDING_DIMENSION
        embedding[0] = float("nan")
        with self.assertRaisesRegex(ValidationError, "finite"):
            EvidenceSearch(query_embedding=embedding)


if __name__ == "__main__":
    unittest.main()
