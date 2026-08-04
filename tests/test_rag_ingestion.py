"""Tests for deterministic chunking and atomic-ingestion payloads."""

import unittest

from matagent.rag.ingestion import (
    CharacterTextChunker,
    DocumentIngestor,
    EvidenceIngestionError,
    ScientificDocument,
)
from matagent.rag.retriever import EMBEDDING_DIMENSION


class FakeEmbeddingProvider:
    model_name = "BAAI/bge-m3"
    dimension = EMBEDDING_DIMENSION

    def __init__(self, *, drop_last: bool = False) -> None:
        self.drop_last = drop_last
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        vectors = [[float(index)] * self.dimension for index, _ in enumerate(texts)]
        return vectors[:-1] if self.drop_last else vectors


class FakeDatabaseClient:
    def __init__(self) -> None:
        self.calls = []

    def post(self, path, payload):
        self.calls.append((path, payload))
        return "12345678-1234-5678-1234-567812345678"


class TextChunkerTests(unittest.TestCase):
    def test_long_text_is_split_with_ordered_overlapping_offsets(self) -> None:
        text = (
            "4H-SiC has high thermal conductivity. " * 8
            + "\n\n"
            + "GaN is also relevant for power electronics. " * 8
        )
        chunks = CharacterTextChunker(chunk_size=180, overlap=30).split(text)

        self.assertGreater(len(chunks), 2)
        self.assertEqual([chunk.index for chunk in chunks], list(range(len(chunks))))
        self.assertTrue(all(len(chunk.content) <= 180 for chunk in chunks))
        self.assertTrue(
            all(
                current.start_char <= previous.end_char
                for previous, current in zip(chunks, chunks[1:])
            )
        )
        self.assertTrue(
            all(text[chunk.start_char : chunk.end_char] == chunk.content for chunk in chunks)
        )

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            CharacterTextChunker(chunk_size=100, overlap=100)


class DocumentIngestorTests(unittest.TestCase):
    def test_document_is_embedded_and_sent_in_one_rpc(self) -> None:
        embedding_provider = FakeEmbeddingProvider()
        database_client = FakeDatabaseClient()
        ingestor = DocumentIngestor(
            chunker=CharacterTextChunker(chunk_size=120, overlap=20),
            embedding_provider=embedding_provider,
            database_client=database_client,
        )
        document = ScientificDocument(
            title="Wide-bandgap materials review",
            text="Diamond and 4H-SiC are candidate materials. " * 10,
            source_url="https://example.org/review",
            doi="10.0000/wbg-review",
            publication_year=2025,
            material_names=["4H-SiC", "Diamond", "4H-SiC"],
        )

        result = ingestor.ingest(document)

        self.assertGreater(result.chunk_count, 1)
        self.assertEqual(len(database_client.calls), 1)
        path, payload = database_client.calls[0]
        self.assertEqual(path, "/rest/v1/rpc/ingest_rag_document")
        self.assertEqual(payload["document_doi"], "10.0000/wbg-review")
        self.assertEqual(
            payload["document_metadata"]["embedding_model"],
            "BAAI/bge-m3",
        )
        self.assertEqual(
            payload["document_metadata"]["embedding_dimension"],
            EMBEDDING_DIMENSION,
        )
        self.assertEqual(
            len(payload["chunks"]),
            len(embedding_provider.calls[0]),
        )
        self.assertEqual(
            payload["chunks"][0]["material_names"],
            ["4H-SiC", "Diamond"],
        )

    def test_vector_count_mismatch_prevents_database_write(self) -> None:
        database_client = FakeDatabaseClient()
        ingestor = DocumentIngestor(
            chunker=CharacterTextChunker(),
            embedding_provider=FakeEmbeddingProvider(drop_last=True),
            database_client=database_client,
        )

        with self.assertRaisesRegex(EvidenceIngestionError, "different number"):
            ingestor.ingest(
                ScientificDocument(title="Paper", text="Scientific evidence")
            )
        self.assertEqual(database_client.calls, [])


if __name__ == "__main__":
    unittest.main()
