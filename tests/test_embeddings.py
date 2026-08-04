"""Tests for hosted embeddings without paid API calls."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from matagent.rag.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingError,
    EmbeddingSettings,
    OpenAICompatibleEmbeddingProvider,
)
from matagent.rag.retriever import EMBEDDING_DIMENSION


class FakeEmbeddings:
    def __init__(self, responses) -> None:
        self._responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self._responses)


class FakeClient:
    def __init__(self, responses) -> None:
        self.embeddings = FakeEmbeddings(responses)


def response(*items):
    return SimpleNamespace(data=list(items))


def item(index: int, value: float = 0.0, dimension: int = EMBEDDING_DIMENSION):
    return SimpleNamespace(index=index, embedding=[value] * dimension)


class HostedEmbeddingTests(unittest.TestCase):
    def test_batching_preserves_original_text_order(self) -> None:
        client = FakeClient(
            [
                response(item(1, 2.0), item(0, 1.0)),
                response(item(0, 3.0)),
            ]
        )
        settings = EmbeddingSettings(
            api_key="secret-not-used",
            batch_size=2,
        )

        vectors = OpenAICompatibleEmbeddingProvider(
            settings,
            client=client,
        ).embed(["first", "second", "third"])

        self.assertEqual([vector[0] for vector in vectors], [1.0, 2.0, 3.0])
        self.assertEqual(len(client.embeddings.calls), 2)
        self.assertEqual(
            client.embeddings.calls[0]["model"],
            "BAAI/bge-m3",
        )
        self.assertEqual(
            client.embeddings.calls[0]["input"],
            ["first", "second"],
        )

    def test_wrong_dimension_is_rejected(self) -> None:
        client = FakeClient([response(item(0, dimension=12))])
        provider = OpenAICompatibleEmbeddingProvider(
            EmbeddingSettings(api_key="secret-not-used"),
            client=client,
        )

        with self.assertRaisesRegex(EmbeddingError, "12 dimensions"):
            provider.embed(["text"])

    def test_blank_input_is_rejected_before_api_call(self) -> None:
        client = FakeClient([])
        provider = OpenAICompatibleEmbeddingProvider(
            EmbeddingSettings(api_key="secret-not-used"),
            client=client,
        )

        with self.assertRaisesRegex(EmbeddingError, "non-empty"):
            provider.embed([" "])
        self.assertEqual(client.embeddings.calls, [])

    def test_settings_hide_key_and_validate_database_dimension(self) -> None:
        environment = {
            "MATAGENT_EMBEDDING_API_KEY": "top-secret",
            "MATAGENT_EMBEDDING_DIMENSION": "768",
        }
        with patch("matagent.rag.embeddings.load_dotenv"), patch.dict(
            "os.environ", environment, clear=True
        ):
            with self.assertRaisesRegex(
                EmbeddingConfigurationError,
                "database contract",
            ):
                EmbeddingSettings.from_environment()

        self.assertNotIn(
            "top-secret",
            repr(EmbeddingSettings(api_key="top-secret")),
        )


if __name__ == "__main__":
    unittest.main()
