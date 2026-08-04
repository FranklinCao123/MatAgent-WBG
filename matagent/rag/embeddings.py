"""Hosted embedding configuration and provider-independent contracts."""

import math
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from dotenv import load_dotenv

from matagent.rag.retriever import EMBEDDING_DIMENSION


class EmbeddingConfigurationError(RuntimeError):
    """Raised when the hosted embedding service is not configured."""


class EmbeddingError(RuntimeError):
    """Raised when embedding generation or response validation fails."""


class EmbeddingProvider(Protocol):
    """Model-independent contract used by ingestion and retrieval."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per input text, preserving input order."""


@dataclass(frozen=True)
class EmbeddingSettings:
    """Hosted embedding settings with credentials excluded from repr."""

    api_key: str = field(repr=False)
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "BAAI/bge-m3"
    dimension: int = EMBEDDING_DIMENSION
    batch_size: int = 32
    timeout_seconds: float = 30.0

    @classmethod
    def from_environment(cls) -> "EmbeddingSettings":
        load_dotenv()
        api_key = os.getenv("MATAGENT_EMBEDDING_API_KEY", "").strip()
        if not api_key:
            raise EmbeddingConfigurationError(
                "MATAGENT_EMBEDDING_API_KEY is not configured in .env."
            )
        try:
            settings = cls(
                api_key=api_key,
                base_url=os.getenv(
                    "MATAGENT_EMBEDDING_BASE_URL",
                    "https://api.siliconflow.cn/v1",
                ).rstrip("/"),
                model=os.getenv("MATAGENT_EMBEDDING_MODEL", "BAAI/bge-m3"),
                dimension=int(
                    os.getenv(
                        "MATAGENT_EMBEDDING_DIMENSION",
                        str(EMBEDDING_DIMENSION),
                    )
                ),
                batch_size=int(os.getenv("MATAGENT_EMBEDDING_BATCH_SIZE", "32")),
                timeout_seconds=float(
                    os.getenv("MATAGENT_EMBEDDING_TIMEOUT_SECONDS", "30")
                ),
            )
        except ValueError as error:
            raise EmbeddingConfigurationError(
                "Embedding numeric settings are invalid."
            ) from error
        if not settings.base_url.startswith("https://"):
            raise EmbeddingConfigurationError("Embedding base URL must use HTTPS.")
        if not settings.model.strip():
            raise EmbeddingConfigurationError("Embedding model must not be empty.")
        if settings.dimension != EMBEDDING_DIMENSION:
            raise EmbeddingConfigurationError(
                f"Embedding dimension must match the database contract "
                f"({EMBEDDING_DIMENSION})."
            )
        if not 1 <= settings.batch_size <= 128:
            raise EmbeddingConfigurationError(
                "Embedding batch size must be between 1 and 128."
            )
        if settings.timeout_seconds <= 0:
            raise EmbeddingConfigurationError(
                "Embedding timeout must be positive."
            )
        return settings


class OpenAICompatibleEmbeddingProvider:
    """Generate embeddings through an OpenAI-compatible hosted endpoint."""

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self._client = client or _create_client(settings)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("Embedding inputs must be non-empty strings.")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.settings.batch_size):
            batch = texts[start : start + self.settings.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(
                model=self.settings.model,
                input=texts,
                encoding_format="float",
            )
        except Exception as error:
            raise EmbeddingError(
                f"Embedding request failed ({type(error).__name__})."
            ) from error

        data = sorted(response.data, key=lambda item: item.index)
        indices = [item.index for item in data]
        if indices != list(range(len(texts))):
            raise EmbeddingError(
                "Embedding response indices do not match the input batch."
            )

        vectors = [list(item.embedding) for item in data]
        for vector in vectors:
            if len(vector) != self.settings.dimension:
                raise EmbeddingError(
                    f"Embedding service returned {len(vector)} dimensions; "
                    f"expected {self.settings.dimension}."
                )
            if not all(math.isfinite(value) for value in vector):
                raise EmbeddingError("Embedding service returned non-finite values.")
        return vectors


def _create_client(settings: EmbeddingSettings) -> Any:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise EmbeddingConfigurationError(
            "Hosted embeddings require the 'openai' Python package."
        ) from error
    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
        max_retries=2,
    )
