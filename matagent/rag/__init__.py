"""Retrieval infrastructure for scientific evidence."""

from matagent.rag.client import SupabaseAPIError, SupabaseDataClient
from matagent.rag.database import (
    DatabaseConfigurationError,
    DatabaseHealth,
    DatabaseHealthError,
    SupabaseSettings,
    check_database,
    settings_from_environment,
)
from matagent.rag.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingSettings,
    OpenAICompatibleEmbeddingProvider,
)
from matagent.rag.retriever import (
    EMBEDDING_DIMENSION,
    EvidenceChunk,
    EvidenceRetriever,
    EvidenceSearch,
)

__all__ = [
    "DatabaseConfigurationError",
    "DatabaseHealth",
    "DatabaseHealthError",
    "EMBEDDING_DIMENSION",
    "EmbeddingConfigurationError",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingSettings",
    "EvidenceChunk",
    "EvidenceRetriever",
    "EvidenceSearch",
    "SupabaseAPIError",
    "SupabaseDataClient",
    "SupabaseSettings",
    "OpenAICompatibleEmbeddingProvider",
    "check_database",
    "settings_from_environment",
]
