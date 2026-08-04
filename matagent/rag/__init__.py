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
from matagent.rag.ingestion import (
    CharacterTextChunker,
    DocumentIngestor,
    EvidenceIngestionError,
    IngestionResult,
    ScientificDocument,
    TextChunk,
)

__all__ = [
    "DatabaseConfigurationError",
    "DatabaseHealth",
    "DatabaseHealthError",
    "DocumentIngestor",
    "EMBEDDING_DIMENSION",
    "EmbeddingConfigurationError",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingSettings",
    "EvidenceChunk",
    "EvidenceRetriever",
    "EvidenceSearch",
    "EvidenceIngestionError",
    "IngestionResult",
    "SupabaseAPIError",
    "SupabaseDataClient",
    "SupabaseSettings",
    "ScientificDocument",
    "TextChunk",
    "CharacterTextChunker",
    "OpenAICompatibleEmbeddingProvider",
    "check_database",
    "settings_from_environment",
]
