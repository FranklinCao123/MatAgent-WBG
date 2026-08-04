"""Validated vector retrieval for scientific evidence chunks."""

import math
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matagent.rag.client import SupabaseDataClient

EMBEDDING_DIMENSION = 1024


class EvidenceSearch(BaseModel):
    """Inputs accepted by the database similarity-search RPC."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    query_embedding: list[float] = Field(
        min_length=EMBEDDING_DIMENSION,
        max_length=EMBEDDING_DIMENSION,
    )
    match_count: int = Field(default=5, ge=1, le=50)
    match_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)
    material_filter: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def embedding_is_finite(self) -> "EvidenceSearch":
        if not all(math.isfinite(value) for value in self.query_embedding):
            raise ValueError("Embedding values must be finite numbers.")
        return self


class EvidenceChunk(BaseModel):
    """A traceable scientific passage returned by semantic search."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: int
    document_id: UUID
    title: str
    content: str
    source_url: str | None = None
    doi: str | None = None
    publication_year: int | None = None
    material_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    similarity: float = Field(ge=-1.0, le=1.0)


class EvidenceRetriever:
    """Retrieve ranked evidence while keeping embedding generation separate."""

    def __init__(self, client: SupabaseDataClient) -> None:
        self._client = client

    def search(self, search: EvidenceSearch) -> list[EvidenceChunk]:
        payload = self._client.post(
            "/rest/v1/rpc/match_rag_chunks",
            search.model_dump(),
        )
        if not isinstance(payload, list):
            raise ValueError("Supabase retrieval response must be a JSON array.")
        return [EvidenceChunk.model_validate(row) for row in payload]
