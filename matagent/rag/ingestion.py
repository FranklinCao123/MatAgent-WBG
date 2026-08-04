"""Scientific document chunking and atomic vector ingestion."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from matagent.rag.embeddings import EmbeddingProvider


class EvidenceIngestionError(RuntimeError):
    """Raised when a document cannot be converted into stored evidence."""


class DataAPIPoster(Protocol):
    def post(self, path: str, payload: dict[str, Any]) -> Any: ...


class ScientificDocument(BaseModel):
    """Validated source document before chunking and embedding."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_type: Literal["paper", "web", "dataset", "manual"] = "paper"
    source_url: str | None = Field(default=None, pattern=r"^https?://")
    doi: str | None = Field(default=None, min_length=1)
    publisher: str | None = None
    publication_year: int | None = Field(default=None, ge=1800, le=2100)
    abstract: str | None = None
    material_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("material_names")
    @classmethod
    def normalize_material_names(cls, names: list[str]) -> list[str]:
        normalized = [name.strip() for name in names if name.strip()]
        return list(dict.fromkeys(normalized))


@dataclass(frozen=True)
class TextChunk:
    """A deterministic text slice with source-character offsets."""

    index: int
    content: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class CharacterTextChunker:
    """Split near paragraph boundaries while retaining contextual overlap."""

    chunk_size: int = 1800
    overlap: int = 200
    minimum_break_ratio: float = 0.6

    def __post_init__(self) -> None:
        if self.chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters.")
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size.")
        if not 0.5 <= self.minimum_break_ratio <= 1.0:
            raise ValueError("minimum_break_ratio must be between 0.5 and 1.0.")

    def split(self, text: str) -> list[TextChunk]:
        if not text.strip():
            raise EvidenceIngestionError("Document text must not be blank.")

        chunks: list[TextChunk] = []
        start = 0
        while start < len(text):
            hard_end = min(start + self.chunk_size, len(text))
            end = self._preferred_break(text, start, hard_end)
            raw = text[start:end]
            left_trim = len(raw) - len(raw.lstrip())
            right_trim = len(raw) - len(raw.rstrip())
            content = raw.strip()
            if content:
                chunks.append(
                    TextChunk(
                        index=len(chunks),
                        content=content,
                        start_char=start + left_trim,
                        end_char=end - right_trim,
                    )
                )
            if end >= len(text):
                break
            next_start = max(start + 1, end - self.overlap)
            start = self._advance_to_boundary(text, next_start, end)
        return chunks

    def _preferred_break(self, text: str, start: int, hard_end: int) -> int:
        if hard_end >= len(text):
            return len(text)
        earliest = start + int(self.chunk_size * self.minimum_break_ratio)
        candidates = []
        for separator in ("\n\n", "\n", ". ", "。", "; ", "；"):
            position = text.rfind(separator, earliest, hard_end)
            if position >= 0:
                candidates.append(position + len(separator))
        return max(candidates, default=hard_end)

    @staticmethod
    def _advance_to_boundary(text: str, start: int, upper_bound: int) -> int:
        while (
            start < upper_bound
            and start > 0
            and not text[start - 1].isspace()
            and not text[start].isspace()
        ):
            start += 1
        return start


@dataclass(frozen=True)
class IngestionResult:
    document_id: UUID
    chunk_count: int


class DocumentIngestor:
    """Chunk, embed, and atomically persist one scientific document."""

    def __init__(
        self,
        *,
        chunker: CharacterTextChunker,
        embedding_provider: EmbeddingProvider,
        database_client: DataAPIPoster,
    ) -> None:
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._database_client = database_client

    def ingest(self, document: ScientificDocument) -> IngestionResult:
        chunks = self._chunker.split(document.text)
        vectors = self._embedding_provider.embed(
            [chunk.content for chunk in chunks]
        )
        if len(vectors) != len(chunks):
            raise EvidenceIngestionError(
                "Embedding provider returned a different number of vectors."
            )

        stored_metadata = {
            **document.metadata,
            "embedding_model": self._embedding_provider.model_name,
            "embedding_dimension": self._embedding_provider.dimension,
        }
        payload = {
            "document_title": document.title,
            "document_source_type": document.source_type,
            "document_source_url": document.source_url,
            "document_doi": document.doi,
            "document_publisher": document.publisher,
            "document_publication_year": document.publication_year,
            "document_abstract": document.abstract,
            "document_metadata": stored_metadata,
            "chunks": [
                {
                    "chunk_index": chunk.index,
                    "content": chunk.content,
                    "token_count": None,
                    "embedding": vector,
                    "material_names": document.material_names,
                    "metadata": {
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                    },
                }
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        }
        response = self._database_client.post(
            "/rest/v1/rpc/ingest_rag_document",
            payload,
        )
        try:
            document_id = UUID(str(response))
        except (TypeError, ValueError, AttributeError) as error:
            raise EvidenceIngestionError(
                "Ingestion RPC returned an invalid document ID."
            ) from error
        return IngestionResult(
            document_id=document_id,
            chunk_count=len(chunks),
        )
