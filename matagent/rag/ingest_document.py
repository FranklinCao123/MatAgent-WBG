"""CLI for ingesting one UTF-8 scientific text document."""

import argparse
from pathlib import Path

from pydantic import ValidationError

from matagent.rag.client import SupabaseAPIError, SupabaseDataClient
from matagent.rag.database import (
    DatabaseConfigurationError,
    settings_from_environment,
)
from matagent.rag.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingError,
    EmbeddingSettings,
    OpenAICompatibleEmbeddingProvider,
)
from matagent.rag.ingestion import (
    CharacterTextChunker,
    DocumentIngestor,
    EvidenceIngestionError,
    ScientificDocument,
)

MAX_SOURCE_BYTES = 2 * 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed and atomically ingest one scientific text document."
    )
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--source-type",
        choices=("paper", "web", "dataset", "manual"),
        default="paper",
    )
    parser.add_argument("--source-url")
    parser.add_argument("--doi")
    parser.add_argument("--publisher")
    parser.add_argument("--year", type=int)
    parser.add_argument("--abstract")
    parser.add_argument(
        "--material",
        action="append",
        default=[],
        help="Canonical material name; repeat for multiple materials.",
    )
    return parser


def read_source(path: Path) -> str:
    if path.suffix.lower() not in {".txt", ".md"}:
        raise EvidenceIngestionError("Only UTF-8 .txt and .md files are supported.")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise EvidenceIngestionError("Source file exceeds the 2 MiB safety limit.")
    return path.read_text(encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    try:
        document = ScientificDocument(
            title=args.title,
            text=read_source(args.file),
            source_type=args.source_type,
            source_url=args.source_url,
            doi=args.doi,
            publisher=args.publisher,
            publication_year=args.year,
            abstract=args.abstract,
            material_names=args.material,
        )
        embedding_settings = EmbeddingSettings.from_environment()
        ingestor = DocumentIngestor(
            chunker=CharacterTextChunker(),
            embedding_provider=OpenAICompatibleEmbeddingProvider(
                embedding_settings
            ),
            database_client=SupabaseDataClient(settings_from_environment()),
        )
        result = ingestor.ingest(document)
    except (
        DatabaseConfigurationError,
        EmbeddingConfigurationError,
        EmbeddingError,
        EvidenceIngestionError,
        SupabaseAPIError,
        ValidationError,
        OSError,
        UnicodeError,
    ) as error:
        raise SystemExit(f"Document ingestion failed: {error}") from error

    print("Document ingestion: OK")
    print(f"Document ID: {result.document_id}")
    print(f"Chunks stored: {result.chunk_count}")
    print(f"Embedding model: {embedding_settings.model}")


if __name__ == "__main__":
    main()
