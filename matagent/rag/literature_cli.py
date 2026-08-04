"""Human-reviewed literature discovery and ingestion CLI."""

import argparse

from matagent.rag.client import SupabaseAPIError, SupabaseDataClient
from matagent.rag.database import DatabaseConfigurationError, settings_from_environment
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
)
from matagent.rag.semantic_scholar import (
    LiteratureSourceError,
    SemanticScholarClient,
    SemanticScholarSettings,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover and ingest literature.")
    commands = parser.add_subparsers(dest="command", required=True)

    search = commands.add_parser("search", help="Preview attributable papers.")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)

    ingest = commands.add_parser("ingest", help="Ingest one reviewed paper.")
    ingest.add_argument("paper_id")
    ingest.add_argument("--material", action="append", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = SemanticScholarClient(SemanticScholarSettings.from_environment())
    try:
        if args.command == "search":
            result = source.search(args.query, limit=args.limit)
            print(f"Total matches reported: {result.total_matches}")
            print(f"Usable results: {len(result.papers)}")
            print(
                "Skipped without abstract or DOI: "
                f"{result.skipped_without_abstract_or_doi}"
            )
            for paper in result.papers:
                oa = "open-access PDF" if paper.open_access_pdf_url else "no OA PDF"
                print(
                    f"- [{paper.paper_id}] {paper.year or 'unknown'} | "
                    f"{paper.title} | DOI {paper.doi} | {oa}"
                )
            return

        paper = source.get_paper(args.paper_id)
        embedding_settings = EmbeddingSettings.from_environment()
        ingestor = DocumentIngestor(
            chunker=CharacterTextChunker(),
            embedding_provider=OpenAICompatibleEmbeddingProvider(
                embedding_settings
            ),
            database_client=SupabaseDataClient(settings_from_environment()),
        )
        result = ingestor.ingest(paper.to_document(args.material))
    except (
        DatabaseConfigurationError,
        EmbeddingConfigurationError,
        EmbeddingError,
        EvidenceIngestionError,
        LiteratureSourceError,
        SupabaseAPIError,
        ValueError,
    ) as error:
        raise SystemExit(f"Literature command failed: {error}") from error

    print("Literature ingestion: OK")
    print(f"Document ID: {result.document_id}")
    print(f"Chunks stored: {result.chunk_count}")
    print(f"Source paper: {paper.paper_id}")


if __name__ == "__main__":
    main()
