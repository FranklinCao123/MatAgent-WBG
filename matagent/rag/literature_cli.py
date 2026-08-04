"""Human-reviewed literature discovery and ingestion CLI."""

import argparse

from matagent.rag.bootstrap import CorpusBootstrapper
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

    bootstrap = commands.add_parser(
        "bootstrap",
        help="Build a deduplicated multi-material corpus plan.",
    )
    bootstrap.add_argument("--material", action="append", required=True)
    bootstrap.add_argument("--papers-per-material", type=int, default=2)
    bootstrap.add_argument("--search-limit", type=int, default=10)
    bootstrap.add_argument("--minimum-abstract-characters", type=int, default=200)
    bootstrap.add_argument(
        "--write",
        action="store_true",
        help="Embed and persist the previewed plan; omitted means dry-run.",
    )
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

        if args.command == "bootstrap":
            plan = CorpusBootstrapper(source).plan(
                args.material,
                papers_per_material=args.papers_per_material,
                search_limit=args.search_limit,
                minimum_abstract_characters=args.minimum_abstract_characters,
            )
            _print_bootstrap_plan(plan)
            if not args.write:
                print("Dry run only: add --write to embed and persist this plan.")
                return

            ingestor = _build_ingestor()
            print("\nWriting selected papers:")
            for selection in plan.selections:
                result = ingestor.ingest(
                    selection.paper.to_document(list(selection.material_names))
                )
                print(
                    f"- DOI {selection.paper.doi} | document {result.document_id} | "
                    f"{result.chunk_count} chunk(s)"
                )
            print(f"Corpus bootstrap: OK ({len(plan.selections)} unique papers)")
            return

        paper = source.get_paper(args.paper_id)
        ingestor = _build_ingestor()
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


def _build_ingestor() -> DocumentIngestor:
    embedding_settings = EmbeddingSettings.from_environment()
    return DocumentIngestor(
        chunker=CharacterTextChunker(),
        embedding_provider=OpenAICompatibleEmbeddingProvider(embedding_settings),
        database_client=SupabaseDataClient(settings_from_environment()),
    )


def _print_bootstrap_plan(plan) -> None:
    print("Corpus bootstrap plan")
    print(f"Materials searched: {', '.join(plan.searched_materials)}")
    print(f"Total matches reported: {plan.total_matches_reported}")
    print(f"Unique papers selected: {len(plan.selections)}")
    print(
        "Selected per material: "
        + ", ".join(
            f"{material}={count}"
            for material, count in plan.selected_per_material.items()
        )
    )
    print(
        "Rejected: "
        + ", ".join(
            f"{reason}={count}"
            for reason, count in plan.rejected_by_reason.items()
        )
    )
    for selection in plan.selections:
        paper = selection.paper
        access = "open-access" if paper.open_access_pdf_url else "no-OA-link"
        print(
            f"- [{paper.paper_id}] {paper.year or 'unknown'} | "
            f"{paper.title} | DOI {paper.doi} | "
            f"venue={paper.venue or 'unknown'} | "
            f"citations={paper.citation_count or 0} | {access} | "
            f"materials={','.join(selection.material_names)}"
        )


if __name__ == "__main__":
    main()
