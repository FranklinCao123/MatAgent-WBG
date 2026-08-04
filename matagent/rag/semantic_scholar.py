"""Semantic Scholar adapter for traceable scientific abstracts."""

import json
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from matagent.rag.ingestion import ScientificDocument


class LiteratureSourceError(RuntimeError):
    """Raised when literature discovery or source validation fails."""


@dataclass(frozen=True)
class SemanticScholarSettings:
    api_key: str | None = field(default=None, repr=False)
    base_url: str = "https://api.semanticscholar.org/graph/v1"
    timeout_seconds: float = 20.0

    @classmethod
    def from_environment(cls) -> "SemanticScholarSettings":
        load_dotenv()
        return cls(
            api_key=os.getenv("MATAGENT_S2_API_KEY") or None,
            base_url=os.getenv(
                "MATAGENT_S2_BASE_URL",
                "https://api.semanticscholar.org/graph/v1",
            ).rstrip("/"),
            timeout_seconds=float(os.getenv("MATAGENT_S2_TIMEOUT_SECONDS", "20")),
        )


class LiteraturePaper(BaseModel):
    """Minimum attributable paper data accepted by the RAG corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str
    title: str
    abstract: str
    doi: str
    url: str
    year: int | None = None
    venue: str | None = None
    authors: list[str]
    citation_count: int | None = None
    publication_types: list[str]
    open_access_pdf_url: str | None = None

    def to_document(self, material_names: list[str]) -> ScientificDocument:
        return ScientificDocument(
            title=self.title,
            text=f"{self.title}\n\n{self.abstract}",
            source_type="paper",
            source_url=self.url,
            doi=self.doi,
            publisher=self.venue,
            publication_year=self.year,
            abstract=self.abstract,
            material_names=material_names,
            metadata={
                "source_api": "Semantic Scholar Academic Graph API",
                "semantic_scholar_paper_id": self.paper_id,
                "authors": self.authors,
                "citation_count": self.citation_count,
                "publication_types": self.publication_types,
                "open_access_pdf_url": self.open_access_pdf_url,
            },
        )


@dataclass(frozen=True)
class LiteratureSearchResult:
    total_matches: int
    papers: list[LiteraturePaper]
    skipped_without_abstract_or_doi: int


FIELDS = ",".join(
    (
        "title",
        "abstract",
        "url",
        "year",
        "venue",
        "authors",
        "externalIds",
        "citationCount",
        "publicationTypes",
        "openAccessPdf",
    )
)


class SemanticScholarClient:
    """Read-only, bounded client for paper search and paper details."""

    def __init__(self, settings: SemanticScholarSettings, *, opener=urlopen) -> None:
        self._settings = settings
        self._opener = opener

    def search(self, query: str, *, limit: int = 5) -> LiteratureSearchResult:
        if not query.strip():
            raise ValueError("Literature search query must not be blank.")
        if not 1 <= limit <= 20:
            raise ValueError("Literature search limit must be between 1 and 20.")
        payload = self._get(
            "/paper/search",
            {"query": query, "limit": limit, "fields": FIELDS},
        )
        papers, skipped = _parse_papers(payload.get("data", []))
        return LiteratureSearchResult(
            total_matches=int(payload.get("total", 0)),
            papers=papers,
            skipped_without_abstract_or_doi=skipped,
        )

    def get_paper(self, paper_id: str) -> LiteraturePaper:
        if not paper_id.strip():
            raise ValueError("Semantic Scholar paper ID must not be blank.")
        payload = self._get(f"/paper/{quote(paper_id, safe='')}", {"fields": FIELDS})
        paper = _parse_paper(payload)
        if paper is None:
            raise LiteratureSourceError(
                "The selected paper must provide both an abstract and DOI."
            )
        return paper

    def _get(self, path: str, parameters: dict[str, Any]) -> dict[str, Any]:
        headers = {"User-Agent": "MatAgent-WBG/0.1"}
        if self._settings.api_key:
            headers["x-api-key"] = self._settings.api_key
        request = Request(
            f"{self._settings.base_url}{path}?{urlencode(parameters)}",
            headers=headers,
        )
        try:
            with self._opener(
                request,
                timeout=self._settings.timeout_seconds,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code == 429:
                detail = (
                    "Semantic Scholar rate limit reached; configure "
                    "MATAGENT_S2_API_KEY for authenticated requests."
                )
            else:
                detail = f"Semantic Scholar returned HTTP {error.code}."
            raise LiteratureSourceError(
                detail
            ) from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise LiteratureSourceError(
                f"Semantic Scholar request failed ({type(error).__name__})."
            ) from error
        if not isinstance(payload, dict):
            raise LiteratureSourceError("Semantic Scholar returned invalid data.")
        return payload


def _parse_papers(rows: list[dict[str, Any]]) -> tuple[list[LiteraturePaper], int]:
    papers = []
    skipped = 0
    for row in rows:
        paper = _parse_paper(row)
        if paper is None:
            skipped += 1
        else:
            papers.append(paper)
    return papers, skipped


def _parse_paper(row: dict[str, Any]) -> LiteraturePaper | None:
    abstract = row.get("abstract")
    doi = (row.get("externalIds") or {}).get("DOI")
    if not abstract or not doi:
        return None
    open_access = row.get("openAccessPdf") or {}
    return LiteraturePaper(
        paper_id=str(row["paperId"]),
        title=str(row["title"]),
        abstract=str(abstract),
        doi=str(doi),
        url=str(row.get("url") or f"https://www.semanticscholar.org/paper/{row['paperId']}"),
        year=row.get("year"),
        venue=row.get("venue") or None,
        authors=[str(author["name"]) for author in row.get("authors", [])],
        citation_count=row.get("citationCount"),
        publication_types=list(row.get("publicationTypes") or []),
        open_access_pdf_url=open_access.get("url"),
    )
