"""Tests for real-literature adaptation without network requests."""

import json
import unittest
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from matagent.rag.semantic_scholar import (
    SemanticScholarClient,
    SemanticScholarSettings,
)


class FakeResponse:
    def __init__(self, payload) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def paper_row(*, abstract="Measured thermal transport.", doi="10.1/example"):
    return {
        "paperId": "abc123",
        "title": "Thermal transport in 4H-SiC",
        "abstract": abstract,
        "url": "https://www.semanticscholar.org/paper/abc123",
        "year": 2025,
        "venue": "Journal of Example Materials",
        "authors": [{"authorId": "1", "name": "A. Researcher"}],
        "externalIds": {"DOI": doi} if doi else {},
        "citationCount": 12,
        "publicationTypes": ["JournalArticle"],
        "openAccessPdf": {"url": "https://example.org/open.pdf"},
    }


class SemanticScholarTests(unittest.TestCase):
    def test_rate_limit_retries_with_server_delay(self) -> None:
        attempts = []
        sleeps = []

        def rate_limited_then_success(request, timeout):
            attempts.append(request)
            if len(attempts) == 1:
                raise HTTPError(
                    request.full_url,
                    429,
                    "rate limited",
                    {"Retry-After": "1.5"},
                    BytesIO(b"{}"),
                )
            return FakeResponse({"total": 1, "data": [paper_row()]})

        client = SemanticScholarClient(
            SemanticScholarSettings(api_key="secret-not-used"),
            opener=rate_limited_then_success,
            sleeper=sleeps.append,
        )

        result = client.search("SiC power devices", limit=1)

        self.assertEqual(len(result.papers), 1)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(sleeps, [1.5])

    def test_search_is_bounded_and_skips_unattributable_records(self) -> None:
        captured = {}

        def fake_open(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse(
                {
                    "total": 100,
                    "data": [paper_row(), paper_row(abstract=None)],
                }
            )

        client = SemanticScholarClient(
            SemanticScholarSettings(api_key="secret-not-used"),
            opener=fake_open,
        )
        result = client.search("wide bandgap semiconductor", limit=2)

        self.assertEqual(len(result.papers), 1)
        self.assertEqual(result.skipped_without_abstract_or_doi, 1)
        query = parse_qs(urlsplit(captured["request"].full_url).query)
        self.assertEqual(query["limit"], ["2"])
        self.assertIn("abstract", query["fields"][0])
        self.assertEqual(
            captured["request"].get_header("X-api-key"),
            "secret-not-used",
        )
        self.assertEqual(captured["timeout"], 20.0)

    def test_paper_becomes_traceable_scientific_document(self) -> None:
        client = SemanticScholarClient(
            SemanticScholarSettings(),
            opener=lambda request, timeout: FakeResponse(paper_row()),
        )

        paper = client.get_paper("abc123")
        document = paper.to_document(["4H-SiC"])

        self.assertEqual(document.doi, "10.1/example")
        self.assertIn(paper.abstract, document.text)
        self.assertEqual(document.material_names, ["4H-SiC"])
        self.assertEqual(
            document.metadata["semantic_scholar_paper_id"],
            "abc123",
        )
        self.assertEqual(
            document.metadata["source_api"],
            "Semantic Scholar Academic Graph API",
        )


if __name__ == "__main__":
    unittest.main()
