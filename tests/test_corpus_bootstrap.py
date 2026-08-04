"""Tests for controlled, rate-limited RAG corpus bootstrapping."""

import unittest

from matagent.material_names import material_aliases, text_mentions_material
from matagent.rag.bootstrap import CorpusBootstrapper
from matagent.rag.semantic_scholar import (
    LiteraturePaper,
    LiteratureSearchResult,
)


def paper(
    paper_id: str,
    doi: str,
    text: str,
    *,
    year: int = 2025,
) -> LiteraturePaper:
    return LiteraturePaper(
        paper_id=paper_id,
        title=text,
        abstract=f"{text} " + "Measured semiconductor transport behavior. " * 8,
        doi=doi,
        url=f"https://example.org/{paper_id}",
        year=year,
        venue="Journal of Test Materials",
        authors=["A. Researcher"],
        citation_count=10,
        publication_types=["JournalArticle"],
    )


class FakeSource:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def search(self, query: str, *, limit: int):
        self.calls.append((query, limit))
        return next(self.results)


class CorpusBootstrapTests(unittest.TestCase):
    def test_aliases_cover_phase_and_polytype_names(self) -> None:
        self.assertEqual(material_aliases("beta-Ga2O3"), ("beta-Ga2O3", "Ga2O3"))
        self.assertEqual(material_aliases("4H-SiC"), ("4H-SiC", "SiC"))
        self.assertTrue(text_mentions_material("Thermal transport in Ga₂O₃", "Ga2O3"))

    def test_plan_filters_deduplicates_and_respects_rate_limit(self) -> None:
        shared = paper("shared", "https://doi.org/10.1/SHARED", "Ga2O3 and SiC")
        short = shared.model_copy(update={"paper_id": "short", "doi": "10.1/short", "abstract": "SiC"})
        irrelevant = paper("other", "10.1/other", "Gallium nitride devices")
        source = FakeSource(
            [
                LiteratureSearchResult(100, [irrelevant, shared], 2),
                LiteratureSearchResult(80, [short, shared], 1),
            ]
        )
        sleeps = []
        bootstrapper = CorpusBootstrapper(
            source,
            request_interval_seconds=1.1,
            sleeper=sleeps.append,
            current_year=2026,
        )

        plan = bootstrapper.plan(
            ["beta-Ga2O3", "4H-SiC"],
            papers_per_material=1,
            search_limit=2,
        )

        self.assertEqual(sleeps, [1.1])
        self.assertEqual(len(plan.selections), 1)
        self.assertEqual(
            plan.selections[0].material_names,
            ("beta-Ga2O3", "4H-SiC"),
        )
        self.assertEqual(plan.selected_per_material["4H-SiC"], 1)
        self.assertEqual(plan.skipped_without_abstract_or_doi, 3)
        self.assertEqual(plan.rejected_by_reason["material_not_mentioned"], 1)
        self.assertEqual(plan.rejected_by_reason["abstract_too_short"], 1)

    def test_invalid_limits_are_rejected_before_search(self) -> None:
        source = FakeSource([])
        bootstrapper = CorpusBootstrapper(source)

        with self.assertRaises(ValueError):
            bootstrapper.plan(["SiC"], papers_per_material=3, search_limit=2)

        self.assertEqual(source.calls, [])


if __name__ == "__main__":
    unittest.main()
