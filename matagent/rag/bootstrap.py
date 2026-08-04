"""Controlled discovery plan for bootstrapping a scientific RAG corpus."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from matagent.material_names import text_mentions_material
from matagent.rag.semantic_scholar import LiteraturePaper, LiteratureSearchResult


class LiteratureSearcher(Protocol):
    def search(self, query: str, *, limit: int) -> LiteratureSearchResult: ...


@dataclass(frozen=True)
class BootstrapSelection:
    paper: LiteraturePaper
    material_names: tuple[str, ...]


@dataclass(frozen=True)
class BootstrapPlan:
    searched_materials: tuple[str, ...]
    selections: tuple[BootstrapSelection, ...]
    selected_per_material: dict[str, int]
    total_matches_reported: int
    skipped_without_abstract_or_doi: int
    rejected_by_reason: dict[str, int]


class CorpusBootstrapper:
    """Search each material slowly, filter records, and deduplicate by DOI."""

    def __init__(
        self,
        source: LiteratureSearcher,
        *,
        request_interval_seconds: float = 1.05,
        sleeper: Callable[[float], None] = time.sleep,
        current_year: int | None = None,
    ) -> None:
        if request_interval_seconds < 1.0:
            raise ValueError("Semantic Scholar requests must be spaced by at least 1 s.")
        self._source = source
        self._request_interval_seconds = request_interval_seconds
        self._sleeper = sleeper
        self._current_year = current_year or date.today().year

    def plan(
        self,
        materials: Sequence[str],
        *,
        papers_per_material: int = 2,
        search_limit: int = 10,
        minimum_abstract_characters: int = 200,
    ) -> BootstrapPlan:
        normalized_materials = tuple(
            dict.fromkeys(material.strip() for material in materials if material.strip())
        )
        if not normalized_materials:
            raise ValueError("At least one material is required.")
        if len(normalized_materials) > 10:
            raise ValueError("At most 10 materials can be bootstrapped per run.")
        if not 1 <= papers_per_material <= 5:
            raise ValueError("papers_per_material must be between 1 and 5.")
        if not papers_per_material <= search_limit <= 20:
            raise ValueError("search_limit must be between papers_per_material and 20.")
        if minimum_abstract_characters < 100:
            raise ValueError("minimum_abstract_characters must be at least 100.")

        by_doi: dict[str, dict] = {}
        selected_per_material = {}
        rejected = {
            "material_not_mentioned": 0,
            "abstract_too_short": 0,
            "future_publication": 0,
        }
        total_matches = 0
        skipped = 0

        for index, material in enumerate(normalized_materials):
            if index:
                self._sleeper(self._request_interval_seconds)
            result = self._source.search(_build_query(material), limit=search_limit)
            total_matches += result.total_matches
            skipped += result.skipped_without_abstract_or_doi
            selected = 0
            for paper in result.papers:
                reason = self._rejection_reason(paper, material, minimum_abstract_characters)
                if reason:
                    rejected[reason] += 1
                    continue
                doi_key = _normalize_doi(paper.doi)
                entry = by_doi.setdefault(
                    doi_key,
                    {"paper": paper, "materials": []},
                )
                if material not in entry["materials"]:
                    entry["materials"].append(material)
                selected += 1
                if selected >= papers_per_material:
                    break
            selected_per_material[material] = selected

        selections = tuple(
            BootstrapSelection(
                paper=entry["paper"],
                material_names=tuple(entry["materials"]),
            )
            for entry in by_doi.values()
        )
        return BootstrapPlan(
            searched_materials=normalized_materials,
            selections=selections,
            selected_per_material=selected_per_material,
            total_matches_reported=total_matches,
            skipped_without_abstract_or_doi=skipped,
            rejected_by_reason=rejected,
        )

    def _rejection_reason(
        self,
        paper: LiteraturePaper,
        material: str,
        minimum_abstract_characters: int,
    ) -> str | None:
        if paper.year is not None and paper.year > self._current_year:
            return "future_publication"
        if len(paper.abstract.strip()) < minimum_abstract_characters:
            return "abstract_too_short"
        if not text_mentions_material(f"{paper.title}\n{paper.abstract}", material):
            return "material_not_mentioned"
        return None


def _build_query(material: str) -> str:
    return (
        f'"{material}" semiconductor thermal conductivity breakdown field '
        "high temperature power devices"
    )


def _normalize_doi(doi: str) -> str:
    normalized = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized
