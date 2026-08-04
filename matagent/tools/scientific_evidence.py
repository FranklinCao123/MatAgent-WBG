"""Tool adapter from a natural-language query to pgvector evidence."""

from matagent.rag.embeddings import EmbeddingProvider
from matagent.rag.retriever import EvidenceRetriever, EvidenceSearch
from matagent.tools.schemas import (
    CandidateEvidenceArguments,
    ScientificEvidenceArguments,
)


class ScientificEvidenceTool:
    """Embed a query and return attributable, validated evidence chunks."""

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        retriever: EvidenceRetriever,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._retriever = retriever

    def search(self, arguments: ScientificEvidenceArguments) -> dict:
        vectors = self._embedding_provider.embed([arguments.query])
        if len(vectors) != 1:
            raise ValueError("Embedding provider must return one query vector.")
        evidence = self._retriever.search(
            EvidenceSearch(
                query_embedding=vectors[0],
                match_count=arguments.top_k,
                match_threshold=arguments.minimum_similarity,
                material_filter=arguments.material_filter,
            )
        )
        return {
            "query": arguments.query,
            "evidence": [item.model_dump(mode="json") for item in evidence],
        }

    def search_candidates(self, arguments: CandidateEvidenceArguments) -> dict:
        candidates = list(dict.fromkeys(arguments.candidates))
        queries = [
            f"{arguments.user_query}. Focus material: {candidate}"
            for candidate in candidates
        ]
        vectors = self._embedding_provider.embed(queries)
        if len(vectors) != len(candidates):
            raise ValueError(
                "Embedding provider returned the wrong number of candidate vectors."
            )

        grouped = {}
        for candidate, vector in zip(candidates, vectors, strict=True):
            evidence = self._retriever.search(
                EvidenceSearch(
                    query_embedding=vector,
                    match_count=arguments.evidence_per_candidate,
                    match_threshold=arguments.minimum_similarity,
                    material_filter=None,
                )
            )
            grouped[candidate] = [
                item.model_dump(mode="json") for item in evidence
            ]
        return {
            "queries": dict(zip(candidates, queries, strict=True)),
            "candidate_evidence": grouped,
        }
