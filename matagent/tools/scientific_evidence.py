"""Tool adapter from a natural-language query to pgvector evidence."""

from matagent.material_names import material_aliases
from matagent.rag.embeddings import EmbeddingProvider
from matagent.rag.retriever import EvidenceRetriever, EvidenceSearch
from matagent.tools.schemas import CandidateEvidenceArguments


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
            evidence_by_chunk = {}
            for material_filter in material_aliases(candidate):
                evidence = self._retriever.search(
                    EvidenceSearch(
                        query_embedding=vector,
                        match_count=arguments.evidence_per_candidate,
                        match_threshold=arguments.minimum_similarity,
                        material_filter=material_filter,
                    )
                )
                for item in evidence:
                    previous = evidence_by_chunk.get(item.chunk_id)
                    if previous is None or item.similarity > previous.similarity:
                        evidence_by_chunk[item.chunk_id] = item
            evidence = sorted(
                evidence_by_chunk.values(),
                key=lambda item: item.similarity,
                reverse=True,
            )[: arguments.evidence_per_candidate]
            grouped[candidate] = [
                item.model_dump(mode="json") for item in evidence
            ]
        return {
            "queries": dict(zip(candidates, queries, strict=True)),
            "candidate_evidence": grouped,
        }
