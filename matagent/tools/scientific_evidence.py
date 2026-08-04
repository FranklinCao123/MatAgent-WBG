"""Tool adapter from a natural-language query to pgvector evidence."""

from matagent.rag.embeddings import EmbeddingProvider
from matagent.rag.retriever import EvidenceRetriever, EvidenceSearch
from matagent.tools.schemas import ScientificEvidenceArguments


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
