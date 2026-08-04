"""Small, deterministic information-retrieval metrics for RAG."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    query: str
    expected_dois: frozenset[str]
    retrieved_dois: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("Evaluation query must not be blank.")
        if not self.expected_dois:
            raise ValueError("Each evaluation case needs an expected DOI.")


@dataclass(frozen=True)
class RetrievalMetrics:
    case_count: int
    k: int
    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    hit_rate_at_k: float


def evaluate_retrieval(
    cases: list[RetrievalEvaluationCase],
    *,
    k: int,
) -> RetrievalMetrics:
    """Macro-average Precision@k, Recall@k, MRR, and HitRate@k."""

    if not cases:
        raise ValueError("At least one retrieval evaluation case is required.")
    if k < 1:
        raise ValueError("k must be positive.")

    precisions = []
    recalls = []
    reciprocal_ranks = []
    hits = []
    for case in cases:
        expected = {_normalize_doi(doi) for doi in case.expected_dois}
        retrieved = [_normalize_doi(doi) for doi in case.retrieved_dois[:k]]
        relevant_positions = [
            index for index, doi in enumerate(retrieved, 1) if doi in expected
        ]
        unique_relevant = len({doi for doi in retrieved if doi in expected})
        precisions.append(unique_relevant / k)
        recalls.append(unique_relevant / len(expected))
        reciprocal_ranks.append(
            1.0 / relevant_positions[0] if relevant_positions else 0.0
        )
        hits.append(1.0 if relevant_positions else 0.0)

    count = len(cases)
    return RetrievalMetrics(
        case_count=count,
        k=k,
        precision_at_k=sum(precisions) / count,
        recall_at_k=sum(recalls) / count,
        mean_reciprocal_rank=sum(reciprocal_ranks) / count,
        hit_rate_at_k=sum(hits) / count,
    )


def _normalize_doi(doi: str) -> str:
    normalized = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized
