"""Shared interfaces and client setup for LLM-backed components."""

from typing import Any, Protocol

from matagent.schemas import GroundedReport, ScreeningRequirements


class RequirementParsingError(RuntimeError):
    """Raised when a requirement parser cannot produce validated output."""


class RequirementParser(Protocol):
    """Contract implemented by offline and LLM-backed parsers."""

    name: str

    def parse(self, user_query: str) -> ScreeningRequirements:
        """Parse a natural-language query into validated requirements."""


class ReportSynthesisError(RuntimeError):
    """Raised when an LLM cannot produce a citation-safe final synthesis."""


class ReportSynthesizer(Protocol):
    """Contract for optional grounded final-report generation."""

    name: str

    def synthesize(
        self,
        *,
        user_query: str,
        requirements: ScreeningRequirements,
        ranked_candidates: list[dict[str, Any]],
        candidate_evidence: dict[str, list[dict[str, Any]]],
    ) -> GroundedReport: ...


def create_deepseek_client(
    *, api_key: str, base_url: str, client: Any | None = None
) -> Any:
    """Return an injected test client or create the OpenAI-compatible client."""

    if not api_key:
        raise ValueError("DeepSeek API key must not be empty.")
    if client is not None:
        return client
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "DeepSeek mode requires the 'openai' Python package."
        ) from error
    return OpenAI(api_key=api_key, base_url=base_url)
