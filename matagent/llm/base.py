"""Shared interfaces and client setup for LLM-backed components."""

from typing import Any, Protocol

from matagent.schemas import ScreeningRequirements


class RequirementParsingError(RuntimeError):
    """Raised when a requirement parser cannot produce validated output."""


class RequirementParser(Protocol):
    """Contract implemented by offline and LLM-backed parsers."""

    name: str

    def parse(self, user_query: str) -> ScreeningRequirements:
        """Parse a natural-language query into validated requirements."""


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
