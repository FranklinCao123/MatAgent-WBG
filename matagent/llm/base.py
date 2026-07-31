"""Shared interfaces for converting user text into screening requirements."""

from typing import Protocol

from matagent.schemas import ScreeningRequirements


class RequirementParsingError(RuntimeError):
    """Raised when a requirement parser cannot produce validated output."""


class RequirementParser(Protocol):
    """Contract implemented by offline and LLM-backed parsers."""

    name: str

    def parse(self, user_query: str) -> ScreeningRequirements:
        """Parse a natural-language query into validated requirements."""

