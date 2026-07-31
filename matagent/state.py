"""Shared LangGraph state definitions."""

from typing import Any, TypedDict

from matagent.schemas import ScreeningRequirements


class AgentState(TypedDict, total=False):
    """Data accumulated while a material-screening request is processed."""

    user_query: str
    requirements: ScreeningRequirements
    candidates: list[dict[str, Any]]
    ranked_candidates: list[dict[str, Any]]
    tool_history: list[dict[str, Any]]
    status: str
    errors: list[str]
    final_report: str
