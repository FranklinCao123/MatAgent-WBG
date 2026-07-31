"""Shared LangGraph state definitions."""

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Data accumulated while a material-screening request is processed."""

    user_query: str
    requirements: dict[str, Any]
    candidates: list[dict[str, Any]]
    ranked_candidates: list[dict[str, Any]]
    tool_history: list[dict[str, Any]]
    status: str
    errors: list[str]
    final_report: str
