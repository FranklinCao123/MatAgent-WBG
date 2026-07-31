"""Shared LangGraph state definitions."""

from typing import Any, TypedDict

from matagent.schemas import RankingPlan, ScreeningRequirements
from matagent.tools import ToolCallRequest, ToolExecutionResult


class AgentState(TypedDict, total=False):
    """Data accumulated while a material-screening request is processed."""

    user_query: str
    requirements: ScreeningRequirements
    ranking_plan: RankingPlan
    pending_tool_calls: list[ToolCallRequest]
    tool_results: list[ToolExecutionResult]
    tool_iteration: int
    candidates: list[dict[str, Any]]
    ranked_candidates: list[dict[str, Any]]
    tool_history: list[dict[str, Any]]
    status: str
    errors: list[str]
    final_report: str
