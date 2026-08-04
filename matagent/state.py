"""Shared LangGraph state definitions."""

from typing import Any, TypedDict

from matagent.schemas import RankingPlan, ScreeningRequirements
from matagent.tools import ToolCallRequest


class AgentState(TypedDict, total=False):
    """Data accumulated while a material-screening request is processed."""

    user_query: str
    material_backend: str
    requirements: ScreeningRequirements | None
    ranking_plan: RankingPlan | None
    pending_tool_calls: list[ToolCallRequest]
    search_diagnostics: dict[str, Any] | None
    candidates: list[dict[str, Any]]
    ranked_candidates: list[dict[str, Any]]
    scientific_evidence: list[dict[str, Any]]
    candidate_evidence: dict[str, list[dict[str, Any]]]
    evidence_query: str | None
    evidence_errors: list[str]
    rag_enabled: bool
    tool_history: list[dict[str, Any]]
    status: str
    errors: list[str]
    final_report: str
    report_limit: int
