"""Core LangGraph nodes for parsing, tool selection, and execution."""

from collections.abc import Callable
from typing import Any

from matagent.llm import (
    RequirementParser,
    RequirementParsingError,
    ToolSelectionError,
    ToolSelector,
)
from matagent.state import AgentState
from matagent.tools import ToolCallRequest, ToolExecutionError, ToolRegistry


def state_update(
    state: AgentState,
    entry: dict[str, Any],
    **updates: Any,
) -> dict[str, Any]:
    return {**updates, "tool_history": [*state.get("tool_history", []), entry]}


def initialize_run(state: AgentState) -> dict[str, Any]:
    """Clear transient values when a checkpoint thread is reused."""

    return {
        "requirements": None,
        "ranking_plan": None,
        "pending_tool_calls": [],
        "search_diagnostics": None,
        "candidates": [],
        "ranked_candidates": [],
        "scientific_evidence": [],
        "candidate_evidence": {},
        "evidence_query": None,
        "evidence_errors": [],
        "tool_history": [],
        "errors": [],
        "status": "started",
    }


def create_evidence_retrieval_node(
    registry: ToolRegistry,
    *,
    top_k: int,
    candidate_limit: int,
) -> Callable[[AgentState], dict[str, Any]]:
    """Call the registered RAG tool after candidate ranking."""

    def retrieve_evidence(state: AgentState) -> dict[str, Any]:
        candidate_names = list(
            dict.fromkeys(
                candidate.get("name") or candidate.get("formula")
                for candidate in state.get("ranked_candidates", [])[:candidate_limit]
            )
        )
        candidate_names = [name for name in candidate_names if name]
        call = ToolCallRequest(
            id="rag_evidence_00",
            name="retrieve_candidate_evidence",
            arguments={
                "user_query": state["user_query"],
                "candidates": candidate_names,
                "evidence_per_candidate": top_k,
                "minimum_similarity": 0.0,
            },
        )
        try:
            result = registry.execute(call)
        except ToolExecutionError as error:
            return state_update(
                state,
                {
                    "step": "retrieve_evidence",
                    "tool": call.name,
                    "status": "error",
                    "error": str(error),
                },
                scientific_evidence=[],
                candidate_evidence={},
                evidence_query=state["user_query"],
                evidence_errors=[*state.get("evidence_errors", []), str(error)],
                status="evidence_error",
            )
        grouped = result.output["candidate_evidence"]
        evidence = [item for items in grouped.values() for item in items]
        return state_update(
            state,
            {
                "step": "retrieve_evidence",
                "tool": call.name,
                "status": "success",
                "evidence_count": len(evidence),
                "candidate_count": len(grouped),
            },
            scientific_evidence=evidence,
            candidate_evidence=grouped,
            evidence_query=state["user_query"],
            status="evidence_retrieved" if evidence else "no_evidence",
        )

    return retrieve_evidence


def create_requirement_parser_node(
    parser: RequirementParser,
) -> Callable[[AgentState], dict[str, Any]]:
    def parse_requirements(state: AgentState) -> dict[str, Any]:
        try:
            requirements = parser.parse(state["user_query"])
        except RequirementParsingError as error:
            return state_update(
                state,
                {
                    "step": "parse_requirements",
                    "parser": parser.name,
                    "status": "error",
                    "error": str(error),
                },
                errors=[*state.get("errors", []), str(error)],
                status="requirement_error",
            )
        return state_update(
            state,
            {
                "step": "parse_requirements",
                "parser": parser.name,
                "status": "success",
                "result": requirements.model_dump(mode="json"),
            },
            requirements=requirements,
            status="requirements_parsed",
        )

    return parse_requirements


def route_after_parsing(state: AgentState) -> str:
    return "report" if state.get("errors") or not state.get("requirements") else "plan"


def create_tool_decision_node(
    selector: ToolSelector,
    tool_specs: list[dict[str, Any]],
) -> Callable[[AgentState], dict[str, Any]]:
    def decide_tools(state: AgentState) -> dict[str, Any]:
        try:
            calls = selector.select(
                user_query=state["user_query"],
                requirements=state["requirements"],
                ranking_plan=state["ranking_plan"],
                tool_specs=tool_specs,
            )
            filters = state["ranking_plan"].candidate_filters
            if filters:
                enforced = {
                    "exclude_elements": filters.exclude_elements,
                    "require_nonmetal": filters.require_nonmetal,
                    "maximum_energy_above_hull_ev_atom": (
                        filters.maximum_energy_above_hull_ev_atom
                    ),
                }
                calls = [
                    call.model_copy(
                        update={"arguments": {**call.arguments, **enforced}}
                    )
                    if call.name == "search_materials"
                    else call
                    for call in calls
                ]
        except ToolSelectionError as error:
            return state_update(
                state,
                {
                    "step": "decide_tools",
                    "selector": selector.name,
                    "status": "error",
                    "error": str(error),
                },
                pending_tool_calls=[],
                errors=[*state.get("errors", []), str(error)],
                status="tool_selection_error",
            )

        status = "tool_calls_planned" if calls else "no_tool_selected"
        return state_update(
            state,
            {
                "step": "decide_tools",
                "selector": selector.name,
                "status": "success" if calls else status,
                "calls": [call.model_dump(mode="json") for call in calls],
            },
            pending_tool_calls=calls,
            status=status,
        )

    return decide_tools


def route_after_tool_decision(state: AgentState) -> str:
    return (
        "report"
        if state.get("errors") or not state.get("pending_tool_calls")
        else "execute"
    )


def create_tool_execution_node(
    registry: ToolRegistry,
) -> Callable[[AgentState], dict[str, Any]]:
    def execute_tools(state: AgentState) -> dict[str, Any]:
        history = list(state.get("tool_history", []))
        errors = list(state.get("errors", []))
        candidates = []
        diagnostic = None

        for call in state.get("pending_tool_calls", []):
            try:
                result = registry.execute(call)
                output = result.output
            except ToolExecutionError as error:
                errors.append(str(error))
                history.append(
                    {
                        "step": "execute_tool",
                        "tool": call.name,
                        "call_id": call.id,
                        "status": "error",
                        "error": str(error),
                    }
                )
                continue

            candidates.extend(output["candidates"])
            diagnostic = {
                "source": output["source"],
                "retrieved_count": output["retrieved_count"],
                "accepted_count": len(output["candidates"]),
                "excluded_count": len(output["excluded"]),
                "excluded": output["excluded"],
                "applied_filters": output["applied_filters"],
            }
            history.append(
                {
                    "step": "execute_tool",
                    "tool": result.name,
                    "call_id": result.call_id,
                    "status": "success",
                    "candidate_count": len(output["candidates"]),
                    "retrieved_count": output["retrieved_count"],
                    "excluded_count": len(output["excluded"]),
                }
            )

        return {
            "pending_tool_calls": [],
            "candidates": candidates,
            "search_diagnostics": diagnostic,
            "errors": errors,
            "status": (
                "tool_error"
                if errors
                else "candidates_found" if candidates else "no_candidates"
            ),
            "tool_history": history,
        }

    return execute_tools


def route_after_tool_execution(state: AgentState) -> str:
    return "report" if state.get("errors") or not state.get("candidates") else "rank"
