"""Construction of the MatAgent-WBG LangGraph workflow."""

from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from matagent.workflow import (
    create_evidence_retrieval_node,
    create_requirement_parser_node,
    create_tool_decision_node,
    create_tool_execution_node,
    create_report_node,
    initialize_run,
    plan_screening,
    rank_candidates,
    route_after_parsing,
    route_after_tool_decision,
    route_after_tool_execution,
)
from matagent.llm import (
    ReportSynthesizer,
    RequirementParser,
    RuleBasedRequirementParser,
    RuleBasedToolSelector,
    ToolSelector,
)
from matagent.state import AgentState
from matagent.tools import (
    CandidateEvidenceArguments,
    MaterialSearchArguments,
    MockMaterialSearchTool,
    ToolRegistry,
)


def default_mock_data_path() -> Path:
    return Path(__file__).resolve().parent.parent / "mock_materials.json"


def build_graph(
    data_path: Path | None = None,
    requirement_parser: RequirementParser | None = None,
    tool_selector: ToolSelector | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    material_backend: str = "mock",
    material_search_tool: Any | None = None,
    report_limit: int = 10,
    scientific_evidence_tool: Any | None = None,
    evidence_top_k: int = 2,
    evidence_candidate_limit: int = 5,
    report_synthesizer: ReportSynthesizer | None = None,
):
    """Build the graph with caller-selectable data and parser backends."""

    if material_backend not in ("mock", "materials-project"):
        raise ValueError(f"Unsupported material backend: {material_backend}")
    if not 1 <= report_limit <= 100:
        raise ValueError("report_limit must be between 1 and 100.")
    if material_backend == "materials-project" and material_search_tool is None:
        raise ValueError("Materials Project backend requires a configured search tool.")
    if not 1 <= evidence_top_k <= 5:
        raise ValueError("evidence_top_k must be between 1 and 5.")
    if not 1 <= evidence_candidate_limit <= 10:
        raise ValueError("evidence_candidate_limit must be between 1 and 10.")
    search_tool = material_search_tool or MockMaterialSearchTool(
        data_path or default_mock_data_path()
    )
    parser = requirement_parser or RuleBasedRequirementParser()
    selector = tool_selector or RuleBasedToolSelector()
    registry = ToolRegistry()
    registry.register(
        name="search_materials",
        description=(
            "Search the available semiconductor material dataset using an exact "
            "band-gap threshold and comparison operator."
        ),
        arguments_model=MaterialSearchArguments,
        handler=search_tool.search,
    )
    tool_specs = registry.tool_specs()
    if scientific_evidence_tool is not None:
        registry.register(
            name="retrieve_candidate_evidence",
            description=(
                "Retrieve attributable scientific passages for ranked materials."
            ),
            arguments_model=CandidateEvidenceArguments,
            handler=scientific_evidence_tool.search_candidates,
        )

    builder = StateGraph(AgentState)
    builder.add_node(
        "initialize_run",
        lambda state: {
            **initialize_run(state),
            "material_backend": material_backend,
            "report_limit": report_limit,
            "rag_enabled": scientific_evidence_tool is not None,
        },
    )
    builder.add_node(
        "parse_requirements",
        create_requirement_parser_node(parser),
    )
    builder.add_node(
        "decide_tools",
        create_tool_decision_node(selector, tool_specs),
    )
    builder.add_node("execute_tools", create_tool_execution_node(registry))
    builder.add_node("plan_screening", plan_screening)
    builder.add_node("rank_candidates", rank_candidates)
    if scientific_evidence_tool is not None:
        builder.add_node(
            "retrieve_evidence",
            create_evidence_retrieval_node(
                registry,
                top_k=evidence_top_k,
                candidate_limit=evidence_candidate_limit,
            ),
        )
    builder.add_node("generate_report", create_report_node(report_synthesizer))

    builder.add_edge(START, "initialize_run")
    builder.add_edge("initialize_run", "parse_requirements")
    builder.add_conditional_edges(
        "parse_requirements",
        route_after_parsing,
        {
            "plan": "plan_screening",
            "report": "generate_report",
        },
    )
    builder.add_edge("plan_screening", "decide_tools")
    builder.add_conditional_edges(
        "decide_tools",
        route_after_tool_decision,
        {
            "execute": "execute_tools",
            "report": "generate_report",
        },
    )
    builder.add_conditional_edges(
        "execute_tools",
        route_after_tool_execution,
        {
            "rank": "rank_candidates",
            "report": "generate_report",
        },
    )
    if scientific_evidence_tool is not None:
        builder.add_edge("rank_candidates", "retrieve_evidence")
        builder.add_edge("retrieve_evidence", "generate_report")
    else:
        builder.add_edge("rank_candidates", "generate_report")
    builder.add_edge("generate_report", END)
    return builder.compile(checkpointer=checkpointer)
