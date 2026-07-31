"""Construction of the MatAgent-WBG LangGraph workflow."""

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from matagent.nodes import (
    create_requirement_parser_node,
    create_material_search_node,
    generate_report,
    rank_candidates,
    route_after_parsing,
    route_after_search,
)
from matagent.llm import RequirementParser, RuleBasedRequirementParser
from matagent.state import AgentState
from matagent.tools import MockMaterialSearchTool


def default_mock_data_path() -> Path:
    return Path(__file__).resolve().parent.parent / "mock_materials.json"


def build_graph(
    data_path: Path | None = None,
    requirement_parser: RequirementParser | None = None,
):
    """Build the graph with caller-selectable data and parser backends."""

    search_tool = MockMaterialSearchTool(data_path or default_mock_data_path())
    parser = requirement_parser or RuleBasedRequirementParser()

    builder = StateGraph(AgentState)
    builder.add_node(
        "parse_requirements",
        create_requirement_parser_node(parser),
    )
    builder.add_node(
        "search_materials",
        create_material_search_node(search_tool),
    )
    builder.add_node("rank_candidates", rank_candidates)
    builder.add_node("generate_report", generate_report)

    builder.add_edge(START, "parse_requirements")
    builder.add_conditional_edges(
        "parse_requirements",
        route_after_parsing,
        {
            "search": "search_materials",
            "report": "generate_report",
        },
    )
    builder.add_conditional_edges(
        "search_materials",
        route_after_search,
        {
            "rank": "rank_candidates",
            "report": "generate_report",
        },
    )
    builder.add_edge("rank_candidates", "generate_report")
    builder.add_edge("generate_report", END)
    return builder.compile()
