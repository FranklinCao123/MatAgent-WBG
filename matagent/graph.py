"""Construction of the MatAgent-WBG LangGraph workflow."""

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from matagent.nodes import (
    create_material_search_node,
    generate_report,
    parse_requirements,
    rank_candidates,
)
from matagent.state import AgentState
from matagent.tools import MockMaterialSearchTool


def default_mock_data_path() -> Path:
    return Path(__file__).resolve().parent.parent / "mock_materials.json"


def build_graph(data_path: Path | None = None):
    """Build the graph, optionally using a caller-provided mock dataset."""

    search_tool = MockMaterialSearchTool(data_path or default_mock_data_path())

    builder = StateGraph(AgentState)
    builder.add_node("parse_requirements", parse_requirements)
    builder.add_node(
        "search_materials",
        create_material_search_node(search_tool),
    )
    builder.add_node("rank_candidates", rank_candidates)
    builder.add_node("generate_report", generate_report)

    builder.add_edge(START, "parse_requirements")
    builder.add_edge("parse_requirements", "search_materials")
    builder.add_edge("search_materials", "rank_candidates")
    builder.add_edge("rank_candidates", "generate_report")
    builder.add_edge("generate_report", END)
    return builder.compile()
