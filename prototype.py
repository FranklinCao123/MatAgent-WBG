"""MatAgent-WBG minimal local LangGraph prototype.

This prototype uses illustrative mock data only. It is intended to validate the
agent workflow, not to provide scientifically reliable material recommendations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


PROJECT_ROOT = Path(__file__).resolve().parent
MOCK_DATA_PATH = PROJECT_ROOT / "mock_materials.json"


class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    user_query: str
    requirements: dict[str, Any]
    candidates: list[dict[str, Any]]
    ranked_candidates: list[dict[str, Any]]
    tool_history: list[dict[str, Any]]
    final_report: str


def parse_requirements(state: AgentState) -> dict[str, Any]:
    """Convert a user query into simple, deterministic screening criteria."""

    query = state["user_query"].lower()
    is_power_electronics = any(
        term in query for term in ("power", "功率", "power electronics", "电力电子")
    )
    is_high_temperature = any(
        term in query for term in ("high temperature", "high-temperature", "高温")
    )

    requirements = {
        "application": "power electronics" if is_power_electronics else "unspecified",
        "minimum_band_gap_ev": 2.0,
        "prefer_high_thermal_conductivity": is_high_temperature,
        "prefer_high_breakdown_field": is_power_electronics,
        "assumptions": [
            "Wide-bandgap screening uses a demonstration threshold of 2.0 eV.",
            "Cost, manufacturability, and supply-chain constraints are not yet evaluated.",
        ],
    }

    history = list(state.get("tool_history", []))
    history.append(
        {
            "step": "parse_requirements",
            "type": "deterministic_parser",
            "result": requirements,
        }
    )
    return {"requirements": requirements, "tool_history": history}


def search_mock_materials(state: AgentState) -> dict[str, Any]:
    """Search the tiny local mock dataset using the parsed band-gap threshold."""

    with MOCK_DATA_PATH.open("r", encoding="utf-8") as file:
        materials = json.load(file)

    minimum_gap = state["requirements"]["minimum_band_gap_ev"]
    candidates = [
        material
        for material in materials
        if material["band_gap_ev"] >= minimum_gap
    ]

    history = list(state.get("tool_history", []))
    history.append(
        {
            "step": "search_mock_materials",
            "type": "local_json_tool",
            "criteria": {"minimum_band_gap_ev": minimum_gap},
            "candidate_count": len(candidates),
        }
    )
    return {"candidates": candidates, "tool_history": history}


def rank_candidates(state: AgentState) -> dict[str, Any]:
    """Rank mock candidates with a transparent demonstration-only heuristic."""

    candidates = state["candidates"]
    if not candidates:
        return {"ranked_candidates": []}

    maxima = {
        "band_gap_ev": max(item["band_gap_ev"] for item in candidates),
        "thermal_conductivity_w_mk": max(
            item["thermal_conductivity_w_mk"] for item in candidates
        ),
        "breakdown_field_mv_cm": max(
            item["breakdown_field_mv_cm"] for item in candidates
        ),
    }
    weights = {
        "band_gap_ev": 0.30,
        "thermal_conductivity_w_mk": 0.35,
        "breakdown_field_mv_cm": 0.35,
    }

    ranked = []
    for material in candidates:
        score = sum(
            weights[property_name]
            * material[property_name]
            / maxima[property_name]
            for property_name in weights
        )
        ranked.append({**material, "demo_score": round(score, 3)})

    ranked.sort(key=lambda item: item["demo_score"], reverse=True)

    history = list(state.get("tool_history", []))
    history.append(
        {
            "step": "rank_candidates",
            "type": "weighted_rule",
            "weights": weights,
        }
    )
    return {"ranked_candidates": ranked, "tool_history": history}


def generate_report(state: AgentState) -> dict[str, str]:
    """Generate a compact Markdown report from the current graph state."""

    requirements = state["requirements"]
    ranked = state["ranked_candidates"]

    lines = [
        "# MatAgent-WBG Local Prototype Report",
        "",
        "> WARNING: This report uses illustrative mock data and must not be used "
        "for scientific or engineering decisions.",
        "",
        "## Interpreted request",
        "",
        f"- Original query: {state['user_query']}",
        f"- Application: {requirements['application']}",
        f"- Minimum demonstration band gap: "
        f"{requirements['minimum_band_gap_ev']} eV",
        "",
        "## Demonstration ranking",
        "",
        "| Rank | Material | Band gap (eV) | Thermal conductivity (W/mK) "
        "| Breakdown field (MV/cm) | Demo score |",
        "|---:|---|---:|---:|---:|---:|",
    ]

    if ranked:
        for index, material in enumerate(ranked, start=1):
            lines.append(
                f"| {index} | {material['name']} | {material['band_gap_ev']} | "
                f"{material['thermal_conductivity_w_mk']} | "
                f"{material['breakdown_field_mv_cm']} | "
                f"{material['demo_score']:.3f} |"
            )
    else:
        lines.append("| - | No mock candidates found | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- All property values are mock values created for workflow testing.",
            "- The score is a simple weighted rule, not a validated scientific model.",
            "- Literature evidence, uncertainty, manufacturability, and cost are not "
            "included in this prototype.",
        ]
    )
    return {"final_report": "\n".join(lines)}


def build_graph():
    """Build and compile the minimal single-agent workflow."""

    builder = StateGraph(AgentState)
    builder.add_node("parse_requirements", parse_requirements)
    builder.add_node("search_mock_materials", search_mock_materials)
    builder.add_node("rank_candidates", rank_candidates)
    builder.add_node("generate_report", generate_report)

    builder.add_edge(START, "parse_requirements")
    builder.add_edge("parse_requirements", "search_mock_materials")
    builder.add_edge("search_mock_materials", "rank_candidates")
    builder.add_edge("rank_candidates", "generate_report")
    builder.add_edge("generate_report", END)
    return builder.compile()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the lightweight MatAgent-WBG local prototype."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="寻找适合高温功率电子器件的宽禁带半导体材料",
        help="Natural-language material screening request.",
    )
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print the tool execution history after the report.",
    )
    args = parser.parse_args()

    graph = build_graph()
    result = graph.invoke({"user_query": args.query, "tool_history": []})
    print(result["final_report"])

    if args.show_trace:
        print("\n## Execution trace\n")
        print(json.dumps(result["tool_history"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
