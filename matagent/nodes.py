"""LangGraph node functions for the local prototype."""

import json
from collections.abc import Callable
from typing import Any

from matagent.schemas import ScreeningRequirements
from matagent.state import AgentState
from matagent.tools import MockMaterialSearchTool


def parse_requirements(state: AgentState) -> dict[str, Any]:
    """Convert a user query into deterministic demonstration criteria."""

    query = state["user_query"].lower()
    is_power_electronics = any(
        term in query for term in ("power", "功率", "power electronics", "电力电子")
    )
    is_high_temperature = any(
        term in query for term in ("high temperature", "high-temperature", "高温")
    )

    requirements = ScreeningRequirements(
        application="power electronics" if is_power_electronics else "unspecified",
        minimum_band_gap_ev=2.0,
        prefer_high_thermal_conductivity=is_high_temperature,
        prefer_high_breakdown_field=is_power_electronics,
        assumptions=[
            "Wide-bandgap screening uses a demonstration threshold of 2.0 eV.",
            "Cost, manufacturability, and supply-chain constraints are not yet evaluated.",
        ],
    )

    history = list(state.get("tool_history", []))
    history.append(
        {
            "step": "parse_requirements",
            "type": "deterministic_parser",
            "result": requirements.model_dump(mode="json"),
        }
    )
    return {"requirements": requirements, "tool_history": history}


def create_material_search_node(
    tool: MockMaterialSearchTool,
) -> Callable[[AgentState], dict[str, Any]]:
    """Inject a search tool into a LangGraph-compatible node function."""

    def search_materials(state: AgentState) -> dict[str, Any]:
        minimum_gap = state["requirements"].minimum_band_gap_ev
        history = list(state.get("tool_history", []))

        try:
            candidates = tool.search(minimum_band_gap_ev=minimum_gap)
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            message = f"{tool.name} failed: {error}"
            history.append(
                {
                    "step": "search_materials",
                    "tool": tool.name,
                    "status": "error",
                    "error": message,
                }
            )
            return {
                "candidates": [],
                "errors": [*state.get("errors", []), message],
                "status": "tool_error",
                "tool_history": history,
            }

        history.append(
            {
                "step": "search_materials",
                "tool": tool.name,
                "status": "success",
                "criteria": {"minimum_band_gap_ev": minimum_gap},
                "candidate_count": len(candidates),
            }
        )
        return {
            "candidates": candidates,
            "status": "candidates_found" if candidates else "no_candidates",
            "tool_history": history,
        }

    return search_materials


def route_after_search(state: AgentState) -> str:
    """Choose whether the workflow should rank candidates or report a problem."""

    if state.get("errors") or not state.get("candidates"):
        return "report"
    return "rank"


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
    return {
        "ranked_candidates": ranked,
        "status": "ranked",
        "tool_history": history,
    }


def generate_report(state: AgentState) -> dict[str, str]:
    """Generate a compact Markdown report from the current graph state."""

    requirements = state["requirements"]
    ranked = state.get("ranked_candidates", [])
    errors = state.get("errors", [])

    lines = [
        "# MatAgent-WBG Local Prototype Report",
        "",
        "> WARNING: This report uses illustrative mock data and must not be used "
        "for scientific or engineering decisions.",
        "",
        "## Interpreted request",
        "",
        f"- Original query: {state['user_query']}",
        f"- Application: {requirements.application}",
        f"- Minimum demonstration band gap: "
        f"{requirements.minimum_band_gap_ev} eV",
        "",
        "## Workflow status",
        "",
        f"- Status before report generation: {state.get('status', 'unknown')}",
    ]

    if errors:
        lines.extend(
            [
                "- The workflow encountered an error:",
                *[f"  - {error}" for error in errors],
            ]
        )

    lines.extend(
        [
        "",
        "## Demonstration ranking",
        "",
        "| Rank | Material | Band gap (eV) | Thermal conductivity (W/mK) "
        "| Breakdown field (MV/cm) | Demo score |",
        "|---:|---|---:|---:|---:|---:|",
        ]
    )

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
    return {"final_report": "\n".join(lines), "status": "completed"}
