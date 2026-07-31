"""LangGraph node functions for the local prototype."""

import json
from collections.abc import Callable
from typing import Any

from matagent.llm import (
    RequirementParser,
    RequirementParsingError,
    RuleBasedRequirementParser,
)
from matagent.schemas import RankingPlan, RankingWeights
from matagent.state import AgentState
from matagent.tools import MockMaterialSearchTool


def create_requirement_parser_node(
    parser: RequirementParser,
) -> Callable[[AgentState], dict[str, Any]]:
    """Inject an offline or LLM-backed parser into a graph node."""

    def parse_requirements_with_parser(state: AgentState) -> dict[str, Any]:
        history = list(state.get("tool_history", []))
        try:
            requirements = parser.parse(state["user_query"])
        except RequirementParsingError as error:
            history.append(
                {
                    "step": "parse_requirements",
                    "parser": parser.name,
                    "status": "error",
                    "error": str(error),
                }
            )
            return {
                "errors": [*state.get("errors", []), str(error)],
                "status": "requirement_error",
                "tool_history": history,
            }

        history.append(
            {
                "step": "parse_requirements",
                "parser": parser.name,
                "status": "success",
                "result": requirements.model_dump(mode="json"),
            }
        )
        return {
            "requirements": requirements,
            "status": "requirements_parsed",
            "tool_history": history,
        }

    return parse_requirements_with_parser


_DEFAULT_PARSER = RuleBasedRequirementParser()
parse_requirements = create_requirement_parser_node(_DEFAULT_PARSER)


def route_after_parsing(state: AgentState) -> str:
    """Stop before tool use when requirement parsing failed."""

    if state.get("errors") or not state.get("requirements"):
        return "report"
    return "plan"


def plan_screening(state: AgentState) -> dict[str, Any]:
    """Translate parsed requirements into an auditable ranking policy."""

    requirements = state["requirements"]
    application = requirements.application.casefold()
    is_high_temperature_application = any(
        term in application for term in ("high-temperature", "high temperature", "高温")
    )
    is_power_application = any(
        term in application for term in ("power", "功率", "电力电子")
    )

    prioritize_thermal = (
        requirements.prefer_high_thermal_conductivity
        or is_high_temperature_application
    )
    prioritize_breakdown = (
        requirements.prefer_high_breakdown_field or is_power_application
    )

    raw_weights = {
        "band_gap_ev": 1.0,
        "thermal_conductivity_w_mk": 1.5 if prioritize_thermal else 1.0,
        "breakdown_field_mv_cm": 1.5 if prioritize_breakdown else 1.0,
    }
    total = sum(raw_weights.values())
    weights = RankingWeights(
        band_gap_ev=raw_weights["band_gap_ev"] / total,
        thermal_conductivity_w_mk=(
            raw_weights["thermal_conductivity_w_mk"] / total
        ),
        breakdown_field_mv_cm=(
            raw_weights["breakdown_field_mv_cm"] / total
        ),
    )

    inferred_requirements = []
    if (
        is_high_temperature_application
        and not requirements.prefer_high_thermal_conductivity
    ):
        inferred_requirements.append(
            "High thermal conductivity was inferred from the high-temperature application."
        )
    if is_power_application and not requirements.prefer_high_breakdown_field:
        inferred_requirements.append(
            "High breakdown field was inferred from the power-device application."
        )

    plan = RankingPlan(
        weights=weights,
        rationale={
            "band_gap_ev": (
                "Band gap is retained as a ranking factor after applying the hard "
                f"minimum threshold of {requirements.minimum_band_gap_ev} eV."
            ),
            "thermal_conductivity_w_mk": (
                "Thermal conductivity receives extra demonstration weight."
                if prioritize_thermal
                else "Thermal conductivity receives the baseline demonstration weight."
            ),
            "breakdown_field_mv_cm": (
                "Breakdown field receives extra demonstration weight."
                if prioritize_breakdown
                else "Breakdown field receives the baseline demonstration weight."
            ),
        },
        inferred_requirements=inferred_requirements,
    )

    history = list(state.get("tool_history", []))
    history.append(
        {
            "step": "plan_screening",
            "type": "deterministic_domain_policy",
            "status": "success",
            "result": plan.model_dump(mode="json"),
        }
    )
    return {
        "ranking_plan": plan,
        "status": "screening_planned",
        "tool_history": history,
    }


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
    weights = state["ranking_plan"].weights.model_dump()

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

    requirements = state.get("requirements")
    ranking_plan = state.get("ranking_plan")
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
    ]

    if requirements is None:
        lines.append("- Structured requirements: unavailable")
    else:
        lines.extend(
            [
                f"- Application: {requirements.application}",
                f"- Minimum band-gap requirement: "
                f"{requirements.minimum_band_gap_ev} eV",
            ]
        )

    lines.extend(
        [
            "",
            "## Workflow status",
            "",
            f"- Status before report generation: {state.get('status', 'unknown')}",
        ]
    )

    if errors:
        lines.extend(
            [
                "- The workflow encountered an error:",
                *[f"  - {error}" for error in errors],
            ]
        )

    if ranking_plan is not None:
        lines.extend(
            [
                "",
                "## Demonstration ranking plan",
                "",
                f"- Band-gap weight: {ranking_plan.weights.band_gap_ev:.1%}",
                "- Thermal-conductivity weight: "
                f"{ranking_plan.weights.thermal_conductivity_w_mk:.1%}",
                "- Breakdown-field weight: "
                f"{ranking_plan.weights.breakdown_field_mv_cm:.1%}",
            ]
        )
        if ranking_plan.inferred_requirements:
            lines.append("- Domain requirements inferred by the planner:")
            lines.extend(
                f"  - {item}" for item in ranking_plan.inferred_requirements
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
