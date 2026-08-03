"""LangGraph node functions for the local prototype."""

from collections.abc import Callable
from typing import Any

from matagent.domain_policy import (
    DEFAULT_MAX_ENERGY_ABOVE_HULL_EV_ATOM,
    DEFAULT_RADIOACTIVE_ELEMENT_EXCLUSIONS,
)
from matagent.llm import (
    RequirementParser,
    RequirementParsingError,
    RuleBasedRequirementParser,
    ToolSelectionError,
    ToolSelector,
)
from matagent.schemas import CandidateFilterPolicy, RankingPlan, RankingWeights
from matagent.state import AgentState
from matagent.tools import ToolExecutionError, ToolRegistry


def initialize_run(state: AgentState) -> dict[str, Any]:
    """Clear transient values that may remain in a reused checkpoint thread."""

    return {
        "requirements": None,
        "ranking_plan": None,
        "pending_tool_calls": [],
        "tool_results": [],
        "search_diagnostics": [],
        "tool_iteration": 0,
        "candidates": [],
        "ranked_candidates": [],
        "tool_history": [],
        "errors": [],
        "status": "started",
    }


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

    if state.get("material_backend") == "materials-project":
        unavailable_priorities = []
        if prioritize_thermal:
            unavailable_priorities.append(
                "Thermal conductivity is a requested priority but is unavailable "
                "from the first Materials Project summary search."
            )
        if prioritize_breakdown:
            unavailable_priorities.append(
                "Breakdown field is a requested priority but is unavailable from "
                "the first Materials Project summary search."
            )
        plan = RankingPlan(
            strategy="materials_project_stability",
            weights=None,
            candidate_filters=CandidateFilterPolicy(
                exclude_elements=list(DEFAULT_RADIOACTIVE_ELEMENT_EXCLUSIONS),
                require_nonmetal=True,
                maximum_energy_above_hull_ev_atom=(
                    DEFAULT_MAX_ENERGY_ABOVE_HULL_EV_ATOM
                ),
                rationale={
                    "exclude_elements": (
                        "Elements without conventionally stable isotopes are "
                        "excluded for the default non-nuclear device workflow."
                    ),
                    "require_nonmetal": (
                        "Metallic entries are outside this semiconductor screening task."
                    ),
                    "maximum_energy_above_hull_ev_atom": (
                        "A 0.1 eV/atom demonstration ceiling retains modestly "
                        "metastable candidates while rejecting high-energy entries."
                    ),
                },
            ),
            rationale={
                "band_gap_ev": (
                    "Band gap is enforced as a hard constraint and used as a "
                    "tie-breaker."
                ),
                "is_stable": "Entries marked stable are ordered before other entries.",
                "energy_above_hull": (
                    "Lower energy above hull is preferred as a thermodynamic "
                    "screening indicator."
                ),
            },
            inferred_requirements=unavailable_priorities,
        )
        history = list(state.get("tool_history", []))
        history.append(
            {
                "step": "plan_screening",
                "type": "materials_project_stability_policy",
                "status": "success",
                "result": plan.model_dump(mode="json"),
            }
        )
        return {
            "ranking_plan": plan,
            "status": "screening_planned",
            "tool_history": history,
        }

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
        strategy="weighted_mock_properties",
        weights=weights,
        rationale={
            "band_gap_ev": (
                "Band gap is retained as a ranking factor after applying the hard "
                f"constraint {requirements.band_gap_operator} "
                f"{requirements.minimum_band_gap_ev} eV."
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


def create_tool_decision_node(
    selector: ToolSelector,
    tool_specs: list[dict[str, Any]],
) -> Callable[[AgentState], dict[str, Any]]:
    """Inject a selector that proposes allow-listed tool calls without running them."""

    def decide_tools(state: AgentState) -> dict[str, Any]:
        history = list(state.get("tool_history", []))
        try:
            calls = selector.select(
                user_query=state["user_query"],
                requirements=state["requirements"],
                ranking_plan=state["ranking_plan"],
                tool_specs=tool_specs,
            )
            filter_policy = state["ranking_plan"].candidate_filters
            if filter_policy is not None:
                enforced_arguments = {
                    "exclude_elements": filter_policy.exclude_elements,
                    "require_nonmetal": filter_policy.require_nonmetal,
                    "maximum_energy_above_hull_ev_atom": (
                        filter_policy.maximum_energy_above_hull_ev_atom
                    ),
                }
                calls = [
                    call.model_copy(
                        update={
                            "arguments": {
                                **call.arguments,
                                **enforced_arguments,
                            }
                        }
                    )
                    if call.name == "search_materials"
                    else call
                    for call in calls
                ]
        except ToolSelectionError as error:
            history.append(
                {
                    "step": "decide_tools",
                    "selector": selector.name,
                    "status": "error",
                    "error": str(error),
                }
            )
            return {
                "pending_tool_calls": [],
                "errors": [*state.get("errors", []), str(error)],
                "status": "tool_selection_error",
                "tool_history": history,
            }

        history.append(
            {
                "step": "decide_tools",
                "selector": selector.name,
                "status": "success" if calls else "no_tool_selected",
                "calls": [call.model_dump(mode="json") for call in calls],
            }
        )
        return {
            "pending_tool_calls": calls,
            "tool_iteration": state.get("tool_iteration", 0) + 1,
            "status": "tool_calls_planned" if calls else "no_tool_selected",
            "tool_history": history,
        }

    return decide_tools


def route_after_tool_decision(state: AgentState) -> str:
    """Execute selected tools or stop when selection failed or returned nothing."""

    if state.get("errors") or not state.get("pending_tool_calls"):
        return "report"
    return "execute"


def create_tool_execution_node(
    registry: ToolRegistry,
) -> Callable[[AgentState], dict[str, Any]]:
    """Validate and execute only tool calls present in the registry."""

    def execute_tools(state: AgentState) -> dict[str, Any]:
        history = list(state.get("tool_history", []))
        results = list(state.get("tool_results", []))
        errors = list(state.get("errors", []))
        search_diagnostics = list(state.get("search_diagnostics", []))
        candidates: list[dict[str, Any]] = []

        for call in state.get("pending_tool_calls", []):
            try:
                result = registry.execute(call)
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

            results.append(result)
            if result.name == "search_materials":
                search_output = result.output
                candidates.extend(search_output["candidates"])
                search_diagnostics.append(
                    {
                        "source": search_output["source"],
                        "retrieved_count": search_output["retrieved_count"],
                        "accepted_count": len(search_output["candidates"]),
                        "excluded_count": len(search_output["excluded"]),
                        "excluded": search_output["excluded"],
                        "applied_filters": search_output["applied_filters"],
                    }
                )
            history.append(
                {
                    "step": "execute_tool",
                    "tool": result.name,
                    "call_id": result.call_id,
                    "status": "success",
                    "candidate_count": len(search_output["candidates"]),
                    "retrieved_count": search_output["retrieved_count"],
                    "excluded_count": len(search_output["excluded"]),
                }
            )

        return {
            "pending_tool_calls": [],
            "tool_results": results,
            "candidates": candidates,
            "search_diagnostics": search_diagnostics,
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
    """Rank successful search results or report an empty/error outcome."""

    if state.get("errors") or not state.get("candidates"):
        return "report"
    return "rank"


def rank_candidates(state: AgentState) -> dict[str, Any]:
    """Rank mock candidates with a transparent demonstration-only heuristic."""

    candidates = state["candidates"]
    if not candidates:
        return {"ranked_candidates": []}

    if state["ranking_plan"].strategy == "materials_project_stability":
        def materials_project_key(material: dict[str, Any]) -> tuple:
            energy_above_hull = material.get("energy_above_hull_ev_atom")
            return (
                material.get("is_stable") is not True,
                float("inf") if energy_above_hull is None else energy_above_hull,
                -material["band_gap_ev"],
            )

        ranked = sorted(candidates, key=materials_project_key)
        history = list(state.get("tool_history", []))
        history.append(
            {
                "step": "rank_candidates",
                "type": "lexicographic_stability_rule",
                "criteria": [
                    "is_stable descending",
                    "energy_above_hull ascending",
                    "band_gap_ev descending",
                ],
            }
        )
        return {
            "ranked_candidates": ranked,
            "status": "ranked",
            "tool_history": history,
        }

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

    uses_materials_project = state.get("material_backend") == "materials-project"
    report_limit = state.get("report_limit", 10)
    warning = (
        "> WARNING: Materials Project values are computed screening data; verify "
        "important candidates against methods, uncertainty, and literature."
        if uses_materials_project
        else "> WARNING: This report uses illustrative mock data and must not be used "
        "for scientific or engineering decisions."
    )
    lines = [
        "# MatAgent-WBG Local Prototype Report",
        "",
        warning,
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
                f"- Band-gap requirement: {requirements.band_gap_operator} "
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

    if ranking_plan is not None and ranking_plan.weights is not None:
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

    if ranking_plan is not None and ranking_plan.weights is None:
        candidate_filters = ranking_plan.candidate_filters
        lines.extend(
            [
                "",
                "## Screening plan",
                "",
                "- Ranking order: stable entries first, then lower energy above hull, "
                "then higher band gap.",
                *[
                    f"- {item}"
                    for item in ranking_plan.inferred_requirements
                ],
            ]
        )
        if candidate_filters is not None:
            lines.extend(
                [
                    "- Hard filters enforced by the local domain policy:",
                    "  - Require nonmetallic Materials Project entries.",
                    "  - Maximum energy above hull: "
                    f"{candidate_filters.maximum_energy_above_hull_ev_atom} eV/atom.",
                    "  - Exclude elements without conventionally stable isotopes: "
                    + ", ".join(candidate_filters.exclude_elements),
                ]
            )

    diagnostics = state.get("search_diagnostics", [])
    if uses_materials_project and diagnostics:
        latest_diagnostic = diagnostics[-1]
        lines.extend(
            [
                "",
                "## Candidate filtering",
                "",
                f"- Records returned by the API: {latest_diagnostic['retrieved_count']}",
                f"- Records accepted locally: {latest_diagnostic['accepted_count']}",
                f"- Records rejected by local verification: "
                f"{latest_diagnostic['excluded_count']}",
                f"- Report display limit: {report_limit}",
                "- The API also applies the same element, metallicity, and "
                "energy-above-hull filters before returning records.",
            ]
        )
        if latest_diagnostic["excluded"]:
            lines.append("- Local rejection details:")
            for item in latest_diagnostic["excluded"][:10]:
                lines.append(
                    f"  - {item['formula']} ({item.get('material_id') or 'no ID'}): "
                    + "; ".join(item["reasons"])
                )

    if uses_materials_project:
        lines.extend(
            [
                "",
                "## Materials Project screening ranking",
                "",
                "| Rank | Formula | MP ID | Band gap (eV) | Stable | "
                "Energy above hull (eV/atom) | Formation energy (eV/atom) |",
                "|---:|---|---|---:|---|---:|---:|",
            ]
        )
    else:
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

    if ranked and uses_materials_project:
        for index, material in enumerate(ranked[:report_limit], start=1):
            stable = material["is_stable"]
            stable_text = "yes" if stable is True else "no" if stable is False else "unknown"
            hull = material["energy_above_hull_ev_atom"]
            formation = material["formation_energy_per_atom_ev"]
            lines.append(
                f"| {index} | {material['formula']} | {material['material_id']} | "
                f"{material['band_gap_ev']:.3f} | {stable_text} | "
                f"{'unknown' if hull is None else f'{hull:.4f}'} | "
                f"{'unknown' if formation is None else f'{formation:.4f}'} |"
            )
    elif ranked:
        for index, material in enumerate(ranked[:report_limit], start=1):
            lines.append(
                f"| {index} | {material['name']} | {material['band_gap_ev']} | "
                f"{material['thermal_conductivity_w_mk']} | "
                f"{material['breakdown_field_mv_cm']} | "
                f"{material['demo_score']:.3f} |"
            )
    elif uses_materials_project:
        lines.append("| - | No candidates found | - | - | - | - | - |")
    else:
        lines.append("| - | No mock candidates found | - | - | - | - |")

    limitations = (
        [
            "- Results come from a capped Materials Project summary query, not an "
            "exhaustive screening campaign.",
            "- Thermal conductivity and breakdown field are not supplied by this "
            "first integration and are not used for ranking.",
            "- Stability ordering is a transparent heuristic, not a validated device "
            "performance model.",
            "- Literature evidence, uncertainty, manufacturability, and cost are not "
            "included yet.",
        ]
        if uses_materials_project
        else [
            "- All property values are mock values created for workflow testing.",
            "- The score is a simple weighted rule, not a validated scientific model.",
            "- Literature evidence, uncertainty, manufacturability, and cost are not "
            "included in this prototype.",
        ]
    )
    lines.extend(["", "## Limitations", "", *limitations])
    return {"final_report": "\n".join(lines), "status": "completed"}
