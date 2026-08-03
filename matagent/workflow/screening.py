"""Deterministic candidate planning and ranking policies."""

from typing import Any

from matagent.domain_policy import (
    DEFAULT_MAX_ENERGY_ABOVE_HULL_EV_ATOM,
    DEFAULT_RADIOACTIVE_ELEMENT_EXCLUSIONS,
)
from matagent.schemas import CandidateFilterPolicy, RankingPlan, RankingWeights
from matagent.state import AgentState
from matagent.workflow.core import state_update


def _planned(state: AgentState, plan: RankingPlan, plan_type: str) -> dict[str, Any]:
    return state_update(
        state,
        {
            "step": "plan_screening",
            "type": plan_type,
            "status": "success",
            "result": plan.model_dump(mode="json"),
        },
        ranking_plan=plan,
        status="screening_planned",
    )


def plan_screening(state: AgentState) -> dict[str, Any]:
    """Translate parsed requirements into an auditable screening policy."""

    requirements = state["requirements"]
    application = requirements.application.casefold()
    high_temperature = any(
        term in application for term in ("high-temperature", "high temperature", "高温")
    )
    power = any(term in application for term in ("power", "功率", "电力电子"))
    prioritize_thermal = requirements.prefer_high_thermal_conductivity or high_temperature
    prioritize_breakdown = requirements.prefer_high_breakdown_field or power

    if state.get("material_backend") == "materials-project":
        unavailable = []
        if prioritize_thermal:
            unavailable.append(
                "Thermal conductivity is requested but unavailable from the "
                "Materials Project summary search."
            )
        if prioritize_breakdown:
            unavailable.append(
                "Breakdown field is requested but unavailable from the Materials "
                "Project summary search."
            )
        plan = RankingPlan(
            strategy="materials_project_stability",
            candidate_filters=CandidateFilterPolicy(
                exclude_elements=list(DEFAULT_RADIOACTIVE_ELEMENT_EXCLUSIONS),
                require_nonmetal=True,
                maximum_energy_above_hull_ev_atom=(
                    DEFAULT_MAX_ENERGY_ABOVE_HULL_EV_ATOM
                ),
                rationale={
                    "exclude_elements": "Exclude radioactive elements by default.",
                    "require_nonmetal": "The target application is semiconducting.",
                    "maximum_energy_above_hull_ev_atom": (
                        "Allow modest metastability up to 0.1 eV/atom."
                    ),
                },
            ),
            rationale={
                "band_gap_ev": "Hard constraint and final tie-breaker.",
                "is_stable": "Stable entries rank before metastable entries.",
                "energy_above_hull": "Lower values rank first.",
            },
            inferred_requirements=unavailable,
        )
        return _planned(state, plan, "materials_project_stability_policy")

    raw = {
        "band_gap_ev": 1.0,
        "thermal_conductivity_w_mk": 1.5 if prioritize_thermal else 1.0,
        "breakdown_field_mv_cm": 1.5 if prioritize_breakdown else 1.0,
    }
    total = sum(raw.values())
    weights = RankingWeights(**{name: value / total for name, value in raw.items()})
    inferred = []
    if high_temperature and not requirements.prefer_high_thermal_conductivity:
        inferred.append(
            "High thermal conductivity was inferred from the high-temperature application."
        )
    if power and not requirements.prefer_high_breakdown_field:
        inferred.append(
            "High breakdown field was inferred from the power-device application."
        )
    plan = RankingPlan(
        strategy="weighted_mock_properties",
        weights=weights,
        rationale={
            "band_gap_ev": (
                "Rank after applying the hard constraint "
                f"{requirements.band_gap_operator} "
                f"{requirements.minimum_band_gap_ev} eV."
            ),
            "thermal_conductivity_w_mk": "Extra weight when prioritized.",
            "breakdown_field_mv_cm": "Extra weight when prioritized.",
        },
        inferred_requirements=inferred,
    )
    return _planned(state, plan, "deterministic_domain_policy")


def rank_candidates(state: AgentState) -> dict[str, Any]:
    candidates = state["candidates"]
    if not candidates:
        return {"ranked_candidates": []}

    plan = state["ranking_plan"]
    if plan.strategy == "materials_project_stability":
        def sort_key(material: dict[str, Any]) -> tuple:
            hull = material.get("energy_above_hull_ev_atom")
            return (
                material.get("is_stable") is not True,
                float("inf") if hull is None else hull,
                -material["band_gap_ev"],
            )

        ranked = sorted(candidates, key=sort_key)
        trace = {
            "step": "rank_candidates",
            "type": "lexicographic_stability_rule",
            "criteria": [
                "is_stable descending",
                "energy_above_hull ascending",
                "band_gap_ev descending",
            ],
        }
    else:
        weights = plan.weights.model_dump()
        maxima = {
            name: max(item[name] for item in candidates) for name in weights
        }
        ranked = [
            {
                **material,
                "demo_score": round(
                    sum(
                        weights[name] * material[name] / maxima[name]
                        for name in weights
                    ),
                    3,
                ),
            }
            for material in candidates
        ]
        ranked.sort(key=lambda item: item["demo_score"], reverse=True)
        trace = {"step": "rank_candidates", "type": "weighted_rule", "weights": weights}

    return state_update(
        state,
        trace,
        ranked_candidates=ranked,
        status="ranked",
    )
