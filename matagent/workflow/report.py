"""Markdown report rendering for workflow results."""

from matagent.state import AgentState


def _section(lines: list[str], title: str, items: list[str]) -> None:
    lines.extend(["", f"## {title}", "", *items])


def _number(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.4f}"


def generate_report(state: AgentState) -> dict[str, str]:
    requirements = state.get("requirements")
    plan = state.get("ranking_plan")
    ranked = state.get("ranked_candidates", [])
    is_live = state.get("material_backend") == "materials-project"
    limit = state.get("report_limit", 10)
    warning = (
        "> WARNING: Materials Project values are computed screening data; verify "
        "important candidates against methods, uncertainty, and literature."
        if is_live
        else "> WARNING: This report uses illustrative mock data and must not be used "
        "for scientific or engineering decisions."
    )
    lines = ["# MatAgent-WBG Local Prototype Report", "", warning]

    request_items = [f"- Original query: {state['user_query']}"]
    if requirements:
        request_items += [
            f"- Application: {requirements.application}",
            f"- Band-gap requirement: {requirements.band_gap_operator} "
            f"{requirements.minimum_band_gap_ev} eV",
        ]
    else:
        request_items.append("- Structured requirements: unavailable")
    _section(lines, "Interpreted request", request_items)

    status_items = [
        f"- Status before report generation: {state.get('status', 'unknown')}"
    ]
    if state.get("errors"):
        status_items += ["- The workflow encountered an error:"] + [
            f"  - {error}" for error in state["errors"]
        ]
    _section(lines, "Workflow status", status_items)

    if plan and plan.weights:
        _section(
            lines,
            "Demonstration ranking plan",
            [
                f"- Band-gap weight: {plan.weights.band_gap_ev:.1%}",
                f"- Thermal-conductivity weight: "
                f"{plan.weights.thermal_conductivity_w_mk:.1%}",
                f"- Breakdown-field weight: {plan.weights.breakdown_field_mv_cm:.1%}",
                *[f"- {item}" for item in plan.inferred_requirements],
            ],
        )
    elif plan:
        filters = plan.candidate_filters
        items = [
            "- Ranking order: stable entries, lower energy above hull, then higher "
            "band gap.",
            *[f"- {item}" for item in plan.inferred_requirements],
        ]
        if filters:
            items += [
                "- Hard filters: nonmetallic entries; energy above hull <= "
                f"{filters.maximum_energy_above_hull_ev_atom} eV/atom; exclude "
                f"{len(filters.exclude_elements)} radioactive-element symbols.",
                "- Full filter arguments are available in the execution trace.",
            ]
        _section(lines, "Screening plan", items)

    diagnostic = state.get("search_diagnostics")
    if is_live and diagnostic:
        items = [
            f"- API records: {diagnostic['retrieved_count']}",
            f"- Locally accepted: {diagnostic['accepted_count']}",
            f"- Locally rejected: {diagnostic['excluded_count']}",
            f"- Report display limit: {limit}",
        ] + [
            f"  - {item['formula']} ({item.get('material_id') or 'no ID'}): "
            + "; ".join(item["reasons"])
            for item in diagnostic["excluded"][:10]
        ]
        _section(lines, "Candidate filtering", items)

    if is_live:
        table = [
            "| Rank | Formula | MP ID | Band gap (eV) | Stable | "
            "Energy above hull (eV/atom) | Formation energy (eV/atom) |",
            "|---:|---|---|---:|---|---:|---:|",
        ]
        for index, material in enumerate(ranked[:limit], 1):
            stable = {True: "yes", False: "no"}.get(material["is_stable"], "unknown")
            table.append(
                f"| {index} | {material['formula']} | {material['material_id']} | "
                f"{material['band_gap_ev']:.3f} | {stable} | "
                f"{_number(material['energy_above_hull_ev_atom'])} | "
                f"{_number(material['formation_energy_per_atom_ev'])} |"
            )
        if not ranked:
            table.append("| - | No candidates found | - | - | - | - | - |")
        _section(lines, "Materials Project screening ranking", table)
        limitations = [
            "- The capped query is not an exhaustive screening campaign.",
            "- Thermal conductivity and breakdown field are not available or ranked.",
            "- Stability ordering is a heuristic, not a device-performance model.",
        ]
    else:
        table = [
            "| Rank | Material | Band gap (eV) | Thermal conductivity (W/mK) "
            "| Breakdown field (MV/cm) | Demo score |",
            "|---:|---|---:|---:|---:|---:|",
            *[
                f"| {index} | {material['name']} | {material['band_gap_ev']} | "
                f"{material['thermal_conductivity_w_mk']} | "
                f"{material['breakdown_field_mv_cm']} | "
                f"{material['demo_score']:.3f} |"
                for index, material in enumerate(ranked[:limit], 1)
            ],
        ]
        if not ranked:
            table.append("| - | No mock candidates found | - | - | - | - |")
        _section(lines, "Demonstration ranking", table)
        limitations = [
            "- Property values are illustrative mock data.",
            "- The weighted score is not a validated scientific model.",
        ]

    limitations.append(
        "- Literature, uncertainty, manufacturability, and cost are not included."
    )
    _section(lines, "Limitations", limitations)
    return {"final_report": "\n".join(lines), "status": "completed"}
