"""Markdown report rendering for workflow results."""

from matagent.llm import ReportSynthesisError, ReportSynthesizer
from matagent.schemas import GroundedReport
from matagent.state import AgentState


def _section(lines: list[str], title: str, items: list[str]) -> None:
    lines.extend(["", f"## {title}", "", *items])


def _number(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.4f}"


def _table_text(value: str, *, limit: int = 240) -> str:
    compact = " ".join(value.split()).replace("|", "\\|")
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _generate_base_report(state: AgentState) -> str:
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
            "- Ranking order: device-relevant chemistry class, stability, lower "
            "energy above hull, then higher band gap.",
            *[f"- {item}" for item in plan.inferred_requirements],
        ]
        if filters:
            items += [
                "- Hard filters: nonmetallic, experimentally observed entries; "
                f"at most {filters.maximum_element_count} elements; energy above hull <= "
                f"{filters.maximum_energy_above_hull_ev_atom} eV/atom.",
                "- Device-chemistry baseline excludes hydrides, halides, noble-gas "
                "solids, and radioactive elements.",
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

    if state.get("rag_enabled"):
        grouped = state.get("candidate_evidence", {})
        evidence_errors = state.get("evidence_errors", [])
        if grouped:
            table = [
                "| Material | Evidence coverage | Source | Similarity | Evidence |",
                "|---|---|---|---:|---|",
            ]
            for material, items in grouped.items():
                coverage = (
                    "none" if not items else "limited" if len(items) == 1 else "multiple"
                )
                if not items:
                    table.append(
                        f"| {_table_text(material)} | none | - | - | "
                        "No matching passage stored. |"
                    )
                    continue
                for index, item in enumerate(items):
                    title = _table_text(item["title"], limit=70)
                    source = (
                        f"[{title}]({item['source_url']})"
                        if item.get("source_url")
                        else title
                    )
                    if item.get("doi"):
                        source += f"; DOI `{item['doi']}`"
                    table.append(
                        f"| {_table_text(material) if index == 0 else ''} | "
                        f"{coverage if index == 0 else ''} | {source} | "
                        f"{item['similarity']:.3f} | "
                        f"{_table_text(item['content'])} |"
                    )
            _section(lines, "Candidate-specific scientific evidence", table)
            limitations.append(
                "- Evidence coverage labels count retrieved passages; they do not "
                "validate property values or experimental quality."
            )
        elif evidence_errors:
            _section(
                lines,
                "Scientific evidence",
                ["- Evidence retrieval failed:"]
                + [f"  - {error}" for error in evidence_errors],
            )
            limitations.append("- No scientific evidence was retrieved.")
        else:
            _section(
                lines,
                "Scientific evidence",
                ["- No matching evidence is currently stored in the RAG database."],
            )
            limitations.append("- The RAG database returned no matching evidence.")

    if not state.get("rag_enabled"):
        limitations.append("- Literature evidence is not enabled for this run.")
    limitations.append("- Uncertainty, manufacturability, and cost are not included.")
    _section(lines, "Limitations", limitations)
    return "\n".join(lines)


def _render_grounded_report(report: GroundedReport) -> str:
    lines = ["## Grounded LLM assessment", "", report.executive_summary]
    for item in report.candidate_assessments:
        lines += [
            "",
            f"### {item.material} — {item.confidence} confidence",
            "",
            item.assessment,
            "",
            "- Evidence DOI: "
            + (", ".join(f"`{doi}`" for doi in item.evidence_dois) or "none retrieved"),
        ]
    if report.caveats:
        lines += ["", "### Synthesis caveats", ""] + [
            f"- {caveat}" for caveat in report.caveats
        ]
    return "\n".join(lines)


def create_report_node(
    synthesizer: ReportSynthesizer | None = None,
):
    """Return deterministic reporting with optional citation-checked synthesis."""

    def generate(state: AgentState) -> dict:
        base_report = _generate_base_report(state)
        if synthesizer is None or not state.get("ranked_candidates"):
            return {"final_report": base_report, "status": "completed"}
        try:
            synthesis = synthesizer.synthesize(
                user_query=state["user_query"],
                requirements=state["requirements"],
                ranked_candidates=state["ranked_candidates"],
                candidate_evidence=state.get("candidate_evidence", {}),
            )
        except ReportSynthesisError as error:
            history = [
                *state.get("tool_history", []),
                {
                    "step": "synthesize_report",
                    "synthesizer": synthesizer.name,
                    "status": "error",
                    "error": str(error),
                },
            ]
            fallback = (
                f"{base_report}\n\n## Grounded LLM assessment\n\n"
                f"- Synthesis unavailable; deterministic report retained: {error}"
            )
            return {
                "final_report": fallback,
                "tool_history": history,
                "status": "completed",
            }

        history = [
            *state.get("tool_history", []),
            {
                "step": "synthesize_report",
                "synthesizer": synthesizer.name,
                "status": "success",
                "candidate_count": len(synthesis.candidate_assessments),
            },
        ]
        return {
            "final_report": f"{base_report}\n\n{_render_grounded_report(synthesis)}",
            "tool_history": history,
            "status": "completed",
        }

    return generate


def generate_report(state: AgentState) -> dict:
    """Backward-compatible deterministic report node."""

    return create_report_node()(state)
