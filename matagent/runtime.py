"""Shared runtime assembly for CLI and web entry points."""

from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver

from matagent.config import (
    LLMSettings,
    MaterialDataSettings,
    build_llm_components,
    build_material_search_tool,
    build_scientific_evidence_tool,
)
from matagent.graph import build_graph


@dataclass(frozen=True)
class RuntimeOptions:
    """Small set of user-facing choices; secrets remain in the environment."""

    mode: str | None = None
    material_backend: str | None = None
    fetch_limit: int | None = None
    report_limit: int = 10
    use_rag: bool = False
    evidence_top_k: int = 2
    evidence_candidate_limit: int = 10


def build_runtime_graph(
    options: RuntimeOptions,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """Build one configured graph without exposing credentials to graph state."""

    llm_settings = LLMSettings.from_environment(mode=options.mode)
    parser, selector, synthesizer = build_llm_components(llm_settings)
    material_settings = MaterialDataSettings.from_environment(
        backend=options.material_backend,
        fetch_limit=options.fetch_limit,
    )
    material_tool = build_material_search_tool(material_settings)
    evidence_tool = build_scientific_evidence_tool() if options.use_rag else None
    return build_graph(
        requirement_parser=parser,
        tool_selector=selector,
        checkpointer=checkpointer,
        material_backend=material_settings.backend,
        material_search_tool=material_tool,
        report_limit=options.report_limit,
        scientific_evidence_tool=evidence_tool,
        evidence_top_k=options.evidence_top_k,
        evidence_candidate_limit=options.evidence_candidate_limit,
        report_synthesizer=synthesizer,
    )
