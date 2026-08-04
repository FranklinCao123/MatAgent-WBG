"""Command-line interface for the local MatAgent-WBG prototype."""

import argparse
import json
from pathlib import Path
import sys

from matagent.config import (
    ConfigurationError,
    LLMSettings,
    MaterialDataSettings,
    build_llm_components,
    build_material_search_tool,
    build_scientific_evidence_tool,
)
from matagent.graph import build_graph
from matagent.persistence import (
    DEFAULT_CHECKPOINT_PATH,
    checkpoint_summaries,
    open_sqlite_checkpointer,
    thread_config,
)


DEFAULT_QUERY = "寻找适合高温功率电子器件的宽禁带半导体材料"


def _configure_utf8_output() -> None:
    """Keep scientific Unicode output portable on Windows terminals and pipes."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def main() -> None:
    _configure_utf8_output()
    parser = argparse.ArgumentParser(
        description="Run the lightweight MatAgent-WBG local prototype."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help="Natural-language material screening request.",
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "deepseek"),
        default=None,
        help="Requirement parser backend; defaults to MATAGENT_LLM_MODE or offline.",
    )
    parser.add_argument(
        "--material-backend",
        choices=("mock", "materials-project"),
        default=None,
        help="Material data backend; defaults to MATAGENT_MATERIAL_BACKEND or mock.",
    )
    parser.add_argument(
        "--fetch-limit",
        type=int,
        default=None,
        help=(
            "Materials Project records fetched before local ranking "
            "(1-100; default: 100)."
        ),
    )
    parser.add_argument(
        "--report-limit",
        type=int,
        default=10,
        help="Maximum ranked candidates shown in the report (1-100; default: 10).",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Retrieve scientific evidence from the configured pgvector store.",
    )
    parser.add_argument(
        "--evidence-top-k",
        type=int,
        default=2,
        help="Evidence chunks retrieved per candidate with --rag (1-5).",
    )
    parser.add_argument(
        "--evidence-candidate-limit",
        type=int,
        default=10,
        help="Ranked candidates covered by RAG when enabled (1-10; default: 10).",
    )
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print the tool execution history after the report.",
    )
    parser.add_argument(
        "--thread-id",
        help="Persist this run under a local LangGraph checkpoint thread.",
    )
    parser.add_argument(
        "--checkpoint-db",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Local SQLite checkpoint path (default: .matagent/checkpoints.sqlite3).",
    )
    parser.add_argument(
        "--show-checkpoints",
        action="store_true",
        help="Print compact checkpoint metadata for the selected thread.",
    )
    args = parser.parse_args()

    try:
        settings = LLMSettings.from_environment(mode=args.mode)
        requirement_parser, tool_selector, report_synthesizer = build_llm_components(
            settings
        )
        material_settings = MaterialDataSettings.from_environment(
            backend=args.material_backend,
            fetch_limit=args.fetch_limit,
        )
        material_search_tool = build_material_search_tool(material_settings)
        scientific_evidence_tool = (
            build_scientific_evidence_tool() if args.rag else None
        )
    except (ConfigurationError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    if args.show_checkpoints and not args.thread_id:
        parser.error("--show-checkpoints requires --thread-id.")

    def run(checkpointer=None, config=None):
        graph = build_graph(
            requirement_parser=requirement_parser,
            tool_selector=tool_selector,
            checkpointer=checkpointer,
            material_backend=material_settings.backend,
            material_search_tool=material_search_tool,
            report_limit=args.report_limit,
            scientific_evidence_tool=scientific_evidence_tool,
            evidence_top_k=args.evidence_top_k,
            evidence_candidate_limit=args.evidence_candidate_limit,
            report_synthesizer=report_synthesizer,
        )
        result = graph.invoke({"user_query": args.query}, config=config)
        summaries = (
            checkpoint_summaries(graph, config)
            if args.show_checkpoints and config is not None
            else []
        )
        return result, summaries

    try:
        if args.thread_id:
            config = thread_config(args.thread_id)
            with open_sqlite_checkpointer(args.checkpoint_db) as checkpointer:
                result, summaries = run(checkpointer, config)
        else:
            result, summaries = run()
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(result["final_report"])

    if args.show_trace:
        print("\n## Execution trace\n")
        print(json.dumps(result["tool_history"], ensure_ascii=False, indent=2))

    if summaries:
        print("\n## Checkpoints\n")
        print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
