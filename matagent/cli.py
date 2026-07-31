"""Command-line interface for the local MatAgent-WBG prototype."""

import argparse
import json
from pathlib import Path

from matagent.config import (
    ConfigurationError,
    LLMSettings,
    build_requirement_parser,
    build_tool_selector,
)
from matagent.graph import build_graph
from matagent.persistence import (
    DEFAULT_CHECKPOINT_PATH,
    checkpoint_summaries,
    open_sqlite_checkpointer,
    thread_config,
)


DEFAULT_QUERY = "寻找适合高温功率电子器件的宽禁带半导体材料"


def main() -> None:
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
        requirement_parser = build_requirement_parser(settings)
        tool_selector = build_tool_selector(settings)
    except (ConfigurationError, RuntimeError) as error:
        parser.error(str(error))

    if args.show_checkpoints and not args.thread_id:
        parser.error("--show-checkpoints requires --thread-id.")

    def run(checkpointer=None, config=None):
        graph = build_graph(
            requirement_parser=requirement_parser,
            tool_selector=tool_selector,
            checkpointer=checkpointer,
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
