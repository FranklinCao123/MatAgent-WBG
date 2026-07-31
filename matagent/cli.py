"""Command-line interface for the local MatAgent-WBG prototype."""

import argparse
import json

from matagent.config import (
    ConfigurationError,
    LLMSettings,
    build_requirement_parser,
)
from matagent.graph import build_graph


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
    args = parser.parse_args()

    try:
        settings = LLMSettings.from_environment(mode=args.mode)
        requirement_parser = build_requirement_parser(settings)
    except (ConfigurationError, RuntimeError) as error:
        parser.error(str(error))

    graph = build_graph(requirement_parser=requirement_parser)
    result = graph.invoke(
        {
            "user_query": args.query,
            "tool_history": [],
            "errors": [],
            "status": "started",
        }
    )
    print(result["final_report"])

    if args.show_trace:
        print("\n## Execution trace\n")
        print(json.dumps(result["tool_history"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
