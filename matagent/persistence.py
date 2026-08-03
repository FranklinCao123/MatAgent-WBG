"""Lightweight local LangGraph checkpoint persistence."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


DEFAULT_CHECKPOINT_PATH = Path(".matagent") / "checkpoints.sqlite3"
ALLOWED_CHECKPOINT_TYPES = [
    ("matagent.schemas", "ScreeningRequirements"),
    ("matagent.schemas", "RankingWeights"),
    ("matagent.schemas", "RankingPlan"),
    ("matagent.tools.schemas", "MaterialSearchArguments"),
    ("matagent.tools.schemas", "MaterialCandidate"),
    ("matagent.tools.schemas", "ToolCallRequest"),
    ("matagent.tools.schemas", "ToolExecutionResult"),
]


@contextmanager
def open_sqlite_checkpointer(
    database_path: Path,
) -> Iterator[SqliteSaver]:
    """Open a local SQLite saver and close its connection after use."""

    resolved_path = database_path.resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(resolved_path, check_same_thread=False)
    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=ALLOWED_CHECKPOINT_TYPES,
    )
    saver = SqliteSaver(connection, serde=serializer)
    saver.setup()
    try:
        yield saver
    finally:
        connection.close()


def thread_config(thread_id: str) -> dict:
    """Build the configuration LangGraph uses to isolate checkpoint threads."""

    normalized = thread_id.strip()
    if not normalized:
        raise ValueError("thread_id must not be empty.")
    return {"configurable": {"thread_id": normalized}}


def checkpoint_summaries(graph, config: dict) -> list[dict]:
    """Return safe, compact metadata without dumping complete scientific state."""

    return [
        {
            "created_at": snapshot.created_at,
            "status": snapshot.values.get("status"),
            "next": list(snapshot.next),
            "step": snapshot.metadata.get("step"),
            "source": snapshot.metadata.get("source"),
        }
        for snapshot in graph.get_state_history(config)
    ]
