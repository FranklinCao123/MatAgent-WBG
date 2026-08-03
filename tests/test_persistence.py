"""Tests for local SQLite checkpoint persistence and thread isolation."""

import tempfile
import unittest
from pathlib import Path

from matagent.graph import build_graph
from matagent.persistence import (
    checkpoint_summaries,
    open_sqlite_checkpointer,
    thread_config,
)


class PersistenceTests(unittest.TestCase):
    def test_completed_state_and_history_are_saved_to_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "checkpoints.sqlite3"
            config = thread_config("saved-thread")

            with open_sqlite_checkpointer(database_path) as checkpointer:
                graph = build_graph(checkpointer=checkpointer)
                result = graph.invoke(
                    {"user_query": "Find power semiconductor materials"},
                    config=config,
                )
                latest = graph.get_state(config)
                summaries = checkpoint_summaries(graph, config)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(latest.values["status"], "completed")
            self.assertEqual(
                latest.values["user_query"],
                "Find power semiconductor materials",
            )
            self.assertGreater(len(summaries), 5)
            self.assertEqual(summaries[0]["status"], "completed")
            self.assertTrue(database_path.exists())
            self.assertGreater(database_path.stat().st_size, 0)

    def test_thread_ids_keep_latest_state_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "checkpoints.sqlite3"
            config_a = thread_config("thread-a")
            config_b = thread_config("thread-b")

            with open_sqlite_checkpointer(database_path) as checkpointer:
                graph = build_graph(checkpointer=checkpointer)
                graph.invoke(
                    {"user_query": "Thread A power query"},
                    config=config_a,
                )
                graph.invoke(
                    {"user_query": "Thread B general query"},
                    config=config_b,
                )

                state_a = graph.get_state(config_a)
                state_b = graph.get_state(config_b)

            self.assertEqual(state_a.values["user_query"], "Thread A power query")
            self.assertEqual(state_b.values["user_query"], "Thread B general query")
            self.assertNotEqual(state_a.config, state_b.config)

    def test_reused_thread_resets_transient_run_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "checkpoints.sqlite3"
            config = thread_config("reused-thread")

            with open_sqlite_checkpointer(database_path) as checkpointer:
                graph = build_graph(checkpointer=checkpointer)
                graph.invoke(
                    {"user_query": "First power query"},
                    config=config,
                )
                second = graph.invoke(
                    {"user_query": "Second general query"},
                    config=config,
                )

            self.assertEqual(second["user_query"], "Second general query")
            self.assertEqual(
                [entry["step"] for entry in second["tool_history"]],
                [
                    "parse_requirements",
                    "plan_screening",
                    "decide_tools",
                    "execute_tool",
                    "rank_candidates",
                ],
            )

    def test_empty_thread_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            thread_config("   ")


if __name__ == "__main__":
    unittest.main()
