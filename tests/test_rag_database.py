"""Tests for Supabase health checking without making network requests."""

import json
import unittest
from unittest.mock import patch

from matagent.rag.database import (
    DatabaseConfigurationError,
    SupabaseSettings,
    check_database,
    settings_from_environment,
)


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            [
                {
                    "database_name": "postgres",
                    "postgres_version": "17.5",
                    "vector_version": "0.8.1",
                }
            ]
        ).encode()


class DatabaseHealthTests(unittest.TestCase):
    def test_health_check_uses_https_rpc_and_returns_safe_metadata(self) -> None:
        captured = {}

        def fake_open(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse()

        settings = SupabaseSettings(
            url="https://project.supabase.co",
            secret_key="secret-not-used",
        )
        health = check_database(settings, opener=fake_open)

        self.assertEqual(health.database_name, "postgres")
        self.assertEqual(health.vector_version, "0.8.1")
        self.assertEqual(captured["timeout"], 15)
        self.assertTrue(
            captured["request"].full_url.endswith(
                "/rest/v1/rpc/matagent_database_health"
            )
        )
        self.assertEqual(captured["request"].method, "POST")

    def test_environment_requires_both_settings(self) -> None:
        with patch("matagent.rag.database.load_dotenv"), patch.dict(
            "os.environ", {}, clear=True
        ):
            with self.assertRaisesRegex(DatabaseConfigurationError, "SUPABASE_URL"):
                settings_from_environment()


if __name__ == "__main__":
    unittest.main()
