"""Tests for Supabase health checking without making network requests."""

import json
import unittest
from unittest.mock import patch

from matagent.rag.client import SupabaseAPIError, SupabaseDataClient
from matagent.rag.database import (
    DatabaseConfigurationError,
    DatabaseHealthError,
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
        client = SupabaseDataClient(settings, opener=fake_open)
        health = check_database(settings, client=client)

        self.assertEqual(health.database_name, "postgres")
        self.assertEqual(health.vector_version, "0.8.1")
        self.assertEqual(captured["timeout"], 15)
        self.assertTrue(
            captured["request"].full_url.endswith(
                "/rest/v1/rpc/matagent_database_health"
            )
        )
        self.assertEqual(captured["request"].method, "POST")
        self.assertEqual(
            captured["request"].get_header("Apikey"), "secret-not-used"
        )
        self.assertIsNone(captured["request"].get_header("Authorization"))

    def test_missing_health_rpc_has_actionable_error(self) -> None:
        class MissingRPCClient:
            def post(self, path, payload):
                raise SupabaseAPIError(
                    "safe error",
                    status_code=404,
                    error_code="PGRST202",
                )

        settings = SupabaseSettings(
            url="https://project.supabase.co",
            secret_key="secret-not-used",
        )
        with self.assertRaisesRegex(DatabaseHealthError, "001_supabase_rag.sql"):
            check_database(settings, client=MissingRPCClient())

    def test_environment_requires_both_settings(self) -> None:
        with patch("matagent.rag.database.load_dotenv"), patch.dict(
            "os.environ", {}, clear=True
        ):
            with self.assertRaisesRegex(DatabaseConfigurationError, "SUPABASE_URL"):
                settings_from_environment()

    def test_environment_rejects_data_api_path(self) -> None:
        environment = {
            "MATAGENT_SUPABASE_URL": "https://project.supabase.co/rest/v1/",
            "MATAGENT_SUPABASE_SECRET_KEY": "secret-not-used",
        }
        with patch("matagent.rag.database.load_dotenv"), patch.dict(
            "os.environ", environment, clear=True
        ):
            with self.assertRaisesRegex(DatabaseConfigurationError, "without /rest/v1"):
                settings_from_environment()


if __name__ == "__main__":
    unittest.main()
