"""Supabase Data API configuration and a safe RAG health check."""

import os
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit

from dotenv import load_dotenv


class DatabaseConfigurationError(RuntimeError):
    """Raised when Supabase configuration is missing."""


class DatabaseHealthError(RuntimeError):
    """Raised when Supabase, PostgreSQL, or pgvector is unavailable."""


class DataAPIPoster(Protocol):
    """Minimal dependency accepted by the health check."""

    def post(self, path: str, payload: dict[str, Any]) -> Any: ...


@dataclass(frozen=True)
class SupabaseSettings:
    """Server-side Supabase credentials, with the secret hidden from repr."""

    url: str
    secret_key: str = field(repr=False)


@dataclass(frozen=True)
class DatabaseHealth:
    """Non-secret database metadata returned by a successful health check."""

    database_name: str
    postgres_version: str
    vector_version: str


def settings_from_environment() -> SupabaseSettings:
    """Load server-only credentials without logging or storing them in state."""

    load_dotenv()
    url = os.getenv("MATAGENT_SUPABASE_URL", "").strip().rstrip("/")
    secret_key = os.getenv("MATAGENT_SUPABASE_SECRET_KEY", "").strip()
    missing = []
    if not url:
        missing.append("MATAGENT_SUPABASE_URL")
    if not secret_key:
        missing.append("MATAGENT_SUPABASE_SECRET_KEY")
    if missing:
        raise DatabaseConfigurationError(
            f"Missing Supabase setting(s): {', '.join(missing)}."
        )
    if not url.startswith("https://"):
        raise DatabaseConfigurationError("MATAGENT_SUPABASE_URL must use HTTPS.")
    parsed_url = urlsplit(url)
    if parsed_url.path not in ("", "/") or parsed_url.query:
        raise DatabaseConfigurationError(
            "MATAGENT_SUPABASE_URL must be the Project URL without /rest/v1 "
            "or query parameters."
        )
    return SupabaseSettings(url=url, secret_key=secret_key)


def check_database(
    settings: SupabaseSettings,
    *,
    client: DataAPIPoster | None = None,
) -> DatabaseHealth:
    """Call the restricted health RPC through Supabase's HTTPS Data API."""

    # Import here avoids a module cycle: the reusable client uses SupabaseSettings.
    from matagent.rag.client import SupabaseAPIError, SupabaseDataClient

    data_client = client or SupabaseDataClient(settings)
    try:
        payload = data_client.post(
            "/rest/v1/rpc/matagent_database_health",
            {},
        )
    except SupabaseAPIError as error:
        if error.error_code == "PGRST202":
            detail = "Run sql/001_supabase_rag.sql in the Supabase SQL Editor."
        else:
            detail = str(error)
        raise DatabaseHealthError(detail) from error

    row = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(row, dict):
        raise DatabaseHealthError("Supabase returned an unexpected response.")
    required = ("database_name", "postgres_version", "vector_version")
    if any(not row.get(field) for field in required):
        raise DatabaseHealthError(
            "The health RPC did not confirm PostgreSQL and pgvector."
        )
    return DatabaseHealth(**{field: str(row[field]) for field in required})
