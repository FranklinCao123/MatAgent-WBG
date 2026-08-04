"""Supabase Data API configuration and a safe RAG health check."""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


class DatabaseConfigurationError(RuntimeError):
    """Raised when Supabase configuration is missing."""


class DatabaseHealthError(RuntimeError):
    """Raised when Supabase, PostgreSQL, or pgvector is unavailable."""


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
    return SupabaseSettings(url=url, secret_key=secret_key)


def check_database(
    settings: SupabaseSettings,
    *,
    opener: Callable[..., Any] = urlopen,
) -> DatabaseHealth:
    """Call the restricted health RPC through Supabase's HTTPS Data API."""

    request = Request(
        f"{settings.url}/rest/v1/rpc/matagent_database_health",
        data=b"{}",
        headers={
            "apikey": settings.secret_key,
            "Authorization": f"Bearer {settings.secret_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if error.code == 404:
            detail = "Run sql/001_supabase_rag.sql in the Supabase SQL Editor."
        else:
            detail = f"Supabase Data API returned HTTP {error.code}."
        raise DatabaseHealthError(detail) from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise DatabaseHealthError(
            f"Supabase HTTPS health check failed ({type(error).__name__})."
        ) from error

    row = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(row, dict):
        raise DatabaseHealthError("Supabase returned an unexpected response.")
    required = ("database_name", "postgres_version", "vector_version")
    if any(not row.get(field) for field in required):
        raise DatabaseHealthError(
            "The health RPC did not confirm PostgreSQL and pgvector."
        )
    return DatabaseHealth(**{field: str(row[field]) for field in required})
