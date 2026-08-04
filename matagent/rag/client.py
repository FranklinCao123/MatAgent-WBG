"""Small server-side client for Supabase's HTTPS Data API."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from matagent.rag.database import SupabaseSettings


class SupabaseAPIError(RuntimeError):
    """Safe Data API error that never includes credentials or response bodies."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class SupabaseDataClient:
    """POST JSON to project-relative Data API paths with a server-only key."""

    def __init__(
        self,
        settings: SupabaseSettings,
        *,
        opener: Callable[..., Any] = urlopen,
        timeout_seconds: float = 15,
    ) -> None:
        self._settings = settings
        self._opener = opener
        self._timeout_seconds = timeout_seconds

    def post(self, path: str, payload: Mapping[str, Any]) -> Any:
        if not path.startswith("/rest/v1/") or "://" in path:
            raise ValueError("Supabase API path must be project-relative.")

        request = Request(
            f"{self._settings.url}{path}",
            data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={
                "apikey": self._settings.secret_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            error_code = _safe_error_code(error)
            label = f", {error_code}" if error_code else ""
            raise SupabaseAPIError(
                f"Supabase Data API request failed (HTTP {error.code}{label}).",
                status_code=error.code,
                error_code=error_code,
            ) from error
        except (URLError, TimeoutError) as error:
            raise SupabaseAPIError(
                f"Supabase HTTPS request failed ({type(error).__name__})."
            ) from error
        except json.JSONDecodeError as error:
            raise SupabaseAPIError("Supabase returned invalid JSON.") from error


def _safe_error_code(error: HTTPError) -> str | None:
    """Read only a machine error code, never a potentially sensitive message."""

    try:
        payload = json.loads(error.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    code = payload.get("code") if isinstance(payload, dict) else None
    return str(code) if code else None
