"""Environment-based application configuration."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from matagent.llm import (
    DeepSeekRequirementParser,
    DeepSeekToolSelector,
    RequirementParser,
    RuleBasedToolSelector,
    ToolSelector,
)
from matagent.llm.base import create_deepseek_client
from matagent.llm.rule_based import RuleBasedRequirementParser
from matagent.tools import MaterialsProjectSearchTool


class ConfigurationError(RuntimeError):
    """Raised when a requested runtime mode is not configured correctly."""


@dataclass(frozen=True)
class LLMSettings:
    """LLM settings loaded without ever writing the API key to graph state."""

    mode: str = "offline"
    api_key: str | None = field(default=None, repr=False)
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"

    @classmethod
    def from_environment(cls, mode: str | None = None) -> "LLMSettings":
        load_dotenv()
        return cls(
            mode=mode or os.getenv("MATAGENT_LLM_MODE", "offline"),
            api_key=os.getenv("MATAGENT_LLM_API_KEY"),
            model=os.getenv("MATAGENT_LLM_MODEL", "deepseek-v4-flash"),
            base_url=os.getenv(
                "MATAGENT_LLM_BASE_URL",
                "https://api.deepseek.com",
            ),
        )


@dataclass(frozen=True)
class MaterialDataSettings:
    """Material-search settings with the remote API key hidden from repr output."""

    backend: str = "mock"
    api_key: str | None = field(default=None, repr=False)
    base_url: str = "https://api.materialsproject.org"
    fetch_limit: int = 100
    timeout_seconds: float = 20.0

    @classmethod
    def from_environment(
        cls,
        *,
        backend: str | None = None,
        fetch_limit: int | None = None,
    ) -> "MaterialDataSettings":
        load_dotenv()
        configured_limit = fetch_limit or int(
            os.getenv("MATAGENT_MP_FETCH_LIMIT", "100")
        )
        return cls(
            backend=backend or os.getenv("MATAGENT_MATERIAL_BACKEND", "mock"),
            api_key=os.getenv("MATAGENT_MP_API_KEY"),
            base_url=os.getenv(
                "MATAGENT_MP_BASE_URL",
                "https://api.materialsproject.org",
            ),
            fetch_limit=configured_limit,
            timeout_seconds=float(os.getenv("MATAGENT_MP_TIMEOUT_SECONDS", "20")),
        )


def build_llm_components(
    settings: LLMSettings,
) -> tuple[RequirementParser, ToolSelector]:
    """Build the matching requirement parser and tool selector once."""

    if settings.mode == "offline":
        return RuleBasedRequirementParser(), RuleBasedToolSelector()
    if settings.mode != "deepseek":
        raise ConfigurationError(f"Unsupported LLM mode: {settings.mode}")
    if not settings.api_key:
        raise ConfigurationError("DeepSeek mode requires MATAGENT_LLM_API_KEY.")

    client = create_deepseek_client(
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
    component_settings = {
        "api_key": settings.api_key,
        "model": settings.model,
        "base_url": settings.base_url,
        "client": client,
    }
    return (
        DeepSeekRequirementParser(**component_settings),
        DeepSeekToolSelector(**component_settings),
    )


def build_material_search_tool(settings: MaterialDataSettings):
    """Build a remote search tool, or return None for the graph's mock default."""

    if settings.backend == "mock":
        return None
    if settings.backend == "materials-project":
        if not settings.api_key:
            raise ConfigurationError(
                "Materials Project mode requires MATAGENT_MP_API_KEY."
            )
        try:
            return MaterialsProjectSearchTool(
                api_key=settings.api_key,
                base_url=settings.base_url,
                max_results=settings.fetch_limit,
                timeout_seconds=settings.timeout_seconds,
            )
        except ValueError as error:
            raise ConfigurationError(str(error)) from error
    raise ConfigurationError(
        f"Unsupported material backend: {settings.backend}"
    )
