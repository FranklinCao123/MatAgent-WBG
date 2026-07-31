"""Environment-based application configuration."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from matagent.llm import DeepSeekRequirementParser, RequirementParser
from matagent.llm.rule_based import RuleBasedRequirementParser


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


def build_requirement_parser(settings: LLMSettings) -> RequirementParser:
    """Build the selected parser while keeping provider logic out of the graph."""

    if settings.mode == "offline":
        return RuleBasedRequirementParser()
    if settings.mode == "deepseek":
        if not settings.api_key:
            raise ConfigurationError(
                "DeepSeek mode requires MATAGENT_LLM_API_KEY."
            )
        return DeepSeekRequirementParser(
            api_key=settings.api_key,
            model=settings.model,
            base_url=settings.base_url,
        )
    raise ConfigurationError(f"Unsupported LLM mode: {settings.mode}")
