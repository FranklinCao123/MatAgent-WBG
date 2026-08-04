"""Offline and DeepSeek selectors for OpenAI-compatible tool calling."""

import json
from typing import Any, Protocol

from matagent.llm.base import create_deepseek_client
from matagent.schemas import RankingPlan, ScreeningRequirements
from matagent.tools import ToolCallRequest


class ToolSelectionError(RuntimeError):
    """Raised when a selector cannot produce usable tool calls."""


class ToolSelector(Protocol):
    """Contract shared by deterministic and LLM-backed tool selectors."""

    name: str

    def select(
        self,
        *,
        user_query: str,
        requirements: ScreeningRequirements,
        ranking_plan: RankingPlan,
        tool_specs: list[dict[str, Any]],
    ) -> list[ToolCallRequest]:
        """Return zero or more proposed calls without executing them."""


class RuleBasedToolSelector:
    """Select the mock search tool deterministically for offline operation."""

    name = "rule_based"

    def select(
        self,
        *,
        user_query: str,
        requirements: ScreeningRequirements,
        ranking_plan: RankingPlan,
        tool_specs: list[dict[str, Any]],
    ) -> list[ToolCallRequest]:
        del user_query, ranking_plan, tool_specs
        return [
            ToolCallRequest(
                id="offline-search-1",
                name="search_materials",
                arguments={
                    "band_gap_threshold_ev": requirements.minimum_band_gap_ev,
                    "band_gap_operator": requirements.band_gap_operator,
                },
            )
        ]


class DeepSeekToolSelector:
    """Ask DeepSeek to select from an allow-listed set of tool schemas."""

    name = "deepseek"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._client = create_deepseek_client(
            api_key=api_key, base_url=base_url, client=client
        )

    def select(
        self,
        *,
        user_query: str,
        requirements: ScreeningRequirements,
        ranking_plan: RankingPlan,
        tool_specs: list[dict[str, Any]],
    ) -> list[ToolCallRequest]:
        context = {
            "user_query": user_query,
            "validated_requirements": requirements.model_dump(mode="json"),
        }
        del ranking_plan
        messages = [
            {
                "role": "system",
                "content": (
                    "You route a semiconductor-screening task to available read-only "
                    "tools. Use validated requirements exactly, preserve strict versus "
                    "inclusive comparison operators, and request only tools needed to "
                    "obtain candidate data. Do not invent tool names or arguments."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False),
            },
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tool_specs,
                tool_choice="auto",
                temperature=0,
                max_tokens=500,
                stream=False,
            )
        except Exception as error:
            raise ToolSelectionError(
                f"DeepSeek tool selection failed ({type(error).__name__})."
            ) from error

        tool_calls = response.choices[0].message.tool_calls or []
        selected = []
        for call in tool_calls:
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError as error:
                raise ToolSelectionError(
                    "DeepSeek returned invalid JSON tool arguments."
                ) from error
            selected.append(
                ToolCallRequest(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                )
            )
        return selected
