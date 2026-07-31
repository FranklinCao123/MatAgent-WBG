"""Allow-listed tool registration, schema export, and execution."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from matagent.tools.schemas import ToolCallRequest, ToolExecutionResult


class ToolExecutionError(RuntimeError):
    """Raised when a requested tool cannot be safely executed."""


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    arguments_model: type[BaseModel]
    handler: Callable[[BaseModel], Any]


class ToolRegistry:
    """Execute only explicitly registered tools with validated arguments."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        arguments_model: type[BaseModel],
        handler: Callable[[BaseModel], Any],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            arguments_model=arguments_model,
            handler=handler,
        )

    def tool_specs(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function schemas for all allowed tools."""

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.arguments_model.model_json_schema(),
                },
            }
            for tool in self._tools.values()
        ]

    def execute(self, call: ToolCallRequest) -> ToolExecutionResult:
        tool = self._tools.get(call.name)
        if tool is None:
            raise ToolExecutionError(f"Tool is not registered: {call.name}")

        try:
            arguments = tool.arguments_model.model_validate(call.arguments)
        except ValidationError as error:
            raise ToolExecutionError(
                f"Invalid arguments for tool {call.name}."
            ) from error

        try:
            output = tool.handler(arguments)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ToolExecutionError(
                f"Tool {call.name} failed ({type(error).__name__}): {error}"
            ) from error

        return ToolExecutionResult(
            call_id=call.id,
            name=call.name,
            output=output,
        )
