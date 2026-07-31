"""Validated schemas for LLM-requested tool calls and results."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MaterialSearchArguments(BaseModel):
    """Arguments accepted by the material-search tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    band_gap_threshold_ev: float = Field(gt=0)
    band_gap_operator: Literal[">", ">="]


class ToolCallRequest(BaseModel):
    """A tool invocation proposed by an LLM or offline selector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class ToolExecutionResult(BaseModel):
    """Serializable result produced by the controlled tool executor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str
    name: str
    output: Any
