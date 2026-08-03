"""Validated schemas for LLM-requested tool calls and results."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MaterialSearchArguments(BaseModel):
    """Arguments accepted by the material-search tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    band_gap_threshold_ev: float = Field(gt=0)
    band_gap_operator: Literal[">", ">="]


class MaterialCandidate(BaseModel):
    """Normalized material record shared by local and remote search backends."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    formula: str = Field(min_length=1)
    band_gap_ev: float = Field(ge=0)
    data_source: Literal["mock", "materials_project"]
    data_status: str = Field(min_length=1)
    material_id: str | None = None
    is_stable: bool | None = None
    energy_above_hull_ev_atom: float | None = None
    formation_energy_per_atom_ev: float | None = None
    thermal_conductivity_w_mk: float | None = Field(default=None, ge=0)
    breakdown_field_mv_cm: float | None = Field(default=None, ge=0)


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
