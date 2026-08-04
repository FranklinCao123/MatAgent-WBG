"""Validated schemas for LLM-requested tool calls and results."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MaterialSearchArguments(BaseModel):
    """Arguments accepted by the material-search tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    band_gap_threshold_ev: float = Field(gt=0)
    band_gap_operator: Literal[">", ">="]
    exclude_elements: list[str] = Field(default_factory=list)
    require_nonmetal: bool = False
    maximum_energy_above_hull_ev_atom: float | None = Field(default=None, ge=0)


class CandidateEvidenceArguments(BaseModel):
    """Arguments for evidence retrieval scoped to ranked candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    user_query: str = Field(min_length=1)
    candidates: list[str] = Field(min_length=1, max_length=10)
    evidence_per_candidate: int = Field(default=2, ge=1, le=5)
    minimum_similarity: float = Field(default=0.0, ge=-1.0, le=1.0)


class MaterialCandidate(BaseModel):
    """Normalized material record shared by local and remote search backends."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    formula: str = Field(min_length=1)
    elements: list[str] = Field(default_factory=list)
    band_gap_ev: float = Field(ge=0)
    data_source: Literal["mock", "materials_project"]
    material_id: str | None = None
    is_stable: bool | None = None
    is_metal: bool | None = None
    energy_above_hull_ev_atom: float | None = None
    formation_energy_per_atom_ev: float | None = None
    thermal_conductivity_w_mk: float | None = Field(default=None, ge=0)
    breakdown_field_mv_cm: float | None = Field(default=None, ge=0)


class ExcludedMaterial(BaseModel):
    """A rejected API record and the explicit local reasons for rejection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    material_id: str | None = None
    formula: str
    reasons: list[str] = Field(min_length=1)


class MaterialSearchResult(BaseModel):
    """Candidates plus auditable filtering diagnostics from a search backend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["mock", "materials_project"]
    retrieved_count: int = Field(ge=0)
    candidates: list[MaterialCandidate]
    excluded: list[ExcludedMaterial] = Field(default_factory=list)
    applied_filters: dict[str, Any] = Field(default_factory=dict)


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
