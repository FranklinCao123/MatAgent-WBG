"""Validated data contracts used by the agent workflow."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ScreeningRequirements(BaseModel):
    """Structured requirements extracted from a user's screening request.

    The model is intentionally strict so malformed LLM output can be detected
    before it reaches scientific tools.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    application: str = Field(
        default="unspecified",
        min_length=1,
        description="Target device or scientific application.",
    )
    minimum_band_gap_ev: float = Field(
        gt=0,
        description="Minimum accepted band gap in electronvolts.",
    )
    band_gap_operator: Literal[">", ">="] = Field(
        description="Whether the band-gap threshold is strict or inclusive.",
    )
    prefer_high_thermal_conductivity: bool = Field(
        default=False,
        description="Whether thermal conductivity should be favored.",
    )
    prefer_high_breakdown_field: bool = Field(
        default=False,
        description="Whether breakdown field should be favored.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Explicit assumptions introduced during requirement parsing.",
    )


class RankingWeights(BaseModel):
    """Normalized weights for the current demonstration ranking properties."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    band_gap_ev: float = Field(ge=0, le=1)
    thermal_conductivity_w_mk: float = Field(ge=0, le=1)
    breakdown_field_mv_cm: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "RankingWeights":
        total = (
            self.band_gap_ev
            + self.thermal_conductivity_w_mk
            + self.breakdown_field_mv_cm
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Ranking weights must sum to 1.0.")
        return self


class CandidateFilterPolicy(BaseModel):
    """Auditable hard filters applied before candidate ranking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exclude_elements: list[str] = Field(default_factory=list)
    require_nonmetal: bool = False
    maximum_energy_above_hull_ev_atom: float | None = Field(default=None, ge=0)
    rationale: dict[str, str] = Field(default_factory=dict)


class RankingPlan(BaseModel):
    """Auditable plan connecting parsed requirements to ranking behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: Literal["weighted_mock_properties", "materials_project_stability"]
    weights: RankingWeights | None = None
    candidate_filters: CandidateFilterPolicy | None = None
    rationale: dict[str, str]
    inferred_requirements: list[str] = Field(default_factory=list)
