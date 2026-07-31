"""Validated data contracts used by the agent workflow."""

from pydantic import BaseModel, ConfigDict, Field


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
        default=2.0,
        gt=0,
        description="Minimum accepted band gap in electronvolts.",
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
