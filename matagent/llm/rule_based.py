"""Deterministic requirement parser used for offline development and tests."""

from matagent.schemas import ScreeningRequirements


class RuleBasedRequirementParser:
    """Parse a small set of Chinese and English keywords without an API."""

    name = "rule_based"

    def parse(self, user_query: str) -> ScreeningRequirements:
        query = user_query.lower()
        is_power_electronics = any(
            term in query
            for term in ("power", "功率", "power electronics", "电力电子")
        )
        is_high_temperature = any(
            term in query
            for term in ("high temperature", "high-temperature", "高温")
        )

        return ScreeningRequirements(
            application=(
                "power electronics" if is_power_electronics else "unspecified"
            ),
            minimum_band_gap_ev=2.0,
            band_gap_operator=">=",
            prefer_high_thermal_conductivity=is_high_temperature,
            prefer_high_breakdown_field=is_power_electronics,
            assumptions=[
                "Wide-bandgap screening uses a demonstration threshold of 2.0 eV.",
                "Cost, manufacturability, and supply-chain constraints are not yet evaluated.",
            ],
        )
