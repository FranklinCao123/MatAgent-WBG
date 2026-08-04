"""Deterministic requirement parser used for offline development and tests."""

import re

from matagent.schemas import ScreeningRequirements


_BAND_GAP_PATTERN = re.compile(
    r"(?:band\s*gap|bandgap|带隙|禁带)(?:[^\d]{0,20})(\d+(?:\.\d+)?)\s*(?:eV|电子伏特)",
    re.IGNORECASE,
)


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
        band_gap_match = _BAND_GAP_PATTERN.search(user_query)
        minimum_band_gap_ev = (
            float(band_gap_match.group(1)) if band_gap_match else 2.0
        )
        strict_threshold = any(
            term in query
            for term in (">", "above", "greater than", "more than", "大于", "高于")
        ) and not any(term in query for term in (">=", "at least", "不少于", "不低于"))
        band_gap_operator = ">" if strict_threshold else ">="

        assumptions = [
            "Cost, manufacturability, and supply-chain constraints are not yet evaluated."
        ]
        if band_gap_match is None:
            assumptions.insert(
                0,
                "No numerical band-gap constraint was found; the offline parser uses "
                "a demonstration threshold of 2.0 eV.",
            )

        return ScreeningRequirements(
            application=(
                "power electronics" if is_power_electronics else "unspecified"
            ),
            minimum_band_gap_ev=minimum_band_gap_ev,
            band_gap_operator=band_gap_operator,
            prefer_high_thermal_conductivity=is_high_temperature,
            prefer_high_breakdown_field=is_power_electronics,
            assumptions=assumptions,
        )
