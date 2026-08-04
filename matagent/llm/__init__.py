"""Requirement-parser implementations for offline and LLM modes."""

from matagent.llm.base import (
    ReportSynthesisError,
    ReportSynthesizer,
    RequirementParser,
    RequirementParsingError,
)
from matagent.llm.deepseek import DeepSeekReportSynthesizer, DeepSeekRequirementParser
from matagent.llm.rule_based import RuleBasedRequirementParser
from matagent.llm.tool_selector import (
    DeepSeekToolSelector,
    RuleBasedToolSelector,
    ToolSelectionError,
    ToolSelector,
)

__all__ = [
    "DeepSeekRequirementParser",
    "DeepSeekReportSynthesizer",
    "DeepSeekToolSelector",
    "ReportSynthesisError",
    "ReportSynthesizer",
    "RequirementParser",
    "RequirementParsingError",
    "RuleBasedRequirementParser",
    "RuleBasedToolSelector",
    "ToolSelectionError",
    "ToolSelector",
]
