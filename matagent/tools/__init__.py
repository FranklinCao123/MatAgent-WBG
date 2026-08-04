"""Scientific tool interfaces and local implementations."""

from matagent.tools.material_search import MockMaterialSearchTool
from matagent.tools.materials_project import MaterialsProjectSearchTool
from matagent.tools.registry import ToolExecutionError, ToolRegistry
from matagent.tools.schemas import (
    CandidateEvidenceArguments,
    MaterialSearchArguments,
    MaterialCandidate,
    ExcludedMaterial,
    MaterialSearchResult,
    ToolCallRequest,
    ToolExecutionResult,
)
from matagent.tools.scientific_evidence import ScientificEvidenceTool

__all__ = [
    "CandidateEvidenceArguments",
    "MaterialSearchArguments",
    "MaterialCandidate",
    "ExcludedMaterial",
    "MaterialSearchResult",
    "ScientificEvidenceTool",
    "MaterialsProjectSearchTool",
    "MockMaterialSearchTool",
    "ToolCallRequest",
    "ToolExecutionError",
    "ToolExecutionResult",
    "ToolRegistry",
]
