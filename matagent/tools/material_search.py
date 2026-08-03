"""Material-search tool implementations."""

import json
from pathlib import Path
from typing import Any

from matagent.tools.schemas import (
    ExcludedMaterial,
    MaterialCandidate,
    MaterialSearchArguments,
    MaterialSearchResult,
)


class MockMaterialSearchTool:
    """Search a tiny JSON dataset through a stable tool interface.

    A future Materials Project implementation can expose the same ``search``
    method, allowing the graph to switch backends without changing its nodes.
    """

    name = "mock_material_search"

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def search(self, arguments: MaterialSearchArguments) -> dict[str, Any]:
        with self.data_path.open("r", encoding="utf-8") as file:
            raw_materials: list[dict[str, Any]] = json.load(file)

        materials = [
            MaterialCandidate.model_validate(
                {
                    "formula": material.get("formula", material["name"]),
                    **material,
                    "data_source": "mock",
                }
            ).model_dump(mode="json")
            for material in raw_materials
        ]

        threshold = arguments.band_gap_threshold_ev
        candidates = []
        excluded = []
        for material in materials:
            accepted = (
                material["band_gap_ev"] > threshold
                if arguments.band_gap_operator == ">"
                else material["band_gap_ev"] >= threshold
            )
            if accepted:
                candidates.append(material)
            else:
                excluded.append(
                    ExcludedMaterial(
                        material_id=material.get("material_id"),
                        formula=material["formula"],
                        reasons=[
                            "band gap does not satisfy "
                            f"{arguments.band_gap_operator} {threshold} eV"
                        ],
                    )
                )

        return MaterialSearchResult(
            source="mock",
            retrieved_count=len(materials),
            candidates=candidates,
            excluded=excluded,
            applied_filters={
                "band_gap_operator": arguments.band_gap_operator,
                "band_gap_threshold_ev": threshold,
            },
        ).model_dump(mode="json")
