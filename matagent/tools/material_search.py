"""Material-search tool implementations."""

import json
from pathlib import Path
from typing import Any

from matagent.tools.schemas import MaterialCandidate, MaterialSearchArguments


class MockMaterialSearchTool:
    """Search a tiny JSON dataset through a stable tool interface.

    A future Materials Project implementation can expose the same ``search``
    method, allowing the graph to switch backends without changing its nodes.
    """

    name = "mock_material_search"

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def search(self, arguments: MaterialSearchArguments) -> list[dict[str, Any]]:
        with self.data_path.open("r", encoding="utf-8") as file:
            raw_materials: list[dict[str, Any]] = json.load(file)

        materials = [
            MaterialCandidate.model_validate(
                {
                    "formula": material.get("formula", material["name"]),
                    "data_status": material.get("data_status", "illustrative_mock"),
                    **material,
                    "data_source": "mock",
                }
            ).model_dump(mode="json")
            for material in raw_materials
        ]

        threshold = arguments.band_gap_threshold_ev
        if arguments.band_gap_operator == ">":
            return [
                material
                for material in materials
                if material["band_gap_ev"] > threshold
            ]

        return [
            material
            for material in materials
            if material["band_gap_ev"] >= threshold
        ]
