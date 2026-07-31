"""Material-search tool implementations."""

import json
from pathlib import Path
from typing import Any

from matagent.tools.schemas import MaterialSearchArguments


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
            materials: list[dict[str, Any]] = json.load(file)

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
