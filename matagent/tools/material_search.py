"""Material-search tool implementations."""

import json
from pathlib import Path
from typing import Any


class MockMaterialSearchTool:
    """Search a tiny JSON dataset through a stable tool interface.

    A future Materials Project implementation can expose the same ``search``
    method, allowing the graph to switch backends without changing its nodes.
    """

    name = "mock_material_search"

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    def search(self, minimum_band_gap_ev: float) -> list[dict[str, Any]]:
        with self.data_path.open("r", encoding="utf-8") as file:
            materials: list[dict[str, Any]] = json.load(file)

        return [
            material
            for material in materials
            if material["band_gap_ev"] >= minimum_band_gap_ev
        ]
