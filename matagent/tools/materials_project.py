"""Lightweight Materials Project REST search implementation."""

import json
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import ValidationError

from matagent.tools.schemas import MaterialCandidate, MaterialSearchArguments


SUMMARY_FIELDS = (
    "material_id",
    "formula_pretty",
    "band_gap",
    "is_stable",
    "energy_above_hull",
    "formation_energy_per_atom",
)


class MaterialsProjectSearchTool:
    """Search a small Materials Project summary result set over HTTPS."""

    name = "materials_project_search"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.materialsproject.org",
        max_results: int = 20,
        timeout_seconds: float = 20.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Materials Project API key must not be empty.")
        if not 1 <= max_results <= 100:
            raise ValueError("Materials Project max_results must be between 1 and 100.")
        if timeout_seconds <= 0:
            raise ValueError("Materials Project timeout must be positive.")

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_results = max_results
        self._timeout_seconds = timeout_seconds
        self._opener = opener

    def search(self, arguments: MaterialSearchArguments) -> list[dict[str, Any]]:
        query = urlencode(
            {
                "band_gap_min": arguments.band_gap_threshold_ev,
                "_fields": ",".join(SUMMARY_FIELDS),
                "_limit": self._max_results,
            }
        )
        request = Request(
            f"{self._base_url}/materials/summary/?{query}",
            headers={
                "Accept": "application/json",
                "X-API-KEY": self._api_key,
                "User-Agent": (
                    "MatAgent-WBG/0.1 "
                    "(+https://github.com/FranklinCao123/MatAgent-WBG)"
                ),
            },
            method="GET",
        )

        try:
            with self._opener(request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code in (401, 403):
                raise ValueError(
                    "Materials Project rejected the API key or access request."
                ) from error
            raise OSError(f"Materials Project HTTP request failed ({error.code}).") from error
        except (URLError, TimeoutError) as error:
            raise OSError("Materials Project could not be reached.") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Materials Project returned invalid JSON.") from error

        documents = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(documents, list):
            raise ValueError("Materials Project response does not contain a data list.")

        candidates = []
        for document in documents:
            try:
                candidate = MaterialCandidate(
                    name=document["formula_pretty"],
                    formula=document["formula_pretty"],
                    material_id=str(document["material_id"]),
                    band_gap_ev=document["band_gap"],
                    is_stable=document.get("is_stable"),
                    energy_above_hull_ev_atom=document.get("energy_above_hull"),
                    formation_energy_per_atom_ev=document.get(
                        "formation_energy_per_atom"
                    ),
                    thermal_conductivity_w_mk=None,
                    breakdown_field_mv_cm=None,
                    data_source="materials_project",
                    data_status="computed_materials_project",
                )
            except (KeyError, TypeError, ValidationError) as error:
                raise ValueError(
                    "Materials Project returned an invalid material record."
                ) from error

            threshold = arguments.band_gap_threshold_ev
            if arguments.band_gap_operator == ">" and candidate.band_gap_ev <= threshold:
                continue
            if arguments.band_gap_operator == ">=" and candidate.band_gap_ev < threshold:
                continue
            candidates.append(candidate.model_dump(mode="json"))

        return candidates
