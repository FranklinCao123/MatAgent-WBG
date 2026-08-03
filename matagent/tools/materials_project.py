"""Lightweight Materials Project REST search implementation."""

import json
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import ValidationError

from matagent.tools.schemas import (
    ExcludedMaterial,
    MaterialCandidate,
    MaterialSearchArguments,
    MaterialSearchResult,
)


SUMMARY_FIELDS = (
    "material_id",
    "formula_pretty",
    "elements",
    "band_gap",
    "is_stable",
    "is_metal",
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
        max_results: int = 100,
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

    def search(self, arguments: MaterialSearchArguments) -> dict[str, Any]:
        query_parameters: dict[str, Any] = {
            "band_gap_min": arguments.band_gap_threshold_ev,
            "_fields": ",".join(SUMMARY_FIELDS),
            "_limit": self._max_results,
            "_sort_fields": "energy_above_hull",
        }
        server_excluded_elements = self._server_excluded_elements(
            arguments.exclude_elements
        )
        if server_excluded_elements:
            query_parameters["exclude_elements"] = ",".join(
                server_excluded_elements
            )
        if arguments.require_nonmetal:
            query_parameters["is_metal"] = "false"
        if arguments.maximum_energy_above_hull_ev_atom is not None:
            query_parameters["energy_above_hull_max"] = (
                arguments.maximum_energy_above_hull_ev_atom
            )
        query = urlencode(query_parameters)
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
            detail = self._safe_http_error_detail(error)
            suffix = f": {detail}" if detail else "."
            raise OSError(
                f"Materials Project HTTP request failed ({error.code}){suffix}"
            ) from error
        except (URLError, TimeoutError) as error:
            raise OSError("Materials Project could not be reached.") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Materials Project returned invalid JSON.") from error

        documents = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(documents, list):
            raise ValueError("Materials Project response does not contain a data list.")

        candidates = []
        excluded = []
        for document in documents:
            try:
                candidate = MaterialCandidate(
                    name=document["formula_pretty"],
                    formula=document["formula_pretty"],
                    elements=[str(element) for element in document.get("elements", [])],
                    material_id=str(document["material_id"]),
                    band_gap_ev=document["band_gap"],
                    is_stable=document.get("is_stable"),
                    is_metal=document.get("is_metal"),
                    energy_above_hull_ev_atom=document.get("energy_above_hull"),
                    formation_energy_per_atom_ev=document.get(
                        "formation_energy_per_atom"
                    ),
                    thermal_conductivity_w_mk=None,
                    breakdown_field_mv_cm=None,
                    data_source="materials_project",
                )
            except (KeyError, TypeError, ValidationError) as error:
                raise ValueError(
                    "Materials Project returned an invalid material record."
                ) from error

            reasons = self._local_exclusion_reasons(candidate, arguments)
            if reasons:
                excluded.append(
                    ExcludedMaterial(
                        material_id=candidate.material_id,
                        formula=candidate.formula,
                        reasons=reasons,
                    )
                )
                continue
            candidates.append(candidate.model_dump(mode="json"))

        return MaterialSearchResult(
            source="materials_project",
            retrieved_count=len(documents),
            candidates=candidates,
            excluded=excluded,
            applied_filters={
                "band_gap_operator": arguments.band_gap_operator,
                "band_gap_threshold_ev": arguments.band_gap_threshold_ev,
                "exclude_elements": arguments.exclude_elements,
                "server_exclude_elements": server_excluded_elements,
                "require_nonmetal": arguments.require_nonmetal,
                "maximum_energy_above_hull_ev_atom": (
                    arguments.maximum_energy_above_hull_ev_atom
                ),
                "fetch_limit": self._max_results,
            },
        ).model_dump(mode="json")

    @staticmethod
    def _server_excluded_elements(elements: list[str]) -> list[str]:
        """Fit an exclusion subset within the API's 60-character query limit."""

        selected = []
        length = 0
        for element in elements:
            additional_length = len(element) + (1 if selected else 0)
            if length + additional_length > 60:
                break
            selected.append(element)
            length += additional_length
        return selected

    @staticmethod
    def _safe_http_error_detail(error: HTTPError) -> str:
        """Extract a bounded server validation message without exposing headers."""

        try:
            payload = json.loads(error.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return ""
        if not isinstance(payload, dict):
            return ""
        detail = payload.get("detail") or payload.get("message") or payload.get("error")
        if isinstance(detail, (dict, list)):
            detail = json.dumps(detail, ensure_ascii=False)
        return str(detail)[:500] if detail else ""

    @staticmethod
    def _local_exclusion_reasons(
        candidate: MaterialCandidate,
        arguments: MaterialSearchArguments,
    ) -> list[str]:
        """Repeat important server filters locally for auditable exact semantics."""

        reasons = []
        threshold = arguments.band_gap_threshold_ev
        if arguments.band_gap_operator == ">" and candidate.band_gap_ev <= threshold:
            reasons.append(f"band gap does not satisfy > {threshold} eV")
        if arguments.band_gap_operator == ">=" and candidate.band_gap_ev < threshold:
            reasons.append(f"band gap does not satisfy >= {threshold} eV")

        excluded_elements = sorted(
            set(candidate.elements).intersection(arguments.exclude_elements)
        )
        if excluded_elements:
            reasons.append(
                "contains excluded element(s): " + ", ".join(excluded_elements)
            )

        if arguments.require_nonmetal:
            if candidate.is_metal is True:
                reasons.append("Materials Project classifies the entry as metallic")
            elif candidate.is_metal is None:
                reasons.append("metallicity classification is unavailable")

        maximum_hull = arguments.maximum_energy_above_hull_ev_atom
        if maximum_hull is not None:
            if candidate.energy_above_hull_ev_atom is None:
                reasons.append("energy above hull is unavailable")
            elif candidate.energy_above_hull_ev_atom > maximum_hull:
                reasons.append(
                    "energy above hull exceeds "
                    f"{maximum_hull} eV/atom"
                )

        return reasons
