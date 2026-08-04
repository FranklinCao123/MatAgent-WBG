"""DeepSeek-backed structured requirement parser."""

import json
from typing import Any

from pydantic import ValidationError

from matagent.llm.base import (
    ReportSynthesisError,
    RequirementParsingError,
    create_deepseek_client,
)
from matagent.schemas import GroundedReport, ScreeningRequirements


SYSTEM_PROMPT = """You extract semiconductor material-screening requirements.
Return one JSON object and no prose. The JSON must follow the supplied schema.
Do not invent numerical thresholds that the user did not provide. If the user
requests wide-bandgap screening without a numerical threshold, use 2.0 eV and
record that choice in assumptions. Preserve uncertainty as an assumption.

JSON schema:
{schema}

Example JSON output:
{{
  "application": "high-temperature power electronics",
  "minimum_band_gap_ev": 3.0,
  "band_gap_operator": ">",
  "prefer_high_thermal_conductivity": true,
  "prefer_high_breakdown_field": true,
  "assumptions": []
}}
"""


class DeepSeekRequirementParser:
    """Call DeepSeek JSON Output and validate the result with Pydantic."""

    name = "deepseek"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._client = create_deepseek_client(
            api_key=api_key, base_url=base_url, client=client
        )

    def parse(self, user_query: str) -> ScreeningRequirements:
        schema = json.dumps(
            ScreeningRequirements.model_json_schema(),
            ensure_ascii=False,
        )
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(schema=schema),
            },
            {"role": "user", "content": user_query},
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=800,
                stream=False,
            )
        except Exception as error:
            raise RequirementParsingError(
                f"DeepSeek request failed ({type(error).__name__})."
            ) from error

        content = response.choices[0].message.content
        if not content:
            raise RequirementParsingError("DeepSeek returned empty content.")

        try:
            payload = json.loads(content)
            return ScreeningRequirements.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as error:
            raise RequirementParsingError(
                "DeepSeek returned invalid screening requirements."
            ) from error


REPORT_SYSTEM_PROMPT = """You write a concise scientific material-screening
assessment from validated tool results. Return one JSON object matching the
provided schema and no prose. Use only listed candidates and listed DOI values.
Do not invent properties, experiments, citations, or certainty. Distinguish
computed database fields from literature evidence. If evidence is absent or
indirect, use low or insufficient confidence and state that limitation.
Keep the executive summary under 100 words, each assessment under 80 words,
and provide at most five short caveats.

JSON schema:
{schema}
"""


class DeepSeekReportSynthesizer:
    """Generate a synthesis and reject invented candidates or DOI values."""

    name = "deepseek"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._client = create_deepseek_client(
            api_key=api_key,
            base_url=base_url,
            client=client,
        )

    def synthesize(
        self,
        *,
        user_query: str,
        requirements: ScreeningRequirements,
        ranked_candidates: list[dict[str, Any]],
        candidate_evidence: dict[str, list[dict[str, Any]]],
    ) -> GroundedReport:
        selected = _evidence_first_candidates(
            ranked_candidates,
            candidate_evidence,
            limit=3,
        )
        candidates = [_candidate_context(item) for item in selected]
        allowed_materials = {item["material"] for item in candidates}
        evidence = {
            material: [
                {
                    "doi": item.get("doi"),
                    "title": item.get("title"),
                    "content": str(item.get("content", ""))[:1200],
                    "similarity": item.get("similarity"),
                }
                for item in items
            ]
            for material, items in candidate_evidence.items()
            if material in allowed_materials
        }
        context = {
            "user_query": user_query,
            "validated_requirements": requirements.model_dump(mode="json"),
            "ranked_candidates": candidates,
            "candidate_evidence": evidence,
        }
        schema = json.dumps(GroundedReport.model_json_schema(), ensure_ascii=False)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": REPORT_SYSTEM_PROMPT.format(schema=schema),
                    },
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=2400,
                stream=False,
            )
            content = response.choices[0].message.content
            report = GroundedReport.model_validate(_load_json_object(content))
        except Exception as error:
            raise ReportSynthesisError(
                f"DeepSeek report synthesis failed ({type(error).__name__})."
            ) from error

        allowed_dois = {
            _normalize_doi(item["doi"])
            for items in evidence.values()
            for item in items
            if item.get("doi")
        }
        for assessment in report.candidate_assessments:
            if assessment.material not in allowed_materials:
                raise ReportSynthesisError(
                    "DeepSeek report referenced an unknown candidate."
                )
            if any(
                _normalize_doi(doi) not in allowed_dois
                for doi in assessment.evidence_dois
            ):
                raise ReportSynthesisError("DeepSeek report invented a DOI citation.")
        return report


def _evidence_first_candidates(
    ranked_candidates: list[dict[str, Any]],
    candidate_evidence: dict[str, list[dict[str, Any]]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Keep scientific rank order while giving retrieved evidence a voice."""

    supported_names = {
        material for material, evidence in candidate_evidence.items() if evidence
    }
    supported = [
        candidate
        for candidate in ranked_candidates
        if (candidate.get("name") or candidate.get("formula")) in supported_names
    ]
    unsupported = [
        candidate
        for candidate in ranked_candidates
        if (candidate.get("name") or candidate.get("formula")) not in supported_names
    ]
    return [*supported, *unsupported][:limit]


def _candidate_context(candidate: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "band_gap_ev",
        "is_stable",
        "energy_above_hull_ev_atom",
        "thermal_conductivity_w_mk",
        "breakdown_field_mv_cm",
        "demo_score",
        "data_source",
    )
    return {
        "material": candidate.get("name") or candidate.get("formula"),
        **{field: candidate.get(field) for field in fields},
    }


def _normalize_doi(doi: str) -> str:
    normalized = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized


def _load_json_object(content: str | None) -> dict[str, Any]:
    """Decode JSON mode output, tolerating only an outer Markdown fence."""

    text = (content or "").strip()
    if text.startswith("```") and text.endswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 : -3].strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON output must be an object.")
    return payload
