"""DeepSeek-backed structured requirement parser."""

import json
from typing import Any

from pydantic import ValidationError

from matagent.llm.base import RequirementParsingError
from matagent.schemas import ScreeningRequirements


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
        if not api_key:
            raise ValueError("DeepSeek API key must not be empty.")

        self.model = model
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "DeepSeek mode requires the 'openai' Python package."
                ) from error
            client = OpenAI(api_key=api_key, base_url=base_url)
        self._client = client

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
