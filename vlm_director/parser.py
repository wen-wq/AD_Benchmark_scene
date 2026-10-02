"""Parse raw model text and normalize the preliminary five-field output contract."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from .schemas.scenario_schema import SchemaError, strict_json_loads, validate_event


@dataclass(frozen=True)
class ParseResult:
    """Keep strict JSON success separate from a recovered fenced JSON response."""

    event: dict[str, Any]
    strict_json_valid: bool
    recovered: bool
    legacy_normalized: bool


def decode_response(text: str) -> tuple[dict[str, Any], bool]:
    """Accept exactly one object, optionally enclosed in one Markdown code fence.

    No eval, regex brace extraction, or replacement of quotes is used. Prose,
    arrays, multiple objects, duplicate keys, and non-finite numbers are rejected.
    """
    if not isinstance(text, str) or not text.strip():
        raise SchemaError("Model returned an empty/non-text response")
    candidate = text.strip()
    strict = True
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n\s*```", candidate,
                         flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate, strict = fence.group(1).strip(), False
    obj = strict_json_loads(candidate)
    if not isinstance(obj, dict):
        raise SchemaError("Model response must be one JSON object")
    return obj, strict


def parse_response(text: str) -> ParseResult:
    """Normalize aliases only when trigger and behavior are already structured.

    The initial corner_case_type/target_actor format is supported. Freeform trigger
    or behavior strings are rejected because guessing their semantics is unsafe
    for an executor. A JSON-string parameters object may be decoded losslessly.
    """
    obj, strict = decode_response(text)
    legacy = "corner_case_type" in obj or "target_actor" in obj
    if legacy:
        if "event" in obj or "actor" in obj:
            raise SchemaError("Cannot mix legacy and canonical event keys")
        obj = copy.deepcopy(obj)
        obj["event"] = obj.pop("corner_case_type", None)
        obj["actor"] = obj.pop("target_actor", None)
        params = obj.get("parameters", {})
        if isinstance(params, str):
            obj["parameters"] = strict_json_loads(params)
    return ParseResult(validate_event(obj), strict, not strict, legacy)
