"""Orchestrate baselines without invoking any scenario generator or executor."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .baselines import generate_rule_event
from .models.vlm_wrapper import GenerationSettings, VLMWrapper, read_bev
from .parser import decode_response, parse_response
from .prompts.director_prompt import PROMPT_VERSION, build_director_prompt
from .schemas.scenario_schema import EVENT_TYPES, Scenario, SchemaError, check_feasibility


@dataclass(frozen=True)
class DirectorResult:
    """Separate the executor event from research metadata and raw response text."""

    prediction: dict[str, Any] | None
    raw_response: str
    report: dict[str, Any]

    @property
    def success(self) -> bool:
        """Unknown geometry may be exported; a known failure may never be exported."""
        return self.prediction is not None


class Director:
    """Common path for rule, text-only, and vision-language baseline evaluation."""

    def __init__(self, baseline: int = 0, model: VLMWrapper | None = None,
                 settings: GenerationSettings | None = None,
                 rule_event: str = "auto") -> None:
        """Validate configuration independently of any optional model imports."""
        if baseline not in (0, 1, 2):
            raise ValueError("baseline must be 0, 1, or 2")
        if baseline == 0 and model is not None:
            raise ValueError("Rule baseline does not use a model")
        if baseline != 0 and model is None:
            raise ValueError("Text/vision baselines require an explicit model backend")
        if rule_event not in ("auto", *EVENT_TYPES):
            raise ValueError("Unknown rule event")
        self.baseline, self.model = baseline, model
        self.settings = settings or GenerationSettings()
        self.rule_event = rule_event

    def run(self, scenario: Scenario | dict[str, Any],
            bev_image: str | Path | None = None) -> DirectorResult:
        """Generate once; preserve invalid output without silently retrying/falling back.

        Transport/setup errors raise directly. Response validation failures return
        diagnostics and raw text, with prediction=None. This keeps failed samples
        in evaluation denominators instead of hiding them with a rule fallback.
        """
        if not isinstance(scenario, Scenario):
            scenario = Scenario.from_dict(scenario)
        if self.baseline == 2 and bev_image is None:
            raise ValueError("Baseline 2 requires --bev / bev_image")
        if self.baseline in (0, 1) and bev_image is not None:
            raise ValueError("Only baseline 2 consumes a BEV image")
        image_path = None
        if bev_image is not None:
            if not self.model.supports_vision:
                raise ValueError("Selected model backend does not support vision")
            image_path, _, _ = read_bev(bev_image)
        started = time.perf_counter()
        raw = (generate_rule_event(scenario, seed=self.settings.seed, event_type=self.rule_event)
               if self.baseline == 0 else self.model.generate(
                   build_director_prompt(scenario, has_bev=image_path is not None),
                   image_path, self.settings))
        report: dict[str, Any] = {
            "baseline": self.baseline, "prompt_version": PROMPT_VERSION,
            "model": ({"backend": "rule", "model_id": "rules-v1", "is_mock": False}
                      if self.model is None else self.model.metadata),
            "generation": asdict(self.settings), "bev_used": image_path is not None,
            "strict_json_valid": False, "json_object_valid": False,
            "schema_valid": False, "feasibility": {"status": "not_checked"},
            "success": False, "error": None,
        }
        prediction = None
        try:
            # Decode first to distinguish syntax validity from schema validity.
            _, report["strict_json_valid"] = decode_response(raw)
            report["json_object_valid"] = True
            parsed = parse_response(raw)
            report.update(schema_valid=True, recovered=parsed.recovered,
                          legacy_normalized=parsed.legacy_normalized)
            report.update(event_type=parsed.event["event"], actor=parsed.event["actor"])
            report["feasibility"] = check_feasibility(parsed.event, scenario)
            if report["feasibility"]["status"] == "fail":
                report["error"] = "; ".join(report["feasibility"]["errors"])
            else:
                prediction = parsed.event
                report["success"] = True
        except SchemaError as exc:
            report["error"] = str(exc)
        report["latency_s"] = round(time.perf_counter() - started, 6)
        return DirectorResult(prediction, raw, report)
