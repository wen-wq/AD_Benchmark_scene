"""Baseline 0: deterministic seeded rules over existing snapshot actors."""

import json
import random

from .schemas.scenario_schema import Scenario, SchemaError, check_feasibility, validate_event


def generate_rule_event(scenario: Scenario, *, seed: int = 0,
                        event_type: str = "auto") -> str:
    """Sample among eligible cut-ins, lead brakes, and pedestrian crossings.

    Every candidate goes through the same schema and feasibility checks as a VLM
    event. Unknown geometry is allowed and reported; known failures are discarded.
    If no candidate exists, fail explicitly instead of spawning an invented actor.
    """
    candidates = []
    for actor in scenario.actors:
        behaviors = []
        if actor["type"] == "vehicle":
            behaviors.append(("cut_in", {
                "type": "lane_change", "target_lane": "ego_lane", "duration_s": 3.0}))
            if actor["speed"] >= 0.5:
                duration = min(1.5, actor["speed"] / 0.5)
                behaviors.append(("sudden_braking", {
                    "type": "brake", "deceleration_mps2": -min(5.0, actor["speed"] / duration),
                    "duration_s": duration}))
        else:
            behaviors.append(("pedestrian_crossing", {
                "type": "cross_lane", "target_lane": "ego_lane", "speed_mps": 1.4}))
        for kind, behavior in behaviors:
            if event_type not in ("auto", kind):
                continue
            candidate = validate_event({
                "event": kind, "actor": actor["id"],
                "trigger": {"type": "distance", "value": 15.0},
                "behavior": behavior,
            })
            if check_feasibility(candidate, scenario)["status"] != "fail":
                candidates.append(candidate)
    if not candidates:
        raise SchemaError(f"No eligible actor for rule event {event_type}; supply a suitable snapshot")
    return json.dumps(random.Random(seed).choice(candidates), ensure_ascii=False, allow_nan=False)
