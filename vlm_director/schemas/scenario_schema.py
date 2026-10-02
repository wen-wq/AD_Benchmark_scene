"""Validate scenario snapshots and executor-facing events without extra dependencies.

Input speeds default to km/h to match the supplied example; normalized snapshots
and all output parameters use SI units. Static checks deliberately distinguish
missing geometry from a failed constraint. They cannot certify collision freedom.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVENT_TYPES = ("cut_in", "sudden_braking", "pedestrian_crossing")
BEHAVIOR_TYPES = {
    "cut_in": "lane_change",
    "sudden_braking": "brake",
    "pedestrian_crossing": "cross_lane",
}


class SchemaError(ValueError):
    """A scenario or model response violates the documented JSON contract."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate keys rather than silently retaining the last value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SchemaError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(text: str) -> Any:
    """Disallow NaN, Infinity, and duplicate keys, which ordinary json accepts."""
    def reject_constant(value: str) -> None:
        raise SchemaError(f"Non-finite JSON constant: {value}")

    try:
        return json.loads(text, object_pairs_hook=_unique_object,
                          parse_constant=reject_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise SchemaError(f"Invalid JSON: {exc}") from exc


def number(value: Any, field: str, minimum: float | None = None,
           maximum: float | None = None) -> float:
    """Check actual finite numbers; booleans and numeric strings are not numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{field} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise SchemaError(f"{field} is too large") from exc
    if not math.isfinite(result):
        raise SchemaError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise SchemaError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise SchemaError(f"{field} must be <= {maximum}")
    return result


def lane_id(value: Any, field: str) -> int | str:
    """Lane IDs may be integers or nonempty simulator-native string IDs."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise SchemaError(f"{field} must be an integer or nonempty string")
    if isinstance(value, str) and not value.strip():
        raise SchemaError(f"{field} must not be empty")
    return value


@dataclass(frozen=True)
class Scenario:
    """Validated snapshot, copied from input and normalized to m/s."""

    data: dict[str, Any]

    @property
    def ego(self) -> dict[str, Any]:
        """Return ego state in the normalized snapshot."""
        return self.data["ego"]

    @property
    def actors(self) -> list[dict[str, Any]]:
        """Return existing actor states; the Director may not invent actors."""
        return self.data["actors"]

    @classmethod
    def from_dict(cls, source: Any) -> Scenario:
        """Validate the snapshot contract without importing the scenario generator."""
        if not isinstance(source, dict):
            raise SchemaError("Scenario must be a JSON object")
        if not all(key in source for key in ("map", "ego", "actors")):
            raise SchemaError(
                "Scenario requires map, ego, actors. The existing output/scenario.json "
                "is a trajectory export; a snapshot adapter is required (see README).")
        data = copy.deepcopy(source)
        map_info = data["map"]
        if not isinstance(map_info, (str, dict)) or not map_info:
            raise SchemaError("map must be a nonempty name or map metadata object")
        unit = data.get("speed_unit", "km/h")
        if unit not in ("km/h", "m/s"):
            raise SchemaError("speed_unit must be km/h or m/s")
        factor = 1 / 3.6 if unit == "km/h" else 1.0
        ego = data["ego"]
        if not isinstance(ego, dict):
            raise SchemaError("ego must be an object")
        ego["lane"] = lane_id(ego.get("lane"), "ego.lane")
        ego["speed"] = number(ego.get("speed"), "ego.speed", 0) * factor
        ego.setdefault("id", "ego")
        if not isinstance(ego["id"], str) or not ego["id"].strip():
            raise SchemaError("ego.id must be a nonempty string")
        if "s_m" in ego:
            number(ego["s_m"], "ego.s_m")
        actors = data["actors"]
        if not isinstance(actors, list):
            raise SchemaError("actors must be an array")
        ids = {ego["id"]}
        for index, actor in enumerate(actors):
            prefix = f"actors[{index}]"
            if not isinstance(actor, dict):
                raise SchemaError(f"{prefix} must be an object")
            actor_id = actor.get("id")
            if not isinstance(actor_id, str) or not actor_id.strip() or actor_id in ids:
                raise SchemaError(f"{prefix}.id must be unique and nonempty")
            ids.add(actor_id)
            if actor.get("type") not in ("vehicle", "pedestrian"):
                raise SchemaError(f"{prefix}.type must be vehicle or pedestrian")
            actor["speed"] = number(actor.get("speed"), f"{prefix}.speed", 0) * factor
            if actor["type"] == "vehicle" or "lane" in actor:
                actor["lane"] = lane_id(actor.get("lane"), f"{prefix}.lane")
            if "s_m" in actor:
                number(actor["s_m"], f"{prefix}.s_m")
            if actor.get("direction", "same") not in ("same", "opposite"):
                raise SchemaError(f"{prefix}.direction must be same or opposite")
        if isinstance(map_info, dict):
            if "lane_width_m" in map_info:
                number(map_info["lane_width_m"], "map.lane_width_m", 2, 6)
            lanes = map_info.get("lane_ids")
            if lanes is not None:
                if not isinstance(lanes, list) or not lanes:
                    raise SchemaError("map.lane_ids must be a nonempty array")
                for lane in lanes:
                    lane_id(lane, "map.lane_ids")
                if len(set(lanes)) != len(lanes):
                    raise SchemaError("map.lane_ids must be unique")
                for state in [ego, *actors]:
                    if "lane" in state and state["lane"] not in lanes:
                        raise SchemaError(f"Unknown lane for {state['id']}")
            pairs = map_info.get("adjacent_lanes")
            if pairs is not None:
                if not isinstance(pairs, list):
                    raise SchemaError("map.adjacent_lanes must be an array of pairs")
                for pair in pairs:
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise SchemaError("Each adjacency entry must contain two lane IDs")
                    for lane in pair:
                        lane_id(lane, "map.adjacent_lanes")
                    if pair[0] == pair[1] or (lanes is not None and any(x not in lanes for x in pair)):
                        raise SchemaError("Invalid adjacency pair")
        data["speed_unit"] = "m/s"
        return cls(data)

    @classmethod
    def from_file(cls, path: str | Path) -> Scenario:
        """Accept UTF-8 and UTF-8 BOM files written by Windows tools."""
        return cls.from_dict(strict_json_loads(Path(path).read_text(encoding="utf-8-sig")))


def _keys(obj: Any, required: set[str], optional: set[str], field: str) -> None:
    """Reject missing and unknown fields so misspelled commands cannot execute."""
    if not isinstance(obj, dict):
        raise SchemaError(f"{field} must be an object")
    missing, extra = required - obj.keys(), obj.keys() - required - optional
    if missing or extra:
        raise SchemaError(f"{field}: missing={sorted(missing)}, unknown={sorted(extra)}")


def validate_event(source: Any) -> dict[str, Any]:
    """Validate one event, leaving scenario-specific physics to check_feasibility."""
    _keys(source, {"event", "actor", "trigger", "behavior"},
          {"schema_version", "parameters"}, "event")
    event = copy.deepcopy(source)
    event.setdefault("schema_version", "1.0")
    if event["schema_version"] != "1.0":
        raise SchemaError("Unsupported schema_version; expected 1.0")
    if not isinstance(event["event"], str) or event["event"] not in EVENT_TYPES:
        raise SchemaError(f"event must be one of {EVENT_TYPES}")
    if not isinstance(event["actor"], str) or not event["actor"].strip():
        raise SchemaError("actor must be a nonempty existing actor ID")
    trigger = event["trigger"]
    _keys(trigger, {"type", "value"}, {"unit", "reference", "operator"}, "trigger")
    if trigger["type"] == "distance":
        trigger.setdefault("unit", "m")
        trigger.setdefault("reference", "ego")
        trigger.setdefault("operator", "lte")
        if (trigger["unit"], trigger["reference"], trigger["operator"]) != ("m", "ego", "lte"):
            raise SchemaError("distance trigger requires m, ego, lte")
        number(trigger["value"], "trigger.value", 0.1, 100)
    elif trigger["type"] == "time":
        trigger.setdefault("unit", "s")
        if trigger["unit"] != "s" or "reference" in trigger or "operator" in trigger:
            raise SchemaError("time trigger uses seconds elapsed from this snapshot")
        number(trigger["value"], "trigger.value", 0, 60)
    else:
        raise SchemaError("trigger.type must be distance or time")
    behavior = event["behavior"]
    kind = event["event"]
    fields = {
        "cut_in": {"type", "target_lane", "duration_s"},
        "sudden_braking": {"type", "deceleration_mps2", "duration_s"},
        "pedestrian_crossing": {"type", "target_lane", "speed_mps"},
    }[kind]
    _keys(behavior, fields, set(), "behavior")
    if behavior["type"] != BEHAVIOR_TYPES[kind]:
        raise SchemaError("event and behavior.type disagree")
    if "target_lane" in behavior:
        lane_id(behavior["target_lane"], "behavior.target_lane")
    if kind == "cut_in":
        number(behavior["duration_s"], "behavior.duration_s", 2, 6)
    elif kind == "sudden_braking":
        number(behavior["deceleration_mps2"], "behavior.deceleration_mps2", -8, -0.5)
        number(behavior["duration_s"], "behavior.duration_s", 0.2, 4)
    else:
        number(behavior["speed_mps"], "behavior.speed_mps", 0.5, 3)
    event.setdefault("parameters", {})
    # Reserve parameters for a future version instead of accepting unchecked control.
    _keys(event["parameters"], set(), set(), "parameters")
    return event


def adjacent_lanes(scenario: Scenario, left: int | str, right: int | str) -> bool | None:
    """Use explicit topology first; integer IDs otherwise imply contiguous lanes."""
    map_info = scenario.data["map"]
    if isinstance(map_info, dict) and "adjacent_lanes" in map_info:
        return any({left, right} == set(pair) for pair in map_info["adjacent_lanes"])
    if isinstance(left, int) and isinstance(right, int):
        return abs(left - right) == 1
    return None


def check_feasibility(event: dict[str, Any], scenario: Scenario) -> dict[str, Any]:
    """Apply necessary static constraints; pass is still not simulator certification.

    Missing s_m, missing string-lane topology, or an unspecified pedestrian origin
    yields unknown. Known contradictions yield fail and must block prediction export.
    """
    errors: list[str] = []
    warnings: list[str] = []
    checked = ["actor_exists"]
    actor = next((a for a in scenario.actors if a["id"] == event["actor"]), None)
    if actor is None:
        errors.append("Target actor does not exist in the input snapshot")
    else:
        kind, behavior = event["event"], event["behavior"]
        checked.extend(["actor_type", "behavior_bounds"])
        expected = "pedestrian" if kind == "pedestrian_crossing" else "vehicle"
        if actor["type"] != expected:
            errors.append(f"{kind} requires an existing {expected}")
        ego_lane = scenario.ego["lane"]
        if kind in ("cut_in", "sudden_braking"):
            checked.extend(["travel_direction", "actor_speed"])
            if actor.get("direction", "same") != "same":
                errors.append("Vehicle must travel in the ego direction")
            if actor["speed"] <= 0:
                errors.append("Vehicle must already be moving")
        if kind == "cut_in":
            checked.extend(["lane_adjacency", "lateral_acceleration"])
            target = behavior["target_lane"]
            if target != "ego_lane" and target != ego_lane:
                errors.append("Baseline cut_in must target the ego lane")
            adjacency = adjacent_lanes(scenario, actor.get("lane", ""), ego_lane)
            if actor.get("lane") == ego_lane or adjacency is False:
                errors.append("Cut-in actor must start in an adjacent lane")
            elif adjacency is None:
                warnings.append("Lane adjacency is unknown; provide map.adjacent_lanes")
            map_info = scenario.data["map"]
            width = map_info.get("lane_width_m", 3.5) if isinstance(map_info, dict) else 3.5
            # Peak acceleration for a quintic lane shift, assuming fixed lane width.
            peak = (10 / math.sqrt(3)) * width / behavior["duration_s"] ** 2
            if peak > 3.0:
                errors.append("Estimated peak lateral acceleration exceeds 3 m/s^2")
        elif kind == "sudden_braking":
            checked.extend(["same_lane", "nonnegative_final_speed"])
            if actor.get("lane") != ego_lane:
                errors.append("Braking actor must be in the ego lane")
            final_speed = actor["speed"] + behavior["deceleration_mps2"] * behavior["duration_s"]
            if final_speed < -1e-6:
                errors.append("Braking duration would require a negative final speed")
        else:
            checked.append("crossing_target")
            if behavior["target_lane"] not in ("ego_lane", ego_lane):
                errors.append("Crossing must target the ego lane")
            if actor.get("lane") == ego_lane:
                errors.append("Pedestrian is already in the target lane")
            elif "lane" not in actor:
                warnings.append("Pedestrian crossing origin is unknown")
        if "s_m" in scenario.ego and "s_m" in actor:
            checked.append("longitudinal_position")
            if actor["s_m"] <= scenario.ego["s_m"]:
                errors.append("Target must be ahead of ego at the snapshot time")
            elif (event["trigger"]["type"] == "distance" and expected == "vehicle"
                  and actor["s_m"] - scenario.ego["s_m"] > event["trigger"]["value"]
                  and actor["speed"] >= scenario.ego["speed"]):
                warnings.append("Distance trigger would not fire at constant speeds; "
                                "executor must verify trigger reachability")
        else:
            warnings.append("Missing route positions s_m; forward placement is unverified")
    return {
        "status": "fail" if errors else ("unknown" if warnings else "pass"),
        "errors": errors, "warnings": warnings, "checked": checked,
        "simulation_verified": False,
    }
