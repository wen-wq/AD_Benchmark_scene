"""Small comparison metrics with explicit denominators and feasibility uncertainty."""

from collections import Counter
import math
from typing import Any

from .schemas.scenario_schema import EVENT_TYPES


def summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Count all attempts, including malformed JSON, input errors, and API failures.

    Diversity is measured on exported events only, and its sample count is reported.
    Static pass is a proxy; unknown geometry is never counted as physically feasible.
    Fixture runs remain labeled as interface tests and are not VLM quality results.
    """
    total = len(reports)
    if not total:
        raise ValueError("Evaluation requires at least one attempt")
    exported = [report for report in reports if report.get("success")]
    events = Counter(report["event_type"] for report in exported)
    entropy = -sum((count / len(exported)) * math.log2(count / len(exported))
                   for count in events.values()) if exported else 0.0
    statuses = Counter(report["feasibility"]["status"] for report in reports)
    return {
        "attempts": total,
        "interface_test_only": any(report["model"].get("is_mock") for report in reports),
        "rate_denominator": "all attempts, including input/model/transport failures",
        "strict_json_valid_rate": sum(r["strict_json_valid"] for r in reports) / total,
        "json_object_valid_rate": sum(r["json_object_valid"] for r in reports) / total,
        "schema_valid_rate": sum(r["schema_valid"] for r in reports) / total,
        "export_success_rate": len(exported) / total,
        "static_feasibility_pass_rate": statuses["pass"] / total,
        "static_feasibility_unknown_rate": statuses["unknown"] / total,
        "static_feasibility_fail_rate": statuses["fail"] / total,
        "feasibility_not_checked_rate": statuses["not_checked"] / total,
        "simulation_verified_rate": 0.0,
        "diversity": {
            "denominator": "exported events", "sample_count": len(exported),
            "event_counts": {event: events[event] for event in EVENT_TYPES},
            "unique_event_types": len(events), "event_type_entropy_bits": entropy,
            "unique_target_actors": len({r["actor"] for r in exported}),
        },
    }
