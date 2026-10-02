"""Shared CLI configuration and output persistence for inference and experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .inference import Director, DirectorResult
from .models.vlm_wrapper import (
    DEFAULT_QWEN_MODEL, CompatibleAPIWrapper, GenerationSettings,
    QwenVLMWrapper, ReplayWrapper,
)
from .schemas.scenario_schema import EVENT_TYPES, Scenario


def add_director_arguments(parser: argparse.ArgumentParser) -> None:
    """Keep single-scenario and batch experiment controls consistent."""
    parser.add_argument("--model", choices=("rule", "qwen", "replay"), default="rule")
    parser.add_argument("--baseline", type=int, choices=(0, 1, 2),
                        help="Default: 0 for rules, 1 for text, 2 when BEV is supplied")
    parser.add_argument("--backend", choices=("transformers", "api"), default="transformers")
    parser.add_argument("--model-id", help="HF checkpoint/path or server's model name")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL"))
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--no-json-mode", action="store_true", help="Disable API response_format")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true", help="Prevent HF weight downloads")
    parser.add_argument("--response-file", type=Path, help="Required for explicit replay only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--event", choices=("auto", *EVENT_TYPES), default="auto",
                        help="Rule baseline event selection")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))


def make_director(args: argparse.Namespace, *, has_bev: bool) -> Director:
    """Resolve baseline explicitly; never substitute rules for a failed VLM call."""
    baseline = args.baseline
    if baseline is None:
        baseline = 0 if args.model == "rule" else (2 if has_bev else 1)
    if (args.model == "rule") != (baseline == 0):
        raise ValueError("--model rule requires baseline 0; --model qwen/replay requires baseline 1 or 2")
    if baseline != 0 and args.event != "auto":
        raise ValueError("--event selects rules and only applies to baseline 0")
    if baseline == 2 and not has_bev:
        raise ValueError("Baseline 2 requires --bev (single) or --bev-dir (batch)")
    if baseline != 2 and has_bev:
        raise ValueError("BEV input requires baseline 2 and a qwen/replay model")
    settings = GenerationSettings(args.seed, args.temperature, args.max_new_tokens)
    model = None
    if args.model == "replay":
        if args.response_file is None or not args.response_file.is_file():
            raise ValueError("--model replay requires an existing --response-file")
        model = ReplayWrapper(args.response_file)
    elif args.model == "qwen":
        if args.backend == "api":
            if not args.base_url:
                raise ValueError("--backend api requires --base-url or LLM_BASE_URL")
            model = CompatibleAPIWrapper(args.base_url, args.model_id or DEFAULT_QWEN_MODEL,
                                         timeout=args.timeout, json_mode=not args.no_json_mode)
        else:
            model = QwenVLMWrapper(args.model_id or DEFAULT_QWEN_MODEL, device=args.device,
                                   local_files_only=args.local_files_only)
    return Director(baseline, model, settings, args.event)


def run_file(director: Director, scenario_path: Path,
             bev_path: Path | None = None) -> DirectorResult:
    """Retain failed samples, including setup/input/API errors, for evaluation."""
    try:
        result = director.run(Scenario.from_file(scenario_path), bev_path)
    except (ValueError, RuntimeError, OSError) as exc:
        result = DirectorResult(None, "", {
            "baseline": director.baseline,
            "model": director.model.metadata if director.model else {
                "backend": "rule", "model_id": "rules-v1", "is_mock": False},
            "generation": asdict(director.settings), "bev_used": False,
            "strict_json_valid": False, "json_object_valid": False,
            "schema_valid": False, "success": False,
            "feasibility": {"status": "not_checked", "simulation_verified": False},
            "error_type": type(exc).__name__, "error": str(exc),
        })
    # Keep input identity outside the executor event. No credentials are recorded.
    inputs = {}
    for name, path in (("scenario", scenario_path), ("bev", bev_path)):
        if path is not None:
            entry = {"path": str(path.resolve()), "sha256": None}
            try:
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                pass  # Missing input is already represented by the sample failure.
            inputs[name] = entry
    result.report["inputs"] = inputs
    return result


def write_json(path: Path, obj: Any) -> None:
    """Write valid JSON through a sibling temporary file and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def save_result(result: DirectorResult, output_dir: Path, stem: str,
                *, protected_paths: tuple[Path, ...] = ()) -> dict[str, str | None]:
    """Persist raw text and diagnostics; export only usable high-level events.

    Repeated runs replace artifacts with the same stem. Remove a previous generated
    prediction on failure so an executor cannot mistakenly read a stale success.
    Guard supplied input paths before any writes/deletes, including custom outputs.
    """
    prediction_path = output_dir / f"{stem}_prediction.json"
    report_path = output_dir / f"{stem}_report.json"
    raw_path = output_dir / f"{stem}_raw.txt"
    protected = {path.resolve() for path in protected_paths}
    for path in (prediction_path, report_path, raw_path):
        if path.resolve() in protected:
            raise ValueError(f"Output would overwrite input: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(result.raw_response, encoding="utf-8")
    write_json(report_path, result.report)
    if result.success:
        write_json(prediction_path, result.prediction)
    else:
        prediction_path.unlink(missing_ok=True)
    return {"prediction": str(prediction_path.resolve()) if result.success else None,
            "report": str(report_path.resolve()), "raw": str(raw_path.resolve())}
