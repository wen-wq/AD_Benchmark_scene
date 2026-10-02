"""Compare JSON validity, static feasibility, and event diversity on snapshot files."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_director.cli import add_director_arguments, make_director, run_file, save_result, write_json
from vlm_director.evaluation import summarize_reports


def main(argv: list[str] | None = None) -> int:
    """Reuse one loaded model, vary seeds, and retain every attempted sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_director_arguments(parser)
    parser.add_argument("--scenarios", nargs="+", type=Path, required=True,
                        help="Snapshot JSON files or directories containing *.json")
    parser.add_argument("--samples-per-scenario", type=int, default=1)
    parser.add_argument("--bev-dir", type=Path, help="Match images by scenario stem (.png/.jpg/.jpeg)")
    args = parser.parse_args(argv)
    if args.samples_per_scenario <= 0:
        parser.error("--samples-per-scenario must be positive")
    try:
        director = make_director(args, has_bev=args.bev_dir is not None)
        files = sorted({path.resolve() for entry in args.scenarios
                        for path in (entry.glob("*.json") if entry.is_dir() else [entry])})
        if not files:
            raise ValueError("No scenario JSON files found")
        initial_settings = director.settings
        records = []
        reports = []
        for index, scenario_path in enumerate(files):
            bev_path = None
            if args.bev_dir is not None:
                matches = [args.bev_dir / (scenario_path.stem + suffix)
                           for suffix in (".png", ".jpg", ".jpeg")]
                # Use a missing path to keep missing-image failures in the denominator.
                bev_path = next((path for path in matches if path.is_file()), matches[0])
            for sample in range(args.samples_per_scenario):
                director.settings = replace(initial_settings, seed=initial_settings.seed + sample)
                result = run_file(director, scenario_path, bev_path)
                protected = tuple(path for path in (scenario_path, bev_path, args.response_file) if path)
                stem = f"{index:04d}_{scenario_path.stem}_sample{sample:03d}"
                paths = save_result(result, args.output_dir / "samples", stem, protected_paths=protected)
                reports.append(result.report)
                records.append({"scenario": str(scenario_path), "seed": director.settings.seed,
                                "bev": str(bev_path.resolve()) if bev_path else None,
                                "report": result.report, "artifacts": paths})
                print(f"[{len(records)}/{len(files) * args.samples_per_scenario}] "
                      f"{scenario_path.name}: {'ok' if result.success else 'failed'}", flush=True)
        summary = summarize_reports(reports)
        output = args.output_dir / "evaluation_report.json"
        if output.resolve() in set(files) or (args.response_file and output.resolve() == args.response_file.resolve()):
            raise ValueError("Evaluation output would overwrite an input")
        write_json(output, {"baseline": director.baseline, "metrics": summary, "samples": records})
        print(json.dumps({"report": str(output.resolve()), "metrics": summary}, indent=2))
        return 0 if all(result["success"] for result in reports) else 1
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
