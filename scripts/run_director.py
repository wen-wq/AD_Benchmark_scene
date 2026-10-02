"""Run one Director prediction from any working directory; no generator imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Direct execution puts scripts/ on sys.path; explicitly expose its project parent.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_director.cli import add_director_arguments, make_director, run_file, save_result


def main(argv: list[str] | None = None) -> int:
    """Write event + diagnostics, returning nonzero on configuration or sample failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_director_arguments(parser)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--bev", type=Path, help="Snapshot-aligned local BEV PNG/JPEG")
    args = parser.parse_args(argv)
    try:
        director = make_director(args, has_bev=args.bev is not None)
        result = run_file(director, args.scenario, args.bev)
        protected = tuple(path for path in (args.scenario, args.bev, args.response_file) if path)
        paths = save_result(result, args.output_dir, args.scenario.stem, protected_paths=protected)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"success": result.success, "baseline": director.baseline,
                      "is_mock": result.report["model"]["is_mock"],
                      "feasibility": result.report["feasibility"]["status"],
                      "error": result.report["error"], **paths}, ensure_ascii=False, indent=2))
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
