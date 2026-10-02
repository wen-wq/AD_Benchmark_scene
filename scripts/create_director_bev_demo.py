"""Create a synthetic, snapshot-aligned BEV illustration for interface demos only.

This utility does not import or modify the existing scenario generator. Pillow is
optional; it is unnecessary for the rule baseline and compatible API client.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_director.schemas.scenario_schema import Scenario


def main() -> None:
    """Render the supplied full demo's known lane IDs and route-relative positions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path,
                        default=PROJECT_ROOT / "examples/director/scenarios/urban_full.json")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "examples/director/bev/urban_full.png")
    args = parser.parse_args()
    from PIL import Image, ImageDraw

    scenario = Scenario.from_file(args.scenario)
    if args.output.resolve() == args.scenario.resolve():
        parser.error("Output must not overwrite the input snapshot")
    if not isinstance(scenario.data["map"], dict) or scenario.data["map"].get("lane_ids") != [1, 2, 3]:
        parser.error("This demo renderer requires map.lane_ids=[1,2,3]")
    if any("s_m" not in state for state in [scenario.ego, *scenario.actors]):
        parser.error("This demo renderer requires s_m on every state")
    image = Image.new("RGB", (640, 720), "#17212b")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), "SYNTHETIC BEV / Director interface demo", fill="white")
    draw.text((24, 42), "Forward is up; state matches urban_full.json", fill="#b5c5d4")
    draw.rectangle((102, 85, 538, 660), fill="#343f49", outline="#d1dbe5", width=2)
    for x in (248, 393):
        for y in range(95, 650, 34):
            draw.line((x, y, x, y + 18), fill="#b5c5d4", width=2)
    for lane, x in ((1, 175), (2, 320), (3, 465)):
        draw.text((x - 20, 95), f"lane {lane}", fill="#b5c5d4")
    colors = {"ego": "#50debd", "vehicle": "#f6b86a", "pedestrian": "#e97aaa"}
    for state in [scenario.ego, *scenario.actors]:
        x = {1: 175, 2: 320, 3: 465}[state["lane"]]
        y = int(610 - (state["s_m"] - scenario.ego["s_m"]) * 12)
        kind = "ego" if state is scenario.ego else state["type"]
        if kind == "pedestrian":
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=colors[kind])
        else:
            draw.rounded_rectangle((x - 19, y - 31, x + 19, y + 31), radius=5, fill=colors[kind])
        draw.text((x - 42, y + 36), state["id"], fill="white")
    draw.text((24, 688), "Illustration only; no simulator or VLM model was run.", fill="#b5c5d4")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
