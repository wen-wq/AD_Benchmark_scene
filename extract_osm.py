"""Extract the PKU route corridor from a Beijing OSM PBF on the remote host."""

import json
import math
import os
from pathlib import Path

import osmium

ROOT = Path(__file__).resolve().parent
ROUTE = json.loads((ROOT / "osrm_route.json").read_text())
POINTS = ROUTE["routes"][0]["geometry"]["coordinates"]
LONS = [p[0] for p in POINTS]
LATS = [p[1] for p in POINTS]
PAD_LAT = 0.002
PAD_LON = 0.0025
BBOX = (min(LONS) - PAD_LON, min(LATS) - PAD_LAT,
        max(LONS) + PAD_LON, max(LATS) + PAD_LAT)

MOTOR = {"motorway", "trunk", "primary", "secondary", "tertiary",
         "unclassified", "residential", "living_street", "service",
         "motorway_link", "trunk_link", "primary_link", "secondary_link",
         "tertiary_link"}


class RoadHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.ways = []
        self.signals = []
        self.gates = []

    def node(self, n):
        if not n.location.valid():
            return
        lon, lat = n.location.lon, n.location.lat
        if not (BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]):
            return
        tags = dict(n.tags)
        if tags.get("highway") in ("traffic_signals", "stop", "crossing", "give_way"):
            self.signals.append({"id": n.id, "lon": lon, "lat": lat, "tags": tags})
        if tags.get("barrier") == "gate" or tags.get("entrance"):
            self.gates.append({"id": n.id, "lon": lon, "lat": lat, "tags": tags})

    def way(self, w):
        tags = dict(w.tags)
        if tags.get("highway") not in MOTOR:
            return
        coords = []
        for n in w.nodes:
            if n.location.valid():
                coords.append([n.location.lon, n.location.lat])
        if len(coords) < 2:
            return
        if not any(BBOX[0] <= p[0] <= BBOX[2] and BBOX[1] <= p[1] <= BBOX[3] for p in coords):
            return
        self.ways.append({"id": w.id, "tags": tags, "coordinates": coords})


handler = RoadHandler()
handler.apply_file(str(ROOT / "beijing.osm.pbf"), locations=True)
out = {
    "source": "OSM Beijing extract; route from OSRM public driving router",
    "bounds_lonlat": BBOX,
    "roads": handler.ways,
    "signals": handler.signals,
    "gates": handler.gates,
}
(ROOT / "corridor_osm.json").write_text(json.dumps(out, ensure_ascii=False))
print("road ways", len(handler.ways), "signals", len(handler.signals), "gates", len(handler.gates), flush=True)
near = []
for w in handler.ways:
    if any(k in w["tags"].get("name", "") for k in ("中关村北大街", "海淀路")):
        near.append((w["id"], w["tags"]))
print("named route roads:")
for row in near[:30]:
    print(row, flush=True)
