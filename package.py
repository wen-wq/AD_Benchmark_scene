"""Bundle the runnable pipeline, pinned inputs and structured scenario data."""
import hashlib
import json
import zipfile
from pathlib import Path

from pku_pipeline import OUT, ROOT


def main():
    sources=[
        "README.md","requirements.txt","run_pipeline.sh","extract_osm.py",
        "llm_provider.py","llm_events.json","pku_pipeline.py","native_render.py",
        "native_mosaic.py",
        "street_images.py","harvest_kartaview.py","analyze_collisions.py",
        "validate.py","make_previews.py",
        "package.py",
        "beijing.osm.pbf","osrm_route.json","corridor_osm.json",
        "output/scenario.json","output/scenario.npz","output/route_and_events.geojson",
        "output/route_overview.png","output/scene_preview.jpg",
        "output/turn_speed_profile.png","output/collision_report.json",
        "output/validation_report.json",
    ]
    files=[ROOT/x for x in sources]
    for folder in (ROOT/"vendor",OUT/"clipgt",OUT/"street_images"):
        files.extend(p for p in folder.rglob("*") if p.is_file() and
                     "__pycache__" not in p.parts and p.suffix!=".pyc")
    missing=[str(p) for p in files if not p.exists()]
    if missing:raise FileNotFoundError(missing)
    destination=OUT/"pku_world_scenario_pipeline_v2.zip"
    manifest=[]
    with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(files,key=lambda x:str(x)):
            rel=p.relative_to(ROOT).as_posix()
            data=p.read_bytes()
            archive.writestr("pku-world-scenario-v2/"+rel,data)
            manifest.append({"path":rel,"bytes":len(data),
                             "sha256":hashlib.sha256(data).hexdigest()})
        archive.writestr("pku-world-scenario-v2/package_manifest.json",
            json.dumps({"files":manifest},indent=2))
    print(destination, destination.stat().st_size,"bytes",len(manifest),"files")


if __name__=="__main__":main()
