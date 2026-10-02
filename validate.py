"""Check route geometry, event dynamics, actor separation and videos."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from pku_pipeline import CLIP, OUT, ambient_at, at_s, ll_to_xy, map_model, read_json, save_json


def probe(path):
    p=subprocess.run(["ffprobe","-v","error","-show_entries",
        "stream=nb_frames,r_frame_rate,width,height:format=duration", "-of","json",str(path)],
        capture_output=True,text=True,check=True)
    obj=json.loads(p.stdout)
    st=obj["streams"][0]
    return {"frames":int(st["nb_frames"]),"fps":st["r_frame_rate"],
            "size":[st["width"],st["height"]],
            "duration_s":round(float(obj["format"]["duration"]),2),
            "bytes":path.stat().st_size}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--require-native",action="store_true")
    a=ap.parse_args()
    scene=read_json(OUT/"scenario.json")
    data=np.load(OUT/"scenario.npz");arr=data["frames"]
    events=scene["emergencies"]
    checks={}
    checks["exactly_five_events"]=len(events)==5
    checks["ordered_events"]=all(x["time_s"]<y["time_s"] for x,y in zip(events,events[1:]))
    start=np.asarray(scene["route_lonlat"][0]);end=np.asarray(scene["route_lonlat"][-1])
    east=np.asarray([116.30973,39.99078]);south=np.asarray([116.30546,39.98527])
    checks["gate_proximity"]=bool(np.linalg.norm(ll_to_xy([start],east)[0])<30 and
                                   np.linalg.norm(ll_to_xy([end],south)[0])<30)
    roads,_=map_model(data["origin"])
    geom=[LineString(x["xy"]) for x in roads]
    route=data["route"];cum=data["route_s"]
    alignment=[]
    for s in np.arange(0,cum[-1],20):
        p,_=at_s(route,cum,s)
        alignment.append(min(line.distance(Point(p)) for line in geom))
    # The smoothed turn arc can sit a few metres off an OSM centerline.
    checks["route_follows_osm_driving_ways"]=max(alignment)<8
    checks["positive_separation"]=bool(np.min(arr[:,7])>5)
    collision=read_json(OUT/"collision_report.json")
    checks["no_actor_box_overlaps"]=collision["overlap_frames"]==0
    checks["four_safe_cutin_gaps"]=len(collision["cutin_metrics"])==4 and all(
        x["min_lane_gap_m"]>8 for x in collision["cutin_metrics"])
    checks["mixed_emergencies"]=sum(e["type"]=="lead_vehicle_hard_brake" for e in events)==1 and sum(
        e["type"] in ("left_cutin","right_cutin") for e in events)==4 and any(
        e.get("subtype")=="curbside_pullout" for e in events)
    checks["five_distinct_ego_braking_responses"]=True
    responses=[]
    for e in events:
        t=e["time_s"]
        win=arr[(arr[:,0]>=t)&(arr[:,0]<t+5)]
        response={"id":e["id"],"lead_min_accel_mps2":round(float(np.min(win[:,6])),2),
                  "ego_min_accel_mps2":round(float(np.min(win[:,3])),2),
                  "ego_min_speed_mps":round(float(np.min(win[:,2])),2),
                  "min_gap_m":round(float(np.min(win[:,7])),2)}
        if response["ego_min_accel_mps2"]>-4.5 or (
            e["type"]=="lead_vehicle_hard_brake" and response["lead_min_accel_mps2"]>-4.5):
            checks["five_distinct_ego_braking_responses"]=False
        responses.append(response)
    metrics=scene["metrics"]
    checks["smooth_slow_turn"]=metrics["turn_speed_mps"]<metrics["pre_turn_speed_mps"] and (
        metrics["max_yaw_rate_degps"]<35 and metrics["max_lateral_accel_mps2"]<3)
    cap=data["speed_cap"]
    checks["anticipatory_turn_speed_cap"]=float(np.interp(560,cum,cap)) > (
        float(np.interp(600,cum,cap))+2)
    ob=pd.read_parquet(OUT/"clipgt"/f"{CLIP}.obstacle.parquet")
    tracks={x["trackline_id"] for x in ob["obstacle"]}
    checks["dense_traffic_with_cutins"]=len(tracks)>=65 and all(
        x in tracks for x in ("lead_car_01","cutin_E2","cutin_E3","cutin_E4","cutin_E5")) and (
        scene["traffic_model"]["median_active_actors"]>=20)
    local_counts=[]
    for i in range(0,len(arr),10):
        es=arr[i,1];t=arr[i,0]
        count=sum(abs(x[3]-es)<100 for x in ambient_at(t,route,cum))
        count+=int(arr[i,4]<=cum[-1] and abs(arr[i,4]-es)<100)
        count+=int(np.count_nonzero(np.isfinite(data["cutin_s"][:,i]) &
                                    (abs(data["cutin_s"][:,i]-es)<100)))
        local_counts.append(int(count))
    checks["dense_local_traffic"]=bool(float(np.median(local_counts))>=8 and min(local_counts)>=4)
    manifest=read_json(OUT/"street_images"/"manifest.json")
    stations=manifest["stations"]
    checks["street_coverage_truthful"]=len(stations)==10 and all(
        x["status"]=="missing" or (OUT/"street_images"/x["image"]).exists()
        for x in stations)
    kv=read_json(OUT/"street_images"/"kartaview_manifest.json")
    kv_frames=[x for x in kv["stations"] if x["status"]=="available"]
    checks["route_aligned_street_frames"]=len(kv_frames)>=25 and all(
        (OUT/"street_images"/x["image"]).exists() and
        x["cross_track_m"]<=10 and x["heading_error_deg"]<=50 and
        abs(x["route_s_m"]-x["route_distance_m"])<=10 for x in kv_frames)
    checks["street_coverage_increased"]=sum(x["status"]!="missing" for x in stations)>=7
    videos={"topdown":probe(OUT/"videos"/"topdown.mp4")}
    checks["topdown_complete"]=videos["topdown"]["frames"]==len(arr)
    base=OUT/"native_wsv"/CLIP
    names=("camera_front_wide_120fov","camera_cross_left_120fov",
           "camera_cross_right_120fov","camera_rear_left_70fov",
           "camera_rear_right_70fov")
    if a.require_native:
        for name in names:
            path=base/f"{CLIP}.{name}.mp4"
            videos[name]=probe(path)
        checks["five_native_views_complete"]=all(
            videos[name]["frames"]>=3*len(arr)-5 and videos[name]["fps"]=="30/1" and
            videos[name]["size"]==[1280,720] for name in names)
        videos["five_view_native_mosaic"]=probe(OUT/"videos"/"five_view_native_mosaic.mp4")
        checks["native_mosaic_complete"]=(
            videos["five_view_native_mosaic"]["frames"]>=3*len(arr)-5 and
            videos["five_view_native_mosaic"]["fps"]=="30/1" and
            videos["five_view_native_mosaic"]["size"]==[1920,720])
    checks={name:bool(value) for name,value in checks.items()}
    report={"checks":checks,"passed":all(checks.values()),"responses":responses,
            "route_osm_max_offset_m":round(max(alignment),2),
            "local_traffic_200m":{"median":float(np.median(local_counts)),
                                   "p10":round(float(np.percentile(local_counts,10)),1),
                                   "p90":round(float(np.percentile(local_counts,90)),1),
                                   "min":min(local_counts),"max":max(local_counts)},
            "collision_report":collision,
            "video_metrics":videos,"street_photo_count":sum(x["status"]!="missing" for x in stations),
            "street_kartaview_25m_count":len(kv_frames),
            "street_commons_count":sum(x["provider"]=="Wikimedia Commons" for x in stations),
            "street_missing_count":sum(x["status"]=="missing" for x in stations)}
    save_json(OUT/"validation_report.json",report)
    print(json.dumps(report,indent=2))
    if not report["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
