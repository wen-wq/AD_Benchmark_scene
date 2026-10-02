"""Reproducible OSM/OSRM -> synthetic PKU World Scenario -> ClipGT + videos.

Run on A6000pro80-VPS. Metric XY is local ENU (X=east,Y=north), while
ClipGT FLU is X=north,Y=west. Road widths and camera intrinsics are estimates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
CLIP = "pku_east_to_south_public_roads"
FPS = 10
CAMERAS = {
    "front_wide": 0,
    "cross_left": 85,
    "cross_right": -85,
    "rear_left": 150,
    "rear_right": -150,
}
AMBIENT_RULES = [
    {"id":f"primary_{lane}_{i:02d}", "kind":"moving", "direction":1,
     "s0":float(-180+i*62+lane*19), "speed":float((8.4,9.2,10.0)[lane]),
     "lateral_m":float((-3.2,3.2,6.4)[lane]), "start_t":0., "min_s":0., "max_s":640.}
    for lane in range(3) for i in range(14) if (lane,i) not in ((1,3),(0,4))
] + [
    {"id":f"haidian_opposite_{i:02d}", "kind":"moving", "direction":-1,
     "s0":995., "speed":8.5,
     "lateral_m":4., "start_t":float(32+i*4.3), "min_s":670., "max_s":995.}
    for i in range(18)
] + [
    {"id":f"haidian_parked_{i:02d}", "kind":"parked", "direction":1,
     "s0":float(915 if i==7 else 665+i*34), "speed":0., "lateral_m":-3.8,
     "start_t":0., "min_s":650., "max_s":995.}
    for i in range(10)
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ll_to_xy(coords, origin):
    a = np.asarray(coords, float)
    return np.column_stack(((a[:, 0]-origin[0])*111320*math.cos(math.radians(origin[1])),
                            (a[:, 1]-origin[1])*111320))


def route_model():
    raw = read_json(ROOT / "osrm_route.json")["routes"][0]
    # OSRM's snap added a small start loop and a final gate access spur.
    ll = np.asarray(raw["geometry"]["coordinates"][4:23], float)
    origin = ll[0].tolist()
    raw_xy = ll_to_xy(ll, origin)
    raw_s = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(raw_xy,axis=0),axis=1))]
    sample_s=np.r_[np.arange(0,raw_s[-1],.5),raw_s[-1]]
    center=np.column_stack((np.interp(sample_s,raw_s,raw_xy[:,0]),
                            np.interp(sample_s,raw_s,raw_xy[:,1])))
    # Gaussian smoothing rounds the OSRM/OSM polyline's piecewise-sharp turn.
    # 16 m spatial sigma creates a city-street turn arc while leaving long
    # straight segments on their mapped centerline.
    center=gaussian_filter1d(center, sigma=32, axis=0, mode="nearest")
    d=np.gradient(center,axis=0)
    tan=d/np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-9)
    left=np.column_stack((-tan[:,1],tan[:,0]))
    # Approximate right-hand lane center; blend from primary to two-way road.
    blend=np.clip((sample_s-610)/50,0,1)
    right_shift=1.6+.4*(blend*blend*(3-2*blend))
    ego_path=center-right_shift[:,None]*left
    s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(ego_path,axis=0),axis=1))]
    grid=np.r_[np.arange(0,s[-1],1.),s[-1]]
    route=np.column_stack((np.interp(grid,s,ego_path[:,0]),
                           np.interp(grid,s,ego_path[:,1])))
    cum=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(route,axis=0),axis=1))]
    route_ll=np.column_stack((origin[0]+route[:,0]/(111320*math.cos(math.radians(origin[1]))),
                              origin[1]+route[:,1]/111320))
    return origin,route,cum,route_ll


def at_s(xy, cum, s):
    s = float(np.clip(s, 0, cum[-1]))
    i = min(np.searchsorted(cum, s, side="right")-1, len(cum)-2)
    v = xy[i+1]-xy[i]
    d = max(np.linalg.norm(v), 1e-6)
    p = xy[i] + (s-cum[i])*v/d
    return p, v/d


def ambient_at(t, route, cum):
    out=[]
    for rule in AMBIENT_RULES:
        if t < rule["start_t"]:continue
        s=rule["s0"]+rule["direction"]*rule["speed"]*(t-rule["start_t"])
        if not (rule["min_s"] <= s <= min(rule["max_s"],cum[-1])):continue
        p,tan=at_s(route,cum,s)
        side=np.array([-tan[1],tan[0]])
        out.append((rule["id"],p+rule["lateral_m"]*side,
                    tan*rule["direction"],s))
    return out


def validate_events(spec, route_length):
    events = spec["events"]
    assert len(events) == 5, "exactly five emergencies required"
    distances = [e["distance_m"] for e in events]
    assert distances == sorted(distances) and len(set(distances)) == 5
    assert all(b-a>=130 for a,b in zip(distances,distances[1:]))
    for e in events:
        assert e["type"] in ("lead_vehicle_hard_brake","left_cutin","right_cutin")
        assert 50 < e["distance_m"] < route_length-50
        assert .2 <= e["ego_reaction_s"] <= 1
        assert 1 <= e["ego_brake_duration_s"] <= 2.5
        if e["type"] == "lead_vehicle_hard_brake":
            assert -8 <= e["lead_decel_mps2"] <= -4
            assert 1 <= e["lead_brake_duration_s"] <= 2.5
        else:
            assert 18 <= e["initial_gap_m"] <= 35
            assert 5 <= e["actor_speed_mps"] <= 10
            assert 2 <= e["cutin_duration_s"] <= 4
            if e.get("subtype") == "curbside_pullout":
                assert e["type"] == "right_cutin"
            if "yield_speed_mps" in e:
                assert 0 < e["yield_speed_mps"] <= 6
                assert 4 <= e["yield_duration_s"] <= 12
            if "hold_speed_mps" in e:
                assert 4 <= e["hold_speed_mps"] <= 8
                assert 3 <= e["hold_duration_s"] <= 7
            if "pre_event_visibility_s" in e:
                assert 1 <= e["pre_event_visibility_s"] <= 12
    assert sum(e["type"] != "lead_vehicle_hard_brake" for e in events)>=2
    return events


def road_width(tags):
    try:
        n = int(tags.get("lanes", ""))
        if 1 <= n <= 8:
            return min(max(n*3.2, 4.5), 26.)
    except ValueError:
        pass
    return {"primary": 14., "secondary": 11., "tertiary": 8.,
            "residential": 6., "service": 5.}.get(tags.get("highway"), 6.)


def map_model(origin):
    osm = read_json(ROOT / "corridor_osm.json")
    roads = []
    for w in osm["roads"]:
        pts = ll_to_xy(w["coordinates"], origin)
        roads.append({"id": w["id"], "xy": pts, "width": road_width(w["tags"]),
                      "name": w["tags"].get("name", ""), "tags": w["tags"]})
    return roads, osm


def speed_profile(route, cum):
    tangent=np.gradient(route,axis=0)
    heading=np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]))
    curvature=gaussian_filter1d(np.gradient(heading,cum),sigma=3)
    cap=np.minimum(10.,np.sqrt(1.7/np.maximum(np.abs(curvature),1e-5)))
    cap=np.clip(cap,4.5,10.)
    for i in range(len(cap)-2,-1,-1):
        ds=cum[i+1]-cum[i]
        cap[i]=min(cap[i],math.sqrt(cap[i+1]**2+2*2.3*ds))
    for i in range(1,len(cap)):
        ds=cum[i]-cum[i-1]
        cap[i]=min(cap[i],math.sqrt(cap[i-1]**2+2*1.6*ds))
    return heading,curvature,cap


def simulate(events, route, cum, fps=FPS):
    dt = 1/fps
    length=cum[-1]
    _,_,cap=speed_profile(route,cum)
    def vcap(s):return float(np.interp(np.clip(s,0,length),cum,cap))
    t=ego_s=0.
    lead_s=47.
    ego_v=vcap(0)
    lead_v=vcap(lead_s)
    active = -1
    start_t = -1.
    rows = []
    marks = []
    cutins = []
    while ego_s < length-.5 and t < 260:
        if active+1 < len(events) and ego_s >= events[active+1]["distance_m"]:
            active += 1
            start_t = t
            e = events[active]
            marks.append({**e,"time_s":round(t,2),"ego_s_m":round(ego_s,2),
                          "lead_s_m":round(lead_s,2)})
            if e["type"] in ("left_cutin", "right_cutin"):
                is_pullout=e.get("subtype")=="curbside_pullout"
                cutins.append({"id":f"cutin_{e['id']}","event_id":e["id"],
                               "s":ego_s+e["initial_gap_m"],
                               "start_s":ego_s+e["initial_gap_m"],
                               "v":0. if is_pullout else e["actor_speed_mps"],
                               "speed":e["actor_speed_mps"],"t0":t,
                               "hold_speed":e.get("hold_speed_mps"),
                               "hold_duration":e.get("hold_duration_s",0),
                               "pre_visibility":e.get("pre_event_visibility_s",260),
                               "is_pullout":is_pullout,
                               "duration":e["cutin_duration_s"],
                               "start_lat":3.2 if e["type"]=="left_cutin" else
                               (-3.8 if is_pullout else -3.2),
                               "history":[float("nan")]*len(rows)})
        gap = lead_s-ego_s-4.6
        lead_target=min(vcap(lead_s),vcap(lead_s+max(20.,lead_v*3.0)))
        if gap>57 and ego_v>=6:lead_target=min(lead_target,ego_v-1.5)
        if gap<34:lead_target=min(12.,lead_target+2.)
        lead_a=float(np.clip((lead_target-lead_v)*1.5,-2.3,2.4))
        if active>=0 and events[active]["type"]=="lead_vehicle_hard_brake" and t-start_t<events[active]["lead_brake_duration_s"]:
            lead_a=events[active]["lead_decel_mps2"]
        ego_target=min(vcap(ego_s),vcap(ego_s+max(25.,ego_v*3.5)))
        if gap<35 and ego_v>lead_v:
            ego_target=min(ego_target,max(0.,lead_v+.2*(gap-22)))
        if active>=0 and events[active].get("yield_speed_mps") is not None:
            e=events[active]
            if t-start_t<e["yield_duration_s"]:
                ego_target=min(ego_target,e["yield_speed_mps"])
        ego_a=float(np.clip((ego_target-ego_v)*1.8,-2.5,1.6))
        if active>=0:
            e=events[active];elapsed=t-start_t
            if e["ego_reaction_s"]<=elapsed<e["ego_reaction_s"]+e["ego_brake_duration_s"]:
                ego_a=-5.0 if e["type"]=="lead_vehicle_hard_brake" else -4.6
        if gap<11:ego_a=-6.0
        lead_v=float(np.clip(lead_v+lead_a*dt,0,12))
        lead_s += lead_v*dt
        # Integrate every merging actor with the same clock as the ego car.
        # Older actors are farther ahead and can serve as leaders.
        for j, actor in enumerate(cutins):
            target=min(11.2,vcap(actor["s"])+1.2)
            if actor["is_pullout"] and t-actor["t0"]<4:
                target=min(target,actor["speed"])
            if actor["hold_speed"] is not None and t-actor["t0"]<actor["hold_duration"]:
                target=min(target,actor["hold_speed"])
            leaders=[(lead_s,lead_v)]
            for prior in cutins[:j]:
                if prior["s"]>actor["s"]+4.6:
                    leaders.append((prior["s"],prior["v"]))
            leader_s,leader_v=min(leaders,key=lambda x:x[0])
            actor_gap=leader_s-actor["s"]-4.6
            if actor_gap<28:
                target=min(target,max(0.,leader_v+.35*(actor_gap-14)))
            accel=float(np.clip((target-actor["v"])*1.5,-5.,1.8))
            if actor_gap<8:accel=-6.
            actor["v"]=max(0.,actor["v"]+accel*dt)
            actor["s"]+=actor["v"]*dt
            actor["history"].append(actor["s"])
        # A completed or completing merge becomes a real leader for ego.
        cutin_leaders=[]
        for actor in cutins:
            u=float(np.clip((t-actor["t0"])/actor["duration"],0,1))
            smooth=u*u*u*(10-15*u+6*u*u)
            lat=actor["start_lat"]*(1-smooth)
            if abs(lat)<2.5 and actor["s"]>ego_s:
                cutin_leaders.append((actor["s"],actor["v"]))
        if cutin_leaders:
            near_s,near_v=min(cutin_leaders,key=lambda x:x[0])
            near_gap=near_s-ego_s-4.6
            if near_gap<42:
                ego_target=min(ego_target,max(0.,near_v+.27*(near_gap-19)))
                ego_a=min(ego_a,float(np.clip((ego_target-ego_v)*1.8,-5.,1.6)))
            if near_gap<11:ego_a=-6.
        ego_v = float(np.clip(ego_v+ego_a*dt, 0, 10))
        ego_s += ego_v*dt
        rows.append((t, ego_s, ego_v, ego_a, lead_s, lead_v, lead_a, gap, active))
        t += dt
    arr = np.asarray(rows)
    if len(marks) != 5:
        raise RuntimeError(f"simulation triggered {len(marks)} emergencies")
    if np.min(arr[:,7]) < 5:
        raise RuntimeError(f"unsafe lead gap {np.min(arr[:,7]):.2f} m")
    tracks=[]
    for actor in cutins:
        series=np.asarray(actor["history"],float)
        if len(series)!=len(arr):
            raise RuntimeError("cut-in track/frame mismatch")
        prior=np.isnan(series)
        series[prior]=(actor["start_s"] if actor["is_pullout"] else
                       actor["start_s"]+actor["speed"]*(arr[prior,0]-actor["t0"]))
        series[arr[:,0]<actor["t0"]-actor["pre_visibility"]]=np.nan
        u=np.clip((arr[:,0]-actor["t0"])/actor["duration"],0,1)
        smooth=u*u*u*(10-15*u+6*u*u)
        lateral=actor["start_lat"]*(1-smooth)
        tracks.append({"id":actor["id"],"event_id":actor["event_id"],
                       "s":series,"lateral":lateral})
    return arr, marks, tracks


def cutin_pose(track,i,route,cum):
    s=float(track["s"][i]);lat=float(track["lateral"][i])
    if not (0<=s<=cum[-1]):return None
    p,tan=at_s(route,cum,s)
    side=np.array([-tan[1],tan[0]])
    p=p+lat*side
    j=max(0,i-1);k=min(len(track["s"])-1,i+1)
    if j!=k and 0<=track["s"][j]<=cum[-1] and 0<=track["s"][k]<=cum[-1]:
        pj,tj=at_s(route,cum,track["s"][j]);pk,tk=at_s(route,cum,track["s"][k])
        v=(pk+track["lateral"][k]*np.array([-tk[1],tk[0]])-
           pj-track["lateral"][j]*np.array([-tj[1],tj[0]]))
        if np.linalg.norm(v)>1e-5:tan=v/np.linalg.norm(v)
    return p,tan


def make_scenario():
    OUT.mkdir(exist_ok=True)
    origin, route, cum, ll = route_model()
    roads, osm = map_model(origin)
    spec = read_json(ROOT / "llm_events.json")
    events = validate_events(spec, cum[-1])
    arr, marks, cutins = simulate(events, route, cum)
    heading,curvature,caps=speed_profile(route,cum)
    turn_idx=int(np.argmax(np.abs(curvature)))
    turn_s=float(cum[turn_idx])
    pre=arr[(arr[:,1]>=turn_s-65)&(arr[:,1]<turn_s-35),2]
    turn=arr[(arr[:,1]>=turn_s-10)&(arr[:,1]<=turn_s+10),2]
    yaw=np.unwrap(np.asarray([math.atan2(at_s(route,cum,s)[1][1],at_s(route,cum,s)[1][0]) for s in arr[:,1]]))
    yaw_rate=np.diff(yaw)/np.diff(arr[:,0])
    lat_accel=arr[:,2]**2*np.interp(arr[:,1],cum,np.abs(curvature))
    traffic_count=[len(ambient_at(r[0],route,cum))+1+
                   sum(cutin_pose(x,i,route,cum) is not None for x in cutins)
                   for i,r in enumerate(arr)]
    np.savez_compressed(OUT / "scenario.npz", frames=arr, route=route, route_s=cum,
                        origin=np.asarray(origin),speed_cap=caps,
                        cutin_s=np.asarray([x["s"] for x in cutins]),
                        cutin_lateral=np.asarray([x["lateral"] for x in cutins]),
                        cutin_ids=np.asarray([x["id"] for x in cutins]))
    save_json(OUT / "scenario.json", {
        "scene_id": CLIP, "coordinate_system": "local ENU meters",
        "origin_lonlat": origin, "route_lonlat": ll.tolist(),
        "route_length_m": round(float(cum[-1]), 2),
        "source": {"route": "OSRM driving GeoJSON", "map": "OpenStreetMap Beijing PBF",
                   "road_widths": "estimated from OSM lane tags or road class"},
        "route_policy": "public roads outside PKU campus; OSRM start loop and gate spur excluded",
        "fps": FPS, "duration_s": round(float(arr[-1,0]), 2),
        "frame_count": len(arr), "road_way_count": len(roads),
        "traffic_model": {"name":"rule-based multilane flow with stopped curbside vehicles",
                          "ambient_vehicle_rules":AMBIENT_RULES,
                          "ambient_vehicle_count":len(AMBIENT_RULES),
                          "median_active_actors":int(np.median(traffic_count)),
                          "peak_active_actors":int(max(traffic_count))},
        "emergencies": marks, "event_spec": spec,
        "metrics": {"minimum_bumper_gap_m": round(float(np.min(arr[:,7])), 2),
                    "minimum_ego_speed_mps": round(float(np.min(arr[:,2])), 2),
                    "ego_hard_brake_frames": int(np.count_nonzero(arr[:,3] <= -4.5)),
                    "turn_center_s_m":round(turn_s,2),
                    "pre_turn_speed_mps":round(float(np.mean(pre)),2),
                    "turn_speed_mps":round(float(np.mean(turn)),2),
                    "max_yaw_rate_degps":round(float(np.max(np.abs(yaw_rate))*180/math.pi),2),
                    "max_lateral_accel_mps2":round(float(np.max(lat_accel)),2)}
    })
    features=[{"type":"Feature","properties":{"name":"PKU East Gate to South Gate, outside public roads"},
               "geometry":{"type":"LineString","coordinates":ll.tolist()}}]
    for m in marks:
        mp,_=at_s(route,cum,m["ego_s_m"])
        lon=origin[0]+mp[0]/(111320*math.cos(math.radians(origin[1])))
        lat=origin[1]+mp[1]/111320
        features.append({"type":"Feature","properties":{"id":m["id"],"time_s":m["time_s"],
            "type":m["type"]},"geometry":{"type":"Point","coordinates":[lon,lat]}})
    save_json(OUT/"route_and_events.geojson",{"type":"FeatureCollection","features":features})
    render_overview(roads,route,cum,marks)
    export_clipgt(arr, route, cum, origin, roads, cutins)
    print(f"Built {len(arr)} frames / {cum[-1]:.1f} m / {arr[-1,0]:.1f} s; events={len(marks)}", flush=True)


def render_overview(roads,route,cum,marks):
    w,h=1000,1300
    lo=np.min(route,axis=0)-90; hi=np.max(route,axis=0)+90
    scale=min((w-80)/(hi[0]-lo[0]),(h-80)/(hi[1]-lo[1]))
    def pix(a):
        a=np.asarray(a)
        return np.int32(np.round(np.column_stack((40+(a[:,0]-lo[0])*scale,
                                                    h-40-(a[:,1]-lo[1])*scale))))
    im=np.full((h,w,3),(29,35,36),np.uint8)
    for road in roads:
        cv2.polylines(im,[pix(road["xy"])],False,(64,69,70),
                      max(2,int(road["width"]*scale)),cv2.LINE_AA)
    cv2.polylines(im,[pix(route)],False,(43,190,236),8,cv2.LINE_AA)
    for m in marks:
        p,_=at_s(route,cum,m["ego_s_m"]);u=tuple(pix([p])[0])
        cv2.circle(im,u,12,(30,130,250),-1,cv2.LINE_AA)
        cv2.putText(im,m["id"],(u[0]+15,u[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.8,(250,250,250),2)
    for name,p in (("EAST GATE",route[0]),("SOUTH GATE",route[-1])):
        u=tuple(pix([p])[0]);cv2.circle(im,u,11,(60,240,100),-1,cv2.LINE_AA)
        cv2.putText(im,name,(u[0]+15,u[1]+4),cv2.FONT_HERSHEY_SIMPLEX,.7,(245,245,245),2)
    cv2.putText(im,"PKU PUBLIC-ROAD WORLD SCENARIO",(28,36),cv2.FONT_HERSHEY_SIMPLEX,.9,(250,250,250),2)
    cv2.putText(im,"OSM road geometry | inferred widths | 5 synthetic emergencies",(28,h-22),
                cv2.FONT_HERSHEY_SIMPLEX,.55,(220,220,220),1)
    cv2.imwrite(str(OUT/"route_overview.png"),im)


def flu_vec(p):
    # World local ENU -> fixed ClipGT FLU (x north, y west, z up).
    return {"x": float(p[1]), "y": float(-p[0]), "z": 0.}


def rail(xy):
    return [flu_vec(p) for p in xy]


def quat_for_tangent(tan):
    # ENU route heading vs fixed FLU x=north,y=west.
    yaw = math.atan2(-tan[0], tan[1])
    return {"x": 0., "y": 0., "z": math.sin(yaw/2), "w": math.cos(yaw/2)}


def export_clipgt(arr, route, cum, origin, roads, cutins):
    dest = OUT / "clipgt"
    dest.mkdir(exist_ok=True)
    sample_s = np.arange(0, cum[-1]+1, 2.)
    sample = np.asarray([at_s(route, cum, s)[0] for s in sample_s])
    tan = np.asarray([at_s(route, cum, s)[1] for s in sample_s])
    normal = np.column_stack((-tan[:,1], tan[:,0]))
    # Approximated lane/road geometry centered on the routed carriageway.
    camera_rows = []
    for name, az in CAMERAS.items():
        canon = {"front_wide":"camera:front:wide:120fov", "cross_left":"camera:cross:left:120fov",
                 "cross_right":"camera:cross:right:120fov", "rear_left":"camera:rear:left:70fov",
                 "rear_right":"camera:rear:right:70fov"}[name]
        # F-theta equidistant model: r = focal_pixels * theta. Horizontal
        # field of view matches the canonical camera name at 1280px width.
        hfov = 70 if name.startswith("rear") else 120
        focal = 640/math.radians(hfov/2)
        # Camera optical axis is local FLU +X, rotated about vehicle Z.
        camera_rows.append({"name": canon, "properties": {
            "polynomial": f"0 {focal:.6f} 0 0 0 0", "polynomial-type": "angle-to-pixeldistance",
            "cx": "640", "cy": "360", "width": "1280", "height": "720",
            "linear-c": "1", "linear-d": "0", "linear-e": "0"},
            "nominalSensor2Rig_FLU": {"t": [1.5, 0., 1.6], "roll-pitch-yaw": [0., 0., float(az)]}})
    rig = {"rig": {"sensors": camera_rows}}
    pd.DataFrame([{"calibration_estimate": {"rig_json": json.dumps(rig)}}]).to_parquet(
        dest/f"{CLIP}.calibration_estimate.parquet")
    ego_rows, obs_rows = [], []
    t0 = 1_700_000_000_000_000
    for i,r in enumerate(arr):
        t, es, ev, ea, ls, lv, la, gap, active = r
        ep, et = at_s(route,cum,es)
        lp, lt = at_s(route,cum,min(ls,cum[-1]))
        ts = int(t0+t*1e6)
        ego_rows.append({"key":{"timestamp_micros":ts},
                         "egomotion_estimate":{"location":flu_vec(ep),"orientation":quat_for_tangent(et)}})
        # Show the lead through the public-road route. Vehicle center has z=0.8m.
        if ls <= cum[-1]:
            c = flu_vec(lp); c["z"] = .8
            obs_rows.append({"key":{"timestamp_micros":ts},"obstacle":{
                "trackline_id":"lead_car_01","category":"car","center":c,
                "size":{"x":4.6,"y":1.9,"z":1.6},"orientation":quat_for_tangent(lt)}})
        for track,p,tan,unused_s in ambient_at(t,route,cum):
            ac=flu_vec(p);ac["z"]=.8
            obs_rows.append({"key":{"timestamp_micros":ts},"obstacle":{
                "trackline_id":track,"category":"car","center":ac,
                "size":{"x":4.5,"y":1.8,"z":1.6},"orientation":quat_for_tangent(tan)}})
        for track in cutins:
            pose=cutin_pose(track,i,route,cum)
            if pose is None:continue
            cp,ct=pose
            cc=flu_vec(cp);cc["z"]=.8
            obs_rows.append({"key":{"timestamp_micros":ts},"obstacle":{
                "trackline_id":track["id"],"category":"car","center":cc,
                "size":{"x":4.6,"y":1.9,"z":1.6},"orientation":quat_for_tangent(ct)}})
    pd.DataFrame(ego_rows).to_parquet(dest/f"{CLIP}.egomotion_estimate.parquet")
    pd.DataFrame(obs_rows).to_parquet(dest/f"{CLIP}.obstacle.parquet")
    lanes = []
    for offset in (-3.2, 0., 3.2):
        # Lane rail pair, expressed in fixed world FLU.
        lanes.append({"lane":{"left_rail":rail(sample+(offset+1.6)*normal),
                               "right_rail":rail(sample+(offset-1.6)*normal)}})
    pd.DataFrame(lanes).to_parquet(dest/f"{CLIP}.lane.parquet")
    lines = []
    for off in (-4.8, -1.6, 1.6, 4.8):
        lines.append({"lane_line":{"line_rail":rail(sample+off*normal),
                                   "colors":["WHITE"],
                                   "styles":["SOLID_SINGLE" if abs(off)>4 else "DASHED_SINGLE"]}})
    pd.DataFrame(lines).to_parquet(dest/f"{CLIP}.lane_line.parquet")
    boundaries = [{"road_boundary":{"location":rail(sample+off*normal)}} for off in (-6.4,6.4)]
    for road in roads:
        pts = road["xy"]
        if len(pts) < 2:
            continue
        tangents = np.gradient(pts, axis=0)
        norms = np.linalg.norm(tangents, axis=1)
        tangents = tangents/np.maximum(norms[:,None], 1e-6)
        side = np.column_stack((-tangents[:,1], tangents[:,0]))
        for sign in (-1,1):
            boundaries.append({"road_boundary":{"location":rail(pts+sign*road["width"]*.5*side)}})
    pd.DataFrame(boundaries).to_parquet(dest/f"{CLIP}.road_boundary.parquet")


def video_writer(path, size, fps=FPS):
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)


def top_frame(roads, route, cum, r, marks, cutins, frame_idx, w=1280, h=720):
    t, es, ev, ea, ls = r[:5]
    p, tan = at_s(route,cum,es)
    heading = math.atan2(tan[1],tan[0])
    ca, sa = math.cos(-heading), math.sin(-heading)
    mat = np.array([[ca,-sa],[sa,ca]])
    scale = 3.1
    def pix(xy):
        q=(np.asarray(xy)-p)@mat.T
        return np.int32(np.round(np.column_stack((w*.42+q[:,0]*scale,h*.5-q[:,1]*scale))))
    def draw_car(pos, direction, color, length=4.6, width=1.9):
        direction=np.asarray(direction,float)
        direction=direction/max(np.linalg.norm(direction),1e-9)
        side=np.array([-direction[1],direction[0]])
        corners=np.array([pos+length*.5*direction+width*.5*side,
                          pos+length*.5*direction-width*.5*side,
                          pos-length*.5*direction-width*.5*side,
                          pos-length*.5*direction+width*.5*side])
        pts=pix(corners)
        cv2.fillConvexPoly(im,pts,color,cv2.LINE_AA)
        cv2.polylines(im,[pts],True,(240,240,240),1,cv2.LINE_AA)
        nose=pix([pos+length*.45*direction])[0]
        cv2.circle(im,tuple(nose),2,(255,255,255),-1,cv2.LINE_AA)
    im=np.full((h,w,3),(28,34,34),np.uint8)
    for road in roads:
        a=road["xy"]
        if np.min(np.linalg.norm(a-p,axis=1))>240: continue
        cv2.polylines(im,[pix(a)],False,(60,64,65),max(2,int(road["width"]*scale)),cv2.LINE_AA)
        cv2.polylines(im,[pix(a)],False,(145,145,140),1,cv2.LINE_AA)
    rp=pix(route)
    cv2.polylines(im,[rp],False,(50,178,232),3,cv2.LINE_AA)
    for m in marks:
        mp,_=at_s(route,cum,m["ego_s_m"])
        uv=pix([mp])[0]
        if -20<uv[0]<w+20 and -20<uv[1]<h+20:
            cv2.circle(im,tuple(uv),9,(25,130,255),-1,cv2.LINE_AA)
            cv2.putText(im,m["id"],tuple(uv+np.array([12,-10])),cv2.FONT_HERSHEY_SIMPLEX,.6,(240,240,240),2)
    if ls<=cum[-1]:
        lp,lt=at_s(route,cum,ls)
        draw_car(lp,lt,(30,85,220))
    for track,ap,atan,as_ in ambient_at(t,route,cum):
        av=pix([ap])[0]
        if -20<av[0]<w+20 and -20<av[1]<h+20:
            draw_car(ap,atan,(220,150,80) if track.startswith("primary") else (160,100,205))
    for track in cutins:
        pose=cutin_pose(track,frame_idx,route,cum)
        if pose is None:continue
        cp,ct=pose;cv=pix([cp])[0]
        if -20<cv[0]<w+20 and -20<cv[1]<h+20:
            draw_car(cp,ct,(40,190,240))
    draw_car(p,tan,(40,240,90))
    cv2.putText(im,f"PKU EAST -> SOUTH | PUBLIC ROADS | {es:.0f}/{cum[-1]:.0f} m | {ev:.1f} m/s",(24,35),cv2.FONT_HERSHEY_SIMPLEX,.72,(240,240,240),2)
    active=[m for m in marks if 0<=t-m["time_s"]<4]
    if active:
        kind=active[-1].get("subtype",active[-1]["type"]).replace("_"," ").upper()
        cv2.putText(im,f"EMERGENCY {active[-1]['id']}: {kind}",(24,70),
                    cv2.FONT_HERSHEY_SIMPLEX,.85,(30,120,255),2)
    return im


def project_ground(points, ego, hdir, cam_az, w, h, z=0):
    a=np.deg2rad(cam_az); f=np.array([math.cos(a),math.sin(a)])
    l=np.array([-f[1],f[0]])
    forward=np.array([hdir[0]*f[0]-hdir[1]*f[1],hdir[0]*f[1]+hdir[1]*f[0]])
    left=np.array([-forward[1],forward[0]])
    d=np.asarray(points)-ego
    depth=d@forward
    lateral=d@left
    fx=w*.55; fy=h*.85
    u=w/2-fx*lateral/np.maximum(depth,.1)
    v=h*.43+fy*(1.6-z)/np.maximum(depth,.1)
    return np.column_stack((u,v)),depth


def camera_frame(roads, route, cum, r, marks, cutins, frame_idx, cam, w=960, h=540):
    t, es, ev, ea, ls=r[:5]
    ego, hdir=at_s(route,cum,es)
    im=np.full((h,w,3),(25,29,36),np.uint8)
    cv2.rectangle(im,(0,int(h*.43)),(w,h),(38,43,45),-1)
    az=CAMERAS[cam]
    segs=[]
    for road in roads:
        a=road["xy"]
        if np.min(np.linalg.norm(a-ego,axis=1))>130: continue
        uv,d=project_ground(a,ego,hdir,az,w,h)
        for i in range(len(a)-1):
            if d[i]>2 and d[i+1]>2 and min(d[i],d[i+1])<125:
                segs.append((max(d[i],d[i+1]),uv[i],uv[i+1],road["width"]))
    for _,a,b,width in sorted(segs,reverse=True,key=lambda x:x[0]):
        if max(abs(a[0]),abs(b[0]),abs(a[1]),abs(b[1]))>10000:continue
        thick=int(np.clip(w*.55*width/max(_,3),2,200))
        cv2.line(im,tuple(np.int32(a)),tuple(np.int32(b)),(72,77,80),thick,cv2.LINE_AA)
        cv2.line(im,tuple(np.int32(a)),tuple(np.int32(b)),(140,150,155),2,cv2.LINE_AA)
    # Route highlight is a semantic control cue.
    uv,d=project_ground(route,ego,hdir,az,w,h)
    for i in range(len(route)-1):
        if d[i]>2 and d[i+1]>2 and min(d[i],d[i+1])<125:
            cv2.line(im,tuple(np.int32(uv[i])),tuple(np.int32(uv[i+1])),(55,178,226),2,cv2.LINE_AA)
    lp,_=at_s(route,cum,min(ls,cum[-1]))
    uv,d=project_ground([lp],ego,hdir,az,w,h)
    if ls<=cum[-1] and 2<d[0]<110:
        cx,by=uv[0]; size=w*.55/d[0]
        bw,bh=int(np.clip(size*1.9,2,180)),int(np.clip(size*1.6,2,160))
        if -bw<cx<w+bw and -bh<by<h+bh:
            cv2.rectangle(im,(int(cx-bw/2),int(by-bh)),(int(cx+bw/2),int(by)),(25,82,224),-1)
            cv2.rectangle(im,(int(cx-bw*.27),int(by-bh*.8)),(int(cx+bw*.27),int(by-bh*.38)),(140,160,180),-1)
    for track,ap,atan,as_ in ambient_at(t,route,cum):
        au,ad=project_ground([ap],ego,hdir,az,w,h)
        if not (2<ad[0]<100):continue
        cx,by=au[0];size=w*.55/ad[0]
        bw,bh=int(np.clip(size*1.8,2,170)),int(np.clip(size*1.5,2,140))
        if -bw<cx<w+bw and -bh<by<h+bh:
            color=(220,150,80) if track.startswith("primary") else (160,100,205)
            cv2.rectangle(im,(int(cx-bw/2),int(by-bh)),(int(cx+bw/2),int(by)),color,-1)
    for track in cutins:
        pose=cutin_pose(track,frame_idx,route,cum)
        if pose is None:continue
        cp,_=pose
        au,ad=project_ground([cp],ego,hdir,az,w,h)
        if not (2<ad[0]<100):continue
        cx,by=au[0];size=w*.55/ad[0]
        bw,bh=int(np.clip(size*1.9,2,170)),int(np.clip(size*1.6,2,150))
        if -bw<cx<w+bw and -bh<by<h+bh:
            cv2.rectangle(im,(int(cx-bw/2),int(by-bh)),(int(cx+bw/2),int(by)),(40,190,240),-1)
    cv2.putText(im,cam.upper().replace("_"," "),(20,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(235,235,235),2)
    cv2.putText(im,f"{t:05.1f}s | s={es:.0f}m",(20,h-18),cv2.FONT_HERSHEY_SIMPLEX,.6,(230,230,230),2)
    active=[m for m in marks if 0<=t-m["time_s"]<4]
    if active: cv2.putText(im,f"{active[-1]['id']} EVENT",(w-195,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(35,125,250),2)
    return im


def render(limit=None, topdown_only=False):
    data=np.load(OUT/"scenario.npz")
    arr,route,cum,origin=(data[k] for k in ("frames","route","route_s","origin"))
    roads,_=map_model(origin)
    marks=read_json(OUT/"scenario.json")["emergencies"]
    cutins=[{"id":str(track_id),"s":data["cutin_s"][j],"lateral":data["cutin_lateral"][j]}
            for j,track_id in enumerate(data["cutin_ids"])]
    videos=OUT/"videos";videos.mkdir(exist_ok=True)
    writers={"topdown":video_writer(videos/"topdown.mp4",(1280,720))}
    if not topdown_only:
        for cam in CAMERAS:writers[cam]=video_writer(videos/f"{cam}.mp4",(960,540))
    n=min(len(arr),limit or len(arr))
    for i,r in enumerate(arr[:n]):
        top=top_frame(roads,route,cum,r,marks,cutins,i)
        writers["topdown"].write(top)
        if i==0:cv2.imwrite(str(OUT/"topdown_start.png"),top)
        if i==n//2:cv2.imwrite(str(OUT/"topdown_mid.png"),top)
        if not topdown_only:
            for cam in CAMERAS:
                frame=camera_frame(roads,route,cum,r,marks,cutins,i,cam)
                writers[cam].write(frame)
                if i==n//2:cv2.imwrite(str(OUT/f"{cam}_mid.png"),frame)
        if i%100==0:print(f"render {i}/{n}",flush=True)
    for wr in writers.values():wr.release()
    if limit is None and not topdown_only:
        inputs=[]
        for cam in CAMERAS: inputs += ["-i",str(videos/f"{cam}.mp4")]
        subprocess.run(["ffmpeg","-y",*inputs,"-filter_complex",
          "[0:v]scale=640:360[a];[1:v]scale=640:360[b];[2:v]scale=640:360[c];[3:v]scale=640:360[d];[4:v]scale=640:360[e];[a][b][c]hstack=inputs=3[top];[d][e]hstack=inputs=2[bottom];[bottom]pad=1920:360:320:0[bot];[top][bot]vstack=inputs=2[v]",
          "-map","[v]","-c:v","libx264","-preset","veryfast","-crf","27",str(videos/"five_view_mosaic.mp4")],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    print("Rendered",n,"frames",flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("command",choices=["build","render","all"])
    ap.add_argument("--limit",type=int)
    ap.add_argument("--topdown-only",action="store_true")
    a=ap.parse_args()
    if a.command in ("build","all"):make_scenario()
    if a.command in ("render","all"):render(a.limit,a.topdown_only)


if __name__=="__main__":main()
