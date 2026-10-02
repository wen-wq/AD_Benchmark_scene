"""Sample route-aligned KartaView driving photos every 25 m where coverage exists.

This pinned sequence was discovered by querying the public nearby-photo API.
The photos are planar, forward-facing stills, not 360-degree panoramas.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from pku_pipeline import OUT, ll_to_xy, read_json, save_json

SEQUENCE_ID="10464985"
SEQUENCE_PAGE=4
OUT_DIR=OUT/"street_images"
KV_DIR=OUT_DIR/"kartaview"
API=f"https://api.openstreetcam.org/2.0/photo/?sequenceId={SEQUENCE_ID}&page={SEQUENCE_PAGE}&itemsPerPage=150"
UA="PKUWorldScenario/0.2 (internal research)"


def get(url):
    req=urllib.request.Request(url,headers={"User-Agent":UA})
    with urllib.request.urlopen(req,timeout=30) as response:
        return response.read(),response.headers.get("Content-Type","")


def heading_diff(a,b):
    return abs((a-b+180)%360-180)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    data=np.load(OUT/"scenario.npz")
    route,cum,origin=(data[k] for k in ("route","route_s","origin"))
    photo_data=json.loads(get(API)[0])["result"]["data"]
    candidates=[]
    for photo in photo_data:
        if photo.get("projection")!="PLANE" or not photo.get("fileurlProc"):
            continue
        lat,lon,heading=(float(photo[k]) for k in ("lat","lng","heading"))
        xy=ll_to_xy([[lon,lat]],origin)[0]
        idx=int(np.argmin(np.sum((route-xy)**2,axis=1)))
        cross=float(np.linalg.norm(route[idx]-xy))
        a=route[min(idx+1,len(route)-1)]-route[max(idx-1,0)]
        route_heading=math.degrees(math.atan2(a[0],a[1]))%360
        err=heading_diff(heading,route_heading)
        if cross>25 or err>50 or cum[idx]>650:
            continue
        candidates.append({"id":str(photo["id"]),"sequence_id":SEQUENCE_ID,
            "sequence_index":int(photo["sequenceIndex"]),"route_s_m":round(float(cum[idx]),1),
            "cross_track_m":round(cross,1),"heading_error_deg":round(err,1),
            "photo_lat":lat,"photo_lon":lon,"heading_deg":heading,
            "shot_date":photo.get("shotDate"),"projection":"PLANE",
            "image_url":photo["fileurlProc"],"api_url":f"https://api.openstreetcam.org/2.0/photo/{photo['id']}",
            "license":"CC BY-SA 4.0","attribution":"© Grab and KartaView Contributors"})
    stations=[]
    used=set()
    for station in range(0,int(cum[-1])+1,25):
        choices=[x for x in candidates if x["id"] not in used and
                 abs(x["route_s_m"]-station)<=16]
        if choices:
            chosen=min(choices,key=lambda x:abs(x["route_s_m"]-station)+
                       .4*x["cross_track_m"]+.08*x["heading_error_deg"])
            used.add(chosen["id"])
            row={"route_distance_m":station,"status":"available",**chosen,
                 "image":f"kartaview/s{station:04d}.jpg"}
        else:
            row={"route_distance_m":station,"status":"missing","image":None}
        stations.append(row)
    available=[x for x in stations if x["status"]=="available"]
    print("aligned_candidates",len(candidates),"selected_25m_frames",len(available),
          "covered_stations",[x["route_distance_m"] for x in available],flush=True)
    if args.dry_run:return
    KV_DIR.mkdir(parents=True,exist_ok=True)
    for row in available:
        dest=OUT_DIR/row["image"]
        if dest.exists() and dest.stat().st_size>10000:
            continue
        blob,ctype=get(row["image_url"])
        im=cv2.imdecode(np.frombuffer(blob,np.uint8),cv2.IMREAD_COLOR)
        if not ctype.startswith("image/") or im is None or im.shape[1]<1000:
            raise RuntimeError(f"bad image for {row['id']}: {ctype}, {len(blob)} bytes")
        dest.write_bytes(blob)
        row["width_px"],row["height_px"]=int(im.shape[1]),int(im.shape[0])
        print(row["route_distance_m"],row["id"],row["cross_track_m"],flush=True)
    # Existing cached photos also need dimensions in the manifest.
    for row in available:
        if "width_px" not in row:
            im=cv2.imread(str(OUT_DIR/row["image"]))
            row["width_px"],row["height_px"]=int(im.shape[1]),int(im.shape[0])
    save_json(OUT_DIR/"kartaview_manifest.json",{
        "sampling_interval_m":25,"sequence_id":SEQUENCE_ID,
        "photo_type":"forward-facing planar driving frame, not panorama",
        "source_api":API,"stations":stations,"available_count":len(available),
        "license":"CC BY-SA 4.0","attribution":"© Grab and KartaView Contributors"})
    base=read_json(OUT_DIR/"manifest.json")
    by_s={x["route_distance_m"]:x for x in available}
    for row in base["stations"]:
        extra=by_s.get(row["route_distance_m"])
        if extra is None:continue
        if row["status"]=="missing":
            row.update({"status":"street_level_photo","provider":"KartaView",
                "image":extra["image"],"image_url":extra["image_url"],
                "page_url":extra["api_url"],"photo_lon":extra["photo_lon"],
                "photo_lat":extra["photo_lat"],"heading_deg":extra["heading_deg"],
                "distance_to_sample_m":extra["cross_track_m"],
                "license":extra["license"],"shot_date":extra["shot_date"]})
        else:
            row["additional_kartaview_reference"]=extra["image"]
    base["note"]=("Commons images are ordinary geotagged photos. KartaView images are "
                  "route-aligned forward-facing planar driving frames, not 360 panoramas. "
                  "Missing stations have no verified image within the alignment thresholds.")
    save_json(OUT_DIR/"manifest.json",base)
    with (OUT_DIR/"manifest.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=("route_distance_m","lon","lat","status","provider","image",
                "distance_to_sample_m","heading_deg","page_url","additional_kartaview_reference")
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(base["stations"])
    thumbs=[]
    for row in available:
        im=cv2.imread(str(OUT_DIR/row["image"]))
        im=cv2.resize(im,(320,240),interpolation=cv2.INTER_AREA)
        cv2.rectangle(im,(0,0),(320,32),(20,24,27),-1)
        cv2.putText(im,f"s={row['route_distance_m']} m",(10,23),
                    cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
        thumbs.append(im)
    if thumbs:
        blank=np.full_like(thumbs[0],(20,24,27))
        rows=[]
        for i in range(0,len(thumbs),5):
            group=thumbs[i:i+5]
            rows.append(np.hstack(group+[blank]*(5-len(group))))
        sheet=np.vstack(rows)
        cv2.imwrite(str(OUT_DIR/"kartaview_contact_sheet.jpg"),sheet,
                    [cv2.IMWRITE_JPEG_QUALITY,88])
    print("100m_coverage",sum(x["status"]!="missing" for x in base["stations"]),
          "/",len(base["stations"]),flush=True)


if __name__=="__main__":main()
