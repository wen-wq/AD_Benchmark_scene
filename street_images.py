"""Collect geotagged Commons photos; optionally fill 100 m stations with Baidu panoramas.

Commons photographs are not continuous street-view panoramas. Missing stations stay
explicitly missing. Set BAIDU_AK for panorama/v2 and use --baidu to request real
panoramas with WGS84 input coordinates.
"""
import argparse
import csv
import json
import math
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from pku_pipeline import OUT, ROOT, at_s, ll_to_xy, read_json, route_model, save_json

UA = "PKUWorldScenario/0.1 (internal research; contact: local operator)"
COMMONS = "https://commons.wikimedia.org/w/api.php"
SELECTED = {
    0: "File:East gate of Peking University (20250605110404).jpg",
    100: "File:Bank of Beijing Outside Peking U. East Gate, May 24, 2010 - panoramio.jpg",
    500: "File:West gate of Zhongguancun No.2 Primary School main campus (20210104134524).jpg",
    600: "File:Zhongguancun No.2 Primary School main campus (20210104134827).jpg",
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read(), r.headers.get("Content-Type", "")


def commons_info(title):
    qs = urllib.parse.urlencode({"action":"query","prop":"imageinfo|coordinates","titles":title,
         "iiprop":"url|extmetadata","iiurlwidth":"1024","format":"json"})
    obj=json.loads(get(COMMONS+"?"+qs)[0])
    page=next(iter(obj["query"]["pages"].values()))
    ii=page.get("imageinfo",[{}])[0]
    meta=ii.get("extmetadata",{})
    coord=page.get("coordinates",[{}])[0]
    return ii.get("thumburl",ii.get("url")), {
        "page_url":ii.get("descriptionurl"),"license":meta.get("LicenseShortName",{}).get("value"),
        "artist_html":meta.get("Artist",{}).get("value"),
        "date_html":meta.get("DateTimeOriginal",{}).get("value"),
        "photo_lon":coord.get("lon"),"photo_lat":coord.get("lat")}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baidu",action="store_true",help="Use BAIDU_AK for missing 100m stations")
    a=p.parse_args()
    origin,route,cum,_=route_model()
    out=OUT/"street_images";out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for s in range(0,int(cum[-1])+1,100):
        xy,_=at_s(route,cum,s)
        lon=origin[0]+xy[0]/(111320*np.cos(np.deg2rad(origin[1])))
        lat=origin[1]+xy[1]/111320
        row={"route_distance_m":s,"lon":round(lon,7),"lat":round(lat,7),
             "status":"missing","provider":None,"image":None}
        if s in SELECTED:
            title=SELECTED[s]
            try:
                url,meta=commons_info(title)
                if url:
                    data,ctype=get(url)
                    if ctype.startswith("image/"):
                        name=f"s{s:04d}_commons.jpg"
                        (out/name).write_bytes(data)
                        if meta["photo_lon"] is not None:
                            dx=(meta["photo_lon"]-lon)*111320*math.cos(math.radians(lat))
                            dy=(meta["photo_lat"]-lat)*111320
                            meta["distance_to_sample_m"]=round(math.hypot(dx,dy),1)
                        row.update({"status":"geotagged_photo","provider":"Wikimedia Commons",
                                    "image":name,"title":title,"image_url":url,**meta})
            except Exception as exc:
                row["error"]=str(exc)
        if a.baidu and row["status"]=="missing":
            ak=os.environ.get("BAIDU_AK")
            if not ak:raise RuntimeError("Set BAIDU_AK before --baidu")
            params={"width":1024,"height":512,"location":f"{lon:.7f},{lat:.7f}",
                    "coordtype":"wgs84ll","fov":90,"heading":0,"pitch":0,"ak":ak}
            url="https://api.map.baidu.com/panorama/v2?"+urllib.parse.urlencode(params)
            try:
                data,ctype=get(url)
                if not ctype.startswith("image/"):
                    raise ValueError(data[:200].decode("utf-8","replace"))
                name=f"s{s:04d}_baidu.jpg";(out/name).write_bytes(data)
                row.update({"status":"panorama","provider":"Baidu panorama/v2",
                            "image":name,"heading_deg":0,"fov_deg":90})
            except Exception as exc:row["error"]=str(exc)
        manifest.append(row)
        print(s,row["status"],row.get("error",""),flush=True)
    save_json(out/"manifest.json",{"sampling_interval_m":100,"stations":manifest,
        "note":"Photos are conditioning references, not route-aligned continuous street-view. Missing stations are explicitly missing."})
    with (out/"manifest.csv").open("w",newline="",encoding="utf-8-sig") as f:
        fields=("route_distance_m","lon","lat","status","provider","image",
                "distance_to_sample_m","page_url")
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(manifest)


if __name__=="__main__":main()
