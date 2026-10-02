"""Find overlapping 2D vehicle rectangles in the generated timeline."""
import json
from collections import Counter
import numpy as np
from shapely.geometry import Polygon

from pku_pipeline import OUT, ambient_at, at_s, cutin_pose, save_json

data=np.load(OUT/"scenario.npz")
arr,route,cum=(data[k] for k in ("frames","route","route_s"))
cutins=[{"id":str(track_id),"s":data["cutin_s"][j],"lateral":data["cutin_lateral"][j]}
        for j,track_id in enumerate(data["cutin_ids"])]

def box(p,t):
    side=np.array([-t[1],t[0]])
    return Polygon([p+2.3*t+.95*side,p+2.3*t-.95*side,
                    p-2.3*t-.95*side,p-2.3*t+.95*side])

hits=[]
for i,r in enumerate(arr):
    t,es,_,_,ls=r[:5]
    actors=[("ego",*at_s(route,cum,es))]
    if ls<=cum[-1]:actors.append(("lead",*at_s(route,cum,ls)))
    actors.extend((name,p,tan) for name,p,tan,s in ambient_at(t,route,cum))
    for track in cutins:
        pose=cutin_pose(track,i,route,cum)
        if pose is not None:actors.append((track["id"],*pose))
    pos=np.asarray([x[1] for x in actors])
    pairs=np.column_stack(np.triu_indices(len(actors),1))
    distances=np.linalg.norm(pos[pairs[:,0]]-pos[pairs[:,1]],axis=1)
    for (j,k),dist in zip(pairs[distances<6],distances[distances<6]):
        if box(pos[j],actors[j][2]).intersects(box(pos[k],actors[k][2])):
            hits.append((round(float(t),1),actors[j][0],actors[k][0],round(float(dist),2)))
scene=json.load(open(OUT/"scenario.json"))
cutin_metrics=[]
for track in cutins:
    mark=next(m for m in scene["emergencies"] if m["id"]==track["id"].split("_")[-1])
    mask=(arr[:,0]>=mark["time_s"])&(arr[:,0]<mark["time_s"]+6)&(abs(track["lateral"])<1.)
    cutin_metrics.append({"id":track["id"],
        "min_lane_gap_m":round(float(np.min(track["s"][mask]-arr[mask,1]-4.6)),2),
        "max_lateral_step_m":round(float(np.max(abs(np.diff(track["lateral"])))),3)})
report={"overlap_frames":len(hits),"first_hits":hits[:20],
        "overlap_pairs":[{"actors":list(pair),"frames":count} for pair,count in
                         Counter(tuple(sorted(x[1:3])) for x in hits).most_common()],
        "cutin_metrics":cutin_metrics}
save_json(OUT/"collision_report.json",report)
print(json.dumps(report,indent=2))
if hits:raise SystemExit(1)
