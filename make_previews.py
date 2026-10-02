"""Make compact review images from the completed top-down render."""
import cv2
import numpy as np

from pku_pipeline import OUT, read_json


def snapshot(cap, time_s):
    cap.set(cv2.CAP_PROP_POS_MSEC, time_s*1000)
    ok, frame=cap.read()
    if not ok:
        raise RuntimeError(f"missing frame at {time_s:.1f}s")
    return cv2.resize(frame,(480,270),interpolation=cv2.INTER_AREA)


def main():
    scene=read_json(OUT/"scenario.json")
    data=np.load(OUT/"scenario.npz")
    arr=data["frames"]
    events={e["id"]:e for e in scene["emergencies"]}
    turn_t=float(arr[np.argmin(abs(arr[:,1]-scene["metrics"]["turn_center_s_m"])),0])
    times=[(10.,"Dense traffic"),
           (events["E2"]["time_s"]+1.5,"Left cut-in"),
           (events["E3"]["time_s"]+1.5,"Right cut-in"),
           (turn_t,"Smooth turn"),
           (events["E4"]["time_s"]+1.5,"Curbside pull-out"),
           (events["E5"]["time_s"]+1.5,"Slow close merge")]
    cap=cv2.VideoCapture(str(OUT/"videos"/"topdown.mp4"))
    tiles=[]
    for t,label in times:
        im=snapshot(cap,t)
        cv2.rectangle(im,(0,0),(480,35),(17,22,25),-1)
        cv2.putText(im,f"{label}  {t:.1f}s",(12,25),cv2.FONT_HERSHEY_SIMPLEX,
                    .65,(255,255,255),2,cv2.LINE_AA)
        tiles.append(im)
    cap.release()
    montage=np.vstack((np.hstack(tiles[:3]),np.hstack(tiles[3:])))
    cv2.imwrite(str(OUT/"scene_preview.jpg"),montage,[cv2.IMWRITE_JPEG_QUALITY,92])

    # Actual ego speed and curvature cap in a 140 m window around the turn.
    x0,x1=520.,660.
    w,h=1000,460
    im=np.full((h,w,3),(29,34,38),np.uint8)
    def pt(s,v):
        return (int(80+(s-x0)/(x1-x0)*865),int(390-v/11*330))
    for v in range(0,11,2):
        cv2.line(im,pt(x0,v),pt(x1,v),(60,65,68),1,cv2.LINE_AA)
        cv2.putText(im,str(v),(43,pt(x0,v)[1]+5),cv2.FONT_HERSHEY_SIMPLEX,.5,(210,210,210),1)
    for s in range(520,661,20):
        cv2.line(im,(pt(s,0)[0],390),(pt(s,10)[0],60),(55,60,63),1,cv2.LINE_AA)
        cv2.putText(im,str(s),(pt(s,0)[0]-14,420),cv2.FONT_HERSHEY_SIMPLEX,.5,(210,210,210),1)
    mask=(arr[:,1]>=x0)&(arr[:,1]<=x1)
    actual=np.array([pt(s,v) for s,v in arr[mask,1:3]],np.int32)
    cap_curve=np.array([pt(s,v) for s,v in zip(data["route_s"],data["speed_cap"])
                        if x0<=s<=x1],np.int32)
    cv2.polylines(im,[cap_curve],False,(230,175,74),3,cv2.LINE_AA)
    cv2.polylines(im,[actual],False,(79,214,247),3,cv2.LINE_AA)
    cv2.putText(im,"TURN SPEED PROFILE",(80,32),cv2.FONT_HERSHEY_SIMPLEX,.9,(245,245,245),2)
    cv2.putText(im,"ego speed",(650,32),cv2.FONT_HERSHEY_SIMPLEX,.55,(79,214,247),2)
    cv2.putText(im,"curve speed cap",(780,32),cv2.FONT_HERSHEY_SIMPLEX,.55,(230,175,74),2)
    cv2.putText(im,"distance along route (m)",(400,448),cv2.FONT_HERSHEY_SIMPLEX,.55,(220,220,220),1)
    cv2.imwrite(str(OUT/"turn_speed_profile.png"),im)


if __name__=="__main__":
    main()
