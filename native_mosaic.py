"""Combine the five native WSV camera videos into one labelled preview."""
import subprocess

from pku_pipeline import CLIP, OUT


def main():
    base=OUT/"native_wsv"/CLIP
    camera_names=["camera_front_wide_120fov","camera_cross_left_120fov",
                  "camera_cross_right_120fov","camera_rear_left_70fov",
                  "camera_rear_right_70fov"]
    labels=["FRONT WIDE","CROSS LEFT","CROSS RIGHT","REAR LEFT","REAR RIGHT"]
    inputs=[]
    for name in camera_names:
        p=base/f"{CLIP}.{name}.mp4"
        if not p.exists():raise FileNotFoundError(p)
        inputs.extend(["-i",str(p)])
    branches=[]
    for i,label in enumerate(labels):
        branches.append(f"[{i}:v]scale=640:360,drawtext=text='{label}':x=14:y=13:"
                        f"fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6[{chr(97+i)}]")
    graph=";".join(branches+["[a][b][c]hstack=inputs=3[top]",
                             "[d][e]hstack=inputs=2,pad=1920:360:320:0[bottom]",
                             "[top][bottom]vstack=inputs=2[v]"])
    dest=OUT/"videos"/"five_view_native_mosaic.mp4"
    cmd=["ffmpeg","-y",*inputs,"-filter_complex",graph,"-map","[v]",
         "-c:v","libx264","-preset","veryfast","-crf","26","-pix_fmt","yuv420p",
         str(dest)]
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    print(dest,dest.stat().st_size,"bytes")


if __name__=="__main__":main()
