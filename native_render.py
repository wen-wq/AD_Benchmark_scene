"""Render the generated ClipGT with the installed NVIDIA Cosmos WSV renderer."""
import argparse
from pathlib import Path

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_loaders import load_scene
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.scripts.local import (
    convert_scene_data_for_rendering, render_multi_camera_tiled)

ROOT = Path(__file__).resolve().parent
CAMERAS = ["camera_front_wide_120fov", "camera_cross_left_120fov",
           "camera_cross_right_120fov", "camera_rear_left_70fov",
           "camera_rear_right_70fov"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-frames", type=int, default=-1)
    p.add_argument("--cameras", default=",".join(CAMERAS))
    a = p.parse_args()
    cams = a.cameras.split(",")
    inp = ROOT/"output"/"clipgt"
    scene = load_scene(inp, camera_names=cams, max_frames=a.max_frames,
                       input_pose_fps=30, resize_resolution_hw=(720, 1280))
    models, poses = convert_scene_data_for_rendering(scene, cams, (720, 1280))
    render_multi_camera_tiled(models, poses, scene, cams,
        str(ROOT/"output"/"native_wsv"), scene.scene_id,
        max_frames=a.max_frames, chunk_output=False, overlay_camera=False,
        alpha=.5, clipgt_path=inp, use_persistent_vbos=True,
        multi_sample=1, simplified_output=True)


if __name__ == "__main__":
    main()
