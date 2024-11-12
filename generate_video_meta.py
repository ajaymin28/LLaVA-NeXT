import json
import os
from decord import VideoReader, cpu

def load_video(video_path):

    vr = VideoReader(video_path)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())

    return video_time, fps

root = "/groups/sernam/datasets/ActionGenome/ActionGenome/videos"

meta_json = {}

for file in os.listdir(root):
    if file.endswith(".mp4"):
        video_path = os.path.join(root, file)
        video_time, fps = load_video(video_path)
        meta_json[file] = {"video_time": video_time, "fps": fps}

with open("/groups/sernam/datasets/ActionGenome/ActionGenome/annotations/video_meta.json", "w") as f:
    json.dump(meta_json, f, indent=4)