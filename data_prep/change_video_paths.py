annot_path = "/lustre/fs1/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/llava_annotations/video_llava_vidvrd_annotations_v7_withtime/"
new_annot_save_path = "/lustre/fs1/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/llava_annotations/video_llava_vidvrd_annotations_v7_withtime_newpath/"
new_video_path_root = "/lustre/fs1/home/jbhol/dso/gits/VRDFormer_VRD/"

import glob
import json
import os


os.makedirs(new_annot_save_path,exist_ok=True)

alljsons = glob.glob(os.path.join(annot_path,"*.json"))
for jsonfile in alljsons:
    with open(jsonfile, "r") as f:
        jsonData = json.loads(f.read())
        jsonfileName = jsonfile.split("/")[-1]

        for annotIdx, annot in enumerate(jsonData):
            # print("video path", annot["video"])
            videofileName= annot["video"].split("/")[-1]
            newvideo_path = os.path.join(new_video_path_root, videofileName)
            annot["video"] = newvideo_path
            jsonData[annotIdx] = annot

            # print("new video path", annot["video"])
        with open(os.path.join(new_annot_save_path, jsonfileName), "w") as f:
            json.dump(jsonData, f)

        print(f"Saved : {os.path.join(new_annot_save_path, jsonfileName)}")