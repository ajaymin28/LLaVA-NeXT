import glob
import json
import os
import argparse
from utils.utilities import chunk_list
import random

random.seed(145)

def main(annot_path, new_annot_save_path, new_video_path_root, shuffle_data=False):
    os.makedirs(new_annot_save_path, exist_ok=True)
    all_data = []
    alljsons = glob.glob(os.path.join(annot_path, "*.json"))
    for jsonfile in alljsons:
        with open(jsonfile, "r") as f:
            jsonData = json.loads(f.read())
            jsonfileName = jsonfile.split("/")[-1]
            for annotIdx, annot in enumerate(jsonData):
                videofileName = annot["video"].split("/")[-1]
                newvideo_path = os.path.join(new_video_path_root, videofileName)
                annot["video"] = newvideo_path
                if shuffle_data:
                    all_data.append(annot)
                else:
                    jsonData[annotIdx] = annot
            if not shuffle_data:
                with open(os.path.join(new_annot_save_path, jsonfileName), "w") as f:
                    json.dump(jsonData, f)
                print(f"Saved : {os.path.join(new_annot_save_path, jsonfileName)}")

    if shuffle_data:
        random.shuffle(all_data)
        chunkedData = chunk_list(list_=all_data,chunk_n=1000)
        for chunk_idx,chunk in enumerate(chunkedData):
            shuffled_data_jsonName = f"videochatgpt_tune_part{chunk_idx}.json"

            with open(os.path.join(new_annot_save_path, shuffled_data_jsonName), "w") as f:
                json.dump(chunk, f)
            print(f"Saved : {os.path.join(new_annot_save_path, shuffled_data_jsonName)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video annotations and update paths.")
    parser.add_argument('--annot_path', type=str, required=True, help='Path to the annotation JSON files.')
    parser.add_argument('--new_annot_save_path', type=str, required=True, help='Path to save the updated annotation JSON files.')
    parser.add_argument('--new_video_path_root', type=str, required=True, help='Root path of the new video files.')
    parser.add_argument('--shuffle_data', action='store_true', help='Shuffle data while saving (default: False)')

    args = parser.parse_args()

    main(args.annot_path, args.new_annot_save_path, args.new_video_path_root, args.shuffle_data)



# import glob
# import json
# import os
# if __name__=="__main__":
#     annot_path = "/lustre/fs1/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/llava_annotations/video_llava_vidvrd_annotations_v7_withtime/"
#     new_annot_save_path = "/lustre/fs1/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/llava_annotations/video_llava_vidvrd_annotations_v7_withtime_newpath/"
#     new_video_path_root = "/lustre/fs1/home/jbhol/dso/gits/VRDFormer_VRD/"

#     os.makedirs(new_annot_save_path,exist_ok=True)

#     alljsons = glob.glob(os.path.join(annot_path,"*.json"))
#     for jsonfile in alljsons:
#         with open(jsonfile, "r") as f:
#             jsonData = json.loads(f.read())
#             jsonfileName = jsonfile.split("/")[-1]

#             for annotIdx, annot in enumerate(jsonData):
#                 # print("video path", annot["video"])
#                 videofileName= annot["video"].split("/")[-1]
#                 newvideo_path = os.path.join(new_video_path_root, videofileName)
#                 annot["video"] = newvideo_path
#                 jsonData[annotIdx] = annot

#                 # print("new video path", annot["video"])
#             with open(os.path.join(new_annot_save_path, jsonfileName), "w") as f:
#                 json.dump(jsonData, f)

#             print(f"Saved : {os.path.join(new_annot_save_path, jsonfileName)}")