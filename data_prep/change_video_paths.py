import glob
import json
import os
import argparse

def main(annot_path, new_annot_save_path, new_video_path_root):
    os.makedirs(new_annot_save_path, exist_ok=True)

    alljsons = glob.glob(os.path.join(annot_path, "*.json"))
    for jsonfile in alljsons:
        with open(jsonfile, "r") as f:
            jsonData = json.loads(f.read())
            jsonfileName = jsonfile.split("/")[-1]

            for annotIdx, annot in enumerate(jsonData):
                videofileName = annot["video"].split("/")[-1]
                newvideo_path = os.path.join(new_video_path_root, videofileName)
                annot["video"] = newvideo_path
                jsonData[annotIdx] = annot

            with open(os.path.join(new_annot_save_path, jsonfileName), "w") as f:
                json.dump(jsonData, f)

            print(f"Saved : {os.path.join(new_annot_save_path, jsonfileName)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video annotations and update paths.")
    parser.add_argument('--annot_path', type=str, required=True, help='Path to the annotation JSON files.')
    parser.add_argument('--new_annot_save_path', type=str, required=True, help='Path to save the updated annotation JSON files.')
    parser.add_argument('--new_video_path_root', type=str, required=True, help='Root path of the new video files.')

    args = parser.parse_args()

    main(args.annot_path, args.new_annot_save_path, args.new_video_path_root)



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