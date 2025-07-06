import json
import os

# INPUT_ANNOTATIONS_PATH = "/home/jbhol/dso/gits/LLaVA-NeXT/data_prep/data/video_llava_vidvrd_annotations_v5_3_shuffled"
# OUTPUT_DIR = "/home/jbhol/dso/gits/LLaVA-NeXT/InternVL/internvl_chat/shell/data/VRD/video_llava_vidvrd_annotations_v5_3_shuffled"
# annotations = os.listdir(f"{INPUT_ANNOTATIONS_PATH}")

INPUT_ANNOTATIONS_PATH = "/home/jbhol/dso/gits/LLaVA-NeXT/InternVL/internvl_chat/shell/data/ActionGenome/Sam"
OUTPUT_DIR = "/home/jbhol/dso/gits/LLaVA-NeXT/InternVL/internvl_chat/shell/data/ActionGenome/Sam_jsonl"
annotations = os.listdir(f"{INPUT_ANNOTATIONS_PATH}")

os.makedirs(OUTPUT_DIR,exist_ok=True)

for annotation in annotations:
    print(f"{INPUT_ANNOTATIONS_PATH}/{annotation}")
    with open(f"{INPUT_ANNOTATIONS_PATH}/{annotation}", "r") as f:
        data = json.load(f)
    for i in range(len(data)):
        for j in range(len(data[i]["conversations"])):
            if data[i]['conversations'][j]['from'] == 'human':
                data[i]['conversations'][j]['value'] = data[i]['conversations'][j]['value'].replace("<video>\n\n      ", "<video>\n")
    
    with open(f"{OUTPUT_DIR}/{annotation}l", "w") as f:
        # save in jsonl format
        for i in range(len(data)):
            f.write(json.dumps(data[i]) + "\n")