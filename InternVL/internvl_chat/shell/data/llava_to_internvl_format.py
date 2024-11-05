import json
import os

annotations = os.listdir("/home/scui/projects/LLaVA-NeXT/data_prep/data/AG_llava_annotations_v5_3")

for annotation in annotations:
    with open(f"/home/scui/projects/LLaVA-NeXT/data_prep/data/AG_llava_annotations_v5_3/{annotation}", "r") as f:
        data = json.load(f)
    for i in range(len(data)):
        for j in range(len(data[i]["conversations"])):
            if data[i]['conversations'][j]['from'] == 'human':
                data[i]['conversations'][j]['value'] = data[i]['conversations'][j]['value'].replace("<video>\n\n      ", "<video>\n")
    
    with open(f"/home/scui/projects/LLaVA-NeXT/InternVL/internvl_chat/shell/data/ActionGenome/AG_internvl_annotations_v5_3/{annotation}l", "w") as f:
        # save in jsonl format

        for i in range(len(data)):
            f.write(json.dumps(data[i]) + "\n")
