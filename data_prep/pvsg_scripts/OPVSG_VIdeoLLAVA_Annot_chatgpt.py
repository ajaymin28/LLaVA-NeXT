import os
import json
import glob
from tqdm import tqdm


data_root = '/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/'
with open(os.path.join(data_root, 'pvsg.json'), 'r') as f:
    anno = json.load(f)

print('Keys inside pvsg.json:', list(anno.keys()))
print('Number of Object Classes:', len(anno['objects']['thing']))
print('Number of Stuff Classes:', len(anno['objects']['stuff']))
print('Number of Relation Classes:', len(anno['relations']))


data = {data_dict['video_id']: data_dict for data_dict in anno['data']}


OUTPUT_JSON_DIR = "/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/video_llava_annotations/"
JSON_Q_PATH = f"{OUTPUT_JSON_DIR}/opvsg_video_q.json"
JSON_A_PATH = f"{OUTPUT_JSON_DIR}/opvsg_video_a.json"
JSON_videochatgpt_tune = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_.json"
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
"""
Question: {
    "video_name": "tumblr_nk172bbdPI1u1lr18o1_250",
    "question_id": "v_tumblr_nk172bbdPI1u1lr18o1_250_0",
    "question": "What does the butterfly do 10 or more than 10 times ?"
},
Answer: {
    "answer": "flap wings",
    "type": 3,
    "question_id": "v_tumblr_nk172bbdPI1u1lr18o1_250_0"
},
"""

video_questions_counter = {}

video_questions = []
video_answers = []
video_gpt_promptanswers = []


def getQnACounter(vid_id):
    global video_questions_counter
    if vid_id not in video_questions_counter:
        video_questions_counter[vid_id] = 0
    else:
        video_questions_counter[vid_id] +=1
    qna_counter = video_questions_counter[vid_id]
    return qna_counter


def getListofCategoryString(vid_objects):
    AnswerString = ""
    category_counters = {}
    for idx, vobj in enumerate(vid_objects):
        category = vobj["category"]
        # makes adult, adult ==> adult0, adult1
        if category not in category_counters:category_counters[category] = 0
        else: category_counters[category] +=1
        cat_count = category_counters[category]
        AnswerString += f"{category}"
        if idx!=len(vid_objects)-1:
            AnswerString +=","  
    return AnswerString
    

def getObjectsRelations(vid_rels):
    AnswerString = ""
    SubObjRel = []
    for idx, vid_r in enumerate(vid_rels):
        frames = vid_r[-1][0]
        rel = vid_r[-2]
        sub = vid_r[0]
        obj = vid_r[1]
        
        if [sub,rel, obj] not in SubObjRel:
            AnswerString += f"[{vid_objects_by_id[sub]['category']},{rel},{vid_objects_by_id[obj]['category']}]"
        if idx!=len(vid_rels)-1:
            AnswerString +=";"
            
    return AnswerString
    
        


annot_cnt = 0
for vid_id, vid_data in tqdm(data.items()):
    #print("processing", vid_id)
    
    
    video_path = f"/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/vidor/videos/{vid_id}.mp4"
    if not os.path.exists(f"/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/vidor/videos/{vid_id}.mp4"):
        continue

    
    """
    Q&A for identifying objects in the scene
    """

    vid_objects = vid_data["objects"] # {'object_id': 2, 'category': 'countertop', 'is_thing': True, 'status': []},
    vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in vid_objects} # for relations
    
    #print(vid_objects_by_id)
    
    vid_rels = vid_data["relations"]
    
    PromptAnswer = {
      "id": annot_cnt,
      "video": video_path,
      "conversations": [
        {
          "from": "human",
          "value": "<video>\nList the objects present in the video."
        },
        {
          "from": "gpt",
          "value": getListofCategoryString(vid_objects)
        }
      ]
    }
    
    
    video_gpt_promptanswers.append(PromptAnswer)
    

    """
    Q&A for relations between objects in the scene
    """
    
    annot_cnt +=1
    
    
    PromptAnswer = {
      "id": annot_cnt,
      "video": video_path,
      "conversations": [
        {
          "from": "human",
          "value": "<video>\nIdentify the relationships between the objects in the scene."
        },
        {
          "from": "gpt",
          "value": getObjectsRelations(vid_rels)
        }
      ]
    }
    
    video_gpt_promptanswers.append(PromptAnswer)
    
    annot_cnt +=1
    
    
print("Saved", annot_cnt, " annotations")

with open(JSON_videochatgpt_tune, "w") as f:
    json.dump(video_gpt_promptanswers,f)

