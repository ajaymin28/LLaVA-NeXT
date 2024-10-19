import os
import json
import glob
from tqdm import tqdm
import random
import re


"""
V4 Changes

[X] Normlize BB
[X] Seperate List of objects and Relationship between the objects (experimental) 
    - [X] Reverting back to check the issues, results are not good (v5)
[X] Keep same questions for all conversation
[X] If BB is not found for any objects keep the list empty instead of passing zeros.
[X] Removed list of categories before providing summary
[X] Fix for multiple frames objects might appeare e.g adult.1_[0,10], adult.1_[30,60]
[X] Taken avg frame for gettig bb for an object e.g adult.1_[0,10] => 5th frame will be taken to get BB
    -issue: sometimes in the average frame the object is occluded which results in no BB for the object.

V5 Changes

[x] removed frame size in response.
[x] removed frame from the objects list
[x] removed Q&A pairs from pvsg dataset
[x] removed summary from the pvsg dataset
    
"""


data_root = '/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/'
with open(os.path.join(data_root, 'pvsg.json'), 'r') as f:
    anno = json.load(f)

print('Keys inside pvsg.json:', list(anno.keys()))
print('Number of Object Classes:', len(anno['objects']['thing']))
print('Number of Stuff Classes:', len(anno['objects']['stuff']))
print('Number of Relation Classes:', len(anno['relations']))

train_ids = anno["split"]['vidor']["train"]
val_ids = anno["split"]['vidor']["val"]

data = {data_dict['video_id']: data_dict for data_dict in anno['data']}


OUTPUT_JSON_DIR = "/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/video_llava_annotations_v5/"
JSON_Q_PATH = f"{OUTPUT_JSON_DIR}/opvsg_video_q.json"
JSON_A_PATH = f"{OUTPUT_JSON_DIR}/opvsg_video_a.json"
JSON_videochatgpt_tune = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_.json"
JSON_videochatgpt_tune_validate = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_validate.json"
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

video_questions_counter = {}
video_questions = []
video_answers = []
video_gpt_promptanswers = []
video_gpt_promptanswers_val = []
annot_cnt = {"train": 0, "val": 0}

prompts_list = {
    
    "summary": ["Describe the video in detail",
                "What is happening in the video?",
                "What is the central narrative or story in the video?",
                "What is the purpose or goal of the video?",
                "What are the key takeaways or lessons from the video?"
                ],

    "identify_subject_objects": [
                        "List the objects present in the video",
                        "What objects, items, or elements appear prominently?", 
                        "Identify any significant objects in the video.",
                        "What objects are visible in the video?",
                        "List the main objects featured in the video.",
                        "what are the main objects featured in the video?"
                        ],
    "identify_predicates": [
                            "List the actions, movements or placements of the objects in the scene.",
                            "Describe any interactions between people or objects in the video.",
                            "Describe any significant gestures or interactions between objects in the scene",
                            "How subjects and objects relates to each other in the video?",
                            "How do the objects interact with their environment in the video?",
                            "Describe any notable physical interactions between objects in the video.",
                            "Describe any interactions that highlight the relationships between objects.",
                            "What actions or events take place in the video?",
                          ]
}


import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import numpy as np

def getboundingBoxOftheObject(data_root, vid_id, frame_id, object_id, normlize_bb=True):
    mask_name = os.path.join(data_root, 'vidor', 'masks', vid_id, f'{str(frame_id).zfill(4)}.png')
    mask = Image.open(mask_name)
    mask = np.array(mask)

    segmentation = np.where(mask == object_id)
    mask_h, mask_w = mask.shape[0],mask.shape[1]
    # maskbb = np.zeros(shape=(mask_h,mask_w,3), dtype=np.uint8)

    # Bounding Box
    bbox = []
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

        if normlize_bb:
           x_min = round(x_min/mask_w,2)
           x_max = round(x_max/mask_w,2)
           y_min = round(y_min/mask_h,2)
           y_max = round(y_max/mask_h,2)

        bbox = [x_min, y_min, x_max, y_max]
        # print(bbox)
        # cv2.rectangle(maskbb, (x_min, y_min), (x_max, y_max), (36,255,12), 2)

    return bbox,[mask_h, mask_w]


def getRandomPrompt(key="summary", static=False):
    if static:
       return prompts_list[key][0]
    return random.choice(prompts_list[key])
    
def getQnACounter(vid_id):
    global video_questions_counter
    if vid_id not in video_questions_counter:
        video_questions_counter[vid_id] = 0
    else:
        video_questions_counter[vid_id] +=1
    qna_counter = video_questions_counter[vid_id]
    return qna_counter


def getFramesForObject(vid_data, Subject_id):
    vid_rels = vid_data["relations"]
    for idx, vid_r in enumerate(vid_rels):
        sub = vid_r[0]
        obj = vid_r[1]
        rel = vid_r[2]
        frames = vid_r[3]
        if sub==Subject_id:
            return frames
    return "None"



def getListofCategoryString(vid_objects, vid_data, addObjectId=False, addFrames=False, addBB=False):
    AnswerString = ""
    # category_counters = {}
    mask_size = None
    for idx, vobj in enumerate(vid_objects):
        category = vobj["category"]
        object_id = vobj["object_id"]
        frames = getFramesForObject(vid_data=vid_data, Subject_id=object_id)
        if frames!="None":
          AnswerString += f"{category}"
          if addObjectId:
            AnswerString += f".{object_id}"
          if addBB:
            sub_bb = []
            avg_frame = int((frames[0][0]+frames[0][1])/2)
            try:
              sub_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_data["video_id"],frame_id=avg_frame, object_id=object_id)
            except FileNotFoundError:
              sub_bb = []

            AnswerString += f".{sub_bb}"
          if addFrames:
            AnswerString += f".{frames}"

          if idx!=len(vid_objects)-1:
            AnswerString +="," 

    #AnswerString += f";image_height_width={mask_size}" 
    return AnswerString

def getObjectsRelations(vid_rels, vid_data):
    AnswerString = ""
    SubObjRel = []
    mask_size = None
    for idx, vid_r in enumerate(vid_rels):
        sub = vid_r[0]
        obj = vid_r[1]
        rel = vid_r[2]
        frames = vid_r[3]

        sub_bb = []
        obj_bb = []
        avg_frame = int((frames[0][0]+frames[0][1])/2)

        try:
          sub_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_data["video_id"],frame_id=avg_frame, object_id=sub)
        except FileNotFoundError:
          pass
        
        try:
          obj_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_data["video_id"],frame_id=avg_frame, object_id=obj)
        except FileNotFoundError:
          pass

        if [sub,rel, obj] not in SubObjRel:
            AnswerString += f"[{vid_objects_by_id[sub]['category']}.{sub}.{sub_bb},{rel},{vid_objects_by_id[obj]['category']}.{obj}.{obj_bb}_[{avg_frame}]]"
            #AnswerString += f"[{vid_objects_by_id[sub]['category']}.{sub}.{sub_bb},{rel},{vid_objects_by_id[obj]['category']}.{obj}].{obj_bb}]"
        if idx!=len(vid_rels)-1:
            AnswerString +=";"

    # AnswerString += f";image_height_width={mask_size}" #v5 removed 
    return AnswerString


def getVideoCaptions(vid_data, correct_object_ids=False):
    object_id_pattern_in_descr = r"\((\d+)\)"

    AnswerString = ""
    vid_caps = vid_data['captions']
    for idx, vid_c in enumerate(vid_caps):
        if correct_object_ids:
           """
           Converts adult (1)  ==> adult.1
           """
           vid_description = re.sub(object_id_pattern_in_descr, r".\1", vid_c["description"])
           vid_description = vid_description.replace(" .",".")
        else:
           vid_description = vid_c["description"]
           
        AnswerString += vid_description
        if idx!=len(vid_rels)-1:
            AnswerString +=","
    return AnswerString

def getVideoQandAPairs(vid_data, correct_object_ids=False):
    QnAPairs = []
    vid_qna = vid_data['qa_pairs']
    for idx, vid_qna in enumerate(vid_qna):
        # time_point = vid_qna["time"]
        Question = vid_qna["question"]
        Answer = vid_qna["answer"]

        if correct_object_ids:
           object_id_pattern_in_descr = r"\((\d+)\)"
           Question = re.sub(object_id_pattern_in_descr, r".\1", Question).replace(" .", ".")
           Answer = re.sub(object_id_pattern_in_descr, r".\1", Answer).replace(" .", ".")


        QnASeq = [{
          "from": "human",
          "value": f"<video>\n{Question}"
        },
        {
          "from": "gpt",
          "value": Answer
        }]
        QnAPairs.append(QnASeq)

    return QnAPairs

def getVideoSummary(vid_data):
    AnswerString = vid_data['summary']
    return AnswerString


def append_annotation(vid_id, annotation):
  global video_gpt_promptanswers, video_gpt_promptanswers_val, annot_cnt
  if vid_id in train_ids:
    annotation["id"] = annot_cnt["train"]
    video_gpt_promptanswers.append(annotation)
    annot_cnt["train"] +=1
  else:
    annotation["id"] = annot_cnt["val"]
    video_gpt_promptanswers_val.append(annotation)
    annot_cnt["val"] +=1
  

for vid_id, vid_data in tqdm(data.items()):
    video_path = f"{data_root}vidor/videos/{vid_id}.mp4"
    if not os.path.exists(video_path):
        continue

    vid_objects = vid_data["objects"] # {'object_id': 2, 'category': 'countertop', 'is_thing': True, 'status': []},
    vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in vid_objects} # for relations
    vid_rels = vid_data["relations"]
    
    # # Q&A for video detailed and short summary
    # PromptAnswer = {
    #   "id": annot_cnt,
    #   "video": video_path,
    #   "conversations": [
    #     {
    #       "from": "human",
    #       "value": f"<video>\n{getRandomPrompt(key='summary', static=True)}"
    #     },
    #     {
    #       "from": "gpt",
    #       "value": getVideoCaptions(vid_data,correct_object_ids=True) + f";In short, {getVideoSummary(vid_data)}"
    #     },
    #   ]
    # }


    # append_annotation(vid_data["video_id"],annotation=PromptAnswer)
    # video_gpt_promptanswers.append(PromptAnswer)
    # annot_cnt +=1

    # # Q&A from the pvsg dataset
    # VideoQnAPairs = getVideoQandAPairs(vid_data=vid_data, correct_object_ids=True)
    # for i,QnAP in enumerate(VideoQnAPairs):
    #     Q,A = QnAP
    #     PromptAnswerQnAPairs = {
    #       "id": annot_cnt,
    #       "video": video_path,
    #       "conversations": [Q,A]
    #     }
    #     append_annotation(vid_data["video_id"],annotation=PromptAnswerQnAPairs)
    #     # video_gpt_promptanswers.append(PromptAnswerQnAPairs)
    #     # annot_cnt +=1

    PromptAnswer = {
      "id": annot_cnt,
      "video": video_path,
      "conversations": [
        # Q&A for identifying objects in the scene
        {
          "from": "human",
          "value": f"<video>\n{getRandomPrompt(key='identify_subject_objects', static=True)}"
        },
        {
          "from": "gpt",
          "value": getListofCategoryString(vid_objects, vid_data, addObjectId=True,addBB=True,addFrames=False)
        },
        {
          "from": "human",
          "value": f"{getRandomPrompt(key='identify_predicates', static=True)}"
        },
        {
          "from": "gpt",
          "value": getObjectsRelations(vid_rels, vid_data)
        }
      ]
    }

    append_annotation(vid_data["video_id"],annotation=PromptAnswer)
    # video_gpt_promptanswers.append(PromptAnswer)
    # annot_cnt +=1

with open(JSON_videochatgpt_tune, "w") as f:
    json.dump(video_gpt_promptanswers,f)

with open(JSON_videochatgpt_tune_validate, "w") as f:
    json.dump(video_gpt_promptanswers_val,f)

print("Saved annotations", annot_cnt)