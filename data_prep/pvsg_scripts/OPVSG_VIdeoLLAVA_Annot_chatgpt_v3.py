import os
import json
import glob
from tqdm import tqdm
import random
import re


data_root = '/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/'
with open(os.path.join(data_root, 'pvsg.json'), 'r') as f:
    anno = json.load(f)

print('Keys inside pvsg.json:', list(anno.keys()))
print('Number of Object Classes:', len(anno['objects']['thing']))
print('Number of Stuff Classes:', len(anno['objects']['stuff']))
print('Number of Relation Classes:', len(anno['relations']))


data = {data_dict['video_id']: data_dict for data_dict in anno['data']}


OUTPUT_JSON_DIR = "/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/video_llava_annotations_v3/"
JSON_Q_PATH = f"{OUTPUT_JSON_DIR}/opvsg_video_q.json"
JSON_A_PATH = f"{OUTPUT_JSON_DIR}/opvsg_video_a.json"
JSON_videochatgpt_tune = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_.json"
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

video_questions_counter = {}
video_questions = []
video_answers = []
video_gpt_promptanswers = []





prompts_list = {
    
    "summary": ["Describe the video scene in detail",
                "What is happening in the video?",
                "What is the central narrative or story in the video?",
                "What is the purpose or goal of the video?",
                "What are the key takeaways or lessons from the video?"
                ],

    "identify_subject_objects": ["What objects, items, or elements appear prominently?", 
                        "Identify any significant objects in the video.",
                        "What objects are visible in the video?",
                        "List the main objects featured in the video.",
                        "what are the main objects featured in the video?"
                        ],

    "identify_predicates": ["Describe any interactions between people or objects in the video.",
                            "Describe any significant gestures or interactions between objects in the scene",
                            "How subjects and objects relates to each other in the video?",
                            "How do the objects interact with their environment in the video?",
                            "Describe any notable physical interactions between objects in the video.",
                            "Describe any interactions that highlight the relationships between objects.",
                            "Summarize the actions, movements or placements of the objects in the scene.",
                            "What actions or events take place in the video?",
                          ]
}


import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import numpy as np

def getboundingBoxOftheObject(data_root, vid_id, frame_id, object_id):
    mask_name = os.path.join(data_root, 'vidor', 'masks', vid_id, f'{str(frame_id).zfill(4)}.png')
    mask = Image.open(mask_name)
    mask = np.array(mask)

    segmentation = np.where(mask == object_id)
    mask_h, mask_w = mask.shape[0],mask.shape[1]
    # maskbb = np.zeros(shape=(mask_h,mask_w,3), dtype=np.uint8)

    # Bounding Box
    bbox = [0, 0, 0, 0]
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))
        bbox = [x_min, y_min, x_max, y_max]
        # print(bbox)
        # cv2.rectangle(maskbb, (x_min, y_min), (x_max, y_max), (36,255,12), 2)

    return bbox,[mask_h, mask_w]


def getRandomPrompt(key="summary"):
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
        frames = vid_r[-1][0]
        rel = vid_r[-2]
        sub = vid_r[0]
        obj = vid_r[1]
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
          sub_bb = []
          try:
            sub_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_data["video_id"],frame_id=frames[0], object_id=object_id)
          except FileNotFoundError:
            sub_bb = []

          AnswerString += f"{category}"
          if addObjectId:
            AnswerString += f".{object_id}"
          if addBB:
            AnswerString += f".{sub_bb}"
          if addFrames:
            AnswerString += f".{frames}"

          # AnswerString += f"{category}.{object_id}.{sub_bb}.{frames}"

          if idx!=len(vid_objects)-1:
            AnswerString +=","  

        # makes adult, adult ==> adult0, adult1
        # if category not in category_counters:category_counters[category] = 0
        # else: category_counters[category] +=1
        # cat_count = category_counters[category]
        # if cat_count==0:
    AnswerString += f";image_height_width={mask_size}" 
    return AnswerString

def getObjectsRelations(vid_rels, vid_data):
    AnswerString = ""
    SubObjRel = []
    mask_size = None
    for idx, vid_r in enumerate(vid_rels):
        frames = vid_r[-1][0]
        rel = vid_r[-2]
        sub = vid_r[0]
        obj = vid_r[1]

        sub_bb = []
        obj_bb = []

        try:
          sub_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_data["video_id"],frame_id=frames[0], object_id=sub)
        except FileNotFoundError:
          sub_bb = []
        
        try:
          obj_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_data["video_id"],frame_id=frames[0], object_id=obj)
        except FileNotFoundError:
          obj_bb = []

        if [sub,rel, obj] not in SubObjRel:
            AnswerString += f"[{vid_objects_by_id[sub]['category']}.{sub}.{sub_bb},{rel},{vid_objects_by_id[obj]['category']}.{obj}.{obj_bb}_{frames}]"
            #AnswerString += f"[{vid_objects_by_id[sub]['category']}.{sub}.{sub_bb},{rel},{vid_objects_by_id[obj]['category']}.{obj}].{obj_bb}]"
        if idx!=len(vid_rels)-1:
            AnswerString +=";"

    AnswerString += f";image_height_width={mask_size}" 
    return AnswerString


def getVideoCaptions(vid_data, correct_object_ids=False):
    object_id_pattern_in_descr = r"\((\d+)\)"

    AnswerString = ""
    vid_caps = vid_data['captions']
    for idx, vid_c in enumerate(vid_caps):
        
        if correct_object_ids:
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
    

annot_cnt = 0
for vid_id, vid_data in tqdm(data.items()):
    
    video_path = f"{data_root}vidor/videos/{vid_id}.mp4"
    if not os.path.exists(video_path):
        continue

    vid_objects = vid_data["objects"] # {'object_id': 2, 'category': 'countertop', 'is_thing': True, 'status': []},
    vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in vid_objects} # for relations
    vid_rels = vid_data["relations"]
    
    PromptAnswer = {
      "id": annot_cnt,
      "video": video_path,
      "conversations": [
        # Q&A for video detailed and short summary
        {
          "from": "human",
          "value": f"<video>\n{getRandomPrompt(key='summary')}"
        },
        {
          "from": "gpt",
          "value": f"{getListofCategoryString(vid_objects, vid_data, addObjectId=True)}\n" +  getVideoCaptions(vid_data,correct_object_ids=True) + f"\n In short, {getVideoSummary(vid_data)}"
        },
      ]
    }

    video_gpt_promptanswers.append(PromptAnswer)
    annot_cnt +=1

    # Q&A from the pvsg dataset
    VideoQnAPairs = getVideoQandAPairs(vid_data=vid_data, correct_object_ids=True)
    
    for i,QnAP in enumerate(VideoQnAPairs):
        Q,A = QnAP
        PromptAnswerQnAPairs = {
          "id": annot_cnt,
          "video": video_path,
          "conversations": [Q,A]
        }
        video_gpt_promptanswers.append(PromptAnswerQnAPairs)
        annot_cnt +=1

    PromptAnswer = {
      "id": annot_cnt,
      "video": video_path,
      "conversations": [
        # Q&A for identifying objects in the scene
        {
          "from": "human",
          "value": f"<video>\n{getRandomPrompt(key='identify_subject_objects')}"
        },
        {
          "from": "gpt",
          "value": getListofCategoryString(vid_objects, vid_data, addObjectId=True,addBB=True,addFrames=True)
        },
        # Q&A for relations between objects in the scene 
        {
          "from": "human",
          "value": f"{getRandomPrompt(key='identify_predicates')}"
        },
        {
          "from": "gpt",
          "value": getObjectsRelations(vid_rels, vid_data) + f";total_frames={vid_data['meta']['num_frames']}"
        }
      ]
    }


    video_gpt_promptanswers.append(PromptAnswer)
    annot_cnt +=1
    
    
print("Saved", annot_cnt, " annotations")

with open(JSON_videochatgpt_tune, "w") as f:
    json.dump(video_gpt_promptanswers,f)

