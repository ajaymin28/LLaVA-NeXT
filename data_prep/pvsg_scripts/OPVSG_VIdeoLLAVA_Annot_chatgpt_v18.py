import os
import json
import glob
from tqdm import tqdm
import random
import re
import threading
import time
import copy
from utils.utilities import create_batch_frames

"""
TODO
"""

class SGSpecialTokens:
    VIDEO_FRAME_ID = "#frameid"
    # SG_START = "#sg"
    SG_END = "#sgend"
    SG_SUBJECT = "#subject"
    SG_SUBJECT_ID = "#subid"
    SG_OBJECT = "#object"
    SG_OBJECT_ID = "#objid"
    SG_PREDICATE = "#sgpred"
    SG_BB_START = "#sgbb"
    SG_BB_END = "#sgbbend"
    SG_BB_X1Y1 = "#bbx1y1"
    SG_BB_X2Y2 = "#bbx2y2"
    # SG_BB_X1 = "#sgx1"
    # SG_BB_X2 = "#sgx2"
    # SG_BB_Y1 = "#sgy1"
    # SG_BB_Y2 = "#sgy2"

    @staticmethod
    def get_tokens():
        members = [attr for attr in dir(SGSpecialTokens) if not callable(getattr(SGSpecialTokens, attr)) and not attr.startswith("__")]
        tokens = []
        for mem in members:
            tokens.append(SGSpecialTokens.__getattribute__(SGSpecialTokens,mem))
        return tokens

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
                          ],
    "SGG": [
       "Generate frame-by-frame scene graph for the provided video",
       "Provide frame-by-frame Scene graph triplets in the form of [Subject-id:Predicate:Object-id]",
       "Generate scene graph for the provided video",
       "Provide scene graph for the provided video",
       "Identify subjects, predicates and objects frame-by-frame in the provided video"
    ],

    "SGG_image": [
       "Generate scene graph for the provided image",
       "Provide Scene graph triplets in the form of [Subject-id:Predicate:Object-id] for the provided image",
       "Generate scene graph for the provided image",
       "Provide scene graph for the provided image",
       "Identify subjects, predicates and objects in the provided image"
    ],

    "SGG_with_bb": [
       "Generate frame-by-frame scene graph for the provided video along with bounding box of each objects",
       "Provide frame-by-frame Scene graph triplets in the form of [Subject-id-[min_x,min_y,max_x,max_y]:Predicate:Object-id-[min_x,min_y,max_x,max_y]]",
       "Generate scene graph for the provided video along with bounding box of each objects",
       "Provide scene graph for the provided video with bounding box location of each objects",
       "Identify Subjects, Predicates and Objects frame-by-frame in the provided video, also provide bounding box location of each subject and object"
    ],

    "sg_localization": [
      #  "Provide bounding box location of [{sub}:{rel}:{obj}] in frame {frame_idx} of the provided video" # {} to be replaced by actual value
      "Provide bounding box location of [{sub}:{rel}:{obj}]" # {} to be replaced by actual value
    ],

    "sg_localization_image": [
      "Provide bounding box location of [{sub}:{rel}:{obj}]" # {} to be replaced by actual value
    ]


}


# import matplotlib.pyplot as plt
# import cv2
from PIL import Image
import numpy as np

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

def getboundingBoxOftheObject(data_root, vid_id, frame_id, object_id, normlize_bb=True, dataset="vidor"):
    mask_name = os.path.join(data_root, dataset, 'masks', vid_id, f'{str(frame_id).zfill(4)}.png')
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
           x_min = round(x_min/mask_w,3)
           x_max = round(x_max/mask_w,3)
           y_min = round(y_min/mask_h,3)
           y_max = round(y_max/mask_h,3)

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
        # rel = vid_r[2]
        frames_ = vid_r[3].copy()
        if Subject_id==sub or Subject_id==obj:
            return frames_
    return "None"
   

def get_frame_range_for_annotations(vid_objects, vid_data,):
  min_frame_idx, max_frame_idx = -1, 0
  frames_for_obj = {}
  for vid_obj_idx, vobj in enumerate(vid_objects):
    category = vobj["category"]
    object_id = vobj["object_id"]
    frames_ = getFramesForObject(vid_data=vid_data, Subject_id=object_id)
    if frames_=="None":
        continue
    
    for frame_range in frames_:
      frame_start, frame_end = frame_range

      if f"{category}{object_id}" not in frames_for_obj:
        frames_for_obj[f"{category}{object_id}"] = {
          "frames": []
        }

      frames_for_obj[f"{category}{object_id}"]["frames"].append(frame_range)

      if min_frame_idx ==-1:
          min_frame_idx = frame_start
      if frame_start<=min_frame_idx:
        min_frame_idx = frame_start
      if frame_end>=max_frame_idx:
        max_frame_idx = frame_end

  return min_frame_idx, max_frame_idx, frames_for_obj

def prepare_image_sg(chunk_vid_data_keys,data, norm_bb=True, dataset="vidor", uniform_sampling_idx=8):
  global llava_image_tune, llava_image_tune_val, image_annot_cnt, pbar, train_ids
  global llava_image_tune_bb, llava_image_tune_val_bb

  print(f"{threading.get_ident()} Started, will process {len(chunk_vid_data_keys)} data")

  for vid_data_key in chunk_vid_data_keys:
    vid_data = data[vid_data_key]
    vid_id = vid_data["video_id"]
    vid_objects = vid_data["objects"] # {'object_id': 2, 'category': 'countertop', 'is_thing': True, 'status': []},
    vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in vid_objects} # for relations
    vid_rels = vid_data["relations"]
    vid_id = vid_data["video_id"]
    total_frames = vid_data["meta"]["num_frames"]

    min_frame_idx, max_frame_idx, frames_for_obj = get_frame_range_for_annotations(vid_objects, vid_data) # drop frames with no annotations
    frames_where_subjobj_rel_is_present = {}

    for frame_idx in range(min_frame_idx, max_frame_idx+1):
      if frame_idx>total_frames:
         continue
      
      if frame_idx not in frames_where_subjobj_rel_is_present.keys():
         frames_where_subjobj_rel_is_present[frame_idx] = {
            "subj_obj_rel": [],
            "subj_obj_bb": [],
            "annot_cnt": 0
         }

      for idx, vid_r in enumerate(vid_rels):
        sub = vid_r[0]
        obj = vid_r[1]
        rel = vid_r[2]
        frames = vid_r[3].copy()
        frame_start, frame_end = frames[0][0], frames[0][1]
        if frame_start>total_frames:
           continue
        if frame_end>total_frames:
           continue

        # if frame_start>=frame_idx and frame_idx<=frame_end: # FIXED CONDITION
        if frame_idx>=frame_start and frame_idx<=frame_end:
          sub_bb, obj_bb, mask_size = get_bb_subj_obj(data_root=data_root,vid_id=vid_id,frame_idx=frame_idx,subject_id=sub,object_id=obj)

          if sum(sub_bb)>=0 and sum(obj_bb)>=0:
            # selected_frame = frame_for_bb_idx
            # break
            frames_where_subjobj_rel_is_present[frame_idx]["subj_obj_rel"].append(vid_r)
            frames_where_subjobj_rel_is_present[frame_idx]["subj_obj_bb"].append([sub_bb, obj_bb])
            frames_where_subjobj_rel_is_present[frame_idx]["annot_cnt"] +=1



    for f_idx, frame_data in frames_where_subjobj_rel_is_present.items():
      AnswerString = "{'"
      AnswerString_with_bb = "{'"
      rel_added = []

      if f_idx>total_frames:
        continue
      
      # data = frames_where_subjobj_rel_is_present[f_idx]
      subj_obj_rel = frame_data["subj_obj_rel"]
      subj_obj_bb = frame_data["subj_obj_bb"]

      subj_obj_loclization_prompts = []

      image_path = os.path.join(data_root, dataset, 'frames', vid_id, f'{str(f_idx).zfill(4)}.png')
      if not os.path.exists(image_path):
        continue
      
      for idx, vid_r in enumerate(subj_obj_rel):
        sub = vid_r[0]
        obj = vid_r[1]
        rel = vid_r[2]
        frames = vid_r[3].copy()
        sub_bb, obj_bb = subj_obj_bb[idx]
        # sub_center = getbbcenter(sub_bb)
        # obj_center = getbbcenter(obj_bb)

        subject_category_name = vid_objects_by_id[sub]['category']
        object_category_name = vid_objects_by_id[obj]['category']

        
        
        subj_obj_rel_entity = f"{vid_objects_by_id[sub]['category']}{sub}-{rel}-{vid_objects_by_id[obj]['category']}{obj}"
        if subj_obj_rel_entity not in rel_added:
          rel_added.append(subj_obj_rel_entity)
          AnswerString += f"[{vid_objects_by_id[sub]['category']}-{sub}:{rel}:{vid_objects_by_id[obj]['category']}-{obj}];"
          AnswerString_with_bb += f"[{vid_objects_by_id[sub]['category']}-{sub}_{sub_bb}:{rel}:{vid_objects_by_id[obj]['category']}-{obj}_{obj_bb}_[{sub_bb}:{obj_bb}]];"
          # AnswerString_with_bb += f"[{vid_objects_by_id[sub]['category']}-{sub}_{sub_center}:{rel}:{vid_objects_by_id[obj]['category']}-{obj}_{obj_center}];"


              
          Prompt1 = getPromptTemplate(media_path=image_path,media_type="image")
          
          convQ = getConvBlock(value=getRandomPrompt(key='sg_localization_image', static=True), 
                            conv_type="human", media_type="<image>", 
                            add_media_token=True)
          convQ["value"] = convQ["value"].replace("{sub}", f"{subject_category_name}-{sub}")
          convQ["value"] = convQ["value"].replace("{rel}", rel)
          convQ["value"] = convQ["value"].replace("{obj}", f"{object_category_name}-{obj}")

          Prompt1["conversations"].append(convQ)

          grounding = f"[{subject_category_name}-{sub}_{sub_bb}:{rel}:{object_category_name}-{obj}_{obj_bb}];"
          
          Prompt1["conversations"].append(getConvBlock(value=grounding,
                        conv_type="gpt", media_type="<image>"))
          

          with lock:
            if vid_id in train_ids:
              Prompt1["id"] = image_annot_cnt["train"]
              llava_image_tune_bb.append(Prompt1)
              image_annot_cnt["train"] +=1
            else:
              Prompt1["id"] = image_annot_cnt["val"]
              llava_image_tune_val_bb.append(Prompt1)
              image_annot_cnt["val"] +=1


      AnswerString +="}#END_REL#"
      AnswerString_with_bb +="}#END_REL#"

      Prompt_sg = getPromptTemplate(media_path=image_path,media_type="image")
      Prompt_sg["conversations"].append(getConvBlock(value=getRandomPrompt(key='SGG_image', static=True),
                            conv_type="human", media_type="<image>", add_media_token=True))
      Prompt_sg["conversations"].append(getConvBlock(value=AnswerString,
                            conv_type="gpt", media_type="<image>"))
      
      with lock:
        if vid_id in train_ids:
          Prompt_sg["id"] = image_annot_cnt["train"]
          llava_image_tune.append(Prompt_sg)
          image_annot_cnt["train"] +=1
        else:
          Prompt_sg["id"] = image_annot_cnt["val"]
          llava_image_tune_val.append(Prompt_sg)
          image_annot_cnt["val"] +=1

    with lock:
      pbar.n +=1
      pbar.last_print_n = pbar.n
      pbar.refresh()
          
  
  print(f"{threading.get_ident()} Exited after processing: {len(chunk_vid_data_keys)} data")   


def getListofCategoryString(vid_objects, vid_data, addObjectId=False, addFrames=False, addBB=False , uniform_sampling_idx=8):
    
    AnswerString = ""
    frame_indices = []
    total_frames = vid_data["meta"]["num_frames"]
    """V11 implementation
    [X] Select frames which covers all objects, avoid repetations
    """

    frames_where_obj_is_present = {}
    min_frame_idx, max_frame_idx, frames_for_obj = get_frame_range_for_annotations(vid_objects, vid_data)

    for frame_idx  in range(min_frame_idx, max_frame_idx+1):
      if frame_idx>total_frames:
         continue

      if frame_idx not in frames_where_obj_is_present.keys():
        frames_where_obj_is_present[frame_idx] ={
          "objects_present": [],
          "object_bb": [],
          "object_cnt": 0
        }

      for vid_obj_idx, vobj in enumerate(vid_objects):
        category = vobj["category"]
        object_id = vobj["object_id"]
        
        try:
          sub_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_data["video_id"],frame_id=frame_idx, object_id=object_id)
        except FileNotFoundError:
          pass

        if sum(sub_bb)>0:
          frames_where_obj_is_present[frame_idx]["objects_present"].append(vobj)
          frames_where_obj_is_present[frame_idx]["object_bb"].append(sub_bb)
          frames_where_obj_is_present[frame_idx]["object_cnt"] +=1

    # Take frames with more objects count first
    frames_with_obj_cnt = [(frames_where_obj_is_present[f_idx]["object_cnt"], f_idx) for f_idx in frames_where_obj_is_present]
    frames_with_obj_cnt = sorted(frames_with_obj_cnt,reverse=True)

    objects_added = []

    """
    Frame wise
    AnswerString = {
      0: "floor-1, wall-1, pillow-4",
      1: "floor-1, wall-1, shelf-4"
      .
      .
      7: "obj1,obj2"
    }
    """

    AnswerString += "{"

    for f_obj_idx, f_obj_cnt in enumerate(frames_with_obj_cnt):
      cnt_,f_idx = f_obj_cnt
      data = frames_where_obj_is_present[f_idx]

      AnswerString += f"{f_obj_idx}:"
      AnswerString +="'"  # start the list of objects string by "'"

      objects_present = data["objects_present"]
      objects_bb = data["object_bb"]

      frame_indices.append(f_idx) # use frame indices where object annotations are present

      for oidx, obj in enumerate(objects_present):
        category = obj["category"]
        object_id = obj["object_id"]

        # object_name_id = f"{category}-{object_id}"
        # if object_name_id not in objects_added:
        #   """This ensures unique objects in the list"""
        #   objects_added.append(object_name_id)

        AnswerString += f"{category}"

        if addObjectId:
          AnswerString += f"-{object_id}"

        if addBB:
          AnswerString += f"-{objects_bb[oidx]}"
        if addFrames:
          AnswerString += f"_[{f_idx}]"

        if oidx!=len(objects_present)-1:
          AnswerString +=","
        else:
          AnswerString +="'"  # finish the list of objects string by "'"
        
        if f_obj_idx>6:
           # TODO: some objects which appears in low count, will not be taken due to object density
           # In order to resolve this issue, need to accomodate all frames in 8 frames
           break
        
        if f_obj_idx!=len(frames_with_obj_cnt)-1:
           AnswerString += f"," # end of current key in dict


    AnswerString += "}"

    return AnswerString, frame_indices

def getbbcenter(bb):
   if len(bb)<4:
    return []
   x1,y1,x2,y2 = bb
   bb_w = (x2 - x1)/2
   bb_h = (y2 - y1)/2
   xcenter = x1 + bb_w
   ycenter = y1 + bb_h
   return [round(xcenter,3), round(ycenter,3)]



def addObjectsRelations_bb_instructions(video_path,vid_data,total_frames, subjobj_rel_frames_data, frame_indices,bb_per_object):
  obj_rel_bb_prompts = []
  vid_objects = vid_data["objects"] # {'object_id': 2, 'category': 'countertop', 'is_thing': True, 'status': []},
  vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in vid_objects} # for relations

  #  vid_rels = vid_data["relations"]
  #  vid_id = vid_data["video_id"]
  for frame_list_idx, frame_idx in enumerate(frame_indices):
    data = subjobj_rel_frames_data[frame_idx]
    subj_obj_rel = data["subj_obj_rel"]
    subj_obj_bb = data["subj_obj_bb"]

    add_video_token= True

    # obj_counter = {}
    # def update_obj_cnt(obj_name,obj_counter):
    #   if obj_name not in obj_counter:
    #     obj_counter[obj_name] = 0
    #   else:
    #     obj_counter[obj_name] += 1
    #   return obj_counter

    PromptAnswer = getPromptTemplate(media_path=video_path, media_type="video")

    for rel_idx, relation in enumerate(subj_obj_rel):
      sub = relation[0]
      obj = relation[1]
      rel = relation[2]
      # frames = relation[3].copy()

      sub_bb, obj_bb = subj_obj_bb[rel_idx]
      if sum(sub_bb)==0 or sum(obj_bb)==0:
         continue

      subject_category_name = vid_objects_by_id[sub]['category']
      object_category_name = vid_objects_by_id[obj]['category']

      if subject_category_name not in bb_per_object.keys():
         bb_per_object[subject_category_name] = 0

      if object_category_name not in bb_per_object.keys():
         bb_per_object[object_category_name] = 0
        
      if bb_per_object[subject_category_name]<100 or bb_per_object[object_category_name]<100:
        convQ = getConvBlock(value=getRandomPrompt(key='sg_localization', static=True), 
                            conv_type="human", media_type="<video>", 
                            add_media_token=add_video_token)
        if add_video_token:
          add_video_token = False

        # "Provide bounding box location of [{sub}:{rel}:{obj}] in frame {frame_idx} of the provided video" # {} to be replaced by actual value
        convQ["value"] = convQ["value"].replace("{sub}", f"{subject_category_name}-{sub}")
        convQ["value"] = convQ["value"].replace("{rel}", rel)
        convQ["value"] = convQ["value"].replace("{obj}", f"{object_category_name}-{obj}")
        # convQ["value"] = convQ["value"].replace("{frame_idx}", str(frame_indices.index(frame_idx)))


        resp = f"[{subject_category_name}-{sub}_{sub_bb}:{rel}:{object_category_name}-{obj}_{obj_bb}|[{sub_bb}:{obj_bb}]];"

        # resp = {f"Frame {frame_list_idx}": resp}

        convA = getConvBlock(value=str(resp), 
                          conv_type="gpt", media_type="<video>", 
                          add_media_token=False)
        
        PromptAnswer["conversations"].append(convQ)
        PromptAnswer["conversations"].append(convA)

        bb_per_object[object_category_name] +=1
        bb_per_object[subject_category_name] +=1


      if len(PromptAnswer["conversations"])>6:
         break
    
    PromptAnswer["frame_indices"] =  frame_indices
    PromptAnswer["total_frames"] = total_frames

    if len(PromptAnswer["conversations"])>=2:
       obj_rel_bb_prompts.append(PromptAnswer)


  return obj_rel_bb_prompts, bb_per_object


def getObjectsRelations(vid_rels, vid_data, norm_frames=True, add_frames=True, uniform_sampling_idx=8):
    AnswerString = ""
    AnswerString_with_bb = ""
    SubObjRel = []
    frame_indices = []
    # mask_size = None

    total_frames = vid_data["meta"]["num_frames"]
    # frame_idxs = [x for x in range(total_frames)]
    # everynth = int(total_frames/uniform_sampling_idx)
    # every8thIndex = frame_idxs[0::everynth+1] 
    # every8thIndex = np.linspace(0, total_frames-1, uniform_sampling_idx, dtype=int)

    vid_objects = vid_data["objects"] # {'object_id': 2, 'category': 'countertop', 'is_thing': True, 'status': []},
    vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in vid_objects} # for relations
    vid_rels = vid_data["relations"]
    vid_id = vid_data["video_id"]


    #batch_of_frames, remaining_frames, batch_rels = create_batch_frames(vid_data=vid_data,totalFrames=total_frames,frame_batch=8)

    min_frame_idx, max_frame_idx, frames_for_obj = get_frame_range_for_annotations(vid_objects, vid_data) # drop frames with no annotations
    frames_where_subjobj_rel_is_present = {}

    for frame_idx in range(min_frame_idx, max_frame_idx+1):
      if frame_idx>total_frames:
         continue
      
      if frame_idx not in frames_where_subjobj_rel_is_present.keys():
         frames_where_subjobj_rel_is_present[frame_idx] = {
            "subj_obj_rel": [],
            "subj_obj_bb": [],
            "annot_cnt": 0
         }

      for idx, vid_r in enumerate(vid_rels):
        sub = vid_r[0]
        obj = vid_r[1]
        rel = vid_r[2]
        frames = vid_r[3].copy()
        frame_start, frame_end = frames[0][0], frames[0][1]
        for frame_range in frames:
          frame_start, frame_end = frame_range
          
          if frame_start>total_frames:
            continue
          if frame_end>total_frames:
            continue

          # if frame_start>=frame_idx and frame_idx<=frame_end: # FIXED CONDITION
          if frame_idx>=frame_start and frame_idx<=frame_end:
            sub_bb, obj_bb, mask_size = get_bb_subj_obj(data_root=data_root,vid_id=vid_id,frame_idx=frame_idx,subject_id=sub,object_id=obj)

            if sum(sub_bb)>0 and sum(obj_bb)>0:
              # selected_frame = frame_for_bb_idx
              # break
              frames_where_subjobj_rel_is_present[frame_idx]["subj_obj_rel"].append(vid_r)
              frames_where_subjobj_rel_is_present[frame_idx]["subj_obj_bb"].append([sub_bb, obj_bb])
              frames_where_subjobj_rel_is_present[frame_idx]["annot_cnt"] +=1


    # TODO V13 and later: instead of sorting and using top 8 frames, divide total frames in 8 batch, and cover all SGs 
    
    # frames_with_subjobj_rel_cnt = [(frames_where_subjobj_rel_is_present[f_idx]["annot_cnt"], f_idx) for f_idx in frames_where_subjobj_rel_is_present]
    # frames_with_subjobj_rel_cnt = sorted(frames_with_subjobj_rel_cnt,reverse=True) # get frames with highest rels annotation first
    overall_annotations = []
    
    # AnswerString_keyval = { 
    #    #  "Frame 0": "","Frame 1": "", "Frame 2": "", #  "Frame 3": "","Frame 4": "","Frame 5": "", #  "Frame 6": "","Frame 7": "",
    # }
    # for i in range(8):
    #    AnswerString_keyval[f"Frame {i}"] = ""

    frame_counter = 0
    annotation_total_frame_count = len(frames_where_subjobj_rel_is_present.keys())
    remaining_frames_after_batching = annotation_total_frame_count%uniform_sampling_idx
    frames_needs_to_be_added = uniform_sampling_idx - remaining_frames_after_batching
    frame_indices = []

    # frame_idx_list= list(frames_where_subjobj_rel_is_present.keys())
    # f_start_idx_for_rem_frames = remaining_frames_after_batching+frames_needs_to_be_added
    # append_for_remaining_frames = frame_idx_list[f_start_idx_for_rem_frames:]

    tripletes_for_current_block = ""
    for key_frame_idx, frame_data in frames_where_subjobj_rel_is_present.items():
       
      subj_obj_rel = frame_data["subj_obj_rel"]
      subj_obj_bb = frame_data["subj_obj_bb"]

      tripletes_for_current_block += f"{SGSpecialTokens.VIDEO_FRAME_ID}"

      # rel_added = []
      for idx, vid_r in enumerate(subj_obj_rel):
        sub = vid_r[0]
        obj = vid_r[1]
        rel = vid_r[2]
        # frames = vid_r[3].copy()
        sub_bb, obj_bb = subj_obj_bb[idx]
        # sub_center = getbbcenter(sub_bb)
        # obj_center = getbbcenter(obj_bb)

        # if sum(sub_bb)>0 and sum(obj_bb)>0:
        #    tripletes_for_current_block += f"[{vid_objects_by_id[sub]['category']}-{sub}:{rel}:{vid_objects_by_id[obj]['category']}-{obj}];"
        # if subj_obj_rel_entity not in rel_added:
            # rel_added.append(subj_obj_rel_entity)

        if sum(sub_bb)>0 and sum(obj_bb)>0:
          # tripletes_for_current_block += f"{SpecialTokens.SG_START}"
          # print(sub_bb, obj_bb)

          sub_category = vid_objects_by_id[sub]['category']
          obj_category = vid_objects_by_id[obj]['category']

          tripletes_for_current_block += f"[{SGSpecialTokens.SG_SUBJECT}'{sub_category}-{SGSpecialTokens.SG_SUBJECT_ID}{sub}-[{SGSpecialTokens.SG_BB_X1Y1}[{sub_bb[0]},{sub_bb[1]}],{SGSpecialTokens.SG_BB_X2Y2}[{sub_bb[2]},{sub_bb[3]}]]'"
          tripletes_for_current_block += f":{SGSpecialTokens.SG_PREDICATE}'{rel}'"
          tripletes_for_current_block += f":{SGSpecialTokens.SG_OBJECT}'{obj_category}-{SGSpecialTokens.SG_OBJECT_ID}{obj}-[{SGSpecialTokens.SG_BB_X1Y1}[{obj_bb[0]},{obj_bb[1]}],{SGSpecialTokens.SG_BB_X2Y2}[{obj_bb[2]},{obj_bb[3]}]]'"
          tripletes_for_current_block += f"];{SGSpecialTokens.SG_END}"

          # tripletes_for_current_block += f"["
          # tripletes_for_current_block += f"{SGSpecialTokens.SG_SUBJECT}{vid_objects_by_id[sub]['category']}-{SGSpecialTokens.SG_SUBJECT_ID}{sub}"
          # # tripletes_for_current_block += f"_[{SGSpecialTokens.SG_BB_X1Y1}[{sub_bb[0]},{sub_bb[1]}],{SGSpecialTokens.SG_BB_X2Y2}[{sub_bb[2]},{sub_bb[3]}]]"
          # tripletes_for_current_block += f":"
          # tripletes_for_current_block += f"{SGSpecialTokens.SG_OBJECT}{vid_objects_by_id[obj]['category']}-{SGSpecialTokens.SG_OBJECT_ID}{obj}"
          # # tripletes_for_current_block += f"_[{SGSpecialTokens.SG_BB_X1Y1}[{obj_bb[0]},{obj_bb[1]}],{SGSpecialTokens.SG_BB_X2Y2}[{obj_bb[2]},{obj_bb[3]}]]"
          # tripletes_for_current_block += f":{SGSpecialTokens.SG_PREDICATE}{rel}"
          # tripletes_for_current_block += f"];{SGSpecialTokens.SG_END}"

          # tripletes_for_current_block += f"[{SGSpecialTokens.SG_SUBJECT} {vid_objects_by_id[sub]['category']} {SGSpecialTokens.SG_SUBJECT_ID}-{sub}"
          # tripletes_for_current_block += f" {SGSpecialTokens.SG_BB_START}"
          # # tripletes_for_current_block += f"[{SGSpecialTokens.SG_BB_X1} {sub_bb[0]},{SGSpecialTokens.SG_BB_Y1} {sub_bb[1]},"
          # # tripletes_for_current_block += f"{SGSpecialTokens.SG_BB_X2} {sub_bb[2]},{SGSpecialTokens.SG_BB_Y2} {sub_bb[3]}]"
          # tripletes_for_current_block += f" [{sub_bb[0]},{sub_bb[1]},{sub_bb[2]},{sub_bb[3]}] "
          # tripletes_for_current_block += f"{SGSpecialTokens.SG_BB_END}"
          # tripletes_for_current_block += f":{SGSpecialTokens.SG_PREDICATE} {rel}:"
          # tripletes_for_current_block += f"{SGSpecialTokens.SG_OBJECT} {vid_objects_by_id[obj]['category']} {SGSpecialTokens.SG_OBJECT_ID}-{obj}"
          # tripletes_for_current_block += f" {SGSpecialTokens.SG_BB_START}"
          # tripletes_for_current_block += f" [{obj_bb[0]},{obj_bb[1]},{obj_bb[2]},{obj_bb[3]}] "
          # tripletes_for_current_block += f"{SGSpecialTokens.SG_BB_END}"
          # tripletes_for_current_block += f"];"
          # tripletes_for_current_block += f"{SpecialTokens.SG_END}"

      # tripletes_for_current_block += "</frameseq>"
      # AnswerString_keyval[f"Frame {frame_counter}"] = tripletes_for_current_frame
      frame_indices.append(key_frame_idx)
      frame_counter +=1

      if len(frame_indices)>=8:
        
        overall_annotations.append({
            "frame_idxes": frame_indices,
            "frames_sgs": tripletes_for_current_block
        })

        tripletes_for_current_block = ""
        frame_counter = 0
        frame_indices = []
        #  for i in range(8): 
        #     AnswerString_keyval[f"Frame {i}"] = ""

    # TODO add remaining annotations for last block

    return overall_annotations,AnswerString,AnswerString_with_bb, frame_indices, frames_where_subjobj_rel_is_present

def getConvBlock(value,conv_type="human", media_type="<image>", add_media_token=False):
   assert conv_type=="human" or conv_type=="gpt"
   assert media_type=="<image>" or media_type=="<video>"
   conv = {"from": conv_type, "value": f"{value}"}
   if add_media_token:
      conv["value"] = f"{media_type}\n{value}"
   else:
      conv["value"] = f"{value}" 

   return conv

def getPromptTemplate(media_path, media_type="image"):
  assert media_type=="image" or media_type=="video"
  Prompt = {
          "id": "TobeUpdated",
          f"{media_type}": f"{media_path}",
          "conversations": [],
          "frame_indices": [],  # selected indices will be passed to model for train and test
          "total_frames": "",
  }
  return Prompt

def prepare_vid_sg_threaded(chunk_vid_data_keys,data, norm_bb=True, dataset="vidor", uniform_sampling_idx=8):
   

   global video_gpt_promptanswers, video_gpt_promptanswers_val, annot_cnt
   global video_gpt_bb_promptanswers, video_gpt_bb_promptanswers_val, video_bb_annot_cnt

   bb_per_object = {}

   for vid_id in chunk_vid_data_keys:
      vid_data = data[vid_id]
      total_frames = data[vid_id]["meta"]["num_frames"]
      video_path = f"{data_root}{dataset}/videos/{vid_id}.mp4"
      if not os.path.exists(video_path):
        continue
      
      # vid_objects = vid_data["objects"] # {'object_id': 2, 'category': 'countertop', 'is_thing': True, 'status': []},
      # convQ = getConvBlock(value=getRandomPrompt(key='identify_subject_objects', static=True), 
      #                     conv_type="human", media_type="<video>", 
      #                     add_media_token=True)
      # AnswerStringObjs, frame_indices_obj = getListofCategoryString(vid_objects, vid_data, addObjectId=True,addBB=False,addFrames=False)
      # convA = getConvBlock(value=AnswerStringObjs, 
      #                     conv_type="gpt", media_type="<image>", 
      #                     add_media_token=False)
      # PromptAnswer["conversations"].append(convQ)
      # PromptAnswer["conversations"].append(convA)

      overall_annotations, AnswerStringRels,AnswerString_with_bb, frame_indices_rel, frames_where_subjobj_rel_is_present = getObjectsRelations(vid_data["relations"], vid_data, uniform_sampling_idx=uniform_sampling_idx, add_frames=False)


      for annot in overall_annotations:

        # SG without grounding
        PromptAnswer = getPromptTemplate(media_path=video_path, media_type="video")

        frame_indices = annot["frame_idxes"]
        tripletes_for_current_block = annot["frames_sgs"]
        
        convQ = getConvBlock(value=getRandomPrompt(key='SGG', static=False), 
                            conv_type="human", media_type="<video>", 
                            add_media_token=True)
        
        convA = getConvBlock(value=tripletes_for_current_block, 
                            conv_type="gpt", media_type="<video>", 
                            add_media_token=False)

        PromptAnswer["conversations"].append(convQ)
        PromptAnswer["conversations"].append(convA)

        # all_frame_indices = list(set(frame_indices_rel + frame_indices_obj))
        # all_frame_indices = list(set(frame_indices_rel))

        PromptAnswer["frame_indices"] =  frame_indices
        PromptAnswer["total_frames"] = total_frames

        with lock:
          if vid_id in train_ids:
            PromptAnswer["id"] = annot_cnt["train"]
            video_gpt_promptanswers.append(PromptAnswer)
            annot_cnt["train"] +=1
          else:
            PromptAnswer["id"] = annot_cnt["val"]
            video_gpt_promptanswers_val.append(PromptAnswer)
            annot_cnt["val"] +=1


      # # SG with grounding
      # PromptAnswer = getPromptTemplate(media_path=video_path, media_type="video")
      # convQ = getConvBlock(value=getRandomPrompt(key='SGG_with_bb', static=False), 
      #                     conv_type="human", media_type="<video>", 
      #                     add_media_token=True)
      # convA = getConvBlock(value=AnswerString_with_bb, 
      #                     conv_type="gpt", media_type="<video>", 
      #                     add_media_token=False)
      # PromptAnswer["conversations"].append(convQ)
      # PromptAnswer["conversations"].append(convA)
      # PromptAnswer["frame_indices"] =  all_frame_indices
      # PromptAnswer["total_frames"] = total_frames

      # with lock:
      #   if vid_id in train_ids:
      #     PromptAnswer["id"] = annot_cnt["train"]
      #     video_gpt_promptanswers.append(PromptAnswer)
      #     annot_cnt["train"] +=1
      #   else:
      #     PromptAnswer["id"] = annot_cnt["val"]
      #     video_gpt_promptanswers_val.append(PromptAnswer)
      #     annot_cnt["val"] +=1
      

      # grounding sg triplets

      # obj_rel_bb_prompts, bb_per_object = addObjectsRelations_bb_instructions(video_path=video_path,
      #                                                          vid_data=vid_data,
      #                                                          total_frames=total_frames,
      #                                                          subjobj_rel_frames_data=frames_where_subjobj_rel_is_present,
      #                                                          frame_indices=frame_indices_rel,bb_per_object=bb_per_object)
      # with lock:
      #   for obj_rel_bb_prmpt in obj_rel_bb_prompts:
      #     if vid_id in train_ids:
      #       obj_rel_bb_prmpt["id"] = annot_cnt["train"] # video_bb_annot_cnt["train"]
      #       video_gpt_bb_promptanswers.append(obj_rel_bb_prmpt)
      #       annot_cnt["train"] +=1
      #     else:
      #       obj_rel_bb_prmpt["id"] = annot_cnt["val"] # video_bb_annot_cnt["val"]
      #       video_gpt_bb_promptanswers_val.append(obj_rel_bb_prmpt)
      #       annot_cnt["val"] +=1
         
      with lock:
        pbar.n +=1
        pbar.last_print_n = pbar.n
        pbar.refresh()

      # append_annotation(vid_data["video_id"],annotation=PromptAnswer)
   

def get_bb_subj_obj(data_root,vid_id,frame_idx,subject_id,object_id):
  sub_bb, obj_bb, mask_size = [], [], None
  try:
    sub_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_id,frame_id=frame_idx, object_id=subject_id)
  except FileNotFoundError:
    #print(f"[Warning] Frame {frame_for_bb_idx} not found for vidor {vid_data['video_id']}")
    pass
  
  try:
    obj_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_id,frame_id=frame_idx, object_id=object_id)
  except FileNotFoundError:
    #print(f"[Warning] Frame {frame_for_bb_idx} not found for vidor {vid_data['video_id']}")
    pass

  return sub_bb, obj_bb, mask_size


def getVideoCaptions(vid_data, correct_object_ids=False):
    vid_objects = vid_data["objects"] # {'object_id': 2, 'category': 'countertop', 'is_thing': True, 'status': []},
    # vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in vid_objects} # for relations
    vid_rels = vid_data["relations"]
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


def chunk_list(list_, chunk_n):
    chunk_n = max(1, chunk_n)
    return (list_[i:i+chunk_n] for i in range(0, len(list_), chunk_n))

if __name__=="__main__":
    n_thread_count = 20
    per_thread_data = 0
    threads = []

    lock = threading.Lock()

    data_root = '/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/'
    with open(os.path.join(data_root, 'pvsg.json'), 'r') as f:
        anno = json.load(f)

    print('Keys inside pvsg.json:', list(anno.keys()))
    print('Number of Object Classes:', len(anno['objects']['thing']))
    print('Number of Stuff Classes:', len(anno['objects']['stuff']))
    print('Number of Relation Classes:', len(anno['relations']))

    dataset = "vidor"

    train_ids = anno["split"][dataset]["train"]
    val_ids = anno["split"][dataset]["val"]
    vidor_ids = train_ids + val_ids
    data = {data_dict['video_id']: data_dict for data_dict in anno['data'] if data_dict['video_id'] in vidor_ids}
    keys = list(data.keys())
    
    total_keys = len(keys)
    data_per_thread = int(total_keys/n_thread_count)
    current_vid_idx = 0
    processedThreadsCount = 0

    chunked_list_gen = chunk_list(list_=keys, chunk_n=data_per_thread)
    chunked_list = []
    for cl in chunked_list_gen:
       chunked_list.append(cl)
    
    n_thread_count = len(chunked_list)
    print("len of chunked list: ", len(chunked_list))


    OUTPUT_JSON_DIR = "/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/video_llava_annotations_v18_4/"
    JSON_llava_image_tune_validate = f"{OUTPUT_JSON_DIR}/llava_image_tune_validate.json"
    JSON_llava_image_tune = f"{OUTPUT_JSON_DIR}/llava_image_tune_.json"

    JSON_llava_image_tune_validate_bb = f"{OUTPUT_JSON_DIR}/llava_image_tune_validate_bb.json"
    JSON_llava_image_tune_bb = f"{OUTPUT_JSON_DIR}/llava_image_tune_bb.json"

    JSON_videochatgpt_tune = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_.json"
    JSON_videochatgpt_tune_validate = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_validate.json"
    # JSON_videochatgpt_tune_with_bb = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_with_bb.json"
    # JSON_videochatgpt_tune_with_bb_validate = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_validate.json"
    JSON_videochatgpt_bb_tune = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_bb.json"
    JSON_videochatgpt_bb_tune_validate = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_bb_validate.json"
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

    video_gpt_promptanswers = []
    video_gpt_promptanswers_val = []

    # video_gpt_promptanswers_with_bb = []
    # video_gpt_promptanswers_with_bb_val = []

    video_gpt_bb_promptanswers=  []
    video_gpt_bb_promptanswers_val =  []

    llava_image_tune = []
    llava_image_tune_val = []

    llava_image_tune_bb = []
    llava_image_tune_val_bb = []

    image_annot_cnt = {"train": 0, "val": 0}
    annot_cnt = {"train": 0, "val": 0}
    video_bb_annot_cnt = {"train": 0, "val": 0}

    print("Total videos ",len(keys))

    """
    Image Annotations
    """

    # pbar = tqdm(total=len(keys))
    # pbar.n = 0
    # pbar.last_print_n = 0
    # pbar.refresh()

    # for ch_idx, chunk_vid_data in enumerate(chunked_list):
    #   T = threading.Thread(target=prepare_image_sg, name=f"Thread{ch_idx+1}", args=(chunk_vid_data,data,True,"vidor"))
    #   T.start()
    #   threads.append(T)
    # for th in threads:
    #    th.join()

    # with open(JSON_llava_image_tune, "w") as f:
    #     json.dump(llava_image_tune,f)

    # with open(JSON_llava_image_tune_validate, "w") as f:
    #     json.dump(llava_image_tune_val,f)

    # with open(JSON_llava_image_tune_bb, "w") as f:
    #     json.dump(llava_image_tune_bb,f)

    # with open(JSON_llava_image_tune_validate_bb, "w") as f:
    #     json.dump(llava_image_tune_val_bb,f)

    # print("Saved annotations", image_annot_cnt)


    """
    Video Annotations
    """

    pbar = tqdm(total=len(keys))
    pbar.n = 0
    pbar.last_print_n = 0
    pbar.refresh()


    for ch_idx, chunk_vid_data in enumerate(chunked_list):
      T = threading.Thread(target=prepare_vid_sg_threaded, name=f"Thread{ch_idx+1}", args=(chunk_vid_data,data,True,dataset,8))
      T.start()
      threads.append(T)
    for th in threads:
       th.join()

    with open(JSON_videochatgpt_tune, "w") as f:
        json.dump(video_gpt_promptanswers,f)
    with open(JSON_videochatgpt_tune_validate, "w") as f:
        json.dump(video_gpt_promptanswers_val,f)
    print("Saved annotations", annot_cnt)

    # with open(JSON_videochatgpt_bb_tune, "w") as f:
    #     json.dump(video_gpt_bb_promptanswers,f)
    # with open(JSON_videochatgpt_bb_tune_validate, "w") as f:
    #     json.dump(video_gpt_bb_promptanswers_val,f)
    # print("Saved annotations", video_bb_annot_cnt)