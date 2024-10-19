import os
import json
import glob
from tqdm import tqdm
import random
import re
import threading
import time

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
       "Provide frame-by-frame Scene graph triplets in the form of [Subject:Predicate:Object]",
       "Generate scene graph for the provided video",
       "Provide scene graph for the provided video",
       "Identify subjects, predicates and objects frame-by-frame in the provided video"
    ],

    "sg_localization": [
       "Provide bounding box location of [{sub}:{rel}:{obj}] in frame {frame_idx} of the provided video" # {} to be replaced by actual value
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

  print(f"{threading.get_ident()} Started, will process {len(chunk_vid_data_keys)} data")

  for vid_data_key in chunk_vid_data_keys:
    vid_data = data[vid_data_key]
    vid_id = vid_data["video_id"]
    vid_objects = vid_data["objects"] # {'object_id': 2, 'category': 'countertop', 'is_thing': True, 'status': []},
    vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in vid_objects} # for relations
    vid_rels = vid_data["relations"]

    total_frames = vid_data["meta"]["num_frames"]
    every8thIndex = np.linspace(0, total_frames-1, uniform_sampling_idx, dtype=int)

    min_frame_idx, max_frame_idx = -1, 0
    for _, vid_r in enumerate(vid_rels):
      """
      Ignore frames where there are no SGs
      """    
      rel_frames = vid_r[3]
      for frameSeq in rel_frames:
        frame_start, frame_end = frameSeq
        if min_frame_idx ==-1:
           min_frame_idx = frame_start
        if frame_start<=min_frame_idx:
          min_frame_idx = frame_start
        if frame_end>=max_frame_idx:
          max_frame_idx = frame_end

    for frame_idx in range(min_frame_idx, max_frame_idx+1):

      if not frame_idx in every8thIndex:
         continue

      image_path = os.path.join(data_root, dataset, 'frames', vid_id, f'{str(frame_idx).zfill(4)}.png')
      if not os.path.exists(image_path):
        continue
      # image = Image.open(image_path)
      
      AnswerString = ""
      ListOfObjects = []
      ListOfObjects_bb = []

      for vid_rel_idx, vid_r in enumerate(vid_rels):
          sub = vid_r[0]
          obj = vid_r[1]
          rel = vid_r[2]
          rel_frames = vid_r[3]

          for frameSeq in rel_frames:
              frame_start, frame_end = frameSeq
              if frame_idx>=frame_start and frame_idx<=frame_end:
                # get subjects objects which has annotations in the current frame
                
                sub_bb = []
                obj_bb = []

                try:
                  sub_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_data["video_id"],
                                                                frame_id=frame_idx, object_id=sub, normlize_bb=norm_bb, dataset=dataset)
                except FileNotFoundError: 
                  #print(f"[Warning] Frame {frame_idx} not found for {dataset} {vid_data['video_id']}")
                  pass
                try:
                  obj_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_data["video_id"],
                                                                frame_id=frame_idx, object_id=obj, normlize_bb=norm_bb, dataset=dataset)
                except FileNotFoundError: 
                  #print(f"[Warning] Frame {frame_idx} not found for {dataset} {vid_data['video_id']}")
                  pass


                if sum(sub_bb)>0:
                  if f"{vid_objects_by_id[sub]['category']}.{sub}" not in ListOfObjects:
                    ListOfObjects.append(f"{vid_objects_by_id[sub]['category']}.{sub}")
                    ListOfObjects_bb.append(sub_bb)
                  
                if sum(obj_bb)>0:
                  if f"{vid_objects_by_id[obj]['category']}.{obj}" not in ListOfObjects:
                    ListOfObjects.append(f"{vid_objects_by_id[obj]['category']}.{obj}")
                    ListOfObjects_bb.append(obj_bb)

                if sum(sub_bb)==0 and sum(obj_bb)==0:
                    continue
                
                frame_for_subject_object = 0 # for single image there is no temporal entity
                AnswerString += f"[{vid_objects_by_id[sub]['category']}-{sub}-{sub_bb}:{rel}:{vid_objects_by_id[obj]['category']}-{obj}-{obj_bb}]_[{frame_for_subject_object}];"
                

      ListOfObjectsAnswerStr = ""
      ListOfObjectsAnswerStr_withbb= ""

      for objidx,objn in enumerate(ListOfObjects):
        ListOfObjectsAnswerStr += f"{objn}"
        ListOfObjectsAnswerStr_withbb += f"{objn}.[{ListOfObjects_bb[objidx]}]"
        if objidx!=len(ListOfObjects)-1:
           ListOfObjectsAnswerStr+=","
           ListOfObjectsAnswerStr_withbb+=","


      Prompt1 = getPromptTemplate(media_path=image_path,media_type="image")

      Prompt1["conversations"].append(getConvBlock(value=getRandomPrompt(key='identify_subject_objects', static=True),
                          conv_type="human", media_type="<image>", add_media_token=True))
      Prompt1["conversations"].append(getConvBlock(value=ListOfObjectsAnswerStr_withbb,
                          conv_type="gpt", media_type="<image>"))
      

      Prompt2 = getPromptTemplate(media_path=image_path,media_type="image")
      Prompt2["conversations"].append(getConvBlock(value=getRandomPrompt(key='identify_predicates', static=True),
                          conv_type="human", media_type="<image>", add_media_token=True))
      Prompt2["conversations"].append(getConvBlock(value=AnswerString,
                          conv_type="gpt", media_type="<image>"))

      for prompt_ in [Prompt1,Prompt2]:
        with lock:
          if vid_id in train_ids:
            prompt_["id"] = image_annot_cnt["train"]
            llava_image_tune.append(prompt_)
            image_annot_cnt["train"] +=1
          else:
            prompt_["id"] = image_annot_cnt["val"]
            llava_image_tune_val.append(prompt_)
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


def addObjectsRelations_bb_instructions(video_path,vid_data,total_frames, subjobj_rel_frames_data, frame_indices):
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

    obj_counter = {}
    def update_obj_cnt(obj_name,obj_counter):
      if obj_name not in obj_counter:
        obj_counter[obj_name] = 0
      else:
        obj_counter[obj_name] += 1

      return obj_counter

    for rel_idx, relation in enumerate(subj_obj_rel):
      sub = relation[0]
      obj = relation[1]
      rel = relation[2]
      # frames = relation[3].copy()

      sub_bb, obj_bb = subj_obj_bb[rel_idx]
      if sum(sub_bb)==0 or sum(obj_bb)==0:
         continue

      PromptAnswer = getPromptTemplate(media_path=video_path, media_type="video")

      convQ = getConvBlock(value=getRandomPrompt(key='sg_localization', static=True), 
                            conv_type="human", media_type="<video>", 
                            add_media_token=add_video_token)
      # if add_video_token:
      #    add_video_token = False

      obj_counter = update_obj_cnt(vid_objects_by_id[sub]['category'],obj_counter)
      obj_counter = update_obj_cnt(vid_objects_by_id[obj]['category'],obj_counter)

      obj_cnt = obj_counter[vid_objects_by_id[obj]['category']]
      sub_cnt = obj_counter[vid_objects_by_id[sub]['category']]

      # "Provide bounding box location of [{sub}:{rel}:{obj}] in frame {frame_idx} of the provided video" # {} to be replaced by actual value
      convQ["value"] = convQ["value"].replace("{sub}", f"{vid_objects_by_id[sub]['category']}-{sub_cnt}")
      convQ["value"] = convQ["value"].replace("{rel}", rel)
      convQ["value"] = convQ["value"].replace("{obj}", f"{vid_objects_by_id[obj]['category']}-{obj_cnt}")
      convQ["value"] = convQ["value"].replace("{frame_idx}", str(frame_indices.index(frame_idx)))


      AnswerString_rel_bb = f""""""
      AnswerString_rel_bb += f"""{vid_objects_by_id[sub]['category']}-{sub_cnt}-{subj_obj_bb[rel_idx][0]}:{rel}:"""
      AnswerString_rel_bb += f"""{vid_objects_by_id[obj]['category']}-{obj_cnt}-{subj_obj_bb[rel_idx][1]}"""

      # AnswerString_rel_bb = f"""The bounding box locations of """
      # AnswerString_rel_bb += f"""[{vid_objects_by_id[sub]['category']}:{rel}:{vid_objects_by_id[obj]['category']}]"""
      # AnswerString_rel_bb += f"""in frame {frame_list_idx} is, {subj_obj_bb[rel_idx][0], subj_obj_bb[rel_idx][1]}"""

      
      convA = getConvBlock(value=AnswerString_rel_bb, 
                        conv_type="gpt", media_type="<video>", 
                        add_media_token=False)
      
      PromptAnswer["conversations"].append(convQ)
      PromptAnswer["conversations"].append(convA)

      # if len(PromptAnswer["conversations"])>4:
      #    break
    
      PromptAnswer["frame_indices"] =  frame_indices
      PromptAnswer["total_frames"] = total_frames
      obj_rel_bb_prompts.append(PromptAnswer)


  return obj_rel_bb_prompts


def getObjectsRelations(vid_rels, vid_data, norm_frames=True, add_frames=True, uniform_sampling_idx=8):
    AnswerString = ""
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

        if frame_start>=frame_idx and frame_idx<=frame_end:
          sub_bb, obj_bb, mask_size = get_bb_subj_obj(data_root=data_root,vid_id=vid_id,frame_idx=frame_idx,subject_id=sub,object_id=obj)

          if sum(sub_bb)>=0 and sum(obj_bb)>=0:
            # selected_frame = frame_for_bb_idx
            # break
            frames_where_subjobj_rel_is_present[frame_idx]["subj_obj_rel"].append(vid_r)
            frames_where_subjobj_rel_is_present[frame_idx]["subj_obj_bb"].append([sub_bb, obj_bb])
            frames_where_subjobj_rel_is_present[frame_idx]["annot_cnt"] +=1


    frames_with_subjobj_rel_cnt = [(frames_where_subjobj_rel_is_present[f_idx]["annot_cnt"], f_idx) for f_idx in frames_where_subjobj_rel_is_present]
    frames_with_subjobj_rel_cnt = sorted(frames_with_subjobj_rel_cnt,reverse=True) # get frames with highest rels annotation first


    AnswerString += "{"

    for f_obj_idx, f_obj_cnt in enumerate(frames_with_subjobj_rel_cnt):
      cnt_, f_idx = f_obj_cnt

      if f_idx>total_frames:
        continue


      rel_added = []

      data = frames_where_subjobj_rel_is_present[f_idx]
      subj_obj_rel = data["subj_obj_rel"]
      subj_obj_bb = data["subj_obj_bb"]

      AnswerString += f"'Frame {f_obj_idx}':"
      AnswerString +="'"  # start the list of relations string by "'"

      frame_indices.append(f_idx)

      obj_counter = {}
      def update_obj_cnt(obj_name,obj_counter):
        if obj_name not in obj_counter:
          obj_counter[obj_name] = 0
        else:
          obj_counter[obj_name] += 1
        return obj_counter

      for idx, vid_r in enumerate(subj_obj_rel):
        sub = vid_r[0]
        obj = vid_r[1]
        rel = vid_r[2]
        frames = vid_r[3].copy()
        sub_bb, obj_bb = subj_obj_bb[idx]

        #   """This ensures unique subject object relation in the list"""
        #   rel_added.append(subj_obj_rel_entity)
        #   frame_indices.append(f_idx)
        #   if not add_frames:
        #     AnswerString += f"[{vid_objects_by_id[sub]['category']}-{sub}-{sub_bb}:{rel}:{vid_objects_by_id[obj]['category']}-{obj}-{obj_bb}]" 
        #   else:
        #     AnswerString += f"[{vid_objects_by_id[sub]['category']}-{sub}-{sub_bb}:{rel}:{vid_objects_by_id[obj]['category']}-{obj}-{obj_bb}_[{f_idx}]]"
        
        obj_counter = update_obj_cnt(vid_objects_by_id[sub]['category'],obj_counter)  # adult-0, adult-1, atult-3 and so on.
        obj_counter = update_obj_cnt(vid_objects_by_id[obj]['category'],obj_counter)
        cnt_of_sub = obj_counter[vid_objects_by_id[sub]['category']]
        cnt_of_obj = obj_counter[vid_objects_by_id[obj]['category']]
        
        subj_obj_rel_entity = f"{sub}{cnt_of_sub}-{rel}-{obj}{cnt_of_obj}"
        if subj_obj_rel_entity not in rel_added:
          rel_added.append(subj_obj_rel_entity)
          AnswerString += f"[{vid_objects_by_id[sub]['category']}-{cnt_of_sub}:{rel}:{vid_objects_by_id[obj]['category']}-{cnt_of_obj}]"
        if idx!=len(subj_obj_rel)-1:
          AnswerString +=";"
        else:
          AnswerString +="'"  # finish the list of triplets string by "'"

      
      if f_obj_idx>6:
           # TODO: some predicates which appears in low count, will not be taken due to object density
           # In order to resolve this issue, need to accomodate all frames in 8 frames
           break
      
      if f_obj_idx!=len(frames_with_subjobj_rel_cnt)-1:
        AnswerString +=","

    
    AnswerString +="}"

    return AnswerString, frame_indices, frames_where_subjobj_rel_is_present

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

   for vid_id in chunk_vid_data_keys:
      vid_data = data[vid_id]
      total_frames = data[vid_id]["meta"]["num_frames"]
      video_path = f"{data_root}{dataset}/videos/{vid_id}.mp4"
      if not os.path.exists(video_path):
            continue
      
      vid_objects = vid_data["objects"] # {'object_id': 2, 'category': 'countertop', 'is_thing': True, 'status': []},


      PromptAnswer = getPromptTemplate(media_path=video_path, media_type="video")

      # convQ = getConvBlock(value=getRandomPrompt(key='identify_subject_objects', static=True), 
      #                     conv_type="human", media_type="<video>", 
      #                     add_media_token=True)
      # AnswerStringObjs, frame_indices_obj = getListofCategoryString(vid_objects, vid_data, addObjectId=True,addBB=False,addFrames=False)
      # convA = getConvBlock(value=AnswerStringObjs, 
      #                     conv_type="gpt", media_type="<image>", 
      #                     add_media_token=False)
      # PromptAnswer["conversations"].append(convQ)
      # PromptAnswer["conversations"].append(convA)

      convQ = getConvBlock(value=getRandomPrompt(key='SGG', static=False), 
                          conv_type="human", media_type="<video>", 
                          add_media_token=True)
      AnswerStringRels, frame_indices_rel, frames_where_subjobj_rel_is_present = getObjectsRelations(vid_data["relations"], vid_data, uniform_sampling_idx=uniform_sampling_idx, add_frames=False)
      convA = getConvBlock(value=AnswerStringRels, 
                          conv_type="gpt", media_type="<video>", 
                          add_media_token=False)

      PromptAnswer["conversations"].append(convQ)
      PromptAnswer["conversations"].append(convA)

      # all_frame_indices = list(set(frame_indices_rel + frame_indices_obj))

      all_frame_indices = list(set(frame_indices_rel))
      PromptAnswer["frame_indices"] =  all_frame_indices
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

      
      obj_rel_bb_prompts = addObjectsRelations_bb_instructions(video_path=video_path,
                                                               vid_data=vid_data,
                                                               total_frames=total_frames,
                                                               subjobj_rel_frames_data=frames_where_subjobj_rel_is_present,
                                                               frame_indices=frame_indices_rel)
      
      with lock:
        for obj_rel_bb_prmpt in obj_rel_bb_prompts:
          if vid_id in train_ids:
            obj_rel_bb_prmpt["id"] = video_bb_annot_cnt["train"]
            video_gpt_bb_promptanswers.append(obj_rel_bb_prmpt)
            video_bb_annot_cnt["train"] +=1
          else:
            obj_rel_bb_prmpt["id"] = video_bb_annot_cnt["val"]
            video_gpt_bb_promptanswers_val.append(obj_rel_bb_prmpt)
            video_bb_annot_cnt["val"] +=1
         
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

    train_ids = anno["split"]['vidor']["train"]
    val_ids = anno["split"]['vidor']["val"]
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


    OUTPUT_JSON_DIR = "/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/video_llava_annotations_v12_2/"
    JSON_llava_image_tune_validate = f"{OUTPUT_JSON_DIR}/llava_image_tune_validate.json"
    JSON_llava_image_tune = f"{OUTPUT_JSON_DIR}/llava_image_tune_.json"
    JSON_videochatgpt_tune = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_.json"
    JSON_videochatgpt_tune_validate = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_validate.json"
    JSON_videochatgpt_bb_tune = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_bb.json"
    JSON_videochatgpt_bb_tune_validate = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_bb_validate.json"
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

    video_gpt_promptanswers = []
    video_gpt_promptanswers_val = []

    video_gpt_bb_promptanswers=  []
    video_gpt_bb_promptanswers_val =  []

    llava_image_tune = []
    llava_image_tune_val = []
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
    # print("Saved annotations", image_annot_cnt)


    """
    Video Annotations
    """

    pbar = tqdm(total=len(keys))
    pbar.n = 0
    pbar.last_print_n = 0
    pbar.refresh()


    for ch_idx, chunk_vid_data in enumerate(chunked_list):
      T = threading.Thread(target=prepare_vid_sg_threaded, name=f"Thread{ch_idx+1}", args=(chunk_vid_data,data,True,"vidor",8))
      T.start()
      threads.append(T)
    for th in threads:
       th.join()

    with open(JSON_videochatgpt_tune, "w") as f:
        json.dump(video_gpt_promptanswers,f)

    with open(JSON_videochatgpt_tune_validate, "w") as f:
        json.dump(video_gpt_promptanswers_val,f)
    print("Saved annotations", annot_cnt)

    with open(JSON_videochatgpt_bb_tune, "w") as f:
        json.dump(video_gpt_bb_promptanswers,f)
    with open(JSON_videochatgpt_bb_tune_validate, "w") as f:
        json.dump(video_gpt_bb_promptanswers_val,f)
    print("Saved annotations", video_bb_annot_cnt)


    