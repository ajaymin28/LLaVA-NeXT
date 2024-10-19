import os
import json
import glob
from tqdm import tqdm
import random
import re
import threading
import time


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

V6 Changes

[x] Normlized Temporal frame value
[x] bb values are rounded upto 3 decimal
    - issue : predicting wrong frame (second frame can be added)

V7 Changes

[X] child,sitting on,chair ==> child:sitting on:chair for easy parsing the results
[] Instead of average frame add frame start and end normlized
    - Above will not solve the temporal prediction problem
    - [X] Added frame wise annotations since Video-LLAVA takes only 8 frames from the video. This will be image-fine-tune dataset
      - issue: large number of frames, takes a lot of time to train

[X] - instead of taking all the frames from start to end taking only min max of annotations

V8 Changes

[X] for list of objects, added frame idx[0, 7] since video-llava takes 8 frames for the input video.
    - This will help when the object is visible in the video from start to end as acts as a temporal entity
    - object may be visible on all frames, but taken only first occurance when asked for a list of objects to avoid redundancy.

[] for relation of objects, add nearest frame_idx from [0,7] and its respective bounding box
    - here also multiple frames are possible from [0,7] but only the first occurance is taken. 


    
"""

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
        rel = vid_r[2]
        frames_ = vid_r[3].copy()
        if sub==Subject_id:
            return frames_
    return "None"


def get_best_frame(uniform_sampled_frames, frame_start_end):
  start, end = frame_start_end
  midpoint = (start + end) // 2
  # Find the nearest integer from the first list
  nearest = min(uniform_sampled_frames, key=lambda x: abs(x - midpoint))
  return nearest
   

def getListofCategoryString(vid_objects, vid_data, addObjectId=False, addFrames=False, addBB=False , uniform_sampling_idx=8):
    
    AnswerString = ""
    mask_size = None
    total_frames = vid_data["meta"]["num_frames"]
    frame_idxs = [x for x in range(total_frames)]
    every8thIndex = frame_idxs[0::uniform_sampling_idx]

    for idx, vobj in enumerate(vid_objects):
        category = vobj["category"]
        object_id = vobj["object_id"]
        frames_ = getFramesForObject(vid_data=vid_data, Subject_id=object_id)
        if frames_!="None":
          AnswerString += f"{category}"
          if addObjectId:
            AnswerString += f".{object_id}"
          if addBB:
            sub_bb = []
            frame_start, frame_end = frames_[0][0], frames_[0][1]
            nearest_frames = None
            nearest_frame_idx = None
            frame_for_object = int((frames_[0][0]+frames_[0][1])/2) # TODO: issue: objects can get ocluded when mid frame is taken
            
            for idx, uniform_sampled_frame_idx in enumerate(every8thIndex):
              # get objects which are present in current frame(nearest from uniform sampeled frames)
              if uniform_sampled_frame_idx>=frame_start and uniform_sampled_frame_idx<=frame_end:
                if nearest_frames is None:
                  nearest_frames = []
                nearest_frames.append(uniform_sampled_frame_idx)
                nearest_frame_idx = idx


            if nearest_frames is not None:
               frame_for_object = nearest_frames[0]
            try:
              sub_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_data["video_id"],frame_id=frame_for_object, object_id=object_id)
            except FileNotFoundError:
              sub_bb = []

            AnswerString += f".{sub_bb}"
          if addFrames:
            AnswerString += f"_[{nearest_frame_idx}]"  # index is fixed from 0 to 7 everytime, irrespective of total frames since video-llava takes only 8 frames

          if idx!=len(vid_objects)-1:
            AnswerString +="," 

    #AnswerString += f";image_height_width={mask_size}" 
    return AnswerString


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

    frame_idxs = [x for x in range(total_frames)]
    everynth = int(total_frames/uniform_sampling_idx)
    every8thIndex = frame_idxs[0::everynth+1]

    # print("every 8th index size ", len(every8thIndex))

    min_frame_idx, max_frame_idx = 1000, 0

    for _, vid_r in enumerate(vid_rels):
      """
      Ignore frames where there are no SGs
      """    
      rel_frames = vid_r[3]
      for frameSeq in rel_frames:
        frame_start, frame_end = frameSeq
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
                
                # nearest_frames = []
                # nearest_frame_idxes = []
                frame_for_subject_object = 0

                # for unisam_idx, uniform_sampled_frame_idx in enumerate(every8thIndex):
                #   # get objects which are present in current frame(nearest from uniform sampeled frames)
                #   if uniform_sampled_frame_idx>=frame_start and uniform_sampled_frame_idx<=frame_end:
                #     # nearest_frames.append(uniform_sampled_frame_idx)
                #     nearest_frame_idxes.append(unisam_idx)
                # if len(nearest_frame_idxes)>0:
                #    frame_for_subject_object = nearest_frame_idxes[0] # this acts as a temporal entity (on scale of 0 to 7)
                # else:
                #    frame_for_subject_object = "-1"

                # if frame_for_subject_object>8:
                #    print("frame_for_subject_object is higher than 8")

                AnswerString += f"[{vid_objects_by_id[sub]['category']}.{sub}.{sub_bb}:{rel}:{vid_objects_by_id[obj]['category']}.{obj}.{obj_bb}]_[{frame_for_subject_object}];"
                

      ListOfObjectsAnswerStr = ""
      ListOfObjectsAnswerStr_withbb= ""

      for objidx,objn in enumerate(ListOfObjects):
        ListOfObjectsAnswerStr += f"{objn}"
        ListOfObjectsAnswerStr_withbb += f"{objn}.[{ListOfObjects_bb[objidx]}]"
        if objidx!=len(ListOfObjects)-1:
           ListOfObjectsAnswerStr+=","
           ListOfObjectsAnswerStr_withbb+=","

      
      PromptAnswer = {
          "id": "TobeUpdated",
          "image": f"{image_path}",
          "conversations": [
            # Q&A for identifying objects in the scene
            {
              "from": "human",
              "value": f"<image>\n{getRandomPrompt(key='identify_subject_objects', static=True)}"
            },
            {
              "from": "gpt",
              "value": ListOfObjectsAnswerStr_withbb
            },
            {
              "from": "human",
              "value": f"{getRandomPrompt(key='identify_predicates', static=True)}"
            },
            {
              "from": "gpt",
              "value": AnswerString
            }
          ]
        }

      with lock:
        if vid_id in train_ids:
          PromptAnswer["id"] = image_annot_cnt["train"]
          llava_image_tune.append(PromptAnswer)
          image_annot_cnt["train"] +=3
        else:
          PromptAnswer["id"] = image_annot_cnt["val"]
          llava_image_tune_val.append(PromptAnswer)
          image_annot_cnt["val"] +=3

    with lock:
      pbar.n +=1
      pbar.last_print_n = pbar.n
      pbar.refresh()
          
  
  print(f"{threading.get_ident()} Exited after processing: {len(chunk_vid_data_keys)} data")



def prepare_image_sg_old(chunk_vid_data_keys,data, norm_bb=True, dataset="vidor", uniform_sampling_idx=8):
  global llava_image_tune, llava_image_tune_val, image_annot_cnt, pbar, train_ids

  print(f"{threading.get_ident()} Started, will process {len(chunk_vid_data_keys)} data")

  for vid_data_key in chunk_vid_data_keys:
    vid_data = data[vid_data_key]
    vid_id = vid_data["video_id"]
    vid_objects = vid_data["objects"] # {'object_id': 2, 'category': 'countertop', 'is_thing': True, 'status': []},
    vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in vid_objects} # for relations
    vid_rels = vid_data["relations"]

    total_frames = vid_data["meta"]["num_frames"]
    frame_idxs = [x for x in range(total_frames)]
    every8thIndex = frame_idxs[0::uniform_sampling_idx]

    min_frame_idx, max_frame_idx = 1000, 0

    for _, vid_r in enumerate(vid_rels):
      """
      Ignore frames where there are no SGs
      """    
      rel_frames = vid_r[3]
      for frameSeq in rel_frames:
        frame_start, frame_end = frameSeq
        if frame_start<=min_frame_idx:
          min_frame_idx = frame_start
        if frame_end>=max_frame_idx:
          max_frame_idx = frame_end

    for frame_idx in range(min_frame_idx, max_frame_idx+1):
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


                if sum(sub_bb)==0 and sum(obj_bb)==0:
                    continue
                

                # nearest_frames = []
                nearest_frame_idxes = []
                frame_for_subject_object = int((frame_start+frame_end)/2)
                for idx, uniform_sampled_frame_idx in enumerate(every8thIndex):
                  # get objects which are present in current frame(nearest from uniform sampeled frames)
                  if uniform_sampled_frame_idx>=frame_start and uniform_sampled_frame_idx<=frame_end:
                    # nearest_frames.append(uniform_sampled_frame_idx)
                    nearest_frame_idxes.append(idx)
                
                
                if len(nearest_frame_idxes)>0:
                   frame_for_subject_object = nearest_frame_idxes[0] # this acts as a temporal entity (on scale of 0 to 7)
                else:
                   frame_for_subject_object = "-1"

                AnswerString += f"[{vid_objects_by_id[sub]['category']}.{sub}.{sub_bb}:{rel}:{vid_objects_by_id[obj]['category']}.{obj}.{obj_bb}]_[{frame_for_subject_object}];"
                
                if f"{vid_objects_by_id[sub]['category']}.{sub}" not in ListOfObjects:
                   ListOfObjects.append(f"{vid_objects_by_id[sub]['category']}.{sub}")
                   ListOfObjects_bb.append(sub_bb)
                if f"{vid_objects_by_id[obj]['category']}.{obj}" not in ListOfObjects:
                   ListOfObjects.append(f"{vid_objects_by_id[obj]['category']}.{obj}")
                   ListOfObjects_bb.append(obj_bb)

      ListOfObjectsAnswerStr = ""
      ListOfObjectsAnswerStr_withbb= ""

      for objidx,objn in enumerate(ListOfObjects):
        ListOfObjectsAnswerStr += f"{objn}"
        ListOfObjectsAnswerStr_withbb += f"{objn}.[{ListOfObjects_bb[objidx]}]"
        if objidx!=len(ListOfObjects)-1:
           ListOfObjectsAnswerStr+=","
           ListOfObjectsAnswerStr_withbb+=","

      
      PromptAnswer = {
          "id": "TobeUpdated",
          "image": f"{image_path}",
          "conversations": [
            # Q&A for identifying objects in the scene
            {
              "from": "human",
              "value": f"<image>\n{getRandomPrompt(key='identify_subject_objects', static=True)}"
            },
            {
              "from": "gpt",
              "value": ListOfObjectsAnswerStr_withbb
            },
            {
              "from": "human",
              "value": f"{getRandomPrompt(key='identify_predicates', static=True)}"
            },
            {
              "from": "gpt",
              "value": AnswerString
            }
          ]
        }
      
      # image_sg = {
      #   "id": "TobeUpdated",
      #   "image": f"{image_path}",
      #   "conversations": [
      #     {
      #       "from": "human",
      #       "value": "<image>\nList the objects present in the image and list the actions, movements or placements of the objects in the scene."
      #     },
      #     {
      #       "from": "gpt",
      #       "value": AnswerString
      #     }
      #   ]
      # }

      # image_sg_objects = {
      #   "id": "TobeUpdated",
      #   "image": f"{image_path}",
      #   "conversations": [
      #     {
      #       "from": "human",
      #       "value": "<image>\nList the objects present in the image"
      #     },
      #     {
      #       "from": "gpt",
      #       "value": ListOfObjectsAnswerStr
      #     }
      #   ]
      # }

      # image_sg_objects_with_bb = {
      #   "id": "TobeUpdated",
      #   "image": f"{image_path}",
      #   "conversations": [
      #     {
      #       "from": "human",
      #       "value": "<image>\nList the objects present in the image,also provide bounding box coordinates for each object."
      #     },
      #     {
      #       "from": "gpt",
      #       "value": ListOfObjectsAnswerStr_withbb
      #     }
      #   ]
      # }

      with lock:
        if vid_id in train_ids:
          PromptAnswer["id"] = image_annot_cnt["train"]
          llava_image_tune.append(PromptAnswer)
          # llava_image_tune.append(image_sg_objects)
          # llava_image_tune.append(image_sg_objects_with_bb)
          image_annot_cnt["train"] +=3
        else:
          PromptAnswer["id"] = image_annot_cnt["val"]
          llava_image_tune_val.append(PromptAnswer)
          # llava_image_tune_val.append(image_sg_objects)
          # llava_image_tune_val.append(image_sg_objects_with_bb)
          image_annot_cnt["val"] +=3

    with lock:
      pbar.n +=1
      pbar.last_print_n = pbar.n
      pbar.refresh()
          
  
  print(f"{threading.get_ident()} Exited after processing: {len(chunk_vid_data_keys)} data")


def prepare_vid_sg_threaded(chunk_vid_data_keys,data, norm_bb=True, dataset="vidor", uniform_sampling_idx=8):
   

   global video_gpt_promptanswers, video_gpt_promptanswers_val, annot_cnt

   for vid_id in chunk_vid_data_keys:
      vid_data = data[vid_id]
      video_path = f"{data_root}vidor/videos/{vid_id}.mp4"
      if not os.path.exists(video_path):
            continue
      
      vid_objects = vid_data["objects"] # {'object_id': 2, 'category': 'countertop', 'is_thing': True, 'status': []},
      # vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in vid_objects} # for relations
      # vid_rels = vid_data["relations"]
    
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
            "value": getObjectsRelations(vid_data["relations"], vid_data, uniform_sampling_idx=uniform_sampling_idx)
          }
        ]
      }


      with lock:
        if vid_id in train_ids:
          PromptAnswer["id"] = annot_cnt["train"]
          video_gpt_promptanswers.append(PromptAnswer)
          annot_cnt["train"] +=1
        else:
          PromptAnswer["id"] = annot_cnt["val"]
          video_gpt_promptanswers_val.append(PromptAnswer)
          annot_cnt["val"] +=1

      
      with lock:
        pbar.n +=1
        pbar.last_print_n = pbar.n
        pbar.refresh()

      # append_annotation(vid_data["video_id"],annotation=PromptAnswer)
   


def getObjectsRelations(vid_rels, vid_data, norm_frames=True, add_frames=True, uniform_sampling_idx=8):
    """
    uniform_sampling_idx=8 for Video-LLAVA

    1. Video-LLAVA uses uniform sampling of every nth Frame and it defaults to 8th.
    2. Taking every 8th frame only here for taking annotations rather than random sampling or all, 
    this will help better bounding box predition and frame predition where event is happening.
    """
    AnswerString = ""
    SubObjRel = []
    # mask_size = None

    total_frames = vid_data["meta"]["num_frames"]
    frame_idxs = [x for x in range(total_frames)]
    everynth = int(total_frames/uniform_sampling_idx)
    every8thIndex = frame_idxs[0::everynth+1]

    vid_objects = vid_data["objects"] # {'object_id': 2, 'category': 'countertop', 'is_thing': True, 'status': []},
    vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in vid_objects} # for relations
    vid_rels = vid_data["relations"]

    for idx, vid_r in enumerate(vid_rels):
        sub = vid_r[0]
        obj = vid_r[1]
        rel = vid_r[2]
        frames = vid_r[3].copy()

        sub_bb_list = []
        obj_bb_list = []
        obj_bb_frame_list = []

        sub_bb = []
        obj_bb = []

        frame_start, frame_end = frames[0][0], frames[0][1]
        nearest_frames = []
        nearest_frames_idxes = []
        # frame_for_object = int((frames[0][0]+frames[0][1])/2)


        for frame_for_bb_idx in range(frame_start, frame_end+1):
           # use frames where subject and object both are visible.
          try:
            sub_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_data["video_id"],frame_id=frame_for_bb_idx, object_id=sub)
          except FileNotFoundError:
            #print(f"[Warning] Frame {frame_for_bb_idx} not found for vidor {vid_data['video_id']}")
            pass
          
          try:
            obj_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_data["video_id"],frame_id=frame_for_bb_idx, object_id=obj)
          except FileNotFoundError:
            #print(f"[Warning] Frame {frame_for_bb_idx} not found for vidor {vid_data['video_id']}")
            pass

          if sum(sub_bb)>0 and sum(obj_bb)>0:
            sub_bb_list.append(sub_bb)
            obj_bb_list.append(obj_bb)
            obj_bb_frame_list.append(frame_for_bb_idx)  # frame where both objects are visible
          else:
            pass
            #  print(sub_bb, obj_bb, vid_r, frame_for_bb_idx)
           
        
        if len(obj_bb_list)>0 and len(obj_bb_list)==len(sub_bb_list):
          bb_index = -1
          min_dist = None
          min_dist_idx = -1
          for uni_sample_idx, uniform_sampled_frame_idx in enumerate(every8thIndex):
            # get objects which are present in current frame(nearest from uniform sampeled frames)
            if uniform_sampled_frame_idx>=frame_start and uniform_sampled_frame_idx<=frame_end:
                
                if uniform_sampled_frame_idx in obj_bb_frame_list:
                  bb_index = obj_bb_frame_list.index(uniform_sampled_frame_idx)
                  # get the index from 0 to 7 which is nearest from <obj_bb_frame_list> because these frame contains bb
                  nearest_frames.append(uniform_sampled_frame_idx) # actual video frame from total frames
                  nearest_frames_idxes.append(uni_sample_idx) # idx are only from 0 to 7 for video-llava for all videos
                else:
                  # Take nearest frame where both obj and sub are visible.
                  for objbb_idx, f_idx in enumerate(obj_bb_frame_list):
                      min_dist_cur = abs(f_idx-uniform_sampled_frame_idx)
                      if min_dist is None:
                          min_dist = min_dist_cur
                          min_dist_idx = objbb_idx
                      else:
                          if min_dist_cur<min_dist:
                              min_dist = min_dist_cur
                              min_dist_idx = objbb_idx

          # TODO: here multiple frames and bb may be available but taken only first.
          # Since BB for each frame may be different, makes no sense sending all frames BB as an output from LLM. 
          # We care about every 8th frame and taken only 1st out of them here.
          # Seperately image dataset for all frames can be trained. [TODO]
          
          if bb_index==-1:
            bb_index = min_dist_idx
            frame_for_subject_object = min_dist_idx
            # nearest_frames.append(min_dist_idx)
            # nearest_frames_idxes.append(idx)
          else:
            frame_for_subject_object = nearest_frames_idxes[0]  # out of 8 frames in which frame the event happens


          if bb_index>=0:
            # print(len(sub_bb_list), "sb bb list len", " bb indexed", bb_index, vid_r, vid_data["video_id"])
            sub_bb = sub_bb_list[bb_index]
            obj_bb = obj_bb_list[bb_index]

            # if norm_frames:
            #    avg_frame = round(avg_frame/total_frames,3)

            if [sub,rel,obj] not in SubObjRel:
                if not add_frames:
                  AnswerString += f"[{vid_objects_by_id[sub]['category']}.{sub}.{sub_bb}:{rel}:{vid_objects_by_id[obj]['category']}.{obj}.{obj_bb}]" 
                else:
                  AnswerString += f"[{vid_objects_by_id[sub]['category']}.{sub}.{sub_bb}:{rel}:{vid_objects_by_id[obj]['category']}.{obj}.{obj_bb}_[{frame_for_subject_object}]]"
                
                #AnswerString += f"[{vid_objects_by_id[sub]['category']}.{sub}.{sub_bb},{rel},{vid_objects_by_id[obj]['category']}.{obj}].{obj_bb}]"
            if idx!=len(vid_rels)-1:
                AnswerString +=";"

    # AnswerString += f";image_height_width={mask_size}" #v5 removed 
    return AnswerString


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

    data = {data_dict['video_id']: data_dict for data_dict in anno['data']}

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


    OUTPUT_JSON_DIR = "/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/video_llava_annotations_v8_2/"
    JSON_llava_image_tune_validate = f"{OUTPUT_JSON_DIR}/llava_image_tune_validate.json"
    JSON_llava_image_tune = f"{OUTPUT_JSON_DIR}/llava_image_tune_.json"
    JSON_videochatgpt_tune = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_.json"
    JSON_videochatgpt_tune_validate = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_validate.json"
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

    # video_questions_counter = {}
    # video_questions = []
    # video_answers = []
    video_gpt_promptanswers = []
    video_gpt_promptanswers_val = []

    llava_image_tune = []
    llava_image_tune_val = []
    image_annot_cnt = {"train": 0, "val": 0}

    annot_cnt = {"train": 0, "val": 0}

    print("Total videos ",len(keys))

    pbar = tqdm(total=len(keys))
    pbar.n = 0
    pbar.last_print_n = 0
    pbar.refresh()


    """
    Image Annotations
    """

    # # for vid_id, vid_data in tqdm(data.items()):
    # #    prepare_image_sg(keys,data,True,"vidor")
    
    for ch_idx, chunk_vid_data in enumerate(chunked_list):
      T = threading.Thread(target=prepare_image_sg, name=f"Thread{ch_idx+1}", args=(chunk_vid_data,data,True,"vidor"))
      T.start()
      threads.append(T)
    for th in threads:
       th.join()

    with open(JSON_llava_image_tune, "w") as f:
        json.dump(llava_image_tune,f)

    with open(JSON_llava_image_tune_validate, "w") as f:
        json.dump(llava_image_tune_val,f)
    print("Saved annotations", image_annot_cnt)


    """
    Video Annotations
    """

    # pbar = tqdm(total=len(keys))
    # pbar.n = 0
    # pbar.last_print_n = 0
    # pbar.refresh()


    # for ch_idx, chunk_vid_data in enumerate(chunked_list):
    #   T = threading.Thread(target=prepare_vid_sg_threaded, name=f"Thread{ch_idx+1}", args=(chunk_vid_data,data,True,"vidor",8))
    #   T.start()
    #   threads.append(T)
    # for th in threads:
    #    th.join()

    # with open(JSON_videochatgpt_tune, "w") as f:
    #     json.dump(video_gpt_promptanswers,f)

    # with open(JSON_videochatgpt_tune_validate, "w") as f:
    #     json.dump(video_gpt_promptanswers_val,f)
    # print("Saved annotations", annot_cnt)



    # for vid_id, vid_data in tqdm(data.items()):
    #     video_path = f"{data_root}vidor/videos/{vid_id}.mp4"
    #     if not os.path.exists(video_path):
    #         continue

    #     vid_objects = vid_data["objects"] # {'object_id': 2, 'category': 'countertop', 'is_thing': True, 'status': []},
    #     vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in vid_objects} # for relations
    #     vid_rels = vid_data["relations"]

    #     PromptAnswer = {
    #       "id": annot_cnt,
    #       "video": video_path,
    #       "conversations": [
    #         # Q&A for identifying objects in the scene
    #         {
    #           "from": "human",
    #           "value": f"<video>\n{getRandomPrompt(key='identify_subject_objects', static=True)}"
    #         },
    #         {
    #           "from": "gpt",
    #           "value": getListofCategoryString(vid_objects, vid_data, addObjectId=True,addBB=True,addFrames=True)
    #         },
    #         {
    #           "from": "human",
    #           "value": f"{getRandomPrompt(key='identify_predicates', static=True)}"
    #         },
    #         {
    #           "from": "gpt",
    #           "value": getObjectsRelations(vid_data["relations"], vid_data)
    #         }
    #       ]
    #     }

    #     append_annotation(vid_data["video_id"],annotation=PromptAnswer)

    