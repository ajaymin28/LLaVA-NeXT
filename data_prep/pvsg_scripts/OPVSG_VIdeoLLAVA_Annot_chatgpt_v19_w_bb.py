import os
import json
import glob
from tqdm import tqdm
import random
import re
import threading
import time
import copy
from utils.utilities import getConvBlock, getPromptTemplate, getRandomPrompt, get_frame_range_for_annotations
from utils.utilities import SGSpecialTokens, getboundingBoxOftheObject
# from utils.utilities import getFramesForObject, create_batch_frames
"""
TODO

Full bounding box[x1,y1,x2,y2] instead of [[x1y1],[x2y2]]

19_w_bb
BB as seperate instructions

"""


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

    
def getQnACounter(vid_id):
    global video_questions_counter
    if vid_id not in video_questions_counter:
        video_questions_counter[vid_id] = 0
    else:
        video_questions_counter[vid_id] +=1
    qna_counter = video_questions_counter[vid_id]
    return qna_counter



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


def addObjectsRelations_bb_instructions(video_path,vid_data,total_frames, subjobj_rel_frames_data,frame_indices,bb_per_object):
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

        curr_frame_idx = frame_indices.index(frame_idx)

        # "Provide bounding box location of [{sub}:{rel}:{obj}] in frame {frame_idx} of the provided video" # {} to be replaced by actual value
        convQ["value"] = convQ["value"].replace("{sub}", f"{SGSpecialTokens.SG_SUBJECT}'{subject_category_name}-{SGSpecialTokens.SG_SUBJECT_ID}{sub}'")
        convQ["value"] = convQ["value"].replace("{rel}", f"{SGSpecialTokens.SG_PREDICATE}'{rel}'")
        convQ["value"] = convQ["value"].replace("{obj}", f"{SGSpecialTokens.SG_OBJECT}'{object_category_name}-{SGSpecialTokens.SG_OBJECT_ID}{obj}'")
        convQ["value"] = convQ["value"].replace("{frame_idx}", str(curr_frame_idx))

        resp = ""
        for fi in range(len(frame_indices)):
          if fi==curr_frame_idx:
             resp += f"{SGSpecialTokens.VIDEO_FRAME_ID}[{SGSpecialTokens.SG_SUBJECT}'{subject_category_name}-{SGSpecialTokens.SG_SUBJECT_ID}{sub}'_{SGSpecialTokens.SG_BB_START}{sub_bb}{SGSpecialTokens.SG_BB_END}:{SGSpecialTokens.SG_PREDICATE}'{rel}':{SGSpecialTokens.SG_OBJECT}'{object_category_name}-{SGSpecialTokens.SG_OBJECT_ID}{obj}'_{SGSpecialTokens.SG_BB_START}{obj_bb}{SGSpecialTokens.SG_BB_END}]];{SGSpecialTokens.SG_END}"
          else:
             resp += f"{SGSpecialTokens.VIDEO_FRAME_ID}{SGSpecialTokens.SG_END}"
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


    overall_annotations = []
    
    frame_counter = 0
    annotation_total_frame_count = len(frames_where_subjobj_rel_is_present.keys())
    remaining_frames_after_batching = annotation_total_frame_count%uniform_sampling_idx
    # frames_needs_to_be_added = uniform_sampling_idx - remaining_frames_after_batching
    frame_indices = []

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

        if sum(sub_bb)>0 and sum(obj_bb)>0:

          sub_category = vid_objects_by_id[sub]['category']
          obj_category = vid_objects_by_id[obj]['category']

          tripletes_for_current_block += f"[{SGSpecialTokens.SG_SUBJECT}'{sub_category}-{SGSpecialTokens.SG_SUBJECT_ID}{sub}'"
          tripletes_for_current_block += f":{SGSpecialTokens.SG_PREDICATE}'{rel}'"
          tripletes_for_current_block += f":{SGSpecialTokens.SG_OBJECT}'{obj_category}-{SGSpecialTokens.SG_OBJECT_ID}{obj}'"
          tripletes_for_current_block += f"];"

      frame_indices.append(key_frame_idx)
      frame_counter +=1

      if len(frame_indices)>=8:
        
        overall_annotations.append({
            "frame_idxes": frame_indices,
            "frames_sgs": tripletes_for_current_block+f"{SGSpecialTokens.SG_END}"
        })

        tripletes_for_current_block = ""
        frame_counter = 0
        frame_indices = []

    # TODO add remaining annotations for last block

    return overall_annotations,AnswerString,AnswerString_with_bb, frame_indices, frames_where_subjobj_rel_is_present


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

        obj_rel_bb_prompts, bb_per_object = addObjectsRelations_bb_instructions(video_path=video_path,
                                                                vid_data=vid_data,
                                                                total_frames=total_frames,
                                                                subjobj_rel_frames_data=frames_where_subjobj_rel_is_present,
                                                                frame_indices=frame_indices,
                                                                bb_per_object=bb_per_object)
        with lock:
          for obj_rel_bb_prmpt in obj_rel_bb_prompts:
            if vid_id in train_ids:
              obj_rel_bb_prmpt["id"] = annot_cnt["train"] # video_bb_annot_cnt["train"]
              video_gpt_bb_promptanswers.append(obj_rel_bb_prmpt)
              annot_cnt["train"] +=1
            else:
              obj_rel_bb_prmpt["id"] = annot_cnt["val"] # video_bb_annot_cnt["val"]
              video_gpt_bb_promptanswers_val.append(obj_rel_bb_prmpt)
              annot_cnt["val"] +=1
         
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


    OUTPUT_JSON_DIR = "/lustre/fs1/home/jbhol/dso/gits/OpenPVSG/data/video_llava_annotations_v19_w_bb/"
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

    with open(JSON_videochatgpt_bb_tune, "w") as f:
        json.dump(video_gpt_bb_promptanswers,f)
    with open(JSON_videochatgpt_bb_tune_validate, "w") as f:
        json.dump(video_gpt_bb_promptanswers_val,f)
    print("Saved annotations", video_bb_annot_cnt)