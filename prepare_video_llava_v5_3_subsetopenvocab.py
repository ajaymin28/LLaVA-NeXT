# import sys
# sys.path.append("../")

"""
Varying list for list of objects and predicates
Subset of Objects and Predicates for openvocab test
"""



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
from utils.utilities import SGSpecialTokens, getboundingBoxOftheObject, unnormbb_vidvrd
from utils.utilities import get_bb_subj_obj
from utils.utilities import chunk_list
import cv2
from utils.utilities import SEEDS, addTriplet
from vidvrd2dataset import VidVRD, VidOR

seed_manager = SEEDS()
random.seed(seed_manager.RANDOM_ANNOTATIONS_SHUFFLE_SEED)
# from utils.utilities import getFramesForObject, create_batch_frames


# import matplotlib.pyplot as plt
# import cv2
# from PIL import Image
# import numpy as np

def get_varying_list(current_block_list, full_list, fix_size=30):
	"""
	1. take current list (shuffle it)
	2. add elements to current list from full list without repeatation that sums to fix_size (shuffle it again)
	3. return the list
	"""
	current_block_list = set(copy.deepcopy(current_block_list))
	full_list = set(copy.deepcopy(full_list))

	newelements = full_list.difference(current_block_list)

	current_block_list = list(current_block_list)
	newelements =  list(newelements)
	newElementsNeeded = 0
	currentElementsSize = len(current_block_list) 
	if currentElementsSize>fix_size:
		## more items than predefined limit
		newElementsNeeded = 0
		pass
	else:
		newElementsNeeded = fix_size - len(current_block_list) 

	if len(newelements)<newElementsNeeded:
		current_block_list = current_block_list + random.sample(newelements,k=len(newelements))
	else:
		current_block_list = current_block_list + random.sample(newelements,k=newElementsNeeded)

	random.shuffle(current_block_list)
	return current_block_list



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
    # annotation_total_frame_count = len(frames_where_subjobj_rel_is_present.keys())
    # remaining_frames_after_batching = annotation_total_frame_count%uniform_sampling_idx
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

def validate_annotations(annot):
		videopath = annot["video"]
		frame_indices = annot["frame_indices"]
		frames_with_error = []
		capture = cv2.VideoCapture(videopath)
		for frame_idx in frame_indices:    
			try:
				capture.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
				ret, frame = capture.read()
				if not ret:
					frames_with_error.append(frame_idx)
			except Exception as e:
					frames_with_error.append(frame_idx)
		
		capture.release()
		if len(frames_with_error)>0:
			print(f"error in indexes for {videopath} frames: {frames_with_error}")      
		return frames_with_error

def prepare_vid_sg(data_root,subset, annotations,norm_bb=True, dataset="vidor", uniform_sampling_idx=8, dataset_meta=None):

	global video_gpt_promptanswers, video_gpt_promptanswers_val, annot_cnt, OUTPUT_JSON_DIR, annotation_files_counter_val
	global video_gpt_bb_promptanswers, video_gpt_bb_promptanswers_val, video_bb_annot_cnt, annotation_files_counter
	global print_sample_annotations

	videos_root = os.path.join(data_root, "videos")
	round_bb_by = 3
	MAX_OBJECTS_LIMIT_IN_PROMPT = 10
	MAX_PREDICATES_LIMIT_IN_PROMPT = 10

	profiling_done = False
	

	# bb_per_object = {}
	for annot_idx, annot in enumerate(annotations):
		# t_start = time.perf_counter()
		# if annot_idx>0:
		# 	break
		overall_annotations = []
        
		frame_h, frame_w = annot["height"], annot["width"]
		frame_count = annot["frame_count"]
		video_id = annot["video_id"]
		video_fps = annot["fps"]
        
		video_path = os.path.join(videos_root, video_id+".mp4")
		
		sub_ob_jects = annot['subject/objects']
		sub_ob_jects_by_id = {obj["tid"]: obj  for obj in sub_ob_jects}
		
		rels = annot['relation_instances']
		trajectories = annot['trajectories']
        
		# total_frames_list = [i for i in range(frame_count)]
		# total_frames_list_bb = [i for i in range(frame_count)]

		frames_dict = {}
		for i in range(frame_count):
			if i not in frames_dict.keys():
				frames_dict[i] = {
					"triplets": [],
					"bbox": []
				}

		# print(f"dict len {len(frames_dict.keys())} frame count : {frame_count}")
		assert len(frames_dict.keys())<=frame_count

		# t_start1 = time.perf_counter()
		for rel in rels:
			begin_fid = rel['begin_fid']
			end_fid = rel['end_fid']
			subject_tid =rel['subject_tid']
			predicate = rel['predicate']
			object_tid = rel['object_tid']

			for activity_range in range(begin_fid,end_fid):
				subj_data = sub_ob_jects_by_id[subject_tid]
				obj_data = sub_ob_jects_by_id[object_tid]

				current_frame_traj = trajectories[activity_range]
				sub_bb, obj_bb = None, None
				for curr_trj in current_frame_traj:
					if curr_trj["tid"]==subject_tid:
						sub_bb = curr_trj["bbox"]
					if curr_trj["tid"]==object_tid:
						obj_bb = curr_trj["bbox"]

				
				if activity_range>frame_count:
					continue
	
				frames_dict[activity_range]["triplets"].append([f"{subj_data['category']}-{subj_data['tid']}", predicate, f"{obj_data['category']}-{obj_data['tid']}"])
				frames_dict[activity_range]["bbox"].append([sub_bb, obj_bb])

		assert len(frames_dict.keys())<=frame_count

		capture = None
		frame_indices = []	
		current_block_triplets = []
		current_block_triplet_data = {
			"subjects": [],
			"objects": [],
			"predicates": []
		}
		if os.path.exists(video_path):

			tripletes_for_current_block = ""
			for frame_idx, frame_data in frames_dict.items():
				if len(frame_data["triplets"])==0:
					continue

				if frame_idx>frame_count:
					continue

				
				tripletes_for_current_block += f"{SGSpecialTokens.VIDEO_FRAME_ID}"

				max_triplets_to_add = 10
				added_triplets = []
				current_frame_triplets = []
				current_frame_subj_obj_pairs = []
				current_frame_subj_obj_pairs_predicate = []
				for index_to_draw, triplet in enumerate(frame_data["triplets"]):
								
					subj = triplet[0]
					predicate = triplet[1]
					obj = triplet[2]
					subj, subj_id = subj.split("-")
					obj, obj_id = obj.split("-")

					if "_" in predicate:
						predicate = predicate.replace("_", " ")
					if "_" in subj:
						subj = subj.replace("_", " ")
					if "_" in obj:
						obj = obj.replace("_", " ")

					if addTriplet([subj,predicate,obj],Objects=dataset_meta["objects"],relations=dataset_meta["predicates"]):

						construct_triplet = f"[{subj}-{subj_id}"
						construct_triplet += f":{obj}-{obj_id}"
						construct_triplet += f":{predicate}"
						construct_triplet += f"];"

						if subj not in current_block_triplet_data["subjects"]:
							current_block_triplet_data["subjects"].append(subj)
						
						if obj not in current_block_triplet_data["objects"]:
							current_block_triplet_data["objects"].append(obj)

						if predicate not in current_block_triplet_data["predicates"]:
							current_block_triplet_data["predicates"].append(predicate)

						if construct_triplet not in added_triplets:
								tripletes_for_current_block += construct_triplet
								added_triplets.append(construct_triplet)
								current_frame_triplets.append([f"{subj}-{subj_id}",f"{predicate}",f"{obj}-{obj_id}"])
							
				

				if len(current_frame_triplets)>0:
					frame_indices.append(frame_idx)
					current_block_triplets.append(current_frame_triplets)
					# print(f"added frame : {frame_idx} Total frame count: {frame_count} {frame_idx>frame_count}")

					if len(frame_indices)>=8:
						overall_annotations.append({
							"frame_idxes": frame_indices,
							"frames_sgs": tripletes_for_current_block+f"{SGSpecialTokens.SG_END}",
							"triplets_list": current_block_triplets,
							"current_block_triplet_data" : copy.deepcopy(current_block_triplet_data)
						})

						tripletes_for_current_block = ""
						frame_indices = []
						current_block_triplets = []
						current_block_triplet_data = {
							"subjects": [],
							"objects": [],
							"predicates": []
						}

		for overall_annot in overall_annotations:
			# t_start2 = time.perf_counter()

			# SG without grounding
			video_path =  f"{video_id}.mp4"  # LLAVA TRAINING TAKES VIDEO ROOT, SO ONLY VID ID is needed
			PromptAnswer = getPromptTemplate(media_path=video_path, media_type="video")

			frame_indices_for_sg_block = copy.deepcopy(overall_annot["frame_idxes"])
			tripletes_for_current_block = copy.deepcopy(overall_annot["frames_sgs"])
			current_block_triplet_data = copy.deepcopy(overall_annot["current_block_triplet_data"])
			
			final_subjects_list = get_varying_list(current_block_list=current_block_triplet_data["subjects"], 
										  full_list=dataset_meta["objects"], 
										  fix_size=MAX_OBJECTS_LIMIT_IN_PROMPT) 

			final_objects_list = get_varying_list(current_block_list=current_block_triplet_data["objects"], 
										  full_list=dataset_meta["objects"], 
										  fix_size=MAX_OBJECTS_LIMIT_IN_PROMPT)

			final_predicates_list = get_varying_list(current_block_list=current_block_triplet_data["predicates"], 
										  full_list=dataset_meta["predicates"], 
										  fix_size=MAX_PREDICATES_LIMIT_IN_PROMPT) # total 132 predicates in vidvrd

			
			TripletQ = getRandomPrompt(key='triplet_prompt', static=False)
			TripletQ = TripletQ.replace("{subjects}", ",".join(final_subjects_list))
			TripletQ = TripletQ.replace("{objects}", ",".join(final_objects_list))
			TripletQ = TripletQ.replace("{predicates}", ",".join(final_predicates_list))

			convQ = getConvBlock(value=TripletQ, 
								conv_type="human", media_type="<video>", 
								add_media_token=True)
			convA = getConvBlock(value=tripletes_for_current_block, 
								conv_type="gpt", media_type="<video>", 
								add_media_token=False)

			PromptAnswer["conversations"].append(convQ)
			PromptAnswer["conversations"].append(convA)

			# subj_obj_pair_with_predicates = {}	
			# for frame_idx, tripletData in enumerate(overall_annot["triplets_list"]):
			# 	for trip_idx , triplet in enumerate(tripletData):
			# 		subject_,predicate_,object_ = triplet
			# 		const_key = f"{subject_}:{object_}"
			# 		if const_key not in subj_obj_pair_with_predicates.keys():
			# 			subj_obj_pair_with_predicates[const_key] = {}

			# 		if frame_idx not in subj_obj_pair_with_predicates[const_key].keys():
			# 			subj_obj_pair_with_predicates[const_key][frame_idx] = []

			# 		subj_obj_pair_with_predicates[const_key][frame_idx].append([subject_,predicate_,object_])
			# counter = 0
			# for subobjpair, subobjpairdata_frame in subj_obj_pair_with_predicates.items():
			# 	if counter>2:
			# 		break

			# 	subject_,object_ = subobjpair.split(":")
			# 	constructed_answer = ""
			# 	for frameid_subobjpair, subobjpairdata in subobjpairdata_frame.items():
			# 		# added_triplet = []
			# 		constructed_answer += f"{SGSpecialTokens.VIDEO_FRAME_ID}"
			# 		for sbob_triplets in subobjpairdata:
			# 			sub_,pred_,obj_ =sbob_triplets
			# 			construct_trip = f"[{sub_}:{obj_}:{pred_}];"
			# 			# if construct_trip not in added_triplet:
			# 			# 	added_triplet.append(construct_trip)
			# 			constructed_answer += f"{construct_trip}"

			# 	# "What is the relationship between [{sub}:{obj}] in the video" # {} to be replaced by actual value
			# 	predict_predicate_Q = getRandomPrompt(key='predict_predicate', static=False)
			# 	predict_predicate_Q = predict_predicate_Q.replace("{sub}", subject_)
			# 	predict_predicate_Q = predict_predicate_Q.replace("{obj}", object_)
			# 	predict_predicate_Q = predict_predicate_Q.replace("{predicates}", ",".join(dataset_meta["predicates"]))

			# 	convQ = getConvBlock(value=predict_predicate_Q, 
			# 					conv_type="human", media_type="<video>", 
			# 					add_media_token=False)
				
			# 	convA = getConvBlock(value=constructed_answer, 
			# 					conv_type="gpt", media_type="<video>", 
			# 					add_media_token=False)
				
			# 	PromptAnswer["conversations"].append(convQ)
			# 	PromptAnswer["conversations"].append(convA)

			# 	counter +=1
			

			# all_frame_indices = list(set(frame_indices_rel + frame_indices_obj))
			# all_frame_indices = list(set(frame_indices_rel))

			PromptAnswer["frame_indices"] =  frame_indices_for_sg_block
			PromptAnswer["total_frames"] = frame_count

		
			if subset=="train":
				PromptAnswer["id"] = annot_cnt["train"]
				video_gpt_promptanswers.append(PromptAnswer)
				annot_cnt["train"] +=1
				# if print_sample_annotations:
				# 	print(video_gpt_promptanswers[0])
				# 	print_sample_annotations = False

				# if len(video_gpt_promptanswers)>=1000:
				# 	JSON_videochatgpt_tune = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_part{annotation_files_counter}.json"
				# 	with open(JSON_videochatgpt_tune, "w") as f:
				# 		json.dump(video_gpt_promptanswers,f)
					
				# 	annotation_files_counter +=1
				# 	video_gpt_promptanswers = []
			else:
				PromptAnswer["id"] = annot_cnt["val"]
				video_gpt_promptanswers_val.append(PromptAnswer)
				annot_cnt["val"] +=1

				# if len(video_gpt_promptanswers_val)>=1000:
				# 	JSON_videochatgpt_tune_validate = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_validate_part{annotation_files_counter_val}.json"
				# 	with open(JSON_videochatgpt_tune_validate, "w") as f:
				# 		json.dump(video_gpt_promptanswers_val,f)
					
				# 	annotation_files_counter_val +=1
				# 	video_gpt_promptanswers_val = []

			# t_start4 = time.perf_counter()	



			# if len(video_gpt_promptanswers)>=0:
			# 	JSON_videochatgpt_tune = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_part{annotation_files_counter}.json"
			# 	with open(JSON_videochatgpt_tune, "w") as f:
			# 		json.dump(video_gpt_promptanswers,f)
			# 	annotation_files_counter +=1
			# 	video_gpt_promptanswers = []

			# if len(video_gpt_promptanswers_val)>=1000:
			# 	JSON_videochatgpt_tune_validate = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_validate_part{annotation_files_counter_val}.json"
			# 	with open(JSON_videochatgpt_tune_validate, "w") as f:
			# 		json.dump(video_gpt_promptanswers_val,f)
			# 	annotation_files_counter_val +=1
			# 	video_gpt_promptanswers_val = []
							

			# obj_rel_bb_prompts, bb_per_object = addObjectsRelations_bb_instructions(video_path=video_path,
			# 														vid_data=vid_data,
			# 														total_frames=total_frames,
			# 														subjobj_rel_frames_data=frames_where_subjobj_rel_is_present,
			# 														frame_indices=frame_indices,
			# 														bb_per_object=bb_per_object)
			# with lock:
			# 	for obj_rel_bb_prmpt in obj_rel_bb_prompts:
			# 		if vid_id in train_ids:
			# 		obj_rel_bb_prmpt["id"] = annot_cnt["train"] # video_bb_annot_cnt["train"]
			# 		video_gpt_bb_promptanswers.append(obj_rel_bb_prmpt)
			# 		annot_cnt["train"] +=1
			# 		else:
			# 		obj_rel_bb_prmpt["id"] = annot_cnt["val"] # video_bb_annot_cnt["val"]
			# 		video_gpt_bb_promptanswers_val.append(obj_rel_bb_prmpt)
			# 		annot_cnt["val"] +=1
			
		# t_start5 = time.perf_counter()	

		# with lock:
		pbar.n +=1
		pbar.last_print_n = pbar.n
		pbar.refresh()

		# if not profiling_done:
		# 	print(f"for one video it took: {t_start5-t_start}")
		# 	print(f"perparing one annotation took: {t_start2_end-t_start2}")
		# 	print(f"saving annotation with lock took:  {t_start4-t_start3}s")
		# 	profiling_done = True
		

		# append_annotation(vid_data["video_id"],annotation=PromptAnswer)
   

# def chunk_list(list_, chunk_n):
#     chunk_n = max(1, chunk_n)
#     return (list_[i:i+chunk_n] for i in range(0, len(list_), chunk_n))

if __name__=="__main__":	

		splits = ["train","test"]
		imagenet_vidvrd_root = "/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd"
		imagenet_vidvrd_video_path = os.path.join(imagenet_vidvrd_root, "videos")

		dataset = VidVRD(imagenet_vidvrd_root, imagenet_vidvrd_video_path, splits)

		n_thread_count = 1
		per_thread_data = 0
		CHUNK_N = 250
		threads = []

		lock = threading.Lock()

		data_root = '/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd'
		test_data_dir = os.path.join(data_root, "test")
		test_anno_files = glob.glob(f"{test_data_dir}/*.json") 

		dataset_meta = {
			"subjects" :[],
			"objects": [],
			"predicates": []
		}

		test_annotations = []
		for test_annot in test_anno_files:
				filename = os.path.basename(test_annot)
				filename = filename.split(".")[0]
				annot = dataset.get_anno(vid=filename)
				test_annotations.append(annot)

				rels = annot['relation_instances']
				sub_ob_jects = annot['subject/objects']

				for rel in rels:
					predicate_wo_underscore = rel["predicate"].replace("_", " ")
					if predicate_wo_underscore not in dataset_meta["predicates"]:
						dataset_meta["predicates"].append(predicate_wo_underscore)
				for sbobjs in sub_ob_jects:
					sub_category = sbobjs["category"]
					if "_" in sub_category:
						sub_category = sub_category.replace("_", " ")
					if sub_category not in dataset_meta["objects"]:
						dataset_meta["objects"].append(sub_category)
					# if sbobjs["category"] not in dataset_meta["objects"]:
					# 	dataset_meta["objects"].append(sbobjs["category"])

		
		train_data_dir = os.path.join(data_root, "train")
		train_anno_files = glob.glob(f"{train_data_dir}/*.json")

		train_annotations = []
		for train_annot in train_anno_files:
				filename = os.path.basename(train_annot)
				filename = filename.split(".")[0]
				annot = dataset.get_anno(vid=filename)
				train_annotations.append(annot)

				rels = annot['relation_instances']
				sub_ob_jects = annot['subject/objects']

				for rel in rels:
					predicate_wo_underscore = rel["predicate"].replace("_", " ")
					if predicate_wo_underscore not in dataset_meta["predicates"]:
						dataset_meta["predicates"].append(predicate_wo_underscore)

				for sbobjs in sub_ob_jects:
					sub_category = sbobjs["category"]
					if "_" in sub_category:
						sub_category = sub_category.replace("_", " ")
					if sub_category not in dataset_meta["objects"]:
						dataset_meta["objects"].append(sub_category)

		annotations = {
				"train": train_annotations,
			#  "test": test_annotations
		}


		Selected_SEED = seed_manager.AG_OBJECT_PREDICATE_PARTIAL_SELECTION_SEEDS[0] ## possible seeds [978, 324]
		random.seed(Selected_SEED)

		 # select half of the elements
		half_length = len(dataset_meta["objects"]) // 2
		# Randomly select half of the elements
		Selected_VRD_Objects = random.sample(dataset_meta["objects"], half_length)
		Selected_VRD_contact_relations = random.sample(dataset_meta["predicates"], len(dataset_meta["predicates"])//4)

		TempALLObjSet = set(dataset_meta["objects"])
		TempAllContactingRel = set(dataset_meta["predicates"])
		ObjectsForEval = list(TempALLObjSet.difference(Selected_VRD_Objects))
		RelForEval = list(TempAllContactingRel.difference(Selected_VRD_contact_relations))

		ObjectPredicateData = {
			"selected_objects": Selected_VRD_Objects,
			"selected_relations":  Selected_VRD_contact_relations,
			"skipped_objects": ObjectsForEval,
			"skipped_relations": RelForEval
		}

		partial_dataset_meta = {
			"subjects" :Selected_VRD_Objects,
			"objects": Selected_VRD_Objects,
			"predicates": Selected_VRD_contact_relations
		}


		dataset = "vidvrd"
		version = f"v5_3_openvocab_{Selected_SEED}"

		OUTPUT_JSON_DIR = f"/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/llava_annotations/video_llava_{dataset}_annotations_{version}/"
		# JSON_llava_image_tune_validate = f"{OUTPUT_JSON_DIR}/llava_image_tune_validate.json"
		# JSON_llava_image_tune = f"{OUTPUT_JSON_DIR}/llava_image_tune_.json"
		# JSON_llava_image_tune_validate_bb = f"{OUTPUT_JSON_DIR}/llava_image_tune_validate_bb.json"
		# JSON_llava_image_tune_bb = f"{OUTPUT_JSON_DIR}/llava_image_tune_bb.json"

		JSON_videochatgpt_tune = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_.json"
		JSON_videochatgpt_tune_validate = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_validate.json"

		JSON_videochatgpt_bb_tune = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_bb.json"
		JSON_videochatgpt_bb_tune_validate = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_bb_validate.json"
		os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

		video_gpt_promptanswers = []
		video_gpt_promptanswers_val = []

		video_gpt_bb_promptanswers=  []
		video_gpt_bb_promptanswers_val =  []

		print_sample_annotations = True

		llava_image_tune = []
		llava_image_tune_val = []

		llava_image_tune_bb = []
		llava_image_tune_val_bb = []

		image_annot_cnt = {"train": 0, "val": 0}
		annot_cnt = {"train": 0, "val": 0}
		video_bb_annot_cnt = {"train": 0, "val": 0}
		annotation_files_counter = 0
		annotation_files_counter_val = 0

		for subset, anno_files in annotations.items():
			total_keys = len(anno_files)
			data_per_thread = int(total_keys/n_thread_count)
			current_vid_idx = 0
			processedThreadsCount = 0

			# chunked_list_gen = chunk_list(list_=anno_files, chunk_n=data_per_thread)
			# chunked_list = []
			# for cl in chunked_list_gen:
			# 	chunked_list.append(cl)
			
			# n_thread_count = len(chunked_list)
			# print("len of chunked list: ", len(chunked_list))
			print("Total videos ",len(anno_files))

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

			pbar = tqdm(total=len(anno_files))
			pbar.n = 0
			pbar.last_print_n = 0
			pbar.refresh()


			prepare_vid_sg(data_root=data_root,
						subset=subset,
						annotations=anno_files,norm_bb=True,dataset=dataset,
						uniform_sampling_idx=8,
						dataset_meta=partial_dataset_meta)


			
			if subset=="train":
				json_annotations = video_gpt_promptanswers
			else:
				json_annotations = video_gpt_promptanswers_val

			print(f"Total annotations prepared: {len(json_annotations)}")


			random.seed(seed_manager.RANDOM_ANNOTATIONS_SHUFFLE_SEED) 
			random.shuffle(json_annotations) # shuffle to get all videos shuffled
			chunked_list_gen = chunk_list(list_=json_annotations,chunk_n=CHUNK_N)

			json_file_counter = 0
			# OUTPUT_JSON_DIR = os.path.join(OUTPUT_JSON_DIR,str(Selected_SEED))
			os.makedirs(OUTPUT_JSON_DIR,exist_ok=True)
			for chunked_annotation_list in chunked_list_gen:
				random.shuffle(chunked_annotation_list) # reshuffle to get all videos shuffled again
				JSON_videochatgpt_tune_validate = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_{subset}_seed_{Selected_SEED}_{dataset}_part{json_file_counter}.json"
				with open(JSON_videochatgpt_tune_validate, "w") as f:
					json.dump(chunked_annotation_list,f, indent=4)
				json_file_counter +=1
				print(f"Saved: {JSON_videochatgpt_tune_validate}")

			try:
				with open(f"{OUTPUT_JSON_DIR}/meta_data.json", "w") as f:
					json.dump(ObjectPredicateData,f, indent=4)
			except Exception as e:
				print(f"error saving : {OUTPUT_JSON_DIR}/meta_data.json")
				pass

			# if subset=="train":
			# 	JSON_videochatgpt_tune = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_part{annotation_files_counter}.json"
			# 	with open(JSON_videochatgpt_tune, "w") as f:
			# 		json.dump(video_gpt_promptanswers,f)
			# else:
			# 	JSON_videochatgpt_tune_validate = f"{OUTPUT_JSON_DIR}/videochatgpt_tune_validate_part{annotation_files_counter_val}.json"
			# 	with open(JSON_videochatgpt_tune_validate, "w") as f:
			# 		json.dump(video_gpt_promptanswers_val,f)

				# with open(JSON_videochatgpt_tune_validate, "w") as f:
				# 	json.dump(video_gpt_promptanswers_val,f)
							
				
		print("Saved annotations", annot_cnt)

		# with open(JSON_videochatgpt_bb_tune, "w") as f:
		#     json.dump(video_gpt_bb_promptanswers,f)
		# with open(JSON_videochatgpt_bb_tune_validate, "w") as f:
		#     json.dump(video_gpt_bb_promptanswers_val,f)
		# print("Saved annotations", video_bb_annot_cnt)










############## TO BE REMOVED 

# def get_bb_subj_obj(data_root,vid_id,frame_idx,subject_id,object_id):
#   sub_bb, obj_bb, mask_size = [], [], None
#   try:
#     sub_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_id,frame_id=frame_idx, object_id=subject_id)
#   except FileNotFoundError:
#     #print(f"[Warning] Frame {frame_for_bb_idx} not found for vidor {vid_data['video_id']}")
#     pass
  
#   try:
#     obj_bb, mask_size = getboundingBoxOftheObject(data_root=data_root,vid_id=vid_id,frame_id=frame_idx, object_id=object_id)
#   except FileNotFoundError:
#     #print(f"[Warning] Frame {frame_for_bb_idx} not found for vidor {vid_data['video_id']}")
#     pass

#   return sub_bb, obj_bb, mask_size


# def getVideoCaptions(vid_data, correct_object_ids=False):
#     vid_objects = vid_data["objects"] # {'object_id': 2, 'category': 'countertop', 'is_thing': True, 'status': []},
#     # vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in vid_objects} # for relations
#     vid_rels = vid_data["relations"]
#     object_id_pattern_in_descr = r"\((\d+)\)"
#     AnswerString = ""
#     vid_caps = vid_data['captions']
#     for idx, vid_c in enumerate(vid_caps):
#         if correct_object_ids:
#            """
#            Converts adult (1)  ==> adult.1
#            """
#            vid_description = re.sub(object_id_pattern_in_descr, r".\1", vid_c["description"])
#            vid_description = vid_description.replace(" .",".")
#         else:
#            vid_description = vid_c["description"]
           
#         AnswerString += vid_description
#         if idx!=len(vid_rels)-1:
#             AnswerString +=","
#     return AnswerString

# def getVideoQandAPairs(vid_data, correct_object_ids=False):
#     QnAPairs = []
#     vid_qna = vid_data['qa_pairs']
#     for idx, vid_qna in enumerate(vid_qna):
#         # time_point = vid_qna["time"]
#         Question = vid_qna["question"]
#         Answer = vid_qna["answer"]

#         if correct_object_ids:
#            object_id_pattern_in_descr = r"\((\d+)\)"
#            Question = re.sub(object_id_pattern_in_descr, r".\1", Question).replace(" .", ".")
#            Answer = re.sub(object_id_pattern_in_descr, r".\1", Answer).replace(" .", ".")


#         QnASeq = [{
#           "from": "human",
#           "value": f"<video>\n{Question}"
#         },
#         {
#           "from": "gpt",
#           "value": Answer
#         }]
#         QnAPairs.append(QnASeq)

#     return QnAPairs

# def getVideoSummary(vid_data):
#     AnswerString = vid_data['summary']
#     return AnswerString