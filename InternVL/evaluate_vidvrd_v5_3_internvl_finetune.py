import os
import json
import glob
import numpy as np
import time
import argparse
import os
import copy
import torch
import sys
sys.path.append("/home/jbhol/dso/gits/LLaVA-NeXT") # add util modules to path

from utils.utilities import eval_tagging_scores
from utils.utilities import pre_clean_prediction_data_v18
from utils.utilities import calculate_accuracy_varying_lengths, remove_ids
from utils.utilities import getRandomPrompt, SGSpecialTokens
from data_prep.vidvrd2dataset import VidVRD, VidOR
from utils.utilities import get_varying_list

# import math
from tqdm import tqdm
# from decord import VideoReader, cpu
# from transformers import AutoConfig

# import cv2
# import base64
# import pickle
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
import random
from typing import Dict

sys.path.append("/home/jbhol/dso/gits/LLaVA-NeXT/InternVL") # add InternVL util modules to path
from internvl_chat.utils.internvl_utils import load_model
from internvl_chat.utils.internvl_utils import get_model_response as get_internvl_model_response

def init_main(args):
    global  model,tokenizer,transform,generation_config
    model,tokenizer,transform,generation_config = load_model(model_path=args.model_path,max_new_tokens=args.max_new_tokens,
    do_sample=args.force_sample)


def get_model_output(prompt,file,batch_of_frames=None):
    sg_outputs = {
        # "objects_list": "",
        "triplets": ""
    }
    outputs = get_internvl_model_response(prompt=prompt,video_path=file,frame_indices=batch_of_frames,
                       model=model,tokenizer=tokenizer, transforms=transform,
                       generation_config=generation_config)
    outputs = outputs.strip()
    sg_outputs["triplets"] = outputs
    return sg_outputs


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    # parser.add_argument("--video_path", help="Path to the video files.", required=False)
    parser.add_argument("--output_dir", help="Directory to save the model results.", required=True)
    # parser.add_argument("--output_name", help="Name of the file for storing results", required=False)
    parser.add_argument("--model-path", type=str, default="OpenGVLab/InternVL2-8B")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--conv-mode", type=str, default=None)
    # parser.add_argument("--chunk-idx", type=int, default=0)
    # parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    # parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    # parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    # parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    # parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    # parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    # parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    # parser.add_argum/ent("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    # parser.add_argument("--for_get_frames_num", type=int, default=8)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default=None) 
    # parser.add_argument("--api_key", type=str, help="OpenAI API key")
    # parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    # parser.add_argument("--add_time_instruction", type=str, default=False)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    print(args)

    init_main(args)

    sg_eval_counts = {
        "total_obj_cnt" : 0,
        "total_pred_cnt" : 0,
        "total_sub_cnt" : 0,
        "correct_obj_pred_cnt" : 0,
        "correct_subj_pred_cnt" : 0,
        "correct_predicate_cnt" : 0,
        "gt_triplets_cnt": 0,
        "pred_triplets_cnt": 0,
        "correct_pred_triplets_cnt": 0,
        "total_predicted_triplets": 0
    }

    GtData = {
        "subjects": [],
        "objects": [],
        "predicates": []
    }

    PredData = {
        "subjects": [],
        "predicates": [],
        "objects": []
    }

    # TODO SET PATHS here for propts
    # exec(open("/home/jbhol/dso/gits/Video-LLaVA/picklePrompt.py").read())
    # defaultPrompt = "None"
    # with open('/home/jbhol/dso/gits/Video-LLaVA/prompts.pkl', 'rb') as handle:
    #     b = pickle.load(handle)
    #     defaultPrompt = b["version_13_sam"]

    # print(defaultPrompt)
    # exit()

    dataset_name = "vidvrd"
    version = args.output_dir

    splits = ["test"]
    imagenet_vidvrd_root = "/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd"
    imagenet_vidvrd_video_path = os.path.join(imagenet_vidvrd_root, "videos")
    dataset = VidVRD(imagenet_vidvrd_root, imagenet_vidvrd_video_path, splits)

    # inference_output_dir  = f"{imagenet_vidvrd_root}/inference_outputs_onevision/{args.output_dir}" 
    # inference_prog_output_dir  = f"{imagenet_vidvrd_root}/inference_outputs_onevision/{args.output_dir}/prog" 
    inference_output_dir  = args.output_dir
    inference_prog_output_dir  = f"{args.output_dir}/prog" 
    os.makedirs(inference_output_dir,exist_ok=True)
    os.makedirs(inference_prog_output_dir,exist_ok=True)

    sg_eval_counts["subsets"] = splits

    test_data_dir = os.path.join(imagenet_vidvrd_root, "test")
    test_anno_files = glob.glob(os.path.join(test_data_dir, "*.json")) 
    val_ids = []

    for test_annot in test_anno_files:
        filename = os.path.basename(test_annot)
        # filename = test_annot.split("/")[-1]
        filename = filename.split(".")[0]
        val_ids.append(filename)

    for val_id_idx, video_id in enumerate(val_ids):
        annot = dataset.get_anno(vid=video_id)
        frame_h, frame_w = annot["height"], annot["width"]
        frame_count = annot["frame_count"]
        video_id = annot["video_id"]
        video_fps = annot["fps"]
        sub_ob_jects = annot['subject/objects']
        sub_ob_jects_by_id = {obj["tid"]: obj  for obj in sub_ob_jects}
        rels = annot['relation_instances']
        trajectories = annot['trajectories']
        for rel in rels:
            begin_fid = rel['begin_fid']
            end_fid = rel['end_fid']
            subject_tid =rel['subject_tid']
            predicate = rel['predicate']
            if "_" in predicate:
                predicate = predicate.replace("_", " ")
            object_tid = rel['object_tid']
            for activity_range in range(begin_fid,end_fid):
                subj_data = sub_ob_jects_by_id[subject_tid]
                obj_data = sub_ob_jects_by_id[object_tid]
                if activity_range>frame_count:
                    continue
                if subj_data['category'] not in GtData["subjects"]:
                    GtData["subjects"].append(subj_data['category'])
                if predicate not in GtData["predicates"]:
                    GtData["predicates"].append(predicate)
                if obj_data['category'] not in GtData["objects"]:
                    GtData["objects"].append(obj_data['category'])



    pbar = tqdm(total=len(val_ids))
    pbar.n = 0
    pbar.last_print_n = 0
    pbar.refresh()
    
    llava_response_json = {}
    llava_raw_response_json = {}
    frame_block = 0
    overall_metric = {
        "subject": {"precision": [], "recall": []},
        "object": {"precision": [], "recall": []},
        "predicate": {"precision": [], "recall": []},
        "triplet": {"precision": [], "recall": []} 
    }

    for val_id_idx, video_id in enumerate(val_ids):

        annot = dataset.get_anno(vid=video_id)

        frame_h, frame_w = annot["height"], annot["width"]
        frame_count = annot["frame_count"]
        video_id = annot["video_id"]
        video_fps = annot["fps"]

        # video_path = os.path.join(videos_root, video_id+".mp4")

        sub_ob_jects = annot['subject/objects']
        sub_ob_jects_by_id = {obj["tid"]: obj  for obj in sub_ob_jects}

        rels = annot['relation_instances']
        trajectories = annot['trajectories']

        frames_dict = {}
        for i in range(frame_count):
            if i not in frames_dict.keys():
                frames_dict[i] = {
                    "triplets": [],
                    "bbox": [],
                    "pred_triplets": []
                }
        # print(frames_dict.keys())

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

        
        capture = None
        overall_annotations = []
        videos_root = os.path.join(imagenet_vidvrd_root, "videos")
        video_path = os.path.join(videos_root, video_id+".mp4")

        if os.path.exists(video_path):
            # print(video_path, "exists")
            # capture = cv2.VideoCapture(video_path)
            frame_indices = []	
            frame_counter = 0
            tripletes_for_current_block = ""
            tripletes_list_for_current_block = []
            current_block_triplet_data = {
                "subjects": [],
                "objects": [],
                "predicates": []
            }
            for frame_idx, frame_data in frames_dict.items():

                if frame_idx>frame_count:
                    continue
                
                # capture.set(cv2.CAP_PROP_POS_FRAMES,frame_idx)
                # ret, frame = capture.read()
                # if not ret:
                #     continue
                # h,w,c = frame.shape
                
                tripletes_for_current_block += f"{SGSpecialTokens.VIDEO_FRAME_ID}"

                max_triplets_to_add = 5
                added_triplets = []
                current_frame_triplets = []
                for index_to_draw, triplet in enumerate(frame_data["triplets"]):
                                
                    subj = triplet[0]
                    predicate = triplet[1]
                    if "_" in predicate:
                        predicate = predicate.replace("_", " ")
                    obj = triplet[2]

                    subj, subj_id = subj.split("-")
                    obj, obj_id = obj.split("-")

                    if subj not in current_block_triplet_data["subjects"]:
                        current_block_triplet_data["subjects"].append(subj)
                    
                    if obj not in current_block_triplet_data["objects"]:
                        current_block_triplet_data["objects"].append(obj)

                    if predicate not in current_block_triplet_data["predicates"]:
                        current_block_triplet_data["predicates"].append(predicate)
                    
                    construct_triplet = f"[{subj}-{subj_id}"
                    construct_triplet += f":{predicate}"
                    construct_triplet += f":{obj}-{obj_id}"
                    construct_triplet += f"];"
                    if construct_triplet not in added_triplets:
                            tripletes_for_current_block += construct_triplet
                            added_triplets.append(construct_triplet)
                            # current_frame_triplets.append([f"{subj}-{subj_id}", f"{predicate}", f"{obj}-{obj_id}"])
                            # v3_1 changes predicate last
                            current_frame_triplets.append([f"{subj}-{subj_id}", f"{obj}-{obj_id}",f"{predicate}"])
                
                
                if len(current_frame_triplets)>0:
                    frame_indices.append(frame_idx)
                    tripletes_list_for_current_block.append(current_frame_triplets)
                    # print("adding index", frame_idx)
                    frame_counter +=1

                    if len(frame_indices)>=8:
                        overall_annotations.append({
                            "frame_idxes": frame_indices,
                            "frames_sgs": tripletes_for_current_block+f"{SGSpecialTokens.SG_END}",
                            "triplets": tripletes_list_for_current_block,
                            "current_block_triplet_data": copy.deepcopy(current_block_triplet_data) 

                        })

                        tripletes_for_current_block = ""
                        frame_counter = 0
                        frame_indices = []
                        tripletes_list_for_current_block = []
                        current_block_triplet_data = {
                            "subjects": [],
                            "objects": [],
                            "predicates": []
                        }

            
            if len(frame_indices)>0:
                frames_needs_to_be_added = 8 - len(frame_indices)  # for last batch match the 8 frames
                for i in range(frames_needs_to_be_added):
                    tripletes_for_current_block = f"{SGSpecialTokens.VIDEO_FRAME_ID}" + tripletes_for_current_block
                    frame_indices.insert(0, frame_indices[0]-1)

                    if len(frame_indices)>=8:
                        overall_annotations.append({
                            "frame_idxes": frame_indices,
                            "frames_sgs": tripletes_for_current_block+f"{SGSpecialTokens.SG_END}",
                            "triplets": tripletes_list_for_current_block,
                            "current_block_triplet_data": current_block_triplet_data
                        })
                        
            # capture.release()


        # print("len of overall annot", len(overall_annotations))
        block_metric = {
            "subject": {"precision": [], "recall": []},
            "object": {"precision": [], "recall": []},
            "predicate": {"precision": [], "recall": []},
            "triplet": {"precision": [], "recall": []}
        }
        last_processed_time = None

        random.seed(42)

        subset_samples = random.sample(overall_annotations, k=min(5, len(overall_annotations)))
        for frame_block_index, overall_annot in enumerate(subset_samples):
            # if frame_block_index%2==0:
            #     continue

            if last_processed_time is None:
                last_processed_time = time.perf_counter()
            
            nowTime = time.perf_counter()
            print(f"Processing video: {video_id} Block {frame_block_index}/{len(subset_samples)} last processed in:{round((nowTime-last_processed_time),4)}")
            last_processed_time = nowTime

            Block_GT_Triplets = overall_annot["triplets"]
            Block_frame_ids = overall_annot["frame_idxes"]

            current_block_triplet_data = copy.deepcopy(overall_annot["current_block_triplet_data"])

            final_subjects_list = get_varying_list(current_block_list=current_block_triplet_data["subjects"], 
                                            full_list=GtData["subjects"], 
                                            fix_size=50) 

            final_objects_list = get_varying_list(current_block_list=current_block_triplet_data["objects"], 
                                            full_list=GtData["objects"], 
                                            fix_size=50)

            final_predicates_list = get_varying_list(current_block_list=current_block_triplet_data["predicates"], 
                                            full_list=GtData["predicates"], 
                                            fix_size=50) # total 132 predicates in vidvrd
            

            TripletQ = getRandomPrompt(key='triplet_prompt', static=False)
            TripletQ = TripletQ.replace("{subjects}", ",".join(final_subjects_list))
            TripletQ = TripletQ.replace("{objects}", ",".join(final_objects_list))
            TripletQ = TripletQ.replace("{predicates}", ",".join(final_predicates_list))


            if video_id not in llava_response_json:
                llava_response_json[video_id] = {}
                llava_raw_response_json[video_id] = {}

            if frame_block_index not in llava_response_json[video_id].keys():
                llava_response_json[video_id][frame_block_index] = {}
                llava_raw_response_json[video_id][frame_block_index] = {}


            
            video_path = os.path.join(videos_root, video_id+".mp4")
            # file = video_path if isinstance(video_path, list) else [video_path]
            args.video_path = video_path

            # import pdb
            # pdb.set_trace()

            outputs_unclean = get_model_output(prompt=TripletQ,file=video_path,batch_of_frames=Block_frame_ids)
            outputs = pre_clean_prediction_data_v18(outputs_unclean["triplets"])



            llava_response_json[video_id][frame_block_index] = {
                # "objects_list": outputs["objects_list"],
                "triplets": outputs,
                "frames": Block_frame_ids,
                "GT_triplets": Block_GT_Triplets
            }

            llava_raw_response_json[video_id][frame_block_index] = {
                "frames": Block_frame_ids,
                "GT_triplets": Block_GT_Triplets,
                "raw": outputs_unclean["triplets"],
                "Prompt": TripletQ,
                "cleaned_output": outputs
            }


            try:
                Block_GT_triplets_woids = remove_ids(Block_GT_Triplets,version="v3_1",remove_indexes=True)
                Block_predicated_triplets_woids = remove_ids(outputs,version="v3_1",remove_indexes=True)
            except Exception as e:
                print(f"error removing ids {e}")
                pass

            frame_metric = {
                "subject": {"precision": [], "recall": []},
                "object": {"precision": [], "recall": []},
                "predicate": {"precision": [], "recall": []},
                "triplet": {"precision": [], "recall": []}
            }
            for fidx, GT_tripdata in enumerate(Block_GT_triplets_woids):
                results = None

                frame_GT_triplets = GT_tripdata
                frame_pred_triplets = []

                try:frame_pred_triplets = Block_predicated_triplets_woids[fidx]
                except Exception as e:
                    pass

                gt_relations = [] # {"triplet": ['adult', 'sitting', 'sofa'], "score": 1.0},
                pred_relations = [] # {"triplet": ['adult', 'sitting', 'sofa'], "score": 1.0},

                gt_all = {"triplet": [],"subject": [],"object": [],"predicate": []}
                pred_all = {"triplet": [],"subject": [],"object": [],"predicate": []}

                for fgt in frame_GT_triplets:
                    fgt_s, fgt_p, fgt_o = fgt  # v3_1 changes
                    gt_all["triplet"].append({"triplet": fgt, "score": 1.0})
                    gt_all["subject"].append({"triplet": fgt_s, "score": 1.0})
                    gt_all["predicate"].append({"triplet": fgt_p, "score": 1.0})
                    gt_all["object"].append({"triplet": fgt_o, "score": 1.0})

                for fpred in frame_pred_triplets:
                    fpred_s, fpred_p, fpred_o  = fpred # v3_1 changes
                    pred_all["triplet"].append({"triplet": fpred, "score": 1.0})
                    pred_all["subject"].append({"triplet": fpred_s, "score": 1.0})
                    pred_all["predicate"].append({"triplet": fpred_p, "score": 1.0})
                    pred_all["object"].append({"triplet": fpred_o, "score": 1.0})

                    if fpred_s not in GtData["subjects"]:
                        if fpred_s not in PredData["subjects"]:
                            PredData["subjects"].append(fpred_s)
                    if fpred_p not in GtData["predicates"]:
                        if fpred_p not in PredData["predicates"]:
                            PredData["predicates"].append(fpred_p)
                    if fpred_o not in GtData["objects"]:
                        if fpred_o not in PredData["objects"]:
                            PredData["objects"].append(fpred_o)

                for fm_key, fmdata in frame_metric.items():
                    """
                    Eval score for each frame
                    """
                    prec, rec, hit_scores = eval_tagging_scores(gt_relations=gt_all[fm_key],pred_relations=pred_all[fm_key],min_pred_num=1)
                    frame_metric[fm_key]["precision"].append(prec)
                    frame_metric[fm_key]["recall"].append(rec)

                
                if len(GT_tripdata)>0 and len(frame_pred_triplets)>0:
                    try:
                        results = calculate_accuracy_varying_lengths(gt_triplets=GT_tripdata,pred_triplets=frame_pred_triplets, remove_duplicates=False)
                    except Exception as e:
                        print(f"error calculating score for vid {video_id} block:{frame_block_index} fidx {fidx} actual_fidx: {Block_frame_ids[fidx]}")

                    if results is not None:
                        sg_eval_counts["correct_pred_triplets_cnt"] +=  results["correct_triplet_cnt"]
                        sg_eval_counts["correct_obj_pred_cnt"] += results["correct_object_cnt"]
                        sg_eval_counts["correct_subj_pred_cnt"] +=  results["correct_subject_cnt"]
                        sg_eval_counts["correct_predicate_cnt"] +=  results["correct_predicate_cnt"]
                        sg_eval_counts["gt_triplets_cnt"] +=  results["total_triplets"]
                        sg_eval_counts["total_predicted_triplets"] += results["total_predicted_triplets"]
                        sg_eval_counts["total_obj_cnt"] +=  results["total_objects"]
                        sg_eval_counts["total_sub_cnt"] +=  results["total_subjects"]
                        sg_eval_counts["total_pred_cnt"] +=  results["total_predicates"] 
                else:
                    pass
                    # print(f"vid {video_id} block:{frame_block_index} fidx {fidx} actual_fidx:{Block_frame_ids[fidx]} lengt: {len(GT_tripdata)} lenpred: {frame_pred_triplets} outputs: {outputs}, unclean: {outputs_unclean}")


            for bm_key, bmdata in block_metric.items():
                """
                    average eval score for each frame and appned it to block
                """
                if len(frame_metric[bm_key]["precision"])>0 and len(frame_metric[bm_key]["recall"])>0:
                    block_metric[bm_key]["precision"].append(np.average(np.array(frame_metric[bm_key]['precision'])))
                    block_metric[bm_key]["recall"].append(np.average(np.array(frame_metric[bm_key]['recall'])))
        
        
        for oam_key, oamdata in overall_metric.items():
            """
                    average eval score for each block and appned it to overall
            """
            if len(block_metric[oam_key]["precision"])>0 and len(block_metric[oam_key]["recall"])>0:

                overall_metric[oam_key]["precision"].append(round(float(np.average(np.array(block_metric[oam_key]['precision']))), 4))
                overall_metric[oam_key]["recall"].append(round(float(np.average(np.array(block_metric[oam_key]['recall']))), 4))

        with open(f"{inference_prog_output_dir}/{val_id_idx}_{len(val_ids)}.txt", "w") as f:
            f.write(json.dumps(overall_metric, indent=4))
        
        pbar.n +=1
        pbar.last_print_n = pbar.n
        pbar.refresh()

        sg_eval_counts["VRDFormer_Logic"] = {}
        total_vid_ids = len(overall_metric["triplet"]["precision"])
        for metric_key, metric_values in overall_metric.items():
            if metric_key not in sg_eval_counts["VRDFormer_Logic"].keys():
                sg_eval_counts["VRDFormer_Logic"][metric_key] = {}
            
            if len(overall_metric[metric_key]["precision"])>0 and len(overall_metric[metric_key]["recall"])>0:
                overall_precision = np.average(np.array(overall_metric[metric_key]["precision"], dtype=np.float32))
                overall_recall = np.average(np.array(overall_metric[metric_key]["recall"], dtype=np.float32))
                sg_eval_counts["VRDFormer_Logic"][metric_key] = {
                    "Precision@1": round(float(overall_precision), 4),
                    "Recall@1": round(float(overall_recall), 4),
                }
        sg_eval_counts["VRDFormer_Logic"]["TotalVideos"] = total_vid_ids

        try:
            sg_eval_counts["dataset_meta"] ={
                "dataset_triplets_existing": GtData,
                "dataset_triplets_new": PredData
            }
        except Exception as e:
            pass

        try:
            # outputfile = f"{inference_output_dir}/{dataset_name}_inference_val_{version}.json"
            outputfile = f"{inference_output_dir}/results.json"
            with open(outputfile, "w") as f:
                json.dump(llava_response_json,f, indent=4)
        except Exception as e:
            print(f"error saving file: {e}")

        try:
            outputfile = f"{inference_output_dir}/{dataset_name}_inference_val_raw_response_{version}.json"
            outputfile = f"{inference_output_dir}/results_raw_response.json"
            with open(outputfile, "w") as f:
                json.dump(llava_raw_response_json,f, indent=4)
        except Exception as e:
            print(f"error saving file: {e}")

        try:
            outputfile = f"{inference_output_dir}/results_eval_data.json"
            with open(outputfile, "w") as f:
                json.dump(sg_eval_counts,f, indent=4)
        except Exception as e:
            print(f"error saving file: {e}")

       