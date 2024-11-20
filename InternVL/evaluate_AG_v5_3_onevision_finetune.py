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
sys.path.append("/root/jbhoi/gits/LLaVA-NeXT") # add util modules to path

from utils.utilities import eval_tagging_scores
from utils.utilities import pre_clean_prediction_data_v18
from utils.utilities import calculate_accuracy_varying_lengths, remove_ids
from utils.utilities import getRandomPrompt, SGSpecialTokens
from utils.utilities import get_AG_annotations_framewise, get_shuffled_list
from utils.utilities import AG_Objects,AG_relations, AG_OBJECTS_ALTERATIVES

import argparse
import os
import copy
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import math
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import AutoConfig

import cv2
import base64
import pickle
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
import random
from typing import Dict
from utils.utilities import get_varying_list

sys.path.append("/root/jbhoi/gits/LLaVA-NeXT/InternVL") # add InternVL util modules to path
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
    parser.add_argument("--start_index", type=int, default=0,required=False)
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

    dataset_name = "ActionGnome"
    dataset_name_to_save = dataset_name
    version = args.output_dir


    continue_eval = False
    if args.start_index!=0:
        dataset_name_to_save += f"start_idx_{args.start_index}"
        # if args.prev_eval_data=="":
        #     raise Exception("Require prev_eval data path to continue previous eval")

    splits = ["test"]
    AG_ROOT = "/root/jbhoi/gits/ActionGenome"
    VIDEO_ROOT_PATH = f"{AG_ROOT}/videos"
    AG_ANNOTATIONS_DIR = f"{AG_ROOT}/Annotations"
    CHUNK_N = 1000 # Q&A will be chunked into CHUNK_N parts
    AG_Annotations,dataset_meta,video_frame_data = get_AG_annotations_framewise(AG_ANNOTATIONS_DIR=AG_ANNOTATIONS_DIR, 
                                                                                subset=splits[0])



    inference_output_dir  = args.output_dir
    inference_prog_output_dir  = f"{args.output_dir}/prog" 
    os.makedirs(inference_output_dir,exist_ok=True)
    os.makedirs(inference_prog_output_dir,exist_ok=True)

    sg_eval_counts["subsets"] = splits

    AG_Prompt = getRandomPrompt(key='AG_Prompt', static=True)
    AG_Prompt = AG_Prompt.replace("{objects_list}",  ",".join(get_shuffled_list(AG_Objects)) )
    AG_Prompt = AG_Prompt.replace("{spatial_relations}", ",".join(get_shuffled_list(AG_relations["spatial"])))
    AG_Prompt = AG_Prompt.replace("{contacting_relations}", ",".join(get_shuffled_list(AG_relations["contacting"])))
    AG_Prompt = AG_Prompt.replace("{attention_relations}", ",".join(get_shuffled_list(AG_relations["attention"])))

    AG_relationsCombined = AG_relations["attention"]+AG_relations["spatial"]+AG_relations["contacting"]

    pbar = tqdm(total=len(AG_Annotations))
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
    

    for val_id_idx,AG_Annotation in enumerate(AG_Annotations):

        if val_id_idx<args.start_index:
            ## To Continue unfinished job
            pbar.n = val_id_idx
            pbar.last_print_n = pbar.n
            pbar.refresh()
            continue

        video_id, video_annotations = AG_Annotation
        video_path = os.path.join(VIDEO_ROOT_PATH,video_id)
        if not os.path.exists(video_path):
            print(f"[ERROR] video doesnt exist at: {video_path}")
            raise FileNotFoundError()
        
        
        block_wise_GT_data = []
        frame_indices = []
        added_GT_triplets_frames = []
        added_GT_triplets_bb = []
        for frame_id, frame_triplets,frame_triplets_bb in video_annotations:
            frame_int_idx = int(frame_id.split(".")[0])
            # print(frame_id, frame_int_idx)
            added_GT_triplets_frames.append(frame_triplets)
            added_GT_triplets_bb.append(list(frame_triplets_bb))
            frame_indices.append(frame_int_idx)

            if len(frame_indices)>=8:
                block_wise_GT_data.append({
                    "frame_idxes": frame_indices,
                    "triplets": added_GT_triplets_frames,
                    "triplets_bb": added_GT_triplets_bb
                })

                frame_indices = []
                added_GT_triplets_frames = []
                added_GT_triplets_bb = []

        
        # print(f"remaining frames: {len(frame_indices)}")
        if len(frame_indices)>0:
            ## add remaning frames
            block_wise_GT_data.append({
                "frame_idxes": frame_indices,
                "triplets": added_GT_triplets_frames,
                "triplets_bb": added_GT_triplets_bb
            })
        block_metric = {
            "subject": {"precision": [], "recall": []},
            "object": {"precision": [], "recall": []},
            "predicate": {"precision": [], "recall": []},
            "triplet": {"precision": [], "recall": []}
        }
        last_processed_time = None

        for frame_block_index, block_data in enumerate(block_wise_GT_data):

            if last_processed_time is None:
                last_processed_time = time.perf_counter()
            
            nowTime = time.perf_counter()
            print(f"Processing video: {video_id} Block {frame_block_index}/{len(block_wise_GT_data)} last processed in:{round((nowTime-last_processed_time),4)}")
            last_processed_time = nowTime

            
            Block_frame_ids = block_data["frame_idxes"]
            Block_GT_Triplets = block_data["triplets"]

            if video_id not in llava_response_json:
                llava_response_json[video_id] = {}
                llava_raw_response_json[video_id] = {}

            if frame_block_index not in llava_response_json[video_id].keys():
                llava_response_json[video_id][frame_block_index] = {}
                llava_raw_response_json[video_id][frame_block_index] = {}


            outputs_unclean = get_model_output(prompt=AG_Prompt,file=video_path,batch_of_frames=Block_frame_ids)
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
                "Prompt": AG_Prompt,
                "cleaned_output": outputs
            }


            try:
                Block_GT_triplets_woids = remove_ids(Block_GT_Triplets,version="v2_1",remove_indexes=True)
                Block_predicated_triplets_woids = remove_ids(outputs,version="v2_1",remove_indexes=True)
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

                    if fpred_s not in AG_Objects:
                        if fpred_s not in PredData["subjects"]:
                            PredData["subjects"].append(fpred_s)
                    if fpred_p not in AG_relationsCombined:
                        if fpred_p not in PredData["predicates"]:
                            PredData["predicates"].append(fpred_p)
                    if fpred_o not in AG_Objects:
                        if fpred_o not in PredData["objects"]:
                            PredData["objects"].append(fpred_o)

                    fpred_s = AG_OBJECTS_ALTERATIVES.get(fpred_s,fpred_s)
                    fpred_o = AG_OBJECTS_ALTERATIVES.get(fpred_o,fpred_o)
                    pred_all["triplet"].append({"triplet": fpred, "score": 1.0})
                    pred_all["subject"].append({"triplet": fpred_s, "score": 1.0})
                    pred_all["predicate"].append({"triplet": fpred_p, "score": 1.0})
                    pred_all["object"].append({"triplet": fpred_o, "score": 1.0})

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

        try:
            with open(f"{inference_prog_output_dir}/{val_id_idx}_{len(AG_Annotations)}.txt", "w") as f:
                f.write(json.dumps(overall_metric, indent=4))
        except Exception as e:
            print(f"error saving file: {inference_prog_output_dir}/{val_id_idx}_{len(AG_Annotations)}.txt")
        
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

       