import os
import json
import glob
import numpy as np
import time
from utils.utilities import eval_tagging_scores
from utils.utilities import pre_clean_prediction_data_v18
from utils.utilities import calculate_accuracy_varying_lengths, remove_ids
from utils.utilities import getRandomPrompt, SGSpecialTokens
from utils.utilities import get_AG_annotations_framewise, get_shuffled_list
from utils.utilities import AG_Objects,AG_relations

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

import multiprocessing as mp
import torch
import time

def set_video(args, video_frame_index=[0,1,2,3,4,5,6,7]):
    video_path = args.video_path
    global input_video, video_time, frame_time
    
    # Check if the video exists
    if os.path.exists(video_path):
        if "gpt4v" != args.model_path:
            input_video,frame_time,video_time = load_video(video_path, args, video_frame_index=video_frame_index)
            input_video = image_processor.preprocess(input_video, return_tensors="pt")["pixel_values"].half().cuda()
            input_video = [input_video]
        else:
            spare_frames,frame_time,video_time = load_video_base64(video_path)
            interval = int(len(input_video) / args.for_get_frames_num)
    else:
        raise FileNotFoundError("Video file not found")

def handle_custom_commands(inp):
    actions = {
        "reset_loop": False,
        "exit_loop": False
    }
    if "setvideo" in inp:
        videoPath = inp.split("=")[-1]
        args.video_path = videoPath
        set_video(args)
        print("new video path set")
        actions["reset_loop"] = True
    if "setframes" in inp:
        frames_idx = inp.split("=")[-1]
        frames_idx = eval(frames_idx)
        set_video(args,video_frame_index=frames_idx)
        actions["reset_loop"] = True
    if inp=="exit":
        actions["exit_loop"] = True
        print("exiting...")
    return actions

def load_video_base64(path):
    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # print(len(base64Frames), "frames read.")
    return base64Frames

def load_video(video_path,args, video_frame_index=[0,1,2,3,4,5,6,7]):
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())
    # frame_idx = [i for i in range(0, len(vr), fps)]
    frame_idx = video_frame_index
    frame_time = [i/fps for i in frame_idx]
    
    # if len(frame_idx) > args.for_get_frames_num or args.force_sample:
    #     sample_fps = args.for_get_frames_num
    #     uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
    #     frame_idx = uniform_sampled_frames.tolist()
    #     frame_time = [i/vr.get_avg_fps() for i in frame_idx]

    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    # print("Selected frames index ", frame_idx)
    return spare_frames,frame_time,video_time

def init_main(args, finetuned=False):

    global model, tokenizer,image_processor,context_len,cfg_pretrained

    # Initialize the model
    if "gpt4v" != args.model_path:
        model_name = get_model_name_from_path(args.model_path)
        # Set model configuration parameters if they exist
        if args.overwrite == True:
            overwrite_config = {}
            if finetuned:
                overwrite_config["vocab_size"] = 152064 # to make finetuning model work https://github.com/LLaVA-VL/LLaVA-NeXT/issues/187#issuecomment-2314195882
                overwrite_config["tie_word_embeddings"] = False
                overwrite_config['use_cache'] = True
            overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
            overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
            overwrite_config["mm_newline_position"] = args.mm_newline_position

            cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

            # import pdb;pdb.set_trace()
            if "qwen" not in args.model_path.lower():
                if "224" in cfg_pretrained.mm_vision_tower:
                    # suppose the length of text tokens is around 1000, from bo's report
                    least_token_number = args.for_get_frames_num*(16//args.mm_spatial_pool_stride)**2 + 1000
                else:
                    least_token_number = args.for_get_frames_num*(24//args.mm_spatial_pool_stride)**2 + 1000

                scaling_factor = math.ceil(least_token_number/4096)
                if scaling_factor >= 2:
                    if "vicuna" in cfg_pretrained._name_or_path.lower():
                        print(float(scaling_factor))
                        overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                    overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                    overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config)
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    else:
        pass

    # import pdb;pdb.set_trace()
    if getattr(model.config, "force_sample", None) is not None:
        args.force_sample = model.config.force_sample
    else:
        args.force_sample = False

    if getattr(model.config, "add_time_instruction", None) is not None:
        args.add_time_instruction = model.config.add_time_instruction
    else:
        args.add_time_instruction = False


    


def get_model_output(args,prompt,video_path,batch_of_frames):
    sg_outputs = {
        # "objects_list": "",
        "triplets": ""
    }

    # file = video_path if isinstance(video_path, list) else [video_path]
    # args.video_path = video_path

    input_video,frame_time,video_time = load_video(video_path, args, video_frame_index=batch_of_frames)
    input_video = image_processor.preprocess(input_video, return_tensors="pt")["pixel_values"].half().cuda()
    input_video = [input_video]

    conv = conv_templates[args.conv_mode].copy()

    qs = prompt
    if args.add_time_instruction:
        time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(input_video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
        qs = f'{time_instruciton}\n{qs}'
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    if tokenizer.pad_token_id is None:
        if "qwen" in tokenizer.name_or_path.lower():
            print("Setting pad token to bos token for qwen model.")
            tokenizer.pad_token_id = 151643
            
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # cur_prompt = question

    with torch.inference_mode():
        # model.update_prompt([[cur_prompt]])
        # import pdb;pdb.set_trace()
        # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
        if "mistral" not in cfg_pretrained._name_or_path.lower():
            output_ids = model.generate(inputs=input_ids, images=input_video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=2048, top_p=0.1,num_beams=1,use_cache=False, stopping_criteria=[stopping_criteria])
            # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
        else:
            output_ids = model.generate(inputs=input_ids, images=input_video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=2048, top_p=0.1, num_beams=1, use_cache=False)
            # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True)


        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # import pdb;pdb.set_trace()
        if "mistral" not in cfg_pretrained._name_or_path.lower():
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]

        outputs = outputs.strip()
        sg_outputs["triplets"] = outputs
    
    return sg_outputs

def init_worker(args, task_queue, result_queue, stop_event):
    """Worker process that initializes the model once and processes prompts as they come."""
    init_main(args,finetuned=True)  # Initialize model once per worker
    while not stop_event.is_set():
        try:
            # Wait for new task
            args,prompt,video_id,video_path,batch_of_frames,frame_block_index = task_queue.get(timeout=1)  # Timeout allows checking for stop_event
            result = get_model_output(args,prompt,video_path,batch_of_frames)
            result_queue.put(video_id,video_path,batch_of_frames,result,frame_block_index)
        except Exception as e:
            if stop_event.is_set():
                break  # Exit if stop event is set
            continue

def start_worker_pool(args, num_workers=3):
    """Initialize worker processes and manage task/result queues."""
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    stop_event = mp.Event()
    workers = []

    # Create worker processes
    for _ in range(num_workers):
        worker = mp.Process(target=init_worker, args=(args, task_queue, result_queue, stop_event))
        worker.start()
        workers.append(worker)

    return task_queue, result_queue, stop_event, workers

def stop_worker_pool(stop_event, workers):
    """Stop all worker processes."""
    stop_event.set()  # Signal all workers to stop
    for worker in workers:
        worker.join()

def submit_inference(task_queue, args, prompt, video_id,video_path, batch_of_frames,frame_block_index):
    """Submit a new inference request to the task queue."""
    task_queue.put((args, prompt, video_id,video_path, batch_of_frames, frame_block_index))

def get_results(result_queue):
    """Fetch all available results from the result queue."""
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    return results


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_path", help="Path to the video files.", required=False)
    parser.add_argument("--output_dir", help="Directory to save the model results.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results", required=False)
    parser.add_argument("--model-path", type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=8)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default=None) 
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--add_time_instruction", type=str, default=False)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    parser.add_argument("--start_index", type=int, default=0,required=False)
    # parser.add_argument("--prev_eval_data", type=str, default="", required=False)
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    print(args)
    # init_main(args,finetuned=True)

    task_queue, result_queue, stop_event, workers = start_worker_pool(args, num_workers=2)

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
    VIDEO_ROOT_PATH = "/groups/sernam/datasets/ActionGenome/Charades_v1_480"
    # OUTPUT_JSON_DIR = "/home/jbhol/dso/gits/ActionGenome/inference/AG_llava_annotations_v5_3"
    AG_ANNOTATIONS_DIR = "/groups/sernam/datasets/ActionGenome/ActionGenome/annot_data"
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
        

        if video_id not in llava_response_json:
            llava_response_json[video_id] = {}
            llava_raw_response_json[video_id] = {}
        
        
        block_wise_GT_data = []
        frame_indices = []
        added_GT_triplets_frames = []
        for frame_id, frame_triplets in video_annotations:
            frame_int_idx = int(frame_id.split(".")[0])
            # print(frame_id, frame_int_idx)
            added_GT_triplets_frames.append(frame_triplets)
            frame_indices.append(frame_int_idx)

            if len(frame_indices)>=8:
                block_wise_GT_data.append({
                    "frame_idxes": frame_indices,
                    "triplets": added_GT_triplets_frames,
                })

                frame_indices = []
                added_GT_triplets_frames = []

        
        print(f"remaining frames: {len(frame_indices)}")
        if len(frame_indices)>0:
            ## add remaning frames
            block_wise_GT_data.append({
                "frame_idxes": frame_indices,
                "triplets": added_GT_triplets_frames,
            })
        block_metric = {
            "subject": {"precision": [], "recall": []},
            "object": {"precision": [], "recall": []},
            "predicate": {"precision": [], "recall": []},
            "triplet": {"precision": [], "recall": []}
        }
        last_processed_time = None

        for frame_block_index, block_data in enumerate(block_wise_GT_data):
            # if last_processed_time is None:
            #     last_processed_time = time.perf_counter()
            # nowTime = time.perf_counter()
            # print(f"Processing video: {video_id} Block {frame_block_index}/{len(block_wise_GT_data)} last processed in:{round((nowTime-last_processed_time),4)}")
            # last_processed_time = nowTime

            Block_frame_ids = block_data["frame_idxes"]
            Block_GT_Triplets = block_data["triplets"]

            if frame_block_index not in llava_response_json[video_id].keys():
                llava_response_json[video_id][frame_block_index] = {}
                llava_raw_response_json[video_id][frame_block_index] = {}
            
            try:
                # Submit inference requests
                submit_inference(task_queue,args,AG_Prompt,video_id,video_path,Block_frame_ids, frame_block_index)
                # Allow some time for processing (adjust as needed)
                # time.sleep(5)
            except Exception as e:
                print(f"error in inference: {e} vid:{video_id} block: {frame_block_index}")
            # finally:
            #     # Clean up and stop worker pool
            #     stop_worker_pool(stop_event, workers)

        while task_queue.qsize()>0:
            time.sleep(1)
            if task_queue.qsize()==0:
                break
        
        # Retrieve results
        results = get_results(result_queue)
        for result in results:
            video_id,video_path,batch_of_frames,trip_results,frame_block_index = results
            outputs_unclean = trip_results
        
            # outputs_unclean = get_model_output(args=args,prompt=AG_Prompt,video_path=video_path,batch_of_frames=Block_frame_ids)
            outputs = pre_clean_prediction_data_v18(outputs_unclean["triplets"])

            block_data = block_wise_GT_data[frame_block_index]
            Block_frame_ids = block_data["frame_idxes"]
            Block_GT_Triplets = block_data["triplets"]


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
                    pred_all["triplet"].append({"triplet": fpred, "score": 1.0})
                    pred_all["subject"].append({"triplet": fpred_s, "score": 1.0})
                    pred_all["predicate"].append({"triplet": fpred_p, "score": 1.0})
                    pred_all["object"].append({"triplet": fpred_o, "score": 1.0})

                    if fpred_s not in AG_Objects:
                        if fpred_s not in PredData["subjects"]:
                            PredData["subjects"].append(fpred_s)
                    if fpred_p not in AG_relationsCombined:
                        if fpred_p not in PredData["predicates"]:
                            PredData["predicates"].append(fpred_p)
                    if fpred_o not in AG_Objects:
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

        with open(f"{inference_prog_output_dir}/{val_id_idx}_{len(AG_Annotations)}.txt", "w") as f:
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
                overall_precision = np.average(np.array(overall_metric[metric_key]["precision"]))
                overall_recall = np.average(np.array(overall_metric[metric_key]["recall"]))
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
            outputfile = f"{inference_output_dir}/{dataset_name_to_save}_inference_val.json"
            # outputfile = f"{inference_output_dir}/results.json"
            with open(outputfile, "w") as f:
                json.dump(llava_response_json,f, indent=4)
        except Exception as e:
            print(f"error saving file: {e}")

        try:
            outputfile = f"{inference_output_dir}/{dataset_name_to_save}_inference_val_raw_response.json"
            # outputfile = f"{inference_output_dir}/results_raw_response.json"
            with open(outputfile, "w") as f:
                json.dump(llava_raw_response_json,f, indent=4)
        except Exception as e:
            print(f"error saving file: {e}")

        try:
            outputfile = f"{inference_output_dir}/{dataset_name_to_save}_results_eval_data.json"
            with open(outputfile, "w") as f:
                json.dump(sg_eval_counts,f, indent=4)
        except Exception as e:
            print(f"error saving file: {e}")


    stop_worker_pool(stop_event, workers)