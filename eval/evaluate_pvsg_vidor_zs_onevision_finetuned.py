import os
import json
import numpy as np
import time
from utils.utilities import eval_tagging_scores
from utils.utilities import pre_clean_prediction_data_v18
from utils.utilities import calculate_accuracy_varying_lengths, remove_ids
from utils.utilities import getRandomPrompt

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


    


def get_model_output(prompt,file,batch_of_frames=None):
    sg_outputs = {
        # "objects_list": "",
        "triplets": ""
    }

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


def get_frame_by_frame_annot(frame_count, rels, sub_ob_jects_by_id):
    frames_dict = {}
    for i in range(frame_count+1):
        if i not in frames_dict.keys():
            frames_dict[i] = {
                "triplets": [],
                "bbox": []
            }
    
    assert len(frames_dict.keys())<=frame_count+1

    # print("total frames ", frame_count)
    # t_start1 = time.perf_counter()
    for rel in rels:
        sub, obj, predicate, annot_frames = rel

        for anno_frame_range in annot_frames:
            f_start, f_end = anno_frame_range
            # print("anno_frame_range ", anno_frame_range)
        

            for f_index in range(f_start,f_end):
                if f_index>frame_count:
                    continue

                subn = sub_ob_jects_by_id[sub]["category"]
                objn = sub_ob_jects_by_id[obj]["category"]

                # subj_data = sub_ob_jects_by_id[subject_tid]
                # obj_data = sub_ob_jects_by_id[object_tid]

                # current_frame_traj = trajectories[activity_range]
                # sub_bb, obj_bb = None, None
                # for curr_trj in current_frame_traj:
                # 	if curr_trj["tid"]==subject_tid:
                # 		sub_bb = curr_trj["bbox"]
                # 	if curr_trj["tid"]==object_tid:
                # 		obj_bb = curr_trj["bbox"]

                frames_dict[f_index]["triplets"].append([f"{subn}-{sub}", predicate, f"{objn}-{obj}"])
                # frames_dict[activity_range]["triplets"].append([f"{subj_data['category']}-{subj_data['tid']}", predicate, f"{obj_data['category']}-{obj_data['tid']}"])
                # frames_dict[activity_range]["bbox"].append([sub_bb, obj_bb])

    assert len(frames_dict.keys())<=frame_count+1

    return frames_dict


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
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    print(args)

    init_main(args,finetuned=True)

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

    dataset_name = "vidor"
    version = args.output_dir

    splits = ["test"]
    # imagenet_vidvrd_root = "/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd"
    # imagenet_vidvrd_video_path = os.path.join(imagenet_vidvrd_root, "videos")
    # dataset = VidVRD(imagenet_vidvrd_root, imagenet_vidvrd_video_path, splits)

    # TODO set path
    data_root = '/home/jbhol/dso/gits/OpenPVSG/data'
    with open(os.path.join(data_root, 'pvsg.json'), 'r') as f:
        anno = json.load(f)
    print('Keys inside pvsg.json:', list(anno.keys()))
    print('Number of Object Classes:', len(anno['objects']['thing']))
    print('Number of Stuff Classes:', len(anno['objects']['stuff']))
    print('Number of Relation Classes:', len(anno['relations']))

    videos_root = os.path.join(data_root,dataset_name,'videos')

    inference_output_dir  = f"{data_root}/inference_outputs_onevision/{args.output_dir}" 
    inference_prog_output_dir  = f"{data_root}/inference_outputs_onevision/{args.output_dir}/prog" 
    os.makedirs(inference_output_dir,exist_ok=True)
    os.makedirs(inference_prog_output_dir,exist_ok=True)


    llava_response_json = {}
    llava_raw_response_json = {}

    opvsg_predictes = anno['relations'] 
    opvsg_objects = anno['objects']['thing'] + anno['objects']['stuff']

    train_ids = anno["split"][dataset_name]["train"]
    val_ids = anno["split"][dataset_name]["val"]

    vidor_ids = train_ids + val_ids

    overall_metric = {
        "subject": {"precision": [], "recall": []},
        "object": {"precision": [], "recall": []},
        "predicate": {"precision": [], "recall": []},
        "triplet": {"precision": [], "recall": []} 
    }

    # use train_ids + val_ids to get GT data preciates/objects
    vidor_data = {data_dict['video_id']: data_dict for data_dict in anno['data'] if data_dict['video_id'] in val_ids}
    for id_idx, video_id in enumerate(val_ids):
        
        video_data = vidor_data[video_id]

        total_frames = video_data["meta"]["num_frames"]
        vid_rels = video_data["relations"]

        vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in video_data["objects"]} 


        frames_dict = get_frame_by_frame_annot(frame_count=total_frames,
                                               rels=vid_rels,
                                               sub_ob_jects_by_id=vid_objects_by_id)
    
        for frame_idx, frame_data in frames_dict.items():
            if len(frame_data["triplets"])==0:
                continue

            added_triplets = []
            current_frame_triplets = []

            for index_to_draw, triplet in enumerate(frame_data["triplets"]):
                if len(triplet)!=3:
                    continue
                subj,predicate,obj = triplet
                subj, subj_id = subj.split("-")
                obj, obj_id = obj.split("-")
                if subj not in GtData["subjects"]:
                    GtData["subjects"].append(subj)
                if obj not in GtData["objects"]:
                    GtData["objects"].append(obj)
                if predicate not in GtData["predicates"]:
                    GtData["predicates"].append(predicate)

    
    pbar = tqdm(total=len(val_ids))
    pbar.n = 0
    pbar.last_print_n = 0
    pbar.refresh()

    # tested = ["0004_11566980553", "0010_8610561401", "0018_4748191834"]

    # use only val ids for evaluation
    vidor_data = {data_dict['video_id']: data_dict for data_dict in anno['data'] if data_dict['video_id'] in val_ids}
    for id_idx, video_id in enumerate(val_ids):
        # if video_id!="0004_11566980553":
        #     continue
        video_data = vidor_data[video_id]

        total_frames = video_data["meta"]["num_frames"]
        vid_rels = video_data["relations"]

        vid_objects_by_id = {data_dict['object_id']: data_dict for data_dict in video_data["objects"]} 


        frames_dict = get_frame_by_frame_annot(frame_count=total_frames,
                                               rels=vid_rels,
                                               sub_ob_jects_by_id=vid_objects_by_id)
        
        overall_annotations = []
        frame_indices = []	
        current_block_triplets = []
        current_block_triplet_data = {
            "subjects": [],
            "objects": [],
            "predicates": []
        }

        for frame_idx, frame_data in frames_dict.items():

            added_triplets = []
            current_frame_triplets = []

            for index_to_draw, triplet in enumerate(frame_data["triplets"]):				
                subj = triplet[0]
                predicate = triplet[1]
                # if "_" in predicate:
                #     predicate = predicate.replace("_", " ")
                obj = triplet[2]
                
                subj, subj_id = subj.split("-")
                obj, obj_id = obj.split("-")
                
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
                        # tripletes_for_current_block += construct_triplet
                        added_triplets.append(construct_triplet)
                        current_frame_triplets.append([f"{subj}-{subj_id}",f"{obj}-{obj_id}",f"{predicate}"])

            
            if len(current_frame_triplets)>0:
                frame_indices.append(frame_idx)
                current_block_triplets.append(current_frame_triplets)
                if len(frame_indices)>=8:
                    overall_annotations.append({
                        "frame_idxes": frame_indices,
                        # "frames_sgs": tripletes_for_current_block+f"{SGSpecialTokens.SG_END}",
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


        block_metric = {
            "subject": {"precision": [], "recall": []},
            "object": {"precision": [], "recall": []},
            "predicate": {"precision": [], "recall": []},
            "triplet": {"precision": [], "recall": []}
        }
        last_processed_time = None
        for frame_block_index, overall_annot in enumerate(overall_annotations):

            if last_processed_time is None:
                last_processed_time = time.perf_counter()
            
            nowTime = time.perf_counter()
            print(f"Processing video: {video_id} Block {frame_block_index}/{len(overall_annotations)} last processed in:{round((nowTime-last_processed_time),4)}")
            last_processed_time = nowTime

            Block_GT_Triplets = overall_annot["triplets_list"]
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
            file = video_path if isinstance(video_path, list) else [video_path]
            args.video_path = video_path
            set_video(args=args, video_frame_index=Block_frame_ids)
            outputs_unclean = get_model_output(prompt=TripletQ,file=file,batch_of_frames=Block_frame_ids)
            outputs = pre_clean_prediction_data_v18(outputs_unclean["triplets"])



            llava_response_json[video_id][frame_block_index] = {
                "triplets": outputs,
                # "scene": outputs["description"],
                # "st_progression": outputs["st_progression"],
                "frames": Block_frame_ids,
                "GT_triplets": Block_GT_Triplets
            }

            llava_raw_response_json[video_id][frame_block_index] = {
                "frames": Block_frame_ids,
                "GT_triplets": Block_GT_Triplets,
                "raw_response": outputs_unclean["triplets"],
                "Prompt": TripletQ,
                # "cleaned_output": outputs
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
                    # print(f"vid {val_id} block:{frame_block_index} fidx {fidx} actual_fidx:{Block_frame_ids[fidx]} lengt: {len(GT_tripdata)} lenpred: {frame_pred_triplets} outputs: {outputs}, unclean: {outputs_unclean}")


            for bm_key, bmdata in block_metric.items():
                """
                    average eval score for each frame and appned it to block
                """
                if len(frame_metric[bm_key]["precision"])>0 and len(frame_metric[bm_key]["recall"])>0:
                    block_metric[bm_key]["precision"].append(np.average(np.array(frame_metric[bm_key]["precision"], dtype=np.float32)))
                    block_metric[bm_key]["recall"].append(np.average(np.array(frame_metric[bm_key]["recall"], dtype=np.float32)))
        
        
        for oam_key, oamdata in overall_metric.items():
            """
                    average eval score for each block and appned it to overall
            """
            if len(block_metric[oam_key]["precision"])>0 and len(block_metric[oam_key]["recall"])>0:
                overall_metric[oam_key]["precision"].append(np.average(np.array(block_metric[oam_key]["precision"], dtype=np.float32)))
                overall_metric[oam_key]["recall"].append(np.average(np.array(block_metric[oam_key]["recall"], dtype=np.float32)))

        with open(f"{inference_prog_output_dir}/{id_idx}_{len(vidor_ids)}.txt", "w") as f:
            f.write(str(overall_metric))
        
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
                    "Precision@1": overall_precision,
                    "Recall@1": overall_recall,
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
            outputfile = f"{inference_output_dir}/{dataset_name}_inference_val_{version}.json"
            with open(outputfile, "w") as f:
                json.dump(llava_response_json, f, indent=4)
        except Exception as e:
            print(f"error saving file: {e}")

        try:
            outputfile = f"{inference_output_dir}/{dataset_name}_inference_val_raw_response_{version}.json"
            with open(outputfile, "w") as f:
                json.dump(llava_raw_response_json, f, indent=4)
        except Exception as e:
            print(f"error saving file: {e}")

        try:
            outputfile = f"{inference_output_dir}/{dataset_name}_inference_val_{version}_eval_data.json"
            with open(outputfile, "w") as f:
                json.dump(sg_eval_counts, f, indent=4)
        except Exception as e:
            print(f"error saving file: {e}")

       