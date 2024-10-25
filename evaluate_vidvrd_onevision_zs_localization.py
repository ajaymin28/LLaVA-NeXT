import os
import json
import glob
import numpy as np
import time
from utils.utilities import eval_tagging_scores
from utils.utilities import pre_clean_prediction_data_v18
from utils.utilities import calculate_accuracy_varying_lengths, remove_ids
from utils.utilities import getRandomPrompt, SGSpecialTokens
from data_prep.vidvrd2dataset import VidVRD, VidOR

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
from utils.misc import remove_entity_index


vidvrd_predicates_numbered = """1.jump right 2.stand left 3.taller 4.jump past 5.jump behind 6.stand front 7.sit next to 8.sit behind 9.sit front 10.next to 11.front 12.stand next to 13.stand behind 14.walk right 15.walk next to 16.walk left 17.walk past 18.walk front 19.walk behind 20.faster 21.larger 22.stand with 23.stand right 24.walk with 25.walk toward 26.walk away 27.stop right 28.stop beneath 29.stand above 30.ride 31.run beneath 32.sit above 33.sit beneath 34.sit left 35.sit right 36.walk above 37.behind 38.watch 39.hold 40.feed 41.touch 42.right 43.left 44.follow 45.move front 46.move beneath 47.chase 48.run left 49.run right 50.lie next to 51.lie behind 52.play 53.move behind 54.jump beneath 55.fly with 56.fly past 57.move right 58.move left 59.swim front 60.swim left 61.move with 62.jump front 63.jump left 64.swim right 65.swim next to 66.jump next to 67.swim with 68.move past 69.bite 70.pull 71.jump toward 72.fight 73.run front 74.run behind 75.sit inside 76.drive 77.lie front 78.stop behind 79.lie left 80.stop left 81.lie right 82.creep behind 83.creep above 84.beneath 85.above 86.fall off 87.stop front 88.run away 89.run next to 90.away 91.jump away 92.fly next to 93.lie beneath 94.jump above 95.lie above 96.walk beneath 97.stand beneath 98.move toward 99.toward 100.past 101.move away 102.run past 103.fly behind 104.fly above 105.fly left 106.lie with 107.creep away 108.creep left 109.creep front 110.run with 111.run toward 112.creep right 113.creep past 114.fly front 115.fly right 116.fly away 117.fly toward 118.stop above 119.stand inside 120.kick 121.run above 122.swim beneath 123.jump with 124.lie inside 125.move above 126.move next to 127.creep next to 128.creep beneath 129.swim behind 130.stop next to 131.stop with 132.creep toward"""
vidvrd_objects_numbered = """1.antelope 2.person 3.dog 4.zebra 5.bicycle 6.horse 7.monkey 8.fox 9.elephant 10.lion 11.giant_panda 12.airplane 13.whale 14.watercraft 15.car 16.bird 17.cattle 18.rabbit 19.snake 20.frisbee 21.motorcycle 22.ball 23.domestic_cat 24.bear 25.red_panda 26.lizard 27.skateboard 28.sheep 29.squirrel 30.bus 31.sofa 32.train 33.turtle 34.tiger 35.hamster"""

# """
# "frame-2": {
#             "objects": ["person-2", "dog-4"],
#             "object_bonding_box": [[0.160,0.245,0.450,0.750], [0.530,0.670,0.610,0.740]]
#         },
#         "frame-3": {
#             "objects": ["person-2", "dog-4"],
#             "object_bonding_box": [[0.120,0.200,0.480,0.770], [0.450,0.600,0.710,0.840]]
#         },
#         "frame-4": {
#             "objects": ["person-2", "dog-4"],
#             "object_bonding_box": [[0.150,0.240,0.470,0.750], [0.360,0.460,0.780,0.920]]
#         }
# """
def get_localization_prompt(OBJECTS_list):
    SG_Tagging_prompt = """
    You are given a video and a list of objects, your task is to detect those objects and give bounding box coordinates for those objects.
    The bounding box coordiantes are in [xmin,ymin,xmax,ymax] normlized values between 0 to 1 floting points (e.g [0.201,0.457,0.672,0.854])

    For example,
    list of objects: ["person-2","dog-4"]
    Response: {
        "results": {
            "objects": ["person-2", "dog-4"],
            "object_bonding_box": [[0.154,0.236,0.458,0.754],[0.236,0.356,0.789,0.873]]
        },
    }

    Now for the given video and list of objects="""+ str(OBJECTS_list) + """, detect bounding boxes. Reponse:"""
    return SG_Tagging_prompt

def get_sg_tagging_prompt_top1(subject_, object_, predicates):
    SG_Tagging_prompt = f"""You are given a relations list as follows : {predicates}, 
    Your task is to select a relation from the provided list for given object pairs e.g <person-1,person-2> that best describes the object pairs in the video.
    Each object is assigned an Id to identify them in the video, use these ids to identify the objects in the video and then give the relationship between them.
    For example, 
    What is the relationship between <person-1, dog-3>? Answer: """+ "{'top-1':'7.sit next to.'}" + """
    What is the relationship between <zebra-1, zebra-2>? Answer: """ + "{'top-1':'2.stand left'}" + f"""

    Now in the given video, What is the relationship between <{subject_},{object_}>? Answer:"""
    return SG_Tagging_prompt

def get_sg_tagging_prompt(subject_, object_, predicates, topk=1):
    if topk==1:
        return get_sg_tagging_prompt_top1(subject_, object_, predicates)

    SG_Tagging_prompt = f"""You are given a relations list as follows : {predicates}, 
    Your task is to select Top 5 relations from the provided list for given object pairs e.g <person-1,person-2> that best describes the object pairs in the video.
    Each object is assigned an Id to identify them in the video, use these ids to identify the objects in the video and then give the relationship between them.
    For example, 
    What is the relationship between <person-1, dog-3>? Answer: """+ "{'top-5':['7.sit next to.','10.next to','3.taller', '9.sit front', '19.walk behind']}" + """
    What is the relationship between <zebra-1, zebra-2>? Answer: """ + "{'top-5':['2.stand left','3.taller', '24.walk with', '58.move left', '22.stand with']}" + f"""

    Now in the given video, What is the relationship between <{subject_},{object_}>? Answer:"""
    return SG_Tagging_prompt

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
    parser.add_argument("--topk", type=int, default=1)
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    print(args)

    finetuned = True
    if args.model_path=="lmms-lab/llava-onevision-qwen2-7b-ov":
        finetuned = False

    init_main(args,finetuned=finetuned)

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

    inference_output_dir  = f"{imagenet_vidvrd_root}/inference_outputs_onevision/{args.output_dir}" 
    inference_prog_output_dir  = f"{imagenet_vidvrd_root}/inference_outputs_onevision/{args.output_dir}/prog" 
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
        # "subject": {"precision": [], "recall": []},
        # "object": {"precision": [], "recall": []},
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
            tripletes_box_for_current_block = []
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
                current_frame_bbox = []
                for index_to_draw, triplet in enumerate(frame_data["triplets"]):

                    bbox = frame_data["bbox"][index_to_draw]
                                
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
                            current_frame_bbox.append(bbox)
                
                
                if len(current_frame_triplets)>0:
                    frame_indices.append(frame_idx)
                    tripletes_list_for_current_block.append(current_frame_triplets)
                    tripletes_box_for_current_block.append(current_frame_bbox)
                    # print("adding index", frame_idx)
                    frame_counter +=1

                    if len(frame_indices)>=8:
                        overall_annotations.append({
                            "frame_idxes": frame_indices,
                            "frames_sgs": tripletes_for_current_block+f"{SGSpecialTokens.SG_END}",
                            "triplets": tripletes_list_for_current_block,
                            "triplets_bbox": tripletes_box_for_current_block,
                            "current_block_triplet_data": copy.deepcopy(current_block_triplet_data) 

                        })

                        tripletes_for_current_block = ""
                        frame_counter = 0
                        frame_indices = []
                        tripletes_list_for_current_block = []
                        tripletes_box_for_current_block= []
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
                            "current_block_triplet_data": current_block_triplet_data,
                            "triplets_bbox": tripletes_box_for_current_block
                        })
                        
            # capture.release()


        # print("len of overall annot", len(overall_annotations))
        block_metric = {
            # "subject": {"precision": [], "recall": []},
            # "object": {"precision": [], "recall": []},
            "predicate": {"precision": [], "recall": []},
            "triplet": {"precision": [], "recall": []}
        }
        last_processed_time = None
        for frame_block_index, overall_annot in enumerate(overall_annotations):
            # if frame_block_index%2==0:
            #     continue

            if last_processed_time is None:
                last_processed_time = time.perf_counter()
            
            nowTime = time.perf_counter()
            print(f"Processing video: {video_id} Block {frame_block_index}/{len(overall_annotations)} last processed in:{round((nowTime-last_processed_time),4)}")
            last_processed_time = nowTime

            Block_GT_Triplets = overall_annot["triplets"]
            Block_frame_ids = overall_annot["frame_idxes"]
            Block_bb = overall_annot["triplets_bbox"]

            # current_block_triplet_data = copy.deepcopy(overall_annot["current_block_triplet_data"])
            # final_subjects_list = get_varying_list(current_block_list=current_block_triplet_data["subjects"], 
            #                                 full_list=GtData["subjects"], 
            #                                 fix_size=50) 

            # final_objects_list = get_varying_list(current_block_list=current_block_triplet_data["objects"], 
            #                                 full_list=GtData["objects"], 
            #                                 fix_size=50)

            # final_predicates_list = get_varying_list(current_block_list=current_block_triplet_data["predicates"], 
            #                                 full_list=GtData["predicates"], 
            #                                 fix_size=50) # total 132 predicates in vidvrd
            # TripletQ = getRandomPrompt(key='triplet_prompt', static=False)
            # TripletQ = TripletQ.replace("{subjects}", ",".join(final_subjects_list))
            # TripletQ = TripletQ.replace("{objects}", ",".join(final_objects_list))
            # TripletQ = TripletQ.replace("{predicates}", ",".join(final_predicates_list))


            if video_id not in llava_response_json:
                llava_response_json[video_id] = {}
                llava_raw_response_json[video_id] = {}

            if frame_block_index not in llava_response_json[video_id].keys():
                llava_response_json[video_id][frame_block_index] = {}
                llava_raw_response_json[video_id][frame_block_index] = {}

            Block_predicated_triplets = []
            unique_gt_triplets = []

            gt_all = {
                "triplet": [],
                "tripletScore": [],
                "subject": [],
                "object": [],
                "predicate": [], 
                "bb": [],
                "sub_obj_pairs":[]}
            pred_all = {"triplet": [],"tripletScore": [],"subject": [],"object": [],"predicate": [], "sub_obj_pairs":[]}

            for gt_trip_frame_idx, frame_gt_triplet_data in enumerate(Block_GT_Triplets):
                frame_bb_data = Block_bb[gt_trip_frame_idx]
                for gtidx, gt_triplet in enumerate(frame_gt_triplet_data):
                    subj, obj, predicate = gt_triplet
                    bb = frame_bb_data[gtidx]

                    if [subj, obj] not in gt_all["sub_obj_pairs"]:
                        gt_all["sub_obj_pairs"].append([subj, obj])
                        gt_all["bb"].append(bb)
                    
                    if gt_triplet not in gt_all["triplet"]:
                        gt_all["triplet"].append(gt_triplet)
                        gt_all["tripletScore"].append({"triplet": gt_triplet, "score": 1.0})

                    if subj not in gt_all["subject"]:
                        gt_all["subject"].append(subj)

                    if obj not in gt_all["object"]:
                        gt_all["object"].append(obj)

                    if predicate not in gt_all["predicate"]:
                        gt_all["predicate"].append(predicate)



            predicted_bb = {}

            video_path = os.path.join(videos_root, video_id+".mp4")
            file = video_path if isinstance(video_path, list) else [video_path]
            args.video_path = video_path
            set_video(args=args, video_frame_index=Block_frame_ids)
            for subobjpair_idx, subobjpair in enumerate(gt_all["sub_obj_pairs"]):
                subj, obj = subobjpair
                bb = gt_all["bb"][subobjpair_idx]
                localization_prompt = get_localization_prompt([subj, obj])
                outputs_unclean = get_model_output(prompt=localization_prompt,file=file,batch_of_frames=Block_frame_ids)

                try:
                    outputformatted = eval(outputs_unclean["triplets"])
                except Exception as e:
                    print(f"error parsing output: {outputs_unclean['triplets']}")
                    continue

                if type(outputformatted)==dict:
                    for frameid, frame_data in outputformatted.items():
                        """
                        "objects": ["person-2", "dog-4"]
                        "object_bonding_box": [[0.154,0.236,0.458,0.754],[0.236,0.356,0.789,0.873]]
                        """
                        objects = frame_data["objects"]
                        objects_bb = frame_data["object_bonding_box"]
                        if len(objects)==len(objects_bb):
                            for obj_idx, pred_obj in enumerate(objects):
                                obj_bb = objects_bb[obj_idx]
                                if pred_obj==subj:
                                    predicted_bb[pred_obj] = {"boxes": [obj_bb], "gt_bbox": [bb[0]]}
                                elif pred_obj==obj:
                                    predicted_bb[pred_obj] = {"boxes": [obj_bb], "gt_bbox": [bb[1]]}
                                # if obj not in predicted_bb:
                                #     predicted_bb[obj] = {"boxes": [], "gt_bbox": [bb]}
                                # predicted_bb[obj]["boxes"].append(obj_bb)
                        else:
                            print(f"object and bounding box list len is different: obj {len(objects)} bb: {len(objects_bb)}")
                else:
                    print(f"invalid type: {type(outputformatted)}: {outputformatted}")
                # except Exception as e:
                #     print(f"error parsing model output : {outputs_unclean}")
                    
            
                # llava_response_json[video_id][frame_block_index] = {
                #     # "objects_list": outputs["objects_list"],
                #     "frames": Block_frame_ids,
                #     "GT_triplets": Block_GT_Triplets,
                #     "GT_BB": Block_bb,
                #     "Pred_BB": predicted_bb
                # }

            llava_raw_response_json[video_id][frame_block_index] = {
                # "objects_list": outputs["objects_list"],
                # "triplets": pred_all["triplet"],
                "frames": Block_frame_ids,
                "GT_triplets": Block_GT_Triplets,
                # "GT_BB": Block_bb,
                "Pred_BB": predicted_bb

            }


        with open(f"{inference_prog_output_dir}/{val_id_idx}_{len(val_ids)}.txt", "w") as f:
            f.write(str(outputs_unclean))
        
        pbar.n +=1
        pbar.last_print_n = pbar.n
        pbar.refresh()


        # try:
        #     outputfile = f"{inference_output_dir}/{dataset_name}_inference_val_{version}.json"
        #     with open(outputfile, "w") as f:
        #         json.dump(str(llava_response_json),f)
        # except Exception as e:
        #     print(f"error saving file: {e}")

        try:
            outputfile = f"{inference_output_dir}/{dataset_name}_inference_val_raw_response_{version}.json"
            with open(outputfile, "w") as f:
                json.dump(str(llava_raw_response_json),f)
        except Exception as e:
            print(f"error saving file: {e}")

    

       