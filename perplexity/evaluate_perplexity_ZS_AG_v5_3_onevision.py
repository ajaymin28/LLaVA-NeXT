import os
import json
import glob
# import matplotlib.pyplot as plt
# from PIL import Image, ImageDraw
import numpy as np
import time
from utils.utilities import eval_tagging_scores
from utils.utilities import pre_clean_prediction_data_onevision_v14_AG
from utils.utilities import get_substring_between
from utils.utilities import calculate_accuracy_varying_lengths, remove_ids
from utils.utilities import getRandomPrompt, SGSpecialTokens
from utils.utilities import get_AG_annotations_framewise, get_shuffled_list
from utils.utilities import unnormbb
from utils.utilities import AG_Objects,AG_relations

import argparse
import os
import copy
import torch
import torch.nn.functional as F

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


    


def get_model_output(prompt,file,batch_of_frames=None,target_output=None,cal_perplexity=False):
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
    perplexities = []
    perplexity = None

    with torch.inference_mode():
        # model.update_prompt([[cur_prompt]])
        # import pdb;pdb.set_trace()
        # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
        
        # if "mistral" not in cfg_pretrained._name_or_path.lower():
        #     output_ids = model.generate(inputs=input_ids, images=input_video, 
        #                                 attention_mask=attention_masks, modalities="video", 
        #                                 do_sample=False, temperature=0.0, max_new_tokens=2048, 
        #                                 top_p=0.1,num_beams=1,use_cache=False, 
        #                                 stopping_criteria=[stopping_criteria],
        #                                 output_scores =True,
        #                                 output_logits=True,
        #                                 return_dict_in_generate=True)
        #     # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
        # else:
        #     output_ids = model.generate(inputs=input_ids, images=input_video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=2048, top_p=0.1, num_beams=1, use_cache=False)
        #     # output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True)

    
        
        if cal_perplexity:

            output_ids =  model.generate(inputs=input_ids, images=input_video, 
                                        attention_mask=attention_masks, modalities="video", 
                                        do_sample=False, temperature=0.0, max_new_tokens=2048, 
                                        top_p=0.1,num_beams=1,use_cache=False, 
                                        stopping_criteria=[stopping_criteria],
                                        # output_scores =True,
                                        output_logits=True,
                                        # output_attentions=True,
                                        return_dict_in_generate=True)
        
            sequences = output_ids["sequences"] # predicted ids
            # score = output_ids["scores"]
            logits = output_ids["logits"]
            # past_keys_values = output_ids["past_key_values"]

            # attentions = output_ids["attentions"]
            # stacked_attentions = torch.stack(attentions, dim=1).cuda()
            
            outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0].strip() #decoded tokens sg

            target = tokenizer(
                    [outputs], return_tensors="pt", padding=True, truncation=True
            )
            target_input_ids = target["input_ids"].cuda()
            target_attention_masks = target_input_ids.ne(tokenizer.pad_token_id).long().cuda() #attn mask for decoded tokens
            target["attension_mask"] = target_attention_masks

            stacked_logits = torch.stack(logits, dim=1).cuda()
            shift_logits = stacked_logits[:, :, :]  # Ignore the last token's logits
            shift_labels = target_input_ids[:, :]   # Skip the first token in the labels

            # Compute log probabilities
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

            # probs = torch.exp(log_probs)
            # # Convert probabilities to percentages
            # percentages = probs * 100
            # # Extract the percentages for the predicted tokens
            # token_percentages = torch.gather(percentages, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
        

             # Convert token IDs back to tokens for display
            # tokens = tokenizer.convert_ids_to_tokens(shift_labels[0].cpu().numpy(), skip_special_tokens=True)

            # Display the tokens and their corresponding log probabilities
            #for token, log_prob in zip(tokens, token_percentages[0].cpu().numpy()): print(f"Token: {token}, Probability%: {log_prob}")

            # Gather the log probabilities for the correct tokens
            target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)


            if target_log_probs.size(-1) != target_attention_masks.size(1):
                outputs = outputs.strip()
                sg_outputs["triplets"] = outputs
                return sg_outputs, None

            # Mask out positions corresponding to padding tokens
            target_log_probs = target_log_probs * target_attention_masks[:, :].to(log_probs.dtype)

            # Compute the mean negative log-likelihood for each sequence
            negative_log_likelihood = -target_log_probs.sum(dim=-1) / target_attention_masks[:, :].sum(dim=-1)

            # Compute perplexity for each sequence
            perplexity = torch.exp(negative_log_likelihood)
            # Take mean of perplexities of each batch
            mean_perplexity_score = torch.mean(perplexity)
            # perplexities = perplexities.tolist()

            perplexities.append(mean_perplexity_score.cpu().numpy())

            # outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0].strip()

        # import pdb;pdb.set_trace()
        if "mistral" not in cfg_pretrained._name_or_path.lower():
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]

        outputs = outputs.strip()
        sg_outputs["triplets"] = outputs
    
    return sg_outputs, perplexities


def generate_answer(model, tokenizer, prompt, input, modality="video"):
    sg_outputs = {
        "question": prompt,
        "answer": ""
    }

    conv = conv_templates["qwen_2"].copy()

    qs = prompt
    # if args.add_time_instruction:
    #     time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(input_video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    #     qs = f'{time_instruciton}\n{qs}'
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


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_path", help="Path to the video files.", required=False)
    parser.add_argument("--output_dir", default="/home/jbhol/dso/gits/LLaVA-NeXT/eval/output", help="Directory to save the model results.", required=False)
    parser.add_argument("--output_name", help="Name of the file for storing results", required=False)
    parser.add_argument("--model-path", type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="qwen_2")
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

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--start_index", type=int, default=0,required=False)
    # parser.add_argument("--prev_eval_data", type=str, default="", required=False)
    return parser.parse_args()

def fixdecimals(box, decimals=2):
    x1,y1,x2,y2 = box
    x1 = round(x1,decimals)
    y1 = round(y1,decimals)
    x2 = round(x2,decimals)
    y2 = round(y2,decimals)
    return [x1,y1,x2,y2]

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

    dataset_name = "ActionGnome"
    dataset_name_to_save = dataset_name

    # TODO SET PATHS here for propts
    exec(open("/home/jbhol/dso/gits/LLaVA-NeXT/picklePrompt.py").read())
    defaultPrompt = "None"
    with open('/home/jbhol/dso/gits/LLaVA-NeXT/prompts.pkl', 'rb') as handle:
        b = pickle.load(handle)
        defaultPrompt = b["version_14_AG_ZS_triplets"]

    print(defaultPrompt)
    version = args.output_dir


    continue_eval = False
    if args.start_index!=0:
        dataset_name_to_save += f"start_idx_{args.start_index}"
        # if args.prev_eval_data=="":
        #     raise Exception("Require prev_eval data path to continue previous eval")

    splits = ["test"]
    VIDEO_ROOT_PATH = "/groups/sernam/datasets/ActionGenome/ActionGenome/videos"
    AG_ANNOTATIONS_DIR = "/groups/sernam/datasets/ActionGenome/ActionGenome/annotations"
    CHUNK_N = 1000 # Q&A will be chunked into CHUNK_N parts
    AG_Annotations,dataset_meta,video_frame_data = get_AG_annotations_framewise(AG_ANNOTATIONS_DIR=AG_ANNOTATIONS_DIR, 
                                                                                subset=splits[0])



    inference_output_dir  = args.output_dir
    inference_prog_output_dir  = f"{args.output_dir}/prog" 
    os.makedirs(inference_output_dir,exist_ok=True)
    os.makedirs(inference_prog_output_dir,exist_ok=True)

    sg_eval_counts["subsets"] = splits

    # AG_Prompt = getRandomPrompt(key='AG_Prompt', static=True)
    # AG_Prompt = AG_Prompt.replace("{objects_list}",  ",".join(get_shuffled_list(AG_Objects)) )
    # AG_Prompt = AG_Prompt.replace("{spatial_relations}", ",".join(get_shuffled_list(AG_relations["spatial"])))
    # AG_Prompt = AG_Prompt.replace("{contacting_relations}", ",".join(get_shuffled_list(AG_relations["contacting"])))
    # AG_Prompt = AG_Prompt.replace("{attention_relations}", ",".join(get_shuffled_list(AG_relations["attention"])))

    exec(open("/home/jbhol/dso/gits/LLaVA-NeXT/picklePrompt.py").read())
    defaultPrompt = "None"
    with open('/home/jbhol/dso/gits/LLaVA-NeXT/prompts.pkl', 'rb') as handle:
        b = pickle.load(handle)
        defaultPrompt = b["version_14_AG_ZS_triplets"]

    # print(defaultPrompt)


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

    random.seed(0)


    alljsons = glob.glob(os.path.join("/home/jbhol/dso/gits/ActionGenome/AG_llava_annotations_v5_3_test", "*.json"))
    perplexities = []
    for jsonfile in alljsons:
        with open(jsonfile, "r") as f:
            jsonData = json.loads(f.read())
            jsonfileName = jsonfile.split("/")[-1]

            total_samples = len(jsonData)

            # Check if we can sample 10 elements
            if total_samples < 10:
                print(f"Only {total_samples} elements available, sampling all.")
                sampled_data = jsonData
            else:
                # Sampling 10 random elements
                sampled_data = random.sample(jsonData, 20)


            for annotIdx, annot in tqdm(enumerate(sampled_data)):
                video_path = annot["video"]

                Block_frame_ids = annot["frame_indices"]

                conversations = annot["conversations"]

                # QPrompt = conversations[0]["value"].replace("<video>\n\n      ", "")
                Answer = conversations[1]["value"]


                file = video_path if isinstance(video_path, list) else [video_path]
                args.video_path = video_path
                set_video(args=args, video_frame_index=Block_frame_ids)

                outputs_unclean, perplexity = get_model_output(prompt=defaultPrompt,
                                                   file=file,
                                                   batch_of_frames=Block_frame_ids,
                                                   cal_perplexity=True,
                                                   target_output=Answer)
                
                if perplexity:
                    perplexities.append(perplexity[0])


    
    samples_tested = len(perplexities)
    perplexities  = np.array(perplexities).mean()

    print(f"mean perplexities for : {samples_tested} samples: {perplexities:.4f}")