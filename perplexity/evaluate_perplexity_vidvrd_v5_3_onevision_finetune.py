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
import torch.nn.functional as F
import pickle

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


def map_tokens_to_words(tokens, probs, words):
    # Split the words into a list
    word_list = words.split()

    
    # Initialize a list to store the average probabilities for each word
    word_probs = []
    
    # Initialize index for tokens and probs
    token_index = 0


    for i in range(len(tokens)):
        firstword = word_list[0]
        current_token = tokens[i]
        current_token = current_token.strip("Ċ")
        current_token = current_token.strip("Ġ")

        if current_token==firstword[0]:
            token_index = i
            break
    
    # Iterate over each word in the word list
    for word in word_list:
        # Initialize a list to store probabilities of tokens that form the current word
        token_probs_for_word = []
        
        # Initialize a variable to store the reconstructed word from tokens
        reconstructed_word = ""
        
        # for i in range(len(tokens)): print(tokens[i])
        # Keep adding tokens until the reconstructed word matches the current word
        while token_index < len(tokens) and reconstructed_word != word:
            current_token = tokens[token_index]
            if current_token=="Ċ" or current_token=="Ġ": # skip new line and spaces
                token_index += 1
                continue

            current_token = current_token.strip("Ċ")
            current_token = current_token.strip("Ġ")
            if len(current_token)>0:
                token_probs_for_word.append(probs[token_index])
                reconstructed_word += current_token
            token_index += 1
        
        # Calculate the average probability for the current word
        if token_probs_for_word:
            avg_prob = sum(token_probs_for_word) / len(token_probs_for_word)
            word_probs.append((word, avg_prob))
    
    return word_probs

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
    word_probs = {}

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
            outputs_cleaned = outputs.replace("\n", " ").replace("   ", " ").replace("  ", " ")
            outputs_cleaned = outputs_cleaned.replace("   ", " ")
            outputs_cleaned = outputs_cleaned.replace("  ", " ")
            outputs_cleaned = outputs_cleaned.replace("#sg_start", "")
            outputs_cleaned = outputs_cleaned.replace("#sg_end", "")
            outputs_cleaned = outputs_cleaned.replace("#frameid", "#frameid ")
            outputs_cleaned = outputs_cleaned.replace(":", " : ")
            outputs_cleaned = outputs_cleaned.replace(";", " ; ")
            outputs_cleaned = outputs_cleaned.strip()

            # try:
            #     outputs_cleaned = eval(outputs_cleaned)
            # except Exception as e:
            #     print("error parsing output")
            # if type(outputs_cleaned)==dict:
            #     outputs_cleaned = str(outputs_cleaned["triplets"])

            target = tokenizer([outputs], return_tensors="pt", padding=True, truncation=True)
            target_input_ids = target["input_ids"].cuda()
            target_attention_masks = target_input_ids.ne(tokenizer.pad_token_id).long().cuda() #attn mask for decoded tokens
            target["attension_mask"] = target_attention_masks

            stacked_logits = torch.stack(logits, dim=1).cuda()
            shift_logits = stacked_logits[:, :, :]  # Ignore the last token's logits
            shift_labels = target_input_ids[:, :]   # Skip the first token in the labels

            # Compute log probabilities
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)


            # Display the tokens and their corresponding log probabilities
            #for token, log_prob in zip(tokens, token_percentages[0].cpu().numpy()): print(f"Token: {token}, Probability%: {log_prob}")

            # Gather the log probabilities for the correct tokens

            # FIX: if logits size is mismatch then adjust the logits or labels

            if log_probs.size(1)==0:
                return None,None
            
            if shift_labels.size(-1)==0:
                return None,None


            # FIX: if logits size is mismatch then adjust the logits or labels
            if log_probs.size(1) != shift_labels.size(1):
                absdiff = abs(log_probs.size(1) - shift_labels.size(1))
                if log_probs.size(1)>shift_labels.size(1):
                    log_probs = log_probs[:,:-absdiff,:]
                else:
                    shift_labels = shift_labels[:,:-absdiff]
                    target_attention_masks = target_attention_masks[:,:-absdiff]


            target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)


            # if target_log_probs.size(-1) != target_attention_masks.size(1):
            #     outputs = outputs.strip()
            #     sg_outputs["triplets"] = outputs
            #     return sg_outputs, None

            # Mask out positions corresponding to padding tokens
            target_log_probs = target_log_probs * target_attention_masks[:, :].to(log_probs.dtype)

            probs = torch.exp(log_probs)
            # Convert probabilities to percentages
            percentages = probs * 100
            # Extract the percentages for the predicted tokens
            token_percentages = torch.gather(percentages, 2, shift_labels.unsqueeze(-1)).squeeze(-1)

            # Convert token IDs back to tokens for display
            tokens = tokenizer.convert_ids_to_tokens(shift_labels[0].cpu().numpy(), skip_special_tokens=True)
            # for token, log_pro in zip(tokens, token_percentages[0].cpu().numpy()): print(f"Token: {token}, Probability%: {log_pro}")

            # Example usage
            # tokens = [".l", "ion", "stand", "ing", "next", "to", "another", "l", "lion"]
            # probs = [0.98, 0.97, 0.76, 0.56, 0.87, 0.99, 0.87, 0.86, 0.90]
            # words = "lion standing next to another lion"

            # if type(outputs_cleaned)==dict:
            #     outputs_cleaned = str(outputs_cleaned["triplets"])

            # Get the word probabilities
            word_probs = map_tokens_to_words(tokens, target_log_probs[0].tolist(), outputs_cleaned)

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
    
    return sg_outputs, perplexities, word_probs


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

    dataset_name = "vidvrd"
    version = args.output_dir

    # splits = ["test"]
    # imagenet_vidvrd_root = "/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/llava_annotations/video_llava_vidvrd_test_annotations_v5_3"
    # imagenet_vidvrd_video_path = os.path.join(imagenet_vidvrd_root, "videos")
    # dataset = VidVRD(imagenet_vidvrd_root, imagenet_vidvrd_video_path, splits)

    # inference_output_dir  = f"{imagenet_vidvrd_root}/inference_outputs_onevision/{args.output_dir}" 
    # inference_prog_output_dir  = f"{imagenet_vidvrd_root}/inference_outputs_onevision/{args.output_dir}/prog" 
    inference_output_dir  = args.output_dir
    inference_prog_output_dir  = f"{args.output_dir}/prog" 
    os.makedirs(inference_output_dir,exist_ok=True)
    os.makedirs(inference_prog_output_dir,exist_ok=True)
    random.seed(0)

    alljsons = glob.glob(os.path.join("/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/llava_annotations/video_llava_vidvrd_test_annotations_v5_3", "*.json"))
    perplexities = []
    perplexities_vids = []
    for jsonidx,jsonfile in enumerate(alljsons):
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

                # if not "ILSVRC2015_train_00087003.mp4" in video_path:
                #     continue
                # if [104, 105, 106, 107, 108, 109, 110, 111]!=Block_frame_ids:
                #     continue

                video_name = video_path.split("/")[-1]
                conversations = annot["conversations"]
                QPrompt = conversations[0]["value"].replace("<video>\n\n      ", "")
                Answer = conversations[1]["value"]

                file = video_path if isinstance(video_path, list) else [video_path]
                args.video_path = video_path
                set_video(args=args, video_frame_index=Block_frame_ids)

                outputs_unclean, perplexity, word_scores = get_model_output(prompt=QPrompt,
                                                   file=file,
                                                   batch_of_frames=Block_frame_ids,
                                                   cal_perplexity=True,
                                                   target_output=Answer)
                
                if perplexity is not None:
                    if not np.isnan(perplexity[0]):
                        perplexities.append(perplexity[0])
                        perplexities_vids.append({
                            "vid_name": video_name,
                            "frames": Block_frame_ids,
                            "word_scores": word_scores,
                            "sg_gt": Answer})

                print(f"perplexity for : {video_path}:{Block_frame_ids}: {perplexities[-1]:.4f}")
                save_dir = os.path.join(inference_output_dir,f"{jsonidx}")
                os.makedirs(save_dir,exist_ok=True)

                
                with open(os.path.join(save_dir,f"output_{annotIdx}_{video_name}.txt"), "w") as f:
                    f.write(outputs_unclean["triplets"])

            samples_tested = len(perplexities)
            perplexities_array  = np.array(perplexities).mean()
            print(f"mean perplexities for : {samples_tested} samples: {perplexities_array:.4f}")
            with open(os.path.join(inference_output_dir,"perplexity_data.pickle"), 'wb') as f:
                pickle.dump({
                    "vids": perplexities_vids,
                    "perplexity": perplexities
                }, f)
