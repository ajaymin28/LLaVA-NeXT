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

# from utils.utilities import eval_tagging_scores
# from utils.utilities import pre_clean_prediction_data_v18
# from utils.utilities import calculate_accuracy_varying_lengths, remove_ids
# from utils.utilities import getRandomPrompt, SGSpecialTokens
from data_prep.vidvrd2dataset import VidVRD, VidOR
# from utils.utilities import get_varying_list

# import math
from tqdm import tqdm
# from decord import VideoReader, cpu
# from transformers import AutoConfig

# import cv2
# import base64
import pickle
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
import random
# from typing import Dict

sys.path.append("/home/jbhol/dso/gits/LLaVA-NeXT/InternVL") # add InternVL util modules to path
from internvl_chat.utils.internvl_utils import load_model
from internvl_chat.utils.internvl_utils import get_model_response as get_internvl_model_response

def init_main(args):
    global  model,tokenizer,transform,generation_config
    model,tokenizer,transform,generation_config = load_model(model_path=args.model_path,max_new_tokens=args.max_new_tokens,
    do_sample=args.force_sample)


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

def get_model_output(prompt,file,batch_of_frames=None, calc_peplexity=False):
    sg_outputs = {
        # "objects_list": "",
        "triplets": ""
    }

    if calc_peplexity:
        outputs, perplexity,word_map  = get_internvl_model_response(prompt=prompt,video_path=file,frame_indices=batch_of_frames,
                       model=model,tokenizer=tokenizer, transforms=transform,
                       cal_perplexity=calc_peplexity,
                       generation_config=generation_config, map_fn=map_tokens_to_words)

        outputs = outputs.strip()
        sg_outputs["triplets"] = outputs
        return sg_outputs, perplexity, word_map

    else:
        outputs = get_internvl_model_response(prompt=prompt,video_path=file,frame_indices=batch_of_frames,
                       model=model,tokenizer=tokenizer, transforms=transform,
                       generation_config=generation_config, map_fn=map_tokens_to_words)
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
    parser.add_argument("--max-new-tokens", type=int, default=2048)
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

                args.video_path = video_path

                outputs_unclean, perplexity, word_map = get_model_output(prompt=QPrompt,file=video_path,batch_of_frames=Block_frame_ids,calc_peplexity=True)
                # outputs = pre_clean_prediction_data_v18(outputs_unclean["triplets"])
                
                if perplexity is not None:
                    perplexity = perplexity[0].cpu().numpy()
                    if not np.isnan(perplexity):
                        perplexities.append(perplexity)
                        perplexities_vids.append({
                            "vid_name": video_name,
                            "frames": Block_frame_ids,
                            "word_scores": word_map,
                            "unclean_output": outputs_unclean,
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

            