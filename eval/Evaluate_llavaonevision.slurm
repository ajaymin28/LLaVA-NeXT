#!/bin/bash

#PVSG_VIDOR_EVAL_SCRIPT = "/home/jbhol/dso/gits/Video-LLaVA/evaluate_pvsg_vidor_zs_onevision_finetuned.py"
#VIDVRD_EVAL_SCRIPT = "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_3_onevision_finetune.py"

#MODEL_PATH = "/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3_e02"
#JOB_NAME = ""
#DATA_OUTPUT_DIR = "[test][onevision]_vidor_zs_e02"


#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=p00_mmtune_lora_e01_annotations_v5_3_AG
#SBATCH --time=3-24:00:00
#SBATCH --output=/lustre/fs1/home/jbhol/dso/jobs/%x_raw_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ja882177@ucf.edu

##export CUDA_HOME=/lustre/fs1/home/jbhol/.local/cuda-12.1
##export PATH=${CUDA_HOME}/bin:${PATH}
##export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lustre/fs1/home/jbhol/.local/cuda-12.1/targets/x86_64-linux/lib

source ~/.bashrc
module load gcc/gcc-11.2.0
module load cuda/cuda-12.1.0

## Llava onevision finetune epoch 1 
###CUDA_VISIBLE_DEVICES=0 python $VIDVRD_EVAL_SCRIPT --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3 --output_dir=[test][onevision]_vidvrd_annotations_v5_3_run2 --conv-mode=qwen_2

## Llava onevision finetune epoch 2
###CUDA_VISIBLE_DEVICES=0 python $VIDVRD_EVAL_SCRIPT --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3_e02 --output_dir=[test][onevision]_vidvrd_annotations_v5_3_e02 --conv-mode=qwen_2

## Llava onevision finetune epoch 3
##CUDA_VISIBLE_DEVICES=0 python $VIDVRD_EVAL_SCRIPT --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3_e03 --output_dir=[test][onevision]_vidvrd_annotations_v5_3_e03 --conv-mode=qwen_2

## Llava onevision finetune epoch 1 vidor zs test
##CUDA_VISIBLE_DEVICES=0 python $PVSG_VIDOR_EVAL_SCRIPT --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3 --output_dir=[test][onevision]_vidor_zs_e01 --conv-mode=qwen_2

## Llava onevision finetune epoch 3 vidor zs test
#CUDA_VISIBLE_DEVICES=0 python $PVSG_VIDOR_EVAL_SCRIPT --model-path=$MODEL_PATH --output_dir=$DATA_OUTPUT_DIR --conv-mode=qwen_2


## LLAVA OV LORA P0 E1 
##CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/evaluate_AG_v5_3_onevision_finetune.py --model-base lmms-lab/llava-onevision-qwen2-7b-si --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_olora256_512_llm --output_dir=/home/jbhol/dso/gits/ActionGenome/AG_llava_annotations/inference_output/[test][onevision]_AG_annotations_v5_3_p0_e01 --conv-mode=qwen_2

## LLAVA OV LORA P0 E1 Finish remaning from 1399
##CUDA_VISIBLE_DEVICES=1 python /home/jbhol/dso/gits/LLaVA-NeXT/evaluate_AG_v5_3_onevision_finetune.py --start_index=1399 --model-base lmms-lab/llava-onevision-qwen2-7b-si --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_olora256_512_llm --output_dir=/home/jbhol/dso/gits/ActionGenome/AG_llava_annotations/inference_output/[test][onevision]_AG_annotations_v5_3_p0_e01_rem --conv-mode=qwen_2


## LLAVA OV LORA P01 E1 
###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/evaluate_AG_v5_3_onevision_finetune.py --model-base lmms-lab/llava-onevision-qwen2-7b-si --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_1_olora256_512_llm --output_dir=/home/jbhol/dso/gits/ActionGenome/AG_llava_annotations/inference_output/[test][onevision]_AG_annotations_v5_3_p01_e01_run2 --conv-mode=qwen_2

## LLAVA OV LORA P02 E1 
###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/evaluate_AG_v5_3_onevision_finetune.py --model-base lmms-lab/llava-onevision-qwen2-7b-si --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_2_olora256_512_llm --output_dir=/home/jbhol/dso/gits/ActionGenome/AG_llava_annotations/inference_output/[test][onevision]_AG_annotations_v5_3_p02_e01 --conv-mode=qwen_2

## LLAVA OV LORA P06 E1 
##CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/evaluate_AG_v5_3_onevision_finetune.py --model-base lmms-lab/llava-onevision-qwen2-7b-si --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_6_olora256_512_llm --output_dir=/home/jbhol/dso/gits/ActionGenome/AG_llava_annotations/inference_output/[test][onevision]_AG_annotations_v5_3_p06_e01 --conv-mode=qwen_2

## LLAVA OV LORA P0_23 E1 
###CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/evaluate_AG_v5_3_onevision_finetune.py --model-base lmms-lab/llava-onevision-qwen2-7b-si --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_23_bash_olora256_512_llm --output_dir=/home/jbhol/dso/gits/ActionGenome/AG_llava_annotations/inference_output/[test][onevision]_AG_annotations_v5_3_p023_e01 --conv-mode=qwen_2


## LLAVA OV LORA All mm_tune P0 E1 
CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/evaluate_AG_v5_3_onevision_finetune.py --model-base lmms-lab/llava-onevision-qwen2-7b-si --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_olora256_all_mm_tune_512_llm --output_dir=/home/jbhol/dso/gits/ActionGenome/AG_llava_annotations/inference_output/[test][onevision]_AG_annotations_v5_3_p00_e01_all_mm_tune --conv-mode=qwen_2

## LLAVA OV LORA All mm_tune P02 E1 
##CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/evaluate_AG_v5_3_onevision_finetune.py --model-base lmms-lab/llava-onevision-qwen2-7b-si --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_2_olora256_all_mm_tune_512_llm --output_dir=/home/jbhol/dso/gits/ActionGenome/AG_llava_annotations/inference_output/[test][onevision]_AG_annotations_v5_3_p02_e01_all_mm_tune --conv-mode=qwen_2




CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/Infer_WOGT_AG_v5_3_onevision_finetune.py \
    --model-path=/groups/sernam/VideoSGG/checkpoints/llava_ov/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_olora256_all_mm_tune_512_llm  \
    --model-base lmms-lab/llava-onevision-qwen2-7b-si \
    --output_dir=/home/jbhol/dso/gits/ActionGenome/AG_llava_annotations/inference_output/wogt/[test][onevision]_wogt_AG_p023_e01 \
    --conv-mode=qwen_2


CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/evaluate_AG_v5_3_onevision_withbb.py \
    --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/work_dirs/AG/vidnorm/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split00_ep01_olora256_512_llm_withbb_vidnorm \
    --model-base lmms-lab/llava-onevision-qwen2-7b-si \
    --output_dir=/home/jbhol/dso/gits/ActionGenome/AG_llava_annotations/inference_output/bb/vidnorm/[test][onevision]_AG_annotations_v5_3_p00_e01_bb \
    --conv-mode=qwen_2