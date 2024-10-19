#!/bin/bash

PVSG_VIDOR_EVAL_SCRIPT = "/home/jbhol/dso/gits/Video-LLaVA/evaluate_pvsg_vidor_zs_onevision_finetuned.py"
VIDVRD_EVAL_SCRIPT = "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_3_onevision_finetune.py"

MODEL_PATH = "/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3_e02"
JOB_NAME = ""
DATA_OUTPUT_DIR = "[test][onevision]_vidor_zs_e02"


#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=$JOB_NAME
#SBATCH --time=3-24:00:00
#SBATCH --output=/lustre/fs1/home/jbhol/dso/jobs/%x_raw_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ja882177@ucf.edu

export CUDA_HOME=/lustre/fs1/home/jbhol/.local/cuda-12.1
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lustre/fs1/home/jbhol/.local/cuda-12.1/targets/x86_64-linux/lib

module load gcc/gcc-11.2.0

## Llava onevision finetune epoch 1 
###CUDA_VISIBLE_DEVICES=0 python $VIDVRD_EVAL_SCRIPT --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3 --output_dir=[test][onevision]_vidvrd_annotations_v5_3_run2 --conv-mode=qwen_2

## Llava onevision finetune epoch 2
###CUDA_VISIBLE_DEVICES=0 python $VIDVRD_EVAL_SCRIPT --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3_e02 --output_dir=[test][onevision]_vidvrd_annotations_v5_3_e02 --conv-mode=qwen_2

## Llava onevision finetune epoch 3
##CUDA_VISIBLE_DEVICES=0 python $VIDVRD_EVAL_SCRIPT --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3_e03 --output_dir=[test][onevision]_vidvrd_annotations_v5_3_e03 --conv-mode=qwen_2

## Llava onevision finetune epoch 1 vidor zs test
##CUDA_VISIBLE_DEVICES=0 python $PVSG_VIDOR_EVAL_SCRIPT --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3 --output_dir=[test][onevision]_vidor_zs_e01 --conv-mode=qwen_2

## Llava onevision finetune epoch 3 vidor zs test
CUDA_VISIBLE_DEVICES=0 python $PVSG_VIDOR_EVAL_SCRIPT --model-path=$MODEL_PATH --output_dir=$DATA_OUTPUT_DIR --conv-mode=qwen_2