#!/bin/bash

#PVSG_VIDOR_EVAL_SCRIPT = "/home/jbhol/dso/gits/Video-LLaVA/evaluate_pvsg_vidor_zs_onevision_finetuned.py"
#VIDVRD_EVAL_SCRIPT = "/home/jbhol/dso/gits/Video-LLaVA/evaluate_vidvrd_v5_3_onevision_finetune.py"

#MODEL_PATH = "/home/jbhol/dso/gits/LLaVA-NeXT/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vrd_v5_3_e02"
#JOB_NAME = ""
#DATA_OUTPUT_DIR = "[test][onevision]_vidor_zs_e02"


#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=p06_AG_llavavidqwen2_lora
#SBATCH --time=2-24:00:00
#SBATCH --output=/lustre/fs1/home/jbhol/dso/jobs/%x_raw_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ja882177@ucf.edu

##export CUDA_HOME=/lustre/fs1/home/jbhol/.local/cuda-12.1
##export PATH=${CUDA_HOME}/bin:${PATH}
##export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lustre/fs1/home/jbhol/.local/cuda-12.1/targets/x86_64-linux/lib

##source ~/.bashrc
module load gcc/gcc-11.2.0
module load cuda/cuda-12.1.0

## LLAVAVID Qwen2 LORA tune p00
##CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/evaluate_AG_v5_3_onevision_finetune.py --model-base lmms-lab/LLaVA-Video-7B-Qwen2 --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/work_dirs/llavaqwen2/llavaqwen2-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split00_all_mm_tune_olora256_512_llm --output_dir=/home/jbhol/dso/gits/ActionGenome/AG_llava_annotations/inference_output/[test][llavavid_qwen2]_AG_annotations_v5_3_p00_e01_lora --conv-mode=qwen_2


## LLAVAVID Qwen2 LORA tune p06
CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/evaluate_AG_v5_3_onevision_finetune.py --model-base lmms-lab/LLaVA-Video-7B-Qwen2 --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/work_dirs/llavaqwen2/llavaqwen2-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split06_all_mm_tune_olora256_512_llm --output_dir=/home/jbhol/dso/gits/ActionGenome/AG_llava_annotations/inference_output/[test][llavavid_qwen2]_AG_annotations_v5_3_p06_e01_lora --conv-mode=qwen_2

