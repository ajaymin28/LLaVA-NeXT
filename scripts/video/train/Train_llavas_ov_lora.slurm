#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=lora_p023_e01_llava_ov_finetune_vrd_v5_3
#SBATCH --time=2-24:00:00
#SBATCH --output=/lustre/fs1/home/jbhol/dso/jobs/%x_raw_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ja882177@ucf.edu

###Load modules

export CUDA_HOME=/lustre/fs1/home/jbhol/.local/cuda-12.1
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lustre/fs1/home/jbhol/.local/cuda-12.1/targets/x86_64-linux/lib
module load gcc/gcc-11.2.0

wandb login 8350cb094ed6779c41f439917838bbe922b6bec2

cd /lustre/fs1/home/jbhol/dso/gits/LLaVA-NeXT/
#v7 with time
##bash /home/jbhol/dso/gits/LLaVA-NeXT/scripts/video/train/LORA_SO400M_Qwen2_7B_ov_to_video_am9_Partial.sh
## full data
bash /home/jbhol/dso/gits/LLaVA-NeXT/scripts/video/train/LORA_SO400M_Qwen2_7B_ov_to_video_am9.sh