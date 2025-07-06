#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --gres-flags=enforce-binding
#SBATCH --job-name=ZS_InternVL_AG
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


#bash /home/jbhol/dso/gits/LLaVA-NeXT/InternVL/Eval_ZS_VRD_InternVL.sh

python /home/jbhol/dso/gits/LLaVA-NeXT/InternVL/evaluate_AG_ZS.py --output_dir=/home/jbhol/dso/gits/ActionGenome/inference_out/internvl/ZS