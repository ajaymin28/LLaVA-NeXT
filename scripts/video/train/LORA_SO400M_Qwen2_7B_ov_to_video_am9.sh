#!/bin/bash

# unset PATH
#unset CUDA_HOME

source ~/.bashrc
##conda activate vlnext
module load gcc/gcc-11.2.0
module load cuda/cuda-12.1.0
# unset LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

# Set up the data folder
IMAGE_FOLDER="/groups/sernam/datasets/ActionGenome/Charades_v1_480"
VIDEO_FOLDER="/groups/sernam/datasets/ActionGenome/Charades_v1_480"
DATA_YAML="scripts/video/train/exp.yaml" # e.g exp.yaml

# source ~/.bashrc
# conda activate vlnext
# module load gcc/gcc-11.2.0
# export CUDA_HOME=/home/jbhol/cuda/cuda-12.1
# export PATH=${CUDA_HOME}/bin:${PATH}
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jbhol/cuda/cuda-12.1/targets/x86_64-linux/lib

############### Prepare Envs #################
#python3 -m pip install flash-attn --no-build-isolation
alias python=python3
############### Show Envs ####################

nvidia-smi

################ Arnold Jobs ################

export WANDB_PROJECT="LLM4VideoSGG"

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
#

BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# Stage 2
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-AG_v5_3_split0_olora256_512_llm"
#PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-ov-si"  
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-si" 
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

## --mm_projector_lr 2e-5
##     --init_lora_weights olora \
# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port="${port_in_cmd}" \
deepspeed --master_port 30000 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --lora_enable True \
    --mm_tunable_parts="mm_language_model" \
    --lora_r 256 --lora_alpha 512 \
    --init_lora_weights olora \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_lr 2e-5 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir /home/jbhol/dso/gits/LLaVA-NeXT/work_dirs/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 8 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2
exit 0;