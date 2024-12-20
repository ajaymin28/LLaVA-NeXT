source ~/.bashrc
conda activate llava

# ## Full tune model
# CUDA_VISIBLE_DEVICES=0 python evaluate_AG_v5_3_onevision_finetune.py \
#     --model-path="/groups/sernam/VideoSGG/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_1_olora256_all_mm_tune_512_llm" \
#     --output_dir="/groups/sernam/VideoSGG/results/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_1_olora256_all_mm_tune_512_llm" --conv-mode="qwen_2"

## LORA model


CUDA_VISIBLE_DEVICES=0 python evaluate_AG_v5_3_onevision_finetune.py \
    --model-base lmms-lab/LLaVA-Video-7B-Qwen2 \
    --model-path /groups/sernam/VideoSGG/checkpoints/llava_video/11.03_17:51-llavaov-AG-v5_3_split00_olora256_512_full_visiontowerlr_2e-6 \
    --output_dir work_dirs/results/llava_video/11.03_17:51-llavaov-AG-v5_3_split00_olora256_512_full_visiontowerlr_2e-6 --conv-mode qwen_2 \