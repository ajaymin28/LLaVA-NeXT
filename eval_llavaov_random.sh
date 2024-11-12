source ~/.bashrc
conda activate llava

# ## Full tune model
# CUDA_VISIBLE_DEVICES=0 python evaluate_AG_v5_3_onevision_finetune.py \
#     --model-path="/groups/sernam/VideoSGG/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_1_olora256_all_mm_tune_512_llm" \
#     --output_dir="/groups/sernam/VideoSGG/results/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_1_olora256_all_mm_tune_512_llm" --conv-mode="qwen_2"

## LORA model


CUDA_VISIBLE_DEVICES=0 python evaluate_AG_v5_3_onevision_finetune_random.py \
    --model-base lmms-lab/llava-onevision-qwen2-7b-si \
    --model-path /groups/sernam/VideoSGG/checkpoints/llava_ov/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_1_olora256_all_mm_tune_512_llm \
    --output_dir /groups/sernam/VideoSGG/results/llava_ov/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_1_olora256_all_mm_tune_512_llm_subset2000/ --conv-mode qwen_2 \
    --subset 2000