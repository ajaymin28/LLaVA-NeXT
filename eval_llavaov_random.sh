source activate base
conda activate llava

# ## Full tune model
# CUDA_VISIBLE_DEVICES=0 python evaluate_AG_v5_3_onevision_finetune.py \
#     --model-path="/groups/sernam/VideoSGG/checkpoints/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_1_olora256_all_mm_tune_512_llm" \
#     --output_dir="/groups/sernam/VideoSGG/results/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_1_olora256_all_mm_tune_512_llm" --conv-mode="qwen_2"

## LORA model


CUDA_VISIBLE_DEVICES=0 python evaluate_AG_v5_3_onevision_finetune_random.py \
    --model-base lmms-lab/llava-onevision-qwen2-7b-si \
    --model-path /root/LLaVA-NeXT/checkpoints/models--ajaymin28--vl-sg-AG-loras/snapshots/843d928c78ce5d8e4677edd28c79f37677fb458a/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_23_bash_olora256_512_llm \
    --output_dir /root/LLaVA-NeXT/results/ag_full_temperature10_100subset\
    --conv-mode qwen_2 \
    --temperature 10.0 \
    --subset /root/LLaVA-NeXT/ag_subset_100.json



                # "--model-path",
                # "/root/LLaVA-NeXT/checkpoints/models--ajaymin28--vl-sg-AG-loras/snapshots/843d928c78ce5d8e4677edd28c79f37677fb458a/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_23_bash_olora256_512_llm",
                # "--output_dir",
                # "test/",
                # "--conv-mode",
                # "qwen_2",
                # "--model-base",
                # "lmms-lab/LLaVA-Video-7B-Qwen2",