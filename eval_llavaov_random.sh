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

CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/evaluate_AG_v5_3_ov_ft_random_w_termporal.py \
    --model-base lmms-lab/llava-onevision-qwen2-7b-si \
    --model-path /home/jbhol/dso/gits/LLaVA-NeXT/work_dirs/vrd/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v7_uprompt_split023_olora256_512_llm \
    --output_dir /home/jbhol/dso/gits/ActionGenome/inference_out/llava_ov/AG/AG_P023_V7_1kSamples \
    --conv-mode qwen_2 \
    --subset 1000


CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/evaluate_vidvrd_v5_3_onevisionWithID_FT_Quad.py \
    --model-base lmms-lab/llava-onevision-qwen2-7b-si \
    --model-path /home/jbhol/dso/gits/LLaVA-NeXT/work_dirs/vrd/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-VRD_v5_3_Quad_split09_olora256_512_llm \
    --output_dir /home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/inference_outputs/llava_ov/[test]lora_llava_ov_vidvrd_annotations_v5_3_p09_e01_Quad_FT \
    --conv-mode qwen_2







                # "--model-path",
                # "/root/LLaVA-NeXT/checkpoints/models--ajaymin28--vl-sg-AG-loras/snapshots/843d928c78ce5d8e4677edd28c79f37677fb458a/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_23_bash_olora256_512_llm",
                # "--output_dir",
                # "test/",
                # "--conv-mode",
                # "qwen_2",
                # "--model-base",
                # "lmms-lab/LLaVA-Video-7B-Qwen2",