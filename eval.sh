source ~/.bashrc
conda activate llava

python evaluate_vidvrd_v5_3_onevision_finetune.py \
     --model-path work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vidvrd_v5_3_shuffled_split0_lora256_512_olora_ft_llm+mmproj+visiontower \
     --output_dir work_dirs/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_to_video_am9_vidvrd_v5_3_shuffled_split0_lora256_512_olora_ft_llm+mmproj+visiontower/inference_test \
     --conv-mode qwen_2 \
     --model-base lmms-lab/llava-onevision-qwen2-7b-si