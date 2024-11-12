source ~/.bashrc
conda activate llava


ckpt="work_dirs/10.28_12:22-llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-vidvrd_v5_3_split0_olora256_1024_llm+vision+adapter+fc_use_rslora"
python evaluate_vidvrd_v5_3_onevision_finetune.py \
     --model-path $ckpt \
     --output_dir $ckpt/inference_test \
     --conv-mode qwen_2 \
     --model-base lmms-lab/llava-onevision-qwen2-7b-si