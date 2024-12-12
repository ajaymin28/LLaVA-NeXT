CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/InternVL/evaluate_AG_v5_3_onevision_finetune.py \
    --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/InternVL/checkpoints/internvl8B/lora_with_tunables/AG/AG_P0_internvl2_lora_r256_llm_merged  \
    --output_dir=/home/jbhol/dso/gits/LLaVA-NeXT/InternVL/output/[test][internvl]_AG_p0_e01



CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/InternVL/evaluate_vidvrd_v5_3_internvl_finetune.py \
    --model-path=/home/jbhol/dso/gits/LLaVA-NeXT/InternVL/checkpoints/internvl8B/lora_with_tunables/VRD/11.20_16:06/VRD_P09_internvl2_lora_r256_llm_merged  \
    --output_dir=/home/jbhol/dso/gits/LLaVA-NeXT/InternVL/output/[test][internvl]_VRD_p09_e01

CUDA_VISIBLE_DEVICES=0 python /home/jbhol/dso/gits/LLaVA-NeXT/InternVL/evaluate_vidvrd_v5_3_internvl_finetune.py \
    --model-path=/groups/sernam/VideoSGG/checkpoints/internvl8B/VRD_P0_internvl2_lora_r256_llm_merged  \
    --output_dir=/home/jbhol/dso/gits/LLaVA-NeXT/InternVL/output/[test][internvl]_VRD_p00_e01



