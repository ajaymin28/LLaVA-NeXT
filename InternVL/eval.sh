

module load gcc/gcc-11.2.0
module load cuda/cuda-12.1.0

python /home/jbhol/dso/gits/LLaVA-NeXT/InternVL/evaluate_vidvrd_v5_3_internvl_finetune.py \
    --model-path="/home/jbhol/dso/gits/LLaVA-NeXT/InternVL/checkpoints/internvl8B/lora_with_tunables/p09/ag_internvl2_lora_r16_llm" \
    --output_dir="/home/jbhol/dso/gits/LLaVA-NeXT/InternVL/outputs/[test][internvl]_vrd_p09_e01" \
    --force_sample=True



python /home/jbhol/dso/gits/LLaVA-NeXT/InternVL/evaluate_vidvrd_v5_3_internvl_finetune.py \
    --model-path="/home/jbhol/dso/gits/LLaVA-NeXT/InternVL/pretrained/InternVL2-8B" \
    --output_dir="/home/jbhol/dso/gits/LLaVA-NeXT/InternVL/outputs/[test][internvl]_vrd_p09_e01" \
    