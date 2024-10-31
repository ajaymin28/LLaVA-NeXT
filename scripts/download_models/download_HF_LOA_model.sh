#!/bin/bash

DOWN_MODEL_PATH=$1
OUTPUT_DIR=$2

mkdir -p $OUTPUT_DIR
wget -O "$OUTPUT_DIR/adapter_config.json" $DOWN_MODEL_PATH/adapter_config.json
wget -O "$OUTPUT_DIR/adapter_model.bin" $DOWN_MODEL_PATH/adapter_model.bin
wget -O "$OUTPUT_DIR/config.json" $DOWN_MODEL_PATH/config.json
wget -O "$OUTPUT_DIR/generation_config.json" $DOWN_MODEL_PATH/generation_config.json
wget -O "$OUTPUT_DIR/non_lora_trainables.bin" $DOWN_MODEL_PATH/non_lora_trainables.bin
wget -O "$OUTPUT_DIR/trainer_state.json" $DOWN_MODEL_PATH/trainer_state.json

## Usage: bash /home/jbhol/dso/gits/LLaVA-NeXT/scripts/download_models/download_HF_LOA_model.sh https://huggingface.co/ajaymin28/vl-sg-AG-loras/resolve/main/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_6_olora256_all_mm_tune_512_llm save/to/dir/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-AG_v5_3_split0_6_olora256_all_mm_tune_512_llm