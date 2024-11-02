#!/bin/bash

DOWN_MODEL_PATH=$1 ##url="https://huggingface.co/ajaymin28/vl-sg-AG-fulltune/resolve/main/llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-ov_AG_v5_3_split0_bash_fulltune"
folder_name=$(basename "$DOWN_MODEL_PATH")
OUTPUT_DIR="$2/$folder_name"

mkdir -p $OUTPUT_DIR

wget -O "$OUTPUT_DIR/model-00001-of-00004.safetensors" $DOWN_MODEL_PATH/model-00001-of-00004.safetensors
wget -O "$OUTPUT_DIR/model-00002-of-00004.safetensors" $DOWN_MODEL_PATH/model-00002-of-00004.safetensors
wget -O "$OUTPUT_DIR/model-00003-of-00004.safetensors" $DOWN_MODEL_PATH/model-00003-of-00004.safetensors
wget -O "$OUTPUT_DIR/model-00004-of-00004.safetensors" $DOWN_MODEL_PATH/model-00004-of-00004.safetensors
wget -O "$OUTPUT_DIR/added_tokens.json" $DOWN_MODEL_PATH/added_tokens.json
wget -O "$OUTPUT_DIR/config.json" $DOWN_MODEL_PATH/config.json
wget -O "$OUTPUT_DIR/generation_config.json" $DOWN_MODEL_PATH/generation_config.json
wget -O "$OUTPUT_DIR/latest" $DOWN_MODEL_PATH/latest
wget -O "$OUTPUT_DIR/merges.txt" $DOWN_MODEL_PATH/merges.txt
wget -O "$OUTPUT_DIR/model.safetensors.index.json" $DOWN_MODEL_PATH/model.safetensors.index.json
wget -O "$OUTPUT_DIR/rng_state_0.pth" $DOWN_MODEL_PATH/rng_state_0.pth
wget -O "$OUTPUT_DIR/rng_state_1.pth" $DOWN_MODEL_PATH/rng_state_1.pth
wget -O "$OUTPUT_DIR/rng_state_2.pth" $DOWN_MODEL_PATH/rng_state_2.pth
wget -O "$OUTPUT_DIR/rng_state_3.pth" $DOWN_MODEL_PATH/rng_state_3.pth
wget -O "$OUTPUT_DIR/scheduler.pt" $DOWN_MODEL_PATH/scheduler.pt
wget -O "$OUTPUT_DIR/special_tokens_map.json" $DOWN_MODEL_PATH/special_tokens_map.json
wget -O "$OUTPUT_DIR/tokenizer.json" $DOWN_MODEL_PATH/tokenizer.json
wget -O "$OUTPUT_DIR/tokenizer_config.json" $DOWN_MODEL_PATH/tokenizer_config.json
wget -O "$OUTPUT_DIR/trainer_state.json" $DOWN_MODEL_PATH/trainer_state.json
wget -O "$OUTPUT_DIR/training_args.bin" $DOWN_MODEL_PATH/training_args.bin
wget -O "$OUTPUT_DIR/vocab.json" $DOWN_MODEL_PATH/vocab.json