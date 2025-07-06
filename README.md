## DSGG

Customized MLLMs for generating Dyanmic Scene Graph Generation.

## Data prep for pvsg, vidvrd and ActionGnome.
Refer to [Data Prep Readme](./data_prep/README.md)


## Full finetuning

Have atleast 5-6 GPUs each GPU utilizes ~40Gb VRAM.

```
bash scripts/video/train/SO400M_Qwen2_7B_ov_to_video_am9.sh
```

## Training only LORA

```
bash scripts/video/train/LORA_SO400M_Qwen2_7B_ov_to_video_am9.sh
```

#### When loading LORA model for inference pass --model-base=lmms-lab/llava-onevision-qwen2-7b-si


## Inference

### To run video llava one vision cli Q&A:

```
python playground/demo/video_demo_cli_multi-turn.py --output_dir=/outputs --output_name=output.json --video_path=path/to/video.mp4 --conv-mode=qwen_2


In cli while code is running you can change videos or set frames by:

- setvideo=/vidvrd/videos/ILSVRC2015_train_00098004.mp4
- setframes=[0,1,2,3,4,5,6]
- exit ==> will exit the cli

```

## Evaluation

```
sbatch ./eval/Evaluate_llavaonevision.slurm
```