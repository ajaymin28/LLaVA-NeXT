## Videos are put here: /groups/sernam/datasets/VidVRD/data/vidvrd/videos

### To run video llava one vision cli Q&A:

```
python /home/jbhol/dso/gits/LLaVA-NeXT/playground/demo/video_demo_cli_multi-turn.py --output_dir=/home/jbhol/dso/gits/LLaVA-NeXT/outputs --output_name=output.json --video_path=/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/videos/ILSVRC2015_train_00098004.mp4 --conv-mode=qwen_2
```

### In cli while code is running you can change videos or set frames by:

- setvideo=/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/videos/ILSVRC2015_train_00098004.mp4
- setframes=[0,1,2,3,4,5,6]
- exit ==> will exit the cli
