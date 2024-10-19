## Data prep for pvsg

Follow pvsg envrionment setup and download data from [PVSG](https://github.com/LilyDaytoy/OpenPVSG)

Alternatively data can be downloaded from huggingface (wget is easier from HF)

- https://huggingface.co/datasets/Jingkang/PVSG
- https://huggingface.co/datasets/shangxd/imagenet-vidvrd
- https://huggingface.co/datasets/shangxd/vidor

## Preparing Q&A annotations for video-llava

### PVSG

```
python /home/jbhol/dso/gits/VideoLLAVAGit/Video-LLaVA/data_prep/OPVSG_VIdeoLLAVA_Annot_chatgpt_v17.py --data_root=/home/jbhol/dso/gits/OpenPVSG/data/ --output_dir=out_dir --dataset=vidor
```

### VidVRD

```

Annoatations are prepared based on annotation segments

e.g 

if triplet [dog, eating, food] is present in [frame-1,frame-30], [frame-70,frame-80]
then we have 
[dog, eating, food]_[frame-1, frame-8]
[dog, eating, food]_[frame-9, frame-15] ...

[dog, eating, food]_[frame-70, frame-78]

for frame 79 and 80 which remains, random sampling is done between frame-70 and frame-80
means add 79 and 80 frames first then random sample between frame-70 and frame-80 till we have 8 frames.


python /home/jbhol/dso/gits/VideoLLAVAGit/Video-LLaVA/data_prep/prepare_video_llava_v6_newsampling.py

```


```

[0,4,8,12,16,20,24,28] take every_nth=4 and then shift by shift_frames=5
[5,9,13,18,22,26,30,34] take every_nth=4 and then again shift by shift_frames=5
[10,14 .....]

Note: value of shift_frames=5 can be changed and every_nth=4 can also be changed based on the requirements

python /home/jbhol/dso/gits/VideoLLAVAGit/Video-LLaVA/data_prep/prepare_video_llava_v7_newsampling.py --data_root=/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/ --output_dir=/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/llava_annotations/v7_wotime --dataset=vidvrd --every_nth=4 --shift_frames=5

```


```
With time blocks triplet_[Frame-start,Frame-end]

[0,4,8,12,16,20,24,28] take every_nth=4 then shift by shift_frames=5
[5,9,13,18,22,26,30,34] take every_nth=4 then again shift by shift_frames=5
[10,14 .....]

Note: value of shift_frames=5 can be changed and every_nth=4 can also be changed based on the requirements

python /home/jbhol/dso/gits/VideoLLAVAGit/Video-LLaVA/data_prep/prepare_video_llava_v7_newsampling_with_time.py --data_root=/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/ --output_dir=/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/llava_annotations/v7_with_time --dataset=vidvrd --every_nth=4 --shift_frames=5
```
