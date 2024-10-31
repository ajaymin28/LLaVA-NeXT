#!/bin/bash

WORK_DIR=/root/jbhoi

mkdir -p $WORK_DIR
cd $WORK_DIR

git clone https://github.com/zhengsipeng/VRDFormer_VRD.git
cd VRDFormer_VRD/data/vidvrd

wget https://huggingface.co/datasets/shangxd/imagenet-vidvrd/resolve/main/vidvrd-annotations.zip
wget https://huggingface.co/datasets/shangxd/imagenet-vidvrd/resolve/main/vidvrd-videos-part1.zip
wget https://huggingface.co/datasets/shangxd/imagenet-vidvrd/resolve/main/vidvrd-videos-part2.zip

sudo apt-get install zip unzip
unzip vidvrd-annotations.zip

mv vidvrd-dataset/train train
mv vidvrd-dataset/test test
mv vidvrd-dataset/videos videos
rm -rf vidvrd-dataset

unzip vidvrd-videos-part1.zip -d videos
mv videos/vidvrd-videos-part1/*.mp4 videos
rm -rf videos/vidvrd-videos-part1
unzip vidvrd-videos-part2.zip -d videos
mv videos/vidvrd-videos-part2/*.mp4 videos
rm -rf videos/vidvrd-videos-part2


# VRDDATA_VIDEOS_PATH = $WORK_DIR/gits/VRDFormer_VRD/data/vidvrd/videos

# cd $WORK_DIR/gits

# git clone https://github.com/ajaymin28/LLaVA-NeXT.git
# cd LLaVA-NeXT

# python data_prep/change_video_paths.py \
#     --annot_path "data_prep/video_llava_vidvrd_annotations_v7_withtime" \
#     --new_annot_save_path "data_prep/video_llava_vidvrd_annotations_v7_withtime_new_path" \
#     --new_video_path_root $VRDDATA_VIDEOS_PATH