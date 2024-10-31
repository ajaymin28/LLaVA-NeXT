WORK_DIR=/root/jbhoi/gits

mkdir -p $WORK_DIR
cd $WORK_DIR

git clone https://github.com/JingweiJ/ActionGenome.git
cd ActionGenome
mkdir -p videos

wget https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip

sudo apt-get install zip unzip

unzip Charades_v1_480.zip -d videos
mv videos/Charades_v1_480/*.mp4 videos
rm -rf videos/Charades_v1_480
mv Charades_v1_480.zip /dev/shm