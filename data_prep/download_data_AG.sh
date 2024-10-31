WORK_DIR=/root/jbhoi/gits

mkdir -p $WORK_DIR
cd $WORK_DIR

git clone https://github.com/JingweiJ/ActionGenome.git
cd ActionGenome

wget https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip