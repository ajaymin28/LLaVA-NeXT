import sys
import os

os.environ["TORCH_LOGS"] = "+dynamo" 
os.environ['TORCHDYNAMO_VERBOSE'] = "1"


sys.path.append(os.getcwd())
from llava.train.train import train

if __name__ == "__main__":
    train()
