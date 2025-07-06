from utils.utilities import load_AG_annotations, AG_Objects, AG_relations
from utils.utilities import getConvBlock, getPromptTemplate, getRandomPrompt, get_shuffled_list, chunk_list
import json
import time
import multiprocessing as mp
from tqdm import tqdm
import os
import random
import cv2
import argparse
import glob
from data_prep.vidvrd2dataset import VidVRD, VidOR

random.seed(145)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process VRD video annotations with specified directories and chunk size.")

    parser.add_argument(
        "--video_root_path",
        type=str,
        default="/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/videos",
        help="Root path to the video files."
    )
    
    parser.add_argument(
        "--subset",
        type=str,
        default="test",
        help="subset to dump (train/val)"
    )

    parser.add_argument(
        "--frame_dir", 
        type=str,
        default="/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd/frames",               
        help="Root folder containing frames to be dumped."
    )
    
    parser.add_argument(
        "--ag_annotations_dir",
        type=str,
        default="/groups/sernam/datasets/ActionGenome/ActionGenome/annot_data",
        help="Directory containing ActionGenome annotation data."
    )

    parser.add_argument(
        "--n_workers",
        type=int,
        default=3,
        help="Number of workers/Processes to use."
    )

    return parser.parse_args()




def init_worker(task_queue,result_queue,stop_event):
    """Worker process that initializes and processes as data comes."""
    while not stop_event.is_set():
        capture = None
        try:
            # Wait for new task
            video_path,video_id,frame_indexes,save_path = task_queue.get(timeout=1)  # Timeout allows checking for stop_event
            os.makedirs(save_path,exist_ok=True)

            ## Extract the frames
            capture = cv2.VideoCapture(video_path)
            # print(video_path)
            for frame_id in frame_indexes:
                capture.set(cv2.CAP_PROP_POS_FRAMES,int(frame_id))
                ret, frame = capture.read()
                if ret:
                    frame_name = f"{frame_id:06}.png"
                    frame_path = os.path.join(save_path,frame_name)
                    cv2.imwrite(frame_path,frame)
            capture.release()
        except Exception as e:
            print(f"Exception e: {e}")
            if capture is not None:
                capture.release()
            if stop_event.is_set():
                break  # Exit if stop event is set
            continue
        # finally:
            # capture.release()

def start_worker_pool(num_workers=3):
    """Initialize worker processes and manage task/result queues."""
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    stop_event = mp.Event()
    workers = []

    # Create worker processes
    for _ in range(num_workers):
        worker = mp.Process(target=init_worker, args=(task_queue, result_queue, stop_event))
        worker.start()
        workers.append(worker)

    return task_queue, result_queue, stop_event, workers

def stop_worker_pool(stop_event, workers):
    """Stop all worker processes."""
    stop_event.set()  # Signal all workers to stop
    for worker in workers:
        worker.join()

def submit_inference(task_queue,video_path,video_id,frame_indexes,save_path):
    """Submit a new request to the task queue."""
    task_queue.put((video_path,video_id,frame_indexes,save_path))



if __name__=="__main__":

    args = parse_arguments()


    dataset_name = "vidvrd"
    # version = args.output_dir
    subset = args.subset

    splits = [subset]
    imagenet_vidvrd_root = "/home/jbhol/dso/gits/VRDFormer_VRD/data/vidvrd"
    # annotations_root = os.path.join(imagenet_vidvrd_root, "annotations")
    VIDEO_ROOT_PATH = os.path.join(imagenet_vidvrd_root, "videos")
    dataset = VidVRD(imagenet_vidvrd_root, VIDEO_ROOT_PATH, splits)

    data_dir = os.path.join(imagenet_vidvrd_root, subset)
    anno_files = glob.glob(os.path.join(data_dir, "*.json")) 

    video_ids = []
    for test_annot in anno_files:
        filename = os.path.basename(test_annot)
        # filename = test_annot.split("/")[-1]
        filename = filename.split(".")[0]
        video_ids.append(filename)

    import pickle

    with open("/home/jbhol/dso/gits/VRDFormer_VRD/data/vrd_val_vid_ids.pickle", "wb") as f:
        pickle.dump({"videoids": video_ids},f)


    task_queue, result_queue, stop_event, workers = start_worker_pool(num_workers=args.n_workers)

    set_count = {"train": 0,"test": 0}
    # video_ids_by_set = { "train": [], "test": [] }

    json_annotations = []
    json_file_counter = 0
    Annotation_counter = 0


    for video_id in video_ids:

        video_path = os.path.join(VIDEO_ROOT_PATH,f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"[ERROR] video doesnt exist at: {video_path}")
            raise FileNotFoundError()

        frame_ids_to_dump = []

        annot = dataset.get_anno(vid=video_id)
        frame_h, frame_w = annot["height"], annot["width"]
        frame_count = annot["frame_count"]
        video_id = annot["video_id"]
        video_fps = annot["fps"]
        sub_ob_jects = annot['subject/objects']
        sub_ob_jects_by_id = {obj["tid"]: obj  for obj in sub_ob_jects}
        rels = annot['relation_instances']
        trajectories = annot['trajectories']
        for rel in rels:
            begin_fid = rel['begin_fid']
            end_fid = rel['end_fid']
            for activity_range in range(begin_fid,end_fid):
                frame_ids_to_dump.append(activity_range)

        
        frame_save_path = os.path.join(args.frame_dir,video_id)

        try:
            # Submit inference requests
            submit_inference(task_queue,video_path=video_path,video_id=video_id,frame_indexes=frame_ids_to_dump,save_path=frame_save_path)
            # Allow some time for processing (adjust as needed)
            # time.sleep(1)
        except Exception as e:
            print(f"error in inference: {e} vid:{video_id}")


    Total = task_queue.qsize()
    Processed = 0
    pbar = tqdm(total=task_queue.qsize())
    pbar.n = 0
    pbar.last_print_n = 0
    pbar.refresh()

    while task_queue.qsize()>0:
        time.sleep(1)
        if task_queue.qsize()==0:
            break

        Processed = Total - task_queue.qsize()
        pbar.n = Processed
        pbar.last_print_n = Processed
        pbar.refresh()
    

    stop_worker_pool(stop_event, workers)

        




        

