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
random.seed(145)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process AG video annotations with specified directories and chunk size.")

    parser.add_argument(
        "--video_root_path",
        type=str,
        default="/groups/sernam/datasets/ActionGenome/Charades_v1_480",
        help="Root path to the video files."
    )
    
    parser.add_argument(
        "--subset",
        type=str,
        default="test",
        help="subset to dump (train/test)"
    )

    parser.add_argument(
        "--frame_dir", 
        type=str,
        default="/groups/sernam/datasets/ActionGenome/frames",               
        help="Root folder containing frames to be dumped."
    )
    
    parser.add_argument(
        "--ag_annotations_dir",
        type=str,
        default="/groups/sernam/datasets/ActionGenome/ActionGenome/annot_data",
        help="Directory containing ActionGenome annotation data."
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
            for frame_id in frame_indexes:
                capture.set(cv2.CAP_PROP_POS_FRAMES,int(frame_id))
                ret, frame = capture.read()
                if ret:
                    frame_name = f"{frame_id:06}.png"
                    frame_path = os.path.join(save_path,frame_name)
                    cv2.imwrite(frame_path,frame)
        except Exception as e:
            if capture is not None:
                capture.release()
            if stop_event.is_set():
                break  # Exit if stop event is set
            continue

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
    
    VIDEO_ROOT_PATH = args.video_root_path
    # OUTPUT_JSON_DIR = args.output_json_dir
    AG_ANNOTATIONS_DIR = args.ag_annotations_dir
    # CHUNK_N = args.chunk_n # Q&A will be chunked into CHUNK_N parts

    task_queue, result_queue, stop_event, workers = start_worker_pool(num_workers=10)


    # os.makedirs(OUTPUT_JSON_DIR,exist_ok=True)

    object_anno, person_anno, frame_list = load_AG_annotations(annotation_dir=AG_ANNOTATIONS_DIR)

    assert set(object_anno.keys()) == set(person_anno.keys())
    assert len(object_anno) == len(frame_list)

    set_count = {"train": 0,"test": 0}
    video_ids_by_set = { "train": [], "test": [] }

    dataset_meta = {
        "objects": [],
        "relationships": {
            "attention": [],
            "spatial": [],
            "contacting": []
        }
    }

    json_annotations = []
    json_file_counter = 0
    Annotation_counter = 0

    # video2frames = {}
    video2frames_full = {}
    for path in frame_list:
        video, frame = path.split('/')
        if video not in video2frames_full:
            video2frames_full[video] =[]
        video2frames_full[video].append(path)
    
    # person data and object data by video frameid
    video_frame_data = {}
    # For each video, dump frames.
    for v in tqdm(video2frames_full):
        # curr_frame_dir = os.path.join(frame_dir, v)
        if v not in video_frame_data.keys():
            video_frame_data[v] = []
        framesToKeep = video2frames_full[v]
        for frameid in framesToKeep:
            objects_annot = object_anno[frameid]
            person_data = person_anno[frameid]
            frameid = frameid.split("/")[-1]
            video_frame_data[v].append([frameid,person_data,objects_annot])



    # get dataset metadata, train/test split
    for videoid, video_data in video_frame_data.items():
        for video_annotation in video_data:
            frameid, person_data,objects_annot = video_annotation

            for objAnnot in objects_annot:
                obj_class = objAnnot["class"]
                obj_bb =  objAnnot["bbox"]   

                if obj_class not in dataset_meta["objects"]:
                    dataset_meta["objects"].append(obj_class)

                attention_relationship = objAnnot["attention_relationship"]
                spatial_relationship = objAnnot["spatial_relationship"]
                contacting_relationship = objAnnot["contacting_relationship"]

                if attention_relationship!=None:
                    for attn_rel in attention_relationship:
                        if attn_rel not in dataset_meta["relationships"]["attention"]:
                            dataset_meta["relationships"]["attention"].append(attn_rel)

                if spatial_relationship!=None:
                    for spa_rel in spatial_relationship:
                        if spa_rel not in dataset_meta["relationships"]["spatial"]:
                            dataset_meta["relationships"]["spatial"].append(spa_rel)

                if contacting_relationship!=None:
                    for cont_rel in contacting_relationship:
                        if cont_rel not in dataset_meta["relationships"]["contacting"]:
                            dataset_meta["relationships"]["contacting"].append(cont_rel)

                metadata = objAnnot["metadata"]
                data_split = metadata["set"]
                if data_split=="train":
                    set_count["train"] +=1
                    if videoid not in video_ids_by_set["train"]:
                        video_ids_by_set["train"].append(videoid)
                else:
                    set_count["test"] +=1
                    if videoid not in video_ids_by_set["test"]:
                        video_ids_by_set["test"].append(videoid)

    assert len(video_ids_by_set["train"])==len(list(set(video_ids_by_set["train"])))
    assert len(video_ids_by_set["test"])==len(list(set(video_ids_by_set["test"])))

    # prepare annotations videoid->blocks->frames->triplets
    overall_annotations = []
    for video_id in tqdm(video_ids_by_set[args.subset]):
        video_data = video_frame_data[video_id]

        frame_block_triplets = []
        for video_annotation in video_data:

            frameid, person_data,objects_annot = video_annotation

            frame_triplets = []
            for objAnnot in objects_annot:
                obj_class = objAnnot["class"]
                obj_bb =  objAnnot["bbox"]   
                metadata = objAnnot["metadata"]
                if objAnnot["visible"]:
                    attention_relationship = objAnnot["attention_relationship"]
                    spatial_relationship = objAnnot["spatial_relationship"]
                    contacting_relationship = objAnnot["contacting_relationship"]

                    for attn_rel in attention_relationship:
                        if "_" in attn_rel: attn_rel = attn_rel.replace("_", " ")
                        trip = ["person", attn_rel, obj_class]
                        frame_triplets.append(trip)

                    for spa_rel in spatial_relationship:
                        if "_" in spa_rel: spa_rel = spa_rel.replace("_", " ")
                        trip = [obj_class, spa_rel, "person"]
                        frame_triplets.append(trip)

                    for cont_rel in contacting_relationship:
                        if "_" in cont_rel: cont_rel = cont_rel.replace("_", " ")
                        trip = ["person", cont_rel, obj_class]
                        frame_triplets.append(trip)

            
            frame_block_triplets.append([frameid,frame_triplets])

        overall_annotations.append([video_id, frame_block_triplets])


    
    for video_id, video_frame_block_data in tqdm(overall_annotations):
        annotation_string = ""
        added_frame_ids = []

        video_path = os.path.join(VIDEO_ROOT_PATH,video_id)
        if not os.path.exists(video_path):
            print(f"[ERROR] video doesnt exist at: {video_path}")
            raise FileNotFoundError()

        frame_save_path = os.path.join(args.frame_dir,video_id)

        frame_ids_to_dump = []
        for frame_id, frame_triplets in video_frame_block_data:
            frame_int_idx = int(frame_id.split(".")[0])
            frame_ids_to_dump.append(frame_int_idx)
        
        
        try:
            # Submit inference requests
            submit_inference(task_queue,video_path=video_path,video_id=video_id,frame_indexes=frame_ids_to_dump,save_path=frame_save_path)
            # Allow some time for processing (adjust as needed)
            # time.sleep(1)
        except Exception as e:
            print(f"error in inference: {e} vid:{video_id}")


    
    # if so many frames are there, wait before moving to next video.
    while task_queue.qsize()>0:
        time.sleep(1)
        if task_queue.qsize()==0:
            break

    stop_worker_pool(stop_event, workers)

        




        

