from utils.utilities import load_AG_annotations
from utils.utilities import getConvBlock, getPromptTemplate, getRandomPrompt, get_shuffled_list, chunk_list
import json
from tqdm import tqdm
import os
import random
import argparse
from utils.utilities import SEEDS

seed_manager = SEEDS()
random.seed(seed_manager.RANDOM_ANNOTATIONS_SHUFFLE_SEED)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process AG video annotations with specified directories and chunk size.")

    parser.add_argument(
        "--video_root_path",
        type=str,
        default="/groups/sernam/datasets/ActionGenome/ActionGenome/videos",
        help="Root path to the video files."
    )
    
    parser.add_argument(
        "--output_json_dir",
        type=str,
        default="/home/jbhol/dso/gits/ActionGenome/annotations/AG_llava_annotations_v5_3",
        help="Directory to save the output JSON annotations."
    )
    
    parser.add_argument(
        "--ag_annotations_dir",
        type=str,
        default="/groups/sernam/datasets/ActionGenome/ActionGenome/annotations",
        help="Directory containing ActionGenome annotation data."
    )
    
    parser.add_argument(
        "--chunk_n",
        type=int,
        default=1000,
        help="Number of chunks for Q&A processing."
    )

    return parser.parse_args()

if __name__=="__main__":

    args = parse_arguments()
    # print("VIDEO_ROOT_PATH:", args.video_root_path)
    # print("OUTPUT_JSON_DIR:", args.output_json_dir)
    # print("AG_ANNOTATIONS_DIR:", args.ag_annotations_dir)
    # print("CHUNK_N:", args.chunk_n)
    
    VIDEO_ROOT_PATH = args.video_root_path
    OUTPUT_JSON_DIR = args.output_json_dir
    AG_ANNOTATIONS_DIR = args.ag_annotations_dir
    CHUNK_N = args.chunk_n # Q&A will be chunked into CHUNK_N parts

    os.makedirs(OUTPUT_JSON_DIR,exist_ok=True)

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

    AG_relations = {
    "attention": ['unsure', 'not looking at', 'looking at'],
    "spatial": ['in front of', 'beneath', 'behind', 'on the side of', 'in', 'above'],
    "contacting": ['not contacting', 'sitting on', 'leaning on', 'other relationship', 'holding', 'touching', 'twisting', 'eating', 
                    'drinking from', 'standing on','wearing','lying on','carrying','wiping','covered by','writing on','have it on the back']
    }

    AG_Objects = ['table','chair','bag','doorway','medicine','cup/glass/bottle','food','floor','broom','shoe','clothes','door','doorknob','groceries',
    'closet/cabinet','laptop','bed','shelf','blanket','sandwich','refrigerator','vacuum','box','light','phone/camera','dish','paper/notebook',
    'mirror','book','sofa/couch','television','pillow','towel','picture','window']

    ## possible seeds [978, 324]
    Selected_SEED = seed_manager.AG_OBJECT_PREDICATE_PARTIAL_SELECTION_SEEDS[1]
    random.seed(Selected_SEED)

    # select half of the elements
    half_length = len(AG_Objects) // 2
    # Randomly select half of the elements
    Selected_AG_Objects = random.sample(AG_Objects, half_length)
    Selected_AG_Objects.append("person")
    Selected_AG_contact_relations = random.sample(AG_relations["contacting"], len(AG_relations["contacting"])//2)

    ObjectPredicateData = {
        "selected_objects": Selected_AG_Objects,
        "selected_relations":  {
            "spatial": AG_relations["spatial"],
            "contacting": Selected_AG_contact_relations,
            "attention": AG_relations["attention"],
        },
        "skipped_objects": [],
        "skipped_relations": []
    }


    def addIf(triplet, Objects, relations):
        global ObjectPredicateData
        """
        Add triplet if entities belongs to predefined selected list.
        """
        subj, pred, obj = triplet
        if subj in Objects and obj in Objects and pred in relations:
            return True
        
        # print(f"{triplet} will be skipped")
        if subj not in ObjectPredicateData["skipped_objects"] and subj not in ObjectPredicateData["selected_objects"]:
            ObjectPredicateData["skipped_objects"].append(subj)
        if obj not in ObjectPredicateData["skipped_objects"] and obj not in ObjectPredicateData["selected_objects"]:
            ObjectPredicateData["skipped_objects"].append(obj)
        if pred not in ObjectPredicateData["skipped_relations"] and pred not in ObjectPredicateData["selected_relations"]["contacting"] \
            and pred not in ObjectPredicateData["selected_relations"]["spatial"] \
            and pred not in ObjectPredicateData["selected_relations"]["attention"]:
            ObjectPredicateData["skipped_relations"].append(pred)
        return False

    # prepare annotations videoid->blocks->frames->triplets
    overall_annotations = []
    for video_id in tqdm(video_ids_by_set["train"]):
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
                        if addIf(triplet=trip,Objects=Selected_AG_Objects, relations=Selected_AG_contact_relations+AG_relations["attention"]+AG_relations["spatial"]):
                            frame_triplets.append(trip)

                    for spa_rel in spatial_relationship:
                        if "_" in spa_rel: spa_rel = spa_rel.replace("_", " ")
                        trip = [obj_class, spa_rel, "person"]
                        if addIf(triplet=trip,Objects=Selected_AG_Objects, relations=Selected_AG_contact_relations+AG_relations["attention"]+AG_relations["spatial"]):
                            frame_triplets.append(trip)

                    for cont_rel in contacting_relationship:
                        if "_" in cont_rel: cont_rel = cont_rel.replace("_", " ")
                        trip = ["person", cont_rel, obj_class]
                        if addIf(triplet=trip,Objects=Selected_AG_Objects, relations=Selected_AG_contact_relations+AG_relations["attention"]+AG_relations["spatial"]):
                            frame_triplets.append(trip)

            if len(frame_triplets)>0:
                frame_block_triplets.append([frameid,frame_triplets])

        overall_annotations.append([video_id, frame_block_triplets])


    for video_id, video_frame_block_data in tqdm(overall_annotations):
        annotation_string = ""
        added_frame_ids = []

        video_path = os.path.join(VIDEO_ROOT_PATH,video_id)
        if not os.path.exists(video_path):
            video_path = video_id
            print(f"[ERROR] video doesnt exist at: {video_path}")
            raise FileNotFoundError()

        for frame_id, frame_triplets in video_frame_block_data:
            frame_int_idx = int(frame_id.split(".")[0])
            # print(frame_id, frame_int_idx)

            annotation_string +="#frameid"

            for triplet in frame_triplets:
                s,p,o = triplet
                annotation_string += f"[{s}:{p}:{o}];"

            added_frame_ids.append(frame_int_idx)

            if len(added_frame_ids)>=8:
                # print(added_frame_ids, annotation_string)

                PromptAnswer = getPromptTemplate(media_path=video_path,media_type="video")

                add_video_token = True

                AG_Prompt = getRandomPrompt(key='AG_Prompt', static=True)
                AG_Prompt = AG_Prompt.replace("{objects_list}",  ",".join(get_shuffled_list(Selected_AG_Objects)) )
                AG_Prompt = AG_Prompt.replace("{spatial_relations}", ",".join(get_shuffled_list(AG_relations["spatial"])))
                AG_Prompt = AG_Prompt.replace("{contacting_relations}", ",".join(get_shuffled_list(Selected_AG_contact_relations)))
                AG_Prompt = AG_Prompt.replace("{attention_relations}", ",".join(get_shuffled_list(AG_relations["attention"])))

                convQ = getConvBlock(value=AG_Prompt, 
                                conv_type="human", media_type="<video>", 
                                add_media_token=add_video_token)
                
                if add_video_token:
                    add_video_token = False

                
                convA = getConvBlock(value=annotation_string+"#sg_end", 
                            conv_type="gpt", media_type="<video>", 
                            add_media_token=False)
            
                PromptAnswer["conversations"].append(convQ)
                PromptAnswer["conversations"].append(convA)

                PromptAnswer["frame_indices"] =  added_frame_ids
                # PromptAnswer["total_frames"] = total_frames
                PromptAnswer["id"] = Annotation_counter
                json_annotations.append(PromptAnswer)

                Annotation_counter+=1

                annotation_string = ""
                added_frame_ids = []

    

    #SET SHUFFLE SEED
    random.seed(seed_manager.RANDOM_ANNOTATIONS_SHUFFLE_SEED) 

    random.shuffle(json_annotations) # shuffle to get all videos shuffled
    chunked_list_gen = chunk_list(list_=json_annotations,chunk_n=CHUNK_N)

    json_file_counter = 0
    OUTPUT_JSON_DIR = os.path.join(OUTPUT_JSON_DIR,str(Selected_SEED))
    os.makedirs(OUTPUT_JSON_DIR,exist_ok=True)
    for chunked_annotation_list in chunked_list_gen:
        random.shuffle(chunked_annotation_list) # reshuffle to get all videos shuffled again
        JSON_videochatgpt_tune_validate = f"{OUTPUT_JSON_DIR}/videochatgpt_randomseed_{Selected_SEED}_tune_AG_part{json_file_counter}.json"
        with open(JSON_videochatgpt_tune_validate, "w") as f:
            json.dump(chunked_annotation_list,f, indent=4)
        json_file_counter +=1
        print(f"Saved: {JSON_videochatgpt_tune_validate}")

    try:
        with open(f"{OUTPUT_JSON_DIR}/meta_data.json", "w") as f:
            json.dump(ObjectPredicateData,f, indent=4)
    except Exception as e:
        print(f"error saving : {OUTPUT_JSON_DIR}/meta_data.json")
        pass


        









        

    


