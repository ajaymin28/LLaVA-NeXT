import os
import json
import glob
import numpy as np
import time
from utils.utilities import eval_tagging_scores_with_bb
from utils.utilities import calculate_accuracy_varying_lengths, remove_ids
from utils.utilities import getRandomPrompt, SGSpecialTokens
from utils.utilities import get_AG_annotations_framewise, get_shuffled_list
from utils.utilities import AG_Objects,AG_relations, AG_OBJECTS_ALTERATIVES
import argparse
import os
from tqdm import tqdm
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def get_fresh_metric():
    return {
        "subject": {"precision": [], "recall": []},
        "object": {"precision": [], "recall": []},
        "predicate": {"precision": [], "recall": []},
        "triplet": {"precision": [], "recall": []} 
    }

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--result_json", type=str, required=True)
    # parser.add_argument("--prev_eval_data", type=str, default="", required=False)
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    print(args)


    sg_eval_counts = {
        "total_obj_cnt" : 0,
        "total_pred_cnt" : 0,
        "total_sub_cnt" : 0,
        "correct_obj_pred_cnt" : 0,
        "correct_subj_pred_cnt" : 0,
        "correct_predicate_cnt" : 0,
        "gt_triplets_cnt": 0,
        "pred_triplets_cnt": 0,
        "correct_pred_triplets_cnt": 0,
        "total_predicted_triplets": 0
    }

    GtData = {
        "subjects": [],
        "objects": [],
        "predicates": []
    }

    novel_pred = {
        "subjects": set(),
        "predicates": set(),
        "objects": set()
    }

    dataset_name = "ActionGnome"
    dataset_name_to_save = dataset_name

    splits = ["test"]
    VIDEO_ROOT_PATH = "/groups/sernam/datasets/ActionGenome/ActionGenome/videos"
    AG_ANNOTATIONS_DIR = "/groups/sernam/datasets/ActionGenome/ActionGenome/annotations"
    AG_Annotations,dataset_meta,video_frame_data = get_AG_annotations_framewise(AG_ANNOTATIONS_DIR=AG_ANNOTATIONS_DIR, 
                                                                                subset=splits[0])

    gt_annotations = {k: {int(vv[0].split(".")[0]): vv[1:] for vv in v} for k,v in AG_Annotations}

    if os.path.isdir(args.result_json):
        predicted_results = {}
        for file in [f for f in os.listdir(args.result_json) if f.endswith(".json") and "bbox" in f]:
            predicted_results.update(json.load(open((os.path.join(args.result_json, file)))))
    else:
        predicted_results = json.load(open(args.result_json))
    inference_output_dir =  os.path.dirname(args.result_json)
    inference_prog_output_dir  = f"{inference_output_dir}/prog" 
    os.makedirs(inference_output_dir,exist_ok=True)
    os.makedirs(inference_prog_output_dir,exist_ok=True)

    sg_eval_counts["subsets"] = splits

    AG_relationsCombined = AG_relations["attention"]+AG_relations["spatial"]+AG_relations["contacting"]

    llava_response_json = {}
    llava_raw_response_json = {}
    frame_block = 0

    overall_metric = {'sgcls': get_fresh_metric(), 'sgdet': get_fresh_metric()}

    for video_name, pred_annotation in tqdm(predicted_results.items()):
        gt_video_annotation = gt_annotations[video_name]

        block_metric = {'sgcls': get_fresh_metric(), 'sgdet': get_fresh_metric()}

        for block_id, pred_block in pred_annotation.items():

            frame_metric = {'sgcls': get_fresh_metric(), 'sgdet': get_fresh_metric()}

            for frame_id, pred_frame_triplets in zip(pred_block['frames'], pred_block['triplets']):

                gt_frame = gt_video_annotation[frame_id]

                gt_relations = [] # {"triplet": ['adult', 'sitting', 'sofa'], "score": 1.0},
                pred_relations = [] # {"triplet": ['adult', 'sitting', 'sofa'], "score": 1.0},
                gt_all = {"triplet": [],"subject": [],"object": [],"predicate": []}
                pred_all = {"triplet": [],"subject": [],"object": [],"predicate": []}

                for fgt, fgt_bb in zip(gt_frame[0], gt_frame[1]):   # [[s, r ,o], [s2, r2, o2]...], [[s_bb, o_bb], [s2_bb, o2_bb]...]
                    fgt_s, fgt_p, fgt_o = fgt  # v3_1 changes
                    fgt_s, fgt_o = (fgt_s, tuple(fgt_bb[0])), (fgt_o, tuple(fgt_bb[1]))
                    gt_all["triplet"].append({"triplet": [fgt_s, fgt_p, fgt_o], "score": 1.0})
                    gt_all["subject"].append({"triplet": fgt_s, "score": 1.0})
                    gt_all["predicate"].append({"triplet": fgt_p, "score": 1.0})
                    gt_all["object"].append({"triplet": fgt_o, "score": 1.0})

                for fpred in pred_frame_triplets:   # [[[s, s', bb, score], r, [o, o', bb, score]], [[s, s', bb, score], r, [o, o', bb, score]] ... ]
                    fpred_s, fpred_p, fpred_o  = fpred # v3_1 changes
                    
                    if fpred_s[0] not in AG_Objects:
                        novel_pred["subjects"].add(fpred_s[0])
                    if fpred_p not in AG_relationsCombined:
                        novel_pred["predicates"].add(fpred_p)
                    if fpred_o[0] not in AG_Objects:
                        novel_pred["objects"].add(fpred_o[0])

                    fpred_s, fpred_o = [AG_OBJECTS_ALTERATIVES.get(fpred_s[0], fpred_s[0]), tuple(fpred_s[2])], [AG_OBJECTS_ALTERATIVES.get(fpred_o[0], fpred_o[0]), tuple(fpred_o[2])]

                    pred_all["triplet"].append({"triplet": [fpred_s, fpred_p, fpred_o], "score": 1.0})
                    pred_all["subject"].append({"triplet": fpred_s, "score": 1.0})
                    pred_all["predicate"].append({"triplet": fpred_p, "score": 1.0})
                    pred_all["object"].append({"triplet": fpred_o, "score": 1.0})

                for k, _ in frame_metric['sgcls'].items():
                    """
                    Eval score for each frame
                    """
                    (prec_sgcls, rec_sgcls, hit_scores_sgcls), (prec_sgdet, rec_sgdet, hit_scores_sgdet) = eval_tagging_scores_with_bb(gt_relations=gt_all[k],pred_relations=pred_all[k],min_pred_num=1, mode=k)
                    frame_metric['sgcls'][k]["precision"].append(prec_sgcls)
                    frame_metric['sgcls'][k]["recall"].append(rec_sgcls)
                    frame_metric['sgdet'][k]["precision"].append(prec_sgdet)
                    frame_metric['sgdet'][k]["recall"].append(rec_sgdet)
                
                if len(gt_frame[0])>0 and len(pred_frame_triplets)>0:
                    try:
                        results = calculate_accuracy_varying_lengths(gt_triplets=gt_frame[0],pred_triplets=[(x[0][0], x[1], x[2][0]) for x in pred_frame_triplets], remove_duplicates=False)
                    except Exception as e:
                        print(f"error calculating score for vid {video_name} block:{block_id} fidx {frame_id}")

                    if results is not None:
                        sg_eval_counts["correct_pred_triplets_cnt"] +=  results["correct_triplet_cnt"]
                        sg_eval_counts["correct_obj_pred_cnt"] += results["correct_object_cnt"]
                        sg_eval_counts["correct_subj_pred_cnt"] +=  results["correct_subject_cnt"]
                        sg_eval_counts["correct_predicate_cnt"] +=  results["correct_predicate_cnt"]
                        sg_eval_counts["gt_triplets_cnt"] +=  results["total_triplets"]
                        sg_eval_counts["total_predicted_triplets"] += results["total_predicted_triplets"]
                        sg_eval_counts["total_obj_cnt"] +=  results["total_objects"]
                        sg_eval_counts["total_sub_cnt"] +=  results["total_subjects"]
                        sg_eval_counts["total_pred_cnt"] +=  results["total_predicates"] 
                else:
                    pass
                    # print(f"vid {video_id} block:{frame_block_index} fidx {fidx} actual_fidx:{Block_frame_ids[fidx]} lengt: {len(GT_tripdata)} lenpred: {frame_pred_triplets} outputs: {outputs}, unclean: {outputs_unclean}")

            for k in block_metric['sgcls'].keys():
                """
                    average eval score for each frame and appned it to block
                """
                if len(frame_metric['sgcls'][k]["precision"])>0 and len(frame_metric['sgcls'][k]["recall"])>0:
                    block_metric['sgcls'][k]["precision"].append(np.average(np.array(frame_metric['sgcls'][k]['precision'])))
                    block_metric['sgcls'][k]["recall"].append(np.average(np.array(frame_metric['sgcls'][k]['recall'])))
                    block_metric['sgdet'][k]["precision"].append(np.average(np.array(frame_metric['sgdet'][k]['precision'])))
                    block_metric['sgdet'][k]["recall"].append(np.average(np.array(frame_metric['sgdet'][k]['recall'])))
         
        for k in overall_metric['sgcls'].keys():
            """
                    average eval score for each block and appned it to overall
            """
            if len(block_metric['sgcls'][k]["precision"])>0 and len(block_metric['sgcls'][k]["recall"])>0:
                overall_metric['sgcls'][k]["precision"].append(round(float(np.average(np.array(block_metric['sgcls'][k]['precision']))), 4))
                overall_metric['sgcls'][k]["recall"].append(round(float(np.average(np.array(block_metric['sgcls'][k]['recall']))), 4))
                overall_metric['sgdet'][k]["precision"].append(round(float(np.average(np.array(block_metric['sgdet'][k]['precision']))), 4))
                overall_metric['sgdet'][k]["recall"].append(round(float(np.average(np.array(block_metric['sgdet'][k]['recall']))), 4))

        sg_eval_counts["VRDFormer_Logic"] = {'sgcls': {}, 'sgdet': {}}
        total_vid_ids = len(overall_metric['sgcls']["triplet"]["precision"])
        for k in overall_metric['sgcls'].keys():
            if k not in sg_eval_counts["VRDFormer_Logic"]['sgcls']:
                sg_eval_counts["VRDFormer_Logic"]['sgcls'][k] = {}
                sg_eval_counts["VRDFormer_Logic"]['sgdet'][k] = {}
            
            if len(overall_metric['sgcls'][k]["precision"])>0 and len(overall_metric['sgcls'][k]["recall"])>0:
                overall_precision_sgcls = np.average(np.array(overall_metric['sgcls'][k]["precision"]))
                overall_recall_sgcls = np.average(np.array(overall_metric['sgcls'][k]["recall"]))
                overall_precision_sgdet = np.average(np.array(overall_metric['sgdet'][k]["precision"]))
                overall_recall_sgdet = np.average(np.array(overall_metric['sgdet'][k]["recall"]))
                sg_eval_counts["VRDFormer_Logic"]['sgcls'][k] = {
                    "Precision@1": round(float(overall_precision_sgcls), 4),
                    "Recall@1": round(float(overall_recall_sgcls), 4),
                }
                sg_eval_counts["VRDFormer_Logic"]['sgdet'][k] = {
                    "Precision@1": round(float(overall_precision_sgdet), 4),
                    "Recall@1": round(float(overall_recall_sgdet), 4),
                }
        sg_eval_counts["VRDFormer_Logic"]["TotalVideos"] = total_vid_ids

    try:
        sg_eval_counts["dataset_meta"] ={
            # "dataset_triplets_existing": GtData,
            "dataset_triplets_new": {k: list(v) for k,v in novel_pred.items()}
        }
    except Exception as e:
        pass

    try:
        outputfile = f"{inference_output_dir}/{dataset_name_to_save}_results_eval_data_with_bb.json"
        with open(outputfile, "w") as f:
            json.dump(sg_eval_counts,f, indent=4)
    except Exception as e:
        print(f"error saving file: {e}")



