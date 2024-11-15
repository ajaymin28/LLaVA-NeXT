import os
import json
import glob
import numpy as np
import time
from utils.utilities import eval_tagging_scores, eval_tagging_scores2
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
    parser.add_argument("--subset_json", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--top_k_list", type=str, default=[1,5,10, 20, 50])
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

    
    VIDEO_ROOT_PATH = "/root/datasets/ActionGenome/videos"
    AG_ANNOTATIONS_DIR = "/root/datasets/ActionGenome/annotations"
    AG_Annotations,dataset_meta,video_frame_data = get_AG_annotations_framewise(AG_ANNOTATIONS_DIR=AG_ANNOTATIONS_DIR, 
                                                                                subset=splits[0])

    gt_annotations = {k: {int(vv[0].split(".")[0]): vv[1:] for vv in v} for k,v in AG_Annotations}

    if os.path.isdir(args.result_json):
        predicted_results = {}
        for file in [f for f in os.listdir(args.result_json) if f.endswith(".json") and "bbox" in f]:
            predicted_results.update(json.load(open((os.path.join(args.result_json, file)))))
    else:
        predicted_results = json.load(open(args.result_json))

    if args.subset_json is not None:
        subset_video_ids = json.load(open(args.subset_json))

        predicted_results = {k:v for k,v in predicted_results.items() if k in subset_video_ids}

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

    pred_count_at_k = {
        k: [0] * len(AG_relationsCombined) for k in args.top_k_list
    }

    pred_pred_hits_at_k = {
        k: [0] * len(AG_relationsCombined) for k in args.top_k_list
    }

    pred_pred_fp_at_k = {
        k: [0] * len(AG_relationsCombined) for k in args.top_k_list
    }


    for video_name, pred_annotation in tqdm(predicted_results.items()):
        gt_video_annotation = gt_annotations[video_name]

        block_metric = {'sgcls': get_fresh_metric(), 'sgdet': get_fresh_metric()}

        for block_id, pred_block in pred_annotation.items():

            frame_metric = {'sgcls': get_fresh_metric(), 'sgdet': get_fresh_metric()}

            for frame_id, pred_frame_triplets in zip(pred_block['frames'], pred_block['triplets']):

                if frame_id not in gt_video_annotation or len(pred_frame_triplets)==0:
                    continue
                gt_frame = gt_video_annotation[frame_id]

                gt_relations = [] # {"triplet": ['adult', 'sitting', 'sofa'], "score": 1.0},
                pred_relations = [] # {"triplet": ['adult', 'sitting', 'sofa'], "score": 1.0},
                gt_all = {"triplet": [],"subject": [],"object": [],"predicate": []}
                pred_all = {"triplet": [],"subject": [],"object": [],"predicate": []}

                for fgt, fgt_bb in zip(gt_frame[0], gt_frame[1]):   # [[s, r ,o], [s2, r2, o2]...], [[s_bb, o_bb], [s2_bb, o2_bb]...]
                    fgt_s, fgt_p, fgt_o = fgt  # v3_1 changes
                    # fgt_s, fgt_o = (fgt_s, tuple(fgt_bb[0])), (fgt_o, tuple(fgt_bb[1]))
                    gt_all["triplet"].append({"triplet": [fgt_s, fgt_p, fgt_o], "score": 1.0})
                    gt_all["subject"].append({"triplet": fgt_s, "score": 1.0})
                    gt_all["predicate"].append({"triplet": fgt_p, "score": 1.0})
                    gt_all["object"].append({"triplet": fgt_o, "score": 1.0})
                    for k in args.top_k_list:
                        pred_count_at_k[k][AG_relationsCombined.index(fgt_p)] += 1

                for frame_triplet_idx, fpred in enumerate(pred_frame_triplets):   # [[[s, s', bb, score], r, [o, o', bb, score]], [[s, s', bb, score], r, [o, o', bb, score]] ... ]
                    fpred_s, fpred_p, fpred_o  = fpred # v3_1 changes
                    
                    if fpred_s not in AG_Objects:
                        novel_pred["subjects"].add(fpred_s)
                    if fpred_p not in AG_relationsCombined:
                        novel_pred["predicates"].add(fpred_p)
                    if fpred_o not in AG_Objects:
                        novel_pred["objects"].add(fpred_o)

                    fpred_s, fpred_o = AG_OBJECTS_ALTERATIVES.get(fpred_s, fpred_s), AG_OBJECTS_ALTERATIVES.get(fpred_o, fpred_o)
            

                    pred_all["triplet"].append({"triplet": [fpred_s, fpred_p, fpred_o], "score": 1.0})
                    pred_all["subject"].append({"triplet": fpred_s, "score": 1.0})
                    pred_all["predicate"].append({"triplet": fpred_p, "score": 1.0})
                    pred_all["object"].append({"triplet": fpred_o, "score": 1.0})

                for k, _ in frame_metric['sgcls'].items():
                    """
                    Eval score for each frame
                    """
                    prec_sgcls, rec_sgcls, hit_scores_sgcls = eval_tagging_scores2(gt_relations=gt_all[k],pred_relations=pred_all[k],min_pred_num=1, pred_pred_hits_at_k=pred_pred_hits_at_k, pred_pred_fp_at_k=pred_pred_fp_at_k)
                    frame_metric['sgcls'][k]["precision"].append(prec_sgcls)
                    frame_metric['sgcls'][k]["recall"].append(rec_sgcls)

                
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
         
        for k in overall_metric['sgcls'].keys():
            """
                    average eval score for each block and appned it to overall
            """
            if len(block_metric['sgcls'][k]["precision"])>0 and len(block_metric['sgcls'][k]["recall"])>0:
                overall_metric['sgcls'][k]["precision"].append(round(float(np.average(np.array(block_metric['sgcls'][k]['precision']))), 4))
                overall_metric['sgcls'][k]["recall"].append(round(float(np.average(np.array(block_metric['sgcls'][k]['recall']))), 4))

        sg_eval_counts["VRDFormer_Logic"] = {'sgcls': {}, 'sgdet': {}}
        total_vid_ids = len(overall_metric['sgcls']["triplet"]["precision"])
        for k in overall_metric['sgcls'].keys():
            if k not in sg_eval_counts["VRDFormer_Logic"]['sgcls']:
                sg_eval_counts["VRDFormer_Logic"]['sgcls'][k] = {}
            
            if len(overall_metric['sgcls'][k]["precision"])>0 and len(overall_metric['sgcls'][k]["recall"])>0:
                overall_precision_sgcls = np.average(np.array(overall_metric['sgcls'][k]["precision"]))
                overall_recall_sgcls = np.average(np.array(overall_metric['sgcls'][k]["recall"]))

                sg_eval_counts["VRDFormer_Logic"]['sgcls'][k] = {
                    "Precision@1": round(float(overall_precision_sgcls), 4),
                    "Recall@1": round(float(overall_recall_sgcls), 4),
                }
        sg_eval_counts["VRDFormer_Logic"]["TotalVideos"] = total_vid_ids

    try:
        sg_eval_counts["dataset_meta"] ={
            # "dataset_triplets_existing": GtData,
            "dataset_triplets_new": {k: list(v) for k,v in novel_pred.items()}
        }

    except Exception as e:
        pass

    per_class_recall = {}

    for k, v in pred_pred_hits_at_k.items():
        avg = 0
        per_class_recall[k] = {}

        for idx in range(len(AG_relationsCombined)):
            tmp_avg = float(pred_pred_hits_at_k[k][idx]) / float(pred_count_at_k[k][idx] + 1e-10)

            avg += tmp_avg
            per_class_recall[k][AG_relationsCombined[idx]]= tmp_avg
        if "mrecall" not in sg_eval_counts["VRDFormer_Logic"]["sgcls"]:
            sg_eval_counts["VRDFormer_Logic"]["sgcls"]['mrecall'] = {}
        sg_eval_counts["VRDFormer_Logic"]["sgcls"]['mrecall'][k] = avg/len(AG_relationsCombined)

    per_class_precision = {}
    for k, v in pred_pred_hits_at_k.items():
        avg = 0
        per_class_precision[k] = {}
        for idx in range(len(AG_relationsCombined)):
            tmp_avg = float(pred_pred_hits_at_k[k][idx]) / float(float(pred_pred_hits_at_k[k][idx]) + float(pred_pred_fp_at_k[k][idx]) + 1e-10)

            avg += tmp_avg
            per_class_precision[k][AG_relationsCombined[idx]]= tmp_avg
        if "mprecision" not in sg_eval_counts["VRDFormer_Logic"]["sgcls"]:
            sg_eval_counts["VRDFormer_Logic"]["sgcls"]['mprecision'] = {}
        sg_eval_counts["VRDFormer_Logic"]["sgcls"]['mprecision'][k] = avg/len(AG_relationsCombined)

    sg_eval_counts["VRDFormer_Logic"]["sgcls"]['per_class_recall'] = per_class_recall
    sg_eval_counts["VRDFormer_Logic"]["sgcls"]['per_class_precision'] = per_class_precision

    try:
        outputfile = f"{inference_output_dir}/{args.output_path}"
        with open(outputfile, "w") as f:
            json.dump(sg_eval_counts,f, indent=4)
    except Exception as e:
        print(f"error saving file: {e}")



