from utils.misc import check_alignment
from utils.utilities import eval_tagging_scores, calculate_accuracy_varying_lengths,remove_ids
import copy
import numpy as np
from tqdm import tqdm

DEBUG = False

def get_frame_eval(vid_id,block_id,frames,GT_Triplets,Pred_Triplets, 
                   sg_eval_counts,all_triplets_pairs,
                   dataset_subjects, dataset_objects, dataset_predicates,
                   check_alinged=False,alinged_subjects=None,alinged_predicates=None,alinged_objects=None):
    
    new_subjects = []
    new_objects = []
    new_predicates = []

    Block_GT_triplets_woids = GT_Triplets
    Block_predicated_triplets_woids = Pred_Triplets


    frame_metric = {
        "subject": {"precision": [], "recall": []},
        "object": {"precision": [], "recall": []},
        "predicate": {"precision": [], "recall": []},
        "triplet": {"precision": [], "recall": []}
    }
    for fidx, GT_tripdata in enumerate(Block_GT_triplets_woids):
        results = None

        if fidx not in all_triplets_pairs[vid_id][f"Block{block_id}"].keys():
            all_triplets_pairs[vid_id][f"Block{block_id}"][fidx] = {
                "gt_triplets": [],
                "pred_triplets":[]
            }


        frame_GT_triplets = GT_tripdata
        frame_pred_triplets = []

        try:frame_pred_triplets = Block_predicated_triplets_woids[fidx]
        except Exception as e:
            pass

        gt_all = {"triplet": [],"subject": [],"object": [],"predicate": []}
        pred_all = {"triplet": [],"subject": [],"object": [],"predicate": []}

        for fgt in frame_GT_triplets:
            fgt_s, fgt_p, fgt_o = fgt  # v3_1 changes
            gt_all["triplet"].append({"triplet": fgt, "score": 1.0})
            gt_all["subject"].append({"triplet": fgt_s, "score": 1.0})
            gt_all["predicate"].append({"triplet": fgt_p, "score": 1.0})
            gt_all["object"].append({"triplet": fgt_o, "score": 1.0})

            all_triplets_pairs[vid_id][f"Block{block_id}"][fidx]["gt_triplets"].append(fgt)

        for fpred_idx, fpred in enumerate(frame_pred_triplets):
            fpred_s, fpred_p, fpred_o  = fpred # v3_1 changes
            # print(fpred)

            if check_alinged:
                fpred_s = check_alignment(entity=fpred_s, alignment_list=alinged_subjects,debug=DEBUG)
                fpred_p = check_alignment(entity=fpred_p, alignment_list=alinged_predicates,debug=DEBUG)
                fpred_o = check_alignment(entity=fpred_o, alignment_list=alinged_objects,debug=DEBUG)
                frame_pred_triplets[fpred_idx] = [fpred_s, fpred_p, fpred_o]

            pred_all["subject"].append({"triplet": fpred_s, "score": 1.0})
            pred_all["predicate"].append({"triplet": fpred_p, "score": 1.0})
            pred_all["object"].append({"triplet": fpred_o, "score": 1.0})
            pred_all["triplet"].append({"triplet": [fpred_s, fpred_p, fpred_o], "score": 1.0})

            all_triplets_pairs[vid_id][f"Block{block_id}"][fidx]["pred_triplets"].append([fpred_s, fpred_p, fpred_o])

            if fpred_s not in dataset_subjects:
                if fpred_s not in new_subjects:
                    new_subjects.append(fpred_s)

            if fpred_p not in dataset_predicates:
                if fpred_p not in new_predicates:
                    new_predicates.append(fpred_p)

            if fpred_o not in dataset_objects:
                if fpred_o not in new_objects:
                    new_objects.append(fpred_o)
        
        # all_triplets_pairs.append([frame_GT_triplets,frame_pred_triplets,vid_id,block_id])
        for fm_key, fmdata in frame_metric.items():
            prec, rec, hit_scores = eval_tagging_scores(gt_relations=gt_all[fm_key],pred_relations=pred_all[fm_key],min_pred_num=1)
            frame_metric[fm_key]["precision"].append(prec)
            frame_metric[fm_key]["recall"].append(rec)
        # print(frame_GT_triplets,"<===>",frame_pred_triplets)


        if len(GT_tripdata)>0 and len(frame_pred_triplets)>0:
            try:
                results = calculate_accuracy_varying_lengths(gt_triplets=GT_tripdata,pred_triplets=frame_pred_triplets, remove_duplicates=False)
            except Exception as e:
                print(f"error calculating score for vid {vid_id} block:{block_id} fidx {fidx} actual_fidx: {frames[fidx]}")

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

    

    return frame_metric, sg_eval_counts, all_triplets_pairs, [new_subjects,new_predicates,new_objects]


def get_block_eval(vid_data,vid_id,all_triplets_pairs,sg_eval_counts,
                   dataset_subjects, dataset_objects,dataset_predicates,check_alinged=False,
                   alinged_subjects=None,alinged_predicates=None,alinged_objects=None):

    new_subjects = []
    new_objects = []
    new_predicates = []
    
    block_metric = {
        "subject": {"precision": [], "recall": []},
        "object": {"precision": [], "recall": []},
        "predicate": {"precision": [], "recall": []},
        "triplet": {"precision": [], "recall": []}
    }
    for block_id, block_data in vid_data.items():
        if f"Block{block_id}" not in all_triplets_pairs[vid_id].keys():
            all_triplets_pairs[vid_id][f"Block{block_id}"] = {}

        pred_triplets = copy.deepcopy(block_data["triplets"])
        frame_GT_triplets = copy.deepcopy(block_data["GT_triplets"])
        frames = block_data["frames"]
        all_triplets_pairs[vid_id][f"Block{block_id}"]["frames"] = str(frames)

        try:
            Block_GT_triplets_woids = remove_ids(frame_GT_triplets,version="v2_1", remove_indexes=True)
            Block_predicated_triplets_woids = remove_ids(pred_triplets,version="v2_1", remove_indexes=True)
        except Exception as e:
            pass

        TRIPLET_DATA = {
            'GT_Triplets': Block_GT_triplets_woids,
            'Pred_Triplets': Block_predicated_triplets_woids,
            'all_triplets_pairs': all_triplets_pairs
        }

        ALIGNMENT_DATA = {
            'check_alinged': check_alinged,
            'alinged_subjects': alinged_subjects,
            'alinged_objects': alinged_objects,
            'alinged_predicates': alinged_predicates
        }

        DATASET_DATA = {
            "dataset_subjects": dataset_subjects,
            "dataset_objects": dataset_objects,
            "dataset_predicates": dataset_predicates
        }

        EVAL_DATA = {'sg_eval_counts': sg_eval_counts}

        frame_metric, sg_eval_counts, all_triplets_pairs, new_entities = get_frame_eval(vid_id=vid_id,
                        block_id=block_id,frames=frames,**DATASET_DATA,**EVAL_DATA,**TRIPLET_DATA,**ALIGNMENT_DATA)
        
        [new_subs,new_preds,new_objs] = new_entities
        new_objects = list(set(new_objs+new_objects))
        new_predicates = list(set(new_predicates+new_preds)) 
        new_subjects = list(set(new_subjects+new_subs))
        


        for bm_key, bmdata in block_metric.items():
            block_metric[bm_key]["precision"].append(np.average(np.array(frame_metric[bm_key]["precision"], dtype=np.float32)))
            block_metric[bm_key]["recall"].append(np.average(np.array(frame_metric[bm_key]["recall"], dtype=np.float32)))

    return block_metric,sg_eval_counts, all_triplets_pairs, [new_subjects,new_predicates,new_objects]
 

def eval_pred_data(data, dataset_subjects, dataset_objects, dataset_predicates,
                   check_alinged=False,alinged_subjects=None,alinged_objects=None,alinged_predicates=None):
    new_subjects = []
    new_objects = []
    new_predicates = []

    pred_data = data

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

    overall_metric = {
        "subject": {"precision": [], "recall": []},
        "object": {"precision": [], "recall": []},
        "predicate": {"precision": [], "recall": []},
        "triplet": {"precision": [], "recall": []} 
    }
    all_triplets_pairs  = {}
    
    for vid_id,vid_data in tqdm(pred_data.items(),total=len(pred_data.keys())):

        if vid_id not in all_triplets_pairs.keys():
            all_triplets_pairs[vid_id] = {}


        ALIGNMENT_DATA = {
            'check_alinged': check_alinged,
            'alinged_subjects': alinged_subjects,
            'alinged_objects': alinged_objects,
            'alinged_predicates': alinged_predicates
        }
        TRIPLET_DATA = {'all_triplets_pairs': all_triplets_pairs}
        EVAL_DATA = {'sg_eval_counts': sg_eval_counts}

        DATASET_DATA = {
            "dataset_subjects": dataset_subjects,
            "dataset_objects": dataset_objects,
            "dataset_predicates": dataset_predicates
        }

        block_metric,sg_eval_counts,all_triplets_pairs, new_entities = get_block_eval(vid_data=vid_data,
                                                                                        vid_id=vid_id,
                                                                                        **DATASET_DATA,
                                                                                        **EVAL_DATA,
                                                                                        **TRIPLET_DATA,
                                                                                        **ALIGNMENT_DATA
                                                                                        )
        [new_subs,new_preds,new_objs] = new_entities
        new_objects = list(set(new_objs+new_objects))
        new_predicates = list(set(new_predicates+new_preds)) 
        new_subjects = list(set(new_subjects+new_subs))
        
        for oam_key, oamdata in overall_metric.items():
            overall_metric[oam_key]["precision"].append(np.average(np.array(block_metric[oam_key]["precision"], dtype=np.float32)))
            overall_metric[oam_key]["recall"].append(np.average(np.array(block_metric[oam_key]["recall"], dtype=np.float32)))

    sg_eval_counts["VRDFormer_Logic"] = {}
    total_vid_ids = len(overall_metric["triplet"]["precision"])
    for metric_key, metric_values in overall_metric.items():
        if metric_key not in sg_eval_counts["VRDFormer_Logic"].keys():
            sg_eval_counts["VRDFormer_Logic"][metric_key] = {}
        overall_precision = np.average(np.array(overall_metric[metric_key]["precision"], dtype=np.float32))
        overall_recall = np.average(np.array(overall_metric[metric_key]["recall"], dtype=np.float32))
        sg_eval_counts["VRDFormer_Logic"][metric_key] = {
            "Precision@1": overall_precision,
            "Recall@1": overall_recall,
        }
    return sg_eval_counts, all_triplets_pairs, [new_subjects,new_predicates,new_objects]