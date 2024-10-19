"""
misc functions
"""


def remove_entity_index(input):
    """
    1.cow ==> cow
    """
    new_data = {}
    for key, item in input.items():
        new_data[key] = item.split(".")[-1]
    return new_data

def check_alignment(entity, alignment_list, debug=False):
    """
    if entity is found for alignment and replacement is not None, the aligned entity is returned.
    """
    if entity in alignment_list.keys():
        if alignment_list[entity]!="None":
            temp_entity = alignment_list[entity]
            if debug:
                print(f"{entity}-->{temp_entity}")
            entity = temp_entity
    return entity

def cal_triplet_acc_score_vrdFormer(data):
    """
    Checks correct predicted count out of total GT count
    """
    total = data["gt_triplets_cnt"]
    correct = data["correct_pred_triplets_cnt"]
    score = (correct*100)/total
    return score